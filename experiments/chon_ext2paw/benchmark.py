"""Reproducible structure-level CHON ext2paw sweep; test only the winner.

Run prepare.py first. Every trial uses the same split and independently fitted
training transforms. Arm checkpoints select on their training objective; arms
are ranked using a COMMON physical density error + 3 * balanced PAW loss.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from poraque.data.cache import load_paw_profiles
from poraque.fields.atomic import resolve_library
from poraque.fields.vasp.augmentation import irreps_blocks, record_layout
from poraque.ml import FieldPairDataset, save_bundle, train
from poraque.ml.config import TrainingConfig
from poraque.ml.data import collate_fields, make_dataloader
from poraque.ml.training import validate_occupancies
from poraque.ml.tasks import resolve_task
from poraque_train import build_operator, cache_tag

ARMS = {
    "baseline": {},
    "lr_low": {"learning_rate": 0.0003},
    "lr_high": {"learning_rate": 0.003},
    "width32": {"width": 32},
    "width64": {"width": 64},
    "layers5": {"n_layers": 5},
    "modes10": {"modes": 10},
    "radial16": {"n_radial": 16},
    "occ1": {"occupancy_weight": 1.0},
    "absolute": {"delta_density": False},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def split_records(materials, seed):
    """Stratify by chemical system; tiny groups remain in training."""
    from poraque.fields.vasp.volumetric import read_volumetric

    groups = defaultdict(list)
    for record in sorted(materials, key=lambda r: r.identifier):
        structure = read_volumetric(record.files["CHGCAR"])[0]
        groups["-".join(sorted(set(structure.elements)))].append(record.identifier)
    rng = np.random.default_rng(seed)
    split = {k: [] for k in ("train", "validation", "test")}
    for key in sorted(groups):
        names = list(rng.permutation(groups[key]))
        n = len(names)
        n_test = max(1, round(n * 0.1)) if n >= 5 else 0
        n_val = max(1, round(n * 0.2)) if n >= 3 else 0
        split["test"].extend(names[:n_test])
        split["validation"].extend(names[n_test:n_test + n_val])
        split["train"].extend(names[n_test + n_val:])
    return {k: sorted(v) for k, v in split.items()}


def datasets(cache, config, split, records, baseline):
    subsets = {}
    for key in ("train", "validation", "test"):
        names = set(split[key])
        subsets[key] = FieldPairDataset(
            cache, "ext2paw", materials=[r for r in records if r.identifier in names],
            baseline=baseline, cache=True, spin="auto")
    source, target = subsets["train"].fit_transforms(seed=config.training.seed)
    for key in ("validation", "test"):
        subsets[key].input_transform = source
        subsets[key].target_transform = target
    return subsets


def audit(dataset, profiles):
    layouts = {z: p["projector_channels"] for z, p in profiles.items()}
    atoms, shapes, lengths = Counter(), Counter(), defaultdict(set)
    charge_norm = moment_norm = 0.0
    negative = points = 0
    for i in range(len(dataset)):
        sites = dataset.site_targets(i)
        for z, n in zip(sites["species"], sites["lengths"]):
            z, n = int(z), int(n)
            assert n == len(record_layout(layouts[z])), (dataset.materials[i].identifier, z, n)
            atoms[str(z)] += 1
            lengths[str(z)].add(n)
        field = dataset.load_fields(i)[1].data
        charge = field[0] if field.ndim == 4 else field
        charge_norm += float(np.square(charge).sum())
        if field.ndim == 4:
            moment_norm += float(np.square(field[1]).sum())
        negative += int((charge < 0).sum())
        points += charge.size
        shapes[str(charge.shape)] += 1
    return {"materials": len(dataset), "atoms_per_element_Z": dict(atoms),
            "lengths": {k: sorted(v) for k, v in lengths.items()},
            "grid_shapes": dict(shapes), "negative_grid_fraction": negative / points,
            "magnetisation_to_charge_rms": float(np.sqrt(moment_norm / charge_norm)),
            "POTCAR_titles": {str(z): p["titel"] for z, p in profiles.items()}}


@torch.no_grad()
def measure(operator, dataset):
    """Physical units, whole structures and pooled PAW blocks per element."""
    operator.model.eval()
    parts, per_material = defaultdict(lambda: np.zeros(3)), []
    physical_error = baseline_error = 0.0
    moment_squared = moment_norm = 0.0
    for i in range(len(dataset)):
        batch = collate_fields([dataset[i]])
        batch = {k: v.to(operator.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        sites = {k: batch[k] for k in ("species", "positions", "atom_mask")}
        field, occupancies = operator.model(batch["input"], batch["cell"], sites=sites)
        decoded = operator.target_transform.inverse(field)
        target = batch["target_physical"]
        if "baseline" in batch:
            decoded = decoded + batch["baseline"]
            target = target + batch["baseline"]
        # One structure, charge channel; no positivity/count postprocessing.
        denominator = torch.linalg.vector_norm(target[:, 0]).clamp(min=1e-20)
        error = float(torch.linalg.vector_norm(decoded[:, 0] - target[:, 0]) / denominator)
        physical_error += error
        count_error = float((decoded[:, 0].sum() - target[:, 0].sum()).abs() /
                            target[:, 0].sum().abs().clamp(min=1e-20))
        if target.shape[1] == 2:
            moment_squared += float(torch.square(decoded[:, 1] - target[:, 1]).sum())
            moment_norm += float(torch.square(target[:, 1]).sum())
        if "baseline" in batch:
            baseline_error += float(torch.linalg.vector_norm(
                batch["baseline"][:, 0] - target[:, 0]) / denominator)
        p = operator.site_transform.inverse(occupancies, batch["species"])[0].cpu().numpy()
        r = batch["paw"][0].cpu().numpy()
        b = operator.site_transform.inverse(torch.zeros_like(batch["paw"]),
                                            batch["species"])[0].cpu().numpy()
        species = batch["species"][0].cpu().numpy()
        for z in np.unique(species[species > 0]):
            chosen = species == z
            for L, indices in irreps_blocks(operator.model.layouts[int(z)]).items():
                indices = indices.reshape(-1)
                for s in range(r.shape[1]):
                    ref, pred, mean = (v[chosen, s][:, indices].astype(float) for v in (r, p, b))
                    parts[(int(z), s, L)] += [np.square(pred - ref).sum(),
                                              np.square(ref).sum(),
                                              np.square(mean - ref).sum()]
        per_material.append({"id": dataset.materials[i].identifier, "charge_relative_l2": error,
                             "electron_count_relative_error": count_error})
    loader = make_dataloader(dataset, batch_size=4, shuffle=False)
    rms, occupancy_loss = validate_occupancies(operator, loader)
    blocks = {f"Z{z}/set{s}/L{L}": {
        "relative_rms": float(np.sqrt(e / n)) if n > 1e-30 else None,
        "mean_baseline_relative_rms": float(np.sqrt(b / n)) if n > 1e-30 else None,
        "target_squared_norm": float(n)} for (z, s, L), (e, n, b) in sorted(parts.items())}
    pooled = np.sum(list(parts.values()), axis=0)
    density_error = physical_error / len(dataset)
    return {"charge_relative_l2": density_error, "occupancy_relative_rms": rms,
            "magnetisation_relative_l2_pooled": float(np.sqrt(moment_squared / moment_norm))
                if moment_norm > 1e-30 else None,
            "electron_count_relative_error": float(np.mean(
                [r["electron_count_relative_error"] for r in per_material])),
            "occupancy_balanced_loss": occupancy_loss,
            "common_score": density_error + 3.0 * occupancy_loss,
            "occupancy_mean_baseline": float(np.sqrt(pooled[2] / pooled[1])),
            "density_atomic_baseline": baseline_error / len(dataset) if operator.baseline else None,
            "blocks": blocks, "per_material": per_material}


def configure(base, arm, seed, epochs, device):
    config = copy.deepcopy(base)
    for key, value in ARMS[arm].items():
        if key in ("width", "modes", "n_layers", "occupancy_weight"):
            config.model.paw[key] = value
        elif key == "n_radial":
            config.model.equivariant[key] = value
        elif key == "delta_density":
            config.data.delta_density = value
        else:
            setattr(config.training, key, value)
    config.training.epochs = epochs
    config.training.init_seed = seed
    config.training.device = device
    config.training.strict_device = True
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_ext2paw_chon.yaml")
    parser.add_argument("--output", default="experiments/chon_ext2paw/results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--evaluate-winner", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    base = TrainingConfig.from_yaml(args.config)
    code_fingerprint = hashlib.sha256(b"".join(
        path.read_bytes() for path in sorted(
            [*ROOT.joinpath("src/poraque").rglob("*.py"),
             ROOT / "scripts/poraque_train.py", Path(__file__).resolve()]))).hexdigest()
    cache = str(Path(base.data.cache) / cache_tag(base.data))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "working_tree.patch").write_text(subprocess.check_output(
        ["git", "diff", "--", "src", "scripts"], text=True))
    raw = FieldPairDataset(cache, "ext2paw", cache=True)
    if args.pilot:
        raw.materials = raw.materials[:48]
    else:
        manifest = json.loads((Path(base.data.paths()[0]) / "manifest.json").read_text())
        expected = {r["material_id"] for r in manifest if r["status"] == "downloaded"}
        assert {r.identifier for r in raw.materials} == expected, "cache is incomplete; finish prepare.py"
    profiles = load_paw_profiles(cache)
    # Geometry-only grid feasibility, independent of labels and fitted
    # statistics. A common band must also be supplied by held-out grids.
    nyquist = min(float(np.min(np.pi * np.asarray(shape) /
                  np.linalg.norm(raw.load_fields(i)[0].grid.cell, axis=1)))
                  for i, shape in enumerate(raw.shapes()))
    base.model.paw["readout_g_max"] = 0.95 * nyquist
    print(f"Common geometry-compatible readout band: {0.95 * nyquist:.6f} 1/Ang", flush=True)
    split = split_records(raw.materials, base.training.seed)
    split_path = output / "split.json"
    if split_path.exists():
        assert json.loads(split_path.read_text()) == split, "use a new output directory for another split"
    write_json(split_path, split)
    if not args.pilot:
        write_json(output / "audit.json", audit(raw, profiles))
    records = raw.materials
    del raw
    reference = resolve_library(base.data.atomic_reference, cache=base.data.cache, log=print)
    variants = {}
    for delta in {True, False} if "absolute" in args.arms else {True}:
        variants[delta] = datasets(cache, base, split, records, reference if delta else None)
    results = []
    for arm in args.arms:
        for seed in args.seeds:
            config = configure(base, arm, seed, args.epochs, args.device)
            directory = output / f"{arm}_s{seed}_e{args.epochs}"
            directory.mkdir(exist_ok=True)
            config.task.name = f"chon_{directory.name}"
            signature = hashlib.sha256(json.dumps({"config": config.to_dict(), "split": split,
                                                   "atomic_reference": reference.fingerprint,
                                                   "code_fingerprint": code_fingerprint},
                                                   sort_keys=True).encode()).hexdigest()
            metrics_path = directory / "metrics.json"
            if metrics_path.exists():
                row = json.loads(metrics_path.read_text())
                assert row["signature"] == signature, "trial configuration changed"
                results.append(row)
                continue
            subsets = variants[config.data.delta_density]
            with (directory / "training.log").open("w") as handle:
                def log(message):
                    print(message, flush=True)
                    handle.write(message + "\n")
                    handle.flush()
                operator = build_operator(resolve_task("ext2paw"), subsets["train"], config, log)
                config.to_yaml(str(directory / "config.yaml"))
                start = time.perf_counter()
                history = train(operator, subsets["train"], validation=subsets["validation"],
                                epochs=args.epochs, batch_size=config.training.batch_size,
                                learning_rate=config.training.learning_rate,
                                weight_decay=config.training.weight_decay,
                                optimizer=config.training.optimizer, scheduler=config.training.scheduler,
                                occupancy_weight=config.model.paw_settings()["occupancy_weight"],
                                eval_every=config.training.eval_epoch,
                                early_stopping=max(args.epochs, config.training.early_stopping),
                                seed=config.training.seed, log=log)
                seconds = time.perf_counter() - start
                score = measure(operator, subsets["validation"])
                checkpoint = directory / "model.poraque"
                save_bundle(str(checkpoint), {"ext2paw": operator}, config=config, paw_profiles=profiles)
                row = {"arm": arm, "seed": seed, "epochs": args.epochs,
                       "parameters": operator.model.n_parameters(), "seconds": seconds,
                       "device": str(operator.device), "threads": args.threads,
                       "signature": signature, "history": history, "validation": score,
                       "code_fingerprint": code_fingerprint,
                       "checkpoint": str(checkpoint), "baseline": config.data.delta_density}
                write_json(metrics_path, row)
                results.append(row)
                log(f"{arm} seed={seed}: common score {score['common_score']:.6f}, {seconds:.1f}s")
            del operator
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()
    # Rank configurations using the mean across ALL requested seeds.
    grouped = defaultdict(list)
    for row in results:
        grouped[row["arm"]].append(row)
    ranking = sorted(grouped, key=lambda name: np.mean(
        [r["validation"]["common_score"] for r in grouped[name]]))
    write_json(output / "summary.json", {"ranking": ranking, "runs": results,
               "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "selection": "mean validation physical charge rel L2 + 3 * balanced occupancy loss"})
    with (output / "summary.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "charge_relative_l2", "occupancy_relative_rms",
                         "balanced_occupancy_loss", "common_score", "seconds", "parameters"])
        for arm in ranking:
            for row in grouped[arm]:
                v = row["validation"]
                writer.writerow([arm, row["seed"], v["charge_relative_l2"], v["occupancy_relative_rms"],
                                 v["occupancy_balanced_loss"], v["common_score"], row["seconds"], row["parameters"]])
    if args.evaluate_winner and not args.pilot:
        from poraque.ml import load_bundle
        winner = ranking[0]
        test_rows = []
        for row in grouped[winner]:
            operator = load_bundle(row["checkpoint"], "ext2paw", device=args.device)
            test_rows.append({"arm": winner, "seed": row["seed"], "metrics": measure(
                operator, variants[row["baseline"]]["test"])})
        write_json(output / "test_winner.json", test_rows)
    print("Ranking:", ranking, flush=True)


if __name__ == "__main__":
    main()
