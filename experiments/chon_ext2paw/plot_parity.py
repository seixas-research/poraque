"""Export DFT/model parity for a saved CHON checkpoint, without training.

Restore the bundle's fitted transforms and atomic baseline. The fixed split is
used only for diagnostics; neither model selection nor test metrics are changed.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from poraque.fields.vasp.augmentation import irreps_blocks
from poraque.ml import FieldPairDataset, load_bundle
from poraque.ml.data import collate_fields
from poraque.vis import ParityAccumulator, TrainingReport

ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O"}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@torch.no_grad()
def generate(checkpoint, split_path, cache, output, device="cpu", threads=2):
    torch.set_num_threads(threads)
    checkpoint, output = Path(checkpoint), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(split_path).read_text())
    names = [name for members in split.values() for name in members]
    assert len(names) == len(set(names)), "overlapping splits"
    operator = load_bundle(str(checkpoint), "ext2paw", device=device)
    operator.model.eval()
    raw = FieldPairDataset(str(cache), "ext2paw", cache=False)
    assert set(names) == {r.identifier for r in raw.materials}
    fields = {key: {part: ParityAccumulator(seed=42) for part in split}
              for key in ("charge", "magnetization", "paw_charge", "paw_magnetization")}
    blocks = {(z, L): ParityAccumulator(seed=42)
              for z in ELEMENTS for L in range(3)}
    for part, identifiers in split.items():
        selected = set(identifiers)
        dataset = FieldPairDataset(str(cache), "ext2paw",
            materials=[r for r in raw.materials if r.identifier in selected],
            baseline=operator.baseline, cache=True,
            input_transform=operator.input_transform,
            target_transform=operator.target_transform)
        assert len(dataset) == len(identifiers)
        for index in range(len(dataset)):
            batch = collate_fields([dataset[index]])
            batch = {k: v.to(operator.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            sites = {k: batch[k] for k in ("species", "positions", "atom_mask")}
            field, paw = operator.model(batch["input"], batch["cell"], sites=sites)
            predicted = operator.target_transform.inverse(field)
            reference = batch["target_physical"]
            if "baseline" in batch:
                predicted = predicted + batch["baseline"]
                reference = reference + batch["baseline"]
            x, y = reference[0].cpu().numpy(), predicted[0].cpu().numpy()
            assert np.isfinite(x).all() and np.isfinite(y).all()
            fields["charge"][part].add(x[0], y[0])
            if x.shape[0] == 2:
                fields["magnetization"][part].add(x[1], y[1])
            x = batch["paw"][0].cpu().numpy()
            y = operator.site_transform.inverse(paw, batch["species"])[0].cpu().numpy()
            y = y[..., :x.shape[-1]]
            mask = batch["paw_mask"][0].bool().cpu().numpy()
            assert np.isfinite(x).all() and np.isfinite(y).all()
            fields["paw_charge"][part].add(x[:, 0][mask], y[:, 0][mask])
            if x.shape[1] == 2:
                fields["paw_magnetization"][part].add(x[:, 1][mask], y[:, 1][mask])
            if part == "test":
                species = batch["species"][0].cpu().numpy()
                for z in ELEMENTS:
                    chosen = species == z
                    if not chosen.any():
                        continue
                    for L, indices in irreps_blocks(operator.model.layouts[z]).items():
                        indices = indices.reshape(-1)
                        blocks[z, L].add(x[chosen, 0][:, indices], y[chosen, 0][:, indices])
        print(f"Parity: {part}, {len(dataset)} structures", flush=True)

    options = {
        "charge": ("parity", r"$\rho$", r"e/Å$^3$", True),
        "magnetization": ("parity_magnetization", r"$m$", r"e/Å$^3$", False),
        "paw_charge": ("parity_paw", "PAW charge occupancies", "", False),
        "paw_magnetization": ("parity_paw_magnetization", "PAW magnetic occupancies", "", False),
    }
    artifacts = []
    for fmt in ("png", "pdf"):
        report = TrainingReport(output, prefix="ext2paw", fmt=fmt,
                                save_data=(fmt == "png"))
        for key, (name, label, unit, log) in options.items():
            if not fields[key]["train"].count:
                continue
            artifacts.append(report.global_parity(
                fields[key]["train"], fields[key]["validation"], name=name,
                label=label, unit=unit, log=log))
            artifacts.append(report.global_parity(
                fields[key]["test"], name=f"{name}_test", label=label,
                unit=unit, log=log, split_labels=("test set", "")))
        # The log figure omits nonpositive pairs; retain an all-value view too.
        for part in ("validation", "test"):
            artifacts.append(report.global_parity(
                fields["charge"][part], name=f"parity_charge_linear_{part}",
                label=r"$\rho$", unit=r"e/Å$^3$", log=False,
                split_labels=(f"{part} set", "")))

        figure, axes = plt.subplots(4, 3, figsize=(11, 12), layout="constrained")
        for row, z in enumerate(ELEMENTS):
            for L in range(3):
                panel, accumulator = axes[row, L], blocks[z, L]
                extent = accumulator.extent(False)
                if extent is None:
                    panel.set_visible(False)
                    continue
                low, high = extent
                edges = np.linspace(low, high, 121)
                counts = accumulator.histogram(edges, False)
                shares = counts / counts.sum()
                mesh = panel.pcolormesh(edges, edges, shares.T,
                    norm=LogNorm(vmin=float(shares[shares > 0].min()),
                                 vmax=float(shares.max())), cmap="viridis")
                panel.plot([low, high], [low, high], "--", color="gray", lw=1)
                panel.set(xlim=(low, high), ylim=(low, high),
                    title=f"{ELEMENTS[z]} · L{L} · rel RMS {accumulator.metrics()['relative_l2']:.1%}",
                    xlabel="DFT PAW coefficient", ylabel="FNO PAW coefficient")
                panel.set_aspect("equal")
                figure.colorbar(mesh, ax=panel, shrink=.7)
                if fmt == "png":
                    report._write_parity_histogram(f"parity_paw_{ELEMENTS[z]}_L{L}_test",
                        ["test set"], [int(counts.sum())], edges, [shares])
        figure.suptitle("PAW charge parity on the test set — native coefficients by element and L")
        path = output / f"ext2paw_parity_paw_blocks_test.{fmt}"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        artifacts.append(str(path))

    provenance = {"checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split": split, "device": str(operator.device),
        "field_resolution": "cached working grids, longest axis 24",
        "normalization_or_clipping": False,
        "sampling": "up to 10000 values per structure, fixed seed 42",
        "metric_note": "pooled parity statistics differ from mean per-structure charge L2",
        "fields": {key: {part: {**a.metrics(), "positive_share": a.positive_share()}
                    for part, a in groups.items()} for key, groups in fields.items()},
        "test_paw_charge_blocks": {f"{ELEMENTS[z]}/L{L}": a.metrics()
                                   for (z, L), a in blocks.items()},
        "artifacts": artifacts}
    (output / "parity_metadata.json").write_text(
        json.dumps(finite_json(provenance), indent=2, allow_nan=False) + "\n")
    print(f"Parity figures exported to {output}", flush=True)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "experiments/chon_ext2paw/results/final/best.poraque")
    parser.add_argument("--split", type=Path, default=ROOT / "experiments/chon_ext2paw/results/final/split.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/chon/res24_potcar_spin")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/chon_ext2paw/results/final/plots")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    generate(args.checkpoint, args.split, args.cache, args.output, args.device, args.threads)


if __name__ == "__main__":
    main()
