"""Select across completed confirmation stages, then evaluate test once.

Stages may have different runner revisions; each keeps its own provenance.
Only comparable 100-epoch, two-seed results on one identical split are accepted.
The deployed checkpoint is chosen by validation, never by test performance.
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import measure, write_json
from poraque.ml import FieldPairDataset, load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/chon_ext2chg/results/final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    test_path = args.output / "test_winner.json"
    assert not test_path.exists(), "test already evaluated; preserve the existing result"
    torch.set_num_threads(args.threads)
    groups = defaultdict(list)
    split = None
    for directory in args.directories:
        local_split = json.loads((directory / "split.json").read_text())
        if split is not None:
            assert split == local_split, "confirmation splits differ"
        split = local_split
        paths = list(directory.glob("*/metrics.json"))
        assert paths, f"no completed trials in {directory}"
        for path in paths:
            row = json.loads(path.read_text())
            assert row["epochs"] == args.epochs
            assert len(row["history"]["train_loss"]) == args.epochs
            assert not row["history"]["stopped_early"]
            assert row["device"] == args.device
            assert Path(row["checkpoint"]).exists(), row["checkpoint"]
            groups[row["arm"]].append(row)
    assert groups, "no completed confirmation trials"
    for arm, rows in groups.items():
        assert sorted(r["seed"] for r in rows) == sorted(args.seeds), (arm, "missing or duplicate seeds")
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["validation"]) & set(split["test"]))
    ranking = sorted(groups, key=lambda arm: np.mean(
        [r["validation"]["common_score"] for r in groups[arm]]))
    winner = ranking[0]
    best = min(groups[winner], key=lambda r: r["validation"]["common_score"])
    summary = {"ranking": ranking, "winner": winner,
        "chosen_seed_by_validation": best["seed"], "epochs": args.epochs,
        "selection": "mean validation physical charge relative L2 + 3 * balanced occupancy loss",
        "validation": {arm: {key: {"mean": float(np.mean([r["validation"][key] for r in rows])),
            "std_across_seeds": float(np.std([r["validation"][key] for r in rows], ddof=1))}
            for key in ("charge_relative_l2", "occupancy_relative_rms", "occupancy_balanced_loss", "common_score")}
            for arm, rows in groups.items()}, "runs": [r for rows in groups.values() for r in rows]}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "selection.json", summary)
    write_json(args.output / "split.json", split)
    cache = str(ROOT / "data/cache/chon/res24_potcar_spin")
    raw = FieldPairDataset(cache, "ext2chg", cache=False)
    records = [r for r in raw.materials if r.identifier in split["test"]]
    assert len(records) == len(split["test"])
    test_rows = []
    for row in groups[winner]:
        operator = load_bundle(row["checkpoint"], "ext2chg", device=args.device)
        dataset = FieldPairDataset(cache, "ext2chg", materials=records,
            baseline=operator.baseline, cache=True,
            input_transform=operator.input_transform, target_transform=operator.target_transform)
        assert len(dataset) == len(split["test"])
        test_rows.append({"arm": winner, "seed": row["seed"], "metrics": measure(operator, dataset)})
    write_json(test_path, test_rows)
    shutil.copy2(best["checkpoint"], args.output / "best.poraque")
    shutil.copy2(Path(best["checkpoint"]).parent / "config.yaml", args.output / "selected_config.yaml")
    from plot_parity import generate
    generate(args.output / "best.poraque", args.output / "split.json", cache,
             args.output / "plots", args.device, args.threads)
    print(json.dumps({"winner": winner, "selected_seed": best["seed"],
                     "test": [{"seed": r["seed"], "charge": r["metrics"]["charge_relative_l2"],
                               "paw": r["metrics"]["occupancy_relative_rms"]} for r in test_rows]}, indent=2))


if __name__ == "__main__":
    main()
