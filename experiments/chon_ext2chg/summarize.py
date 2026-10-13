"""Export completed trials as a Markdown table and standalone PDF/PNG plots."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--combined-output", type=Path)
    args = parser.parse_args()
    exports = [(directory, [json.loads(p.read_text())
                for p in sorted(directory.glob("*/metrics.json"))]) for directory in args.directories]
    if args.combined_output:
        args.combined_output.mkdir(parents=True, exist_ok=True)
        exports = [(args.combined_output, [row for _, rows in exports for row in rows])]
    for directory, rows in exports:
        if not rows:
            continue
        groups = defaultdict(list)
        for row in rows:
            groups[row["arm"]].append(row)
        ranking = sorted(groups, key=lambda k: np.mean(
            [r["validation"]["common_score"] for r in groups[k]]))
        if args.combined_output:
            (directory / "summary.json").write_text(json.dumps({
                "source_directories": [str(p) for p in args.directories],
                "ranking": ranking, "runs": rows}, indent=2) + "\n")
        lines = ["| Configuration | seeds | ρ: relative L2 (%) | PAW: RMS (%) | PAW loss | common score | s/epoch | parameters |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm in ranking:
            members = groups[arm]
            def value(key, multiplier=1):
                values = [r["validation"][key] * multiplier for r in members]
                mean = np.mean(values)
                return f"{mean:.3f}" + (f" ± {np.std(values, ddof=1):.3f}" if len(values) > 1 else "")
            timing = np.mean([np.mean(r["history"]["seconds_per_epoch"]) for r in members])
            lines.append(f"| {arm} | {','.join(str(r['seed']) for r in members)} | "
                         f"{value('charge_relative_l2', 100)} | {value('occupancy_relative_rms', 100)} | "
                         f"{value('occupancy_balanced_loss')} | {value('common_score')} | "
                         f"{timing:.2f} | {members[0]['parameters']:,} |")
        (directory / "table.md").write_text("\n".join(lines) + "\n")
        distributions = []
        for row in rows:
            materials = row["validation"]["per_material"]
            errors = [m["charge_relative_l2"] for m in materials]
            distributions.append({"arm": row["arm"], "seed": row["seed"],
                "charge_error_median": float(np.median(errors)),
                "charge_error_p90": float(np.percentile(errors, 90)),
                "worst_five": sorted(materials, key=lambda m: m["charge_relative_l2"],
                                     reverse=True)[:5]})
        (directory / "error_distribution.json").write_text(
            json.dumps(distributions, indent=2) + "\n")
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
        colors = plt.get_cmap("tab20" if len(ranking) > 10 else "tab10")
        for index, arm in enumerate(ranking):
            members = groups[arm]
            color = colors(index)
            for j, row in enumerate(members):
                h = row["history"]
                axes[0].plot(h["val_epoch"], np.asarray(h["val_field_error"]) * 100,
                             color=color, alpha=0.85, label=f"{index + 1}. {arm}" if j == 0 else None)
                axes[1].plot(h["val_epoch"], np.asarray(h["val_occupancy_error"]) * 100,
                             color=color, alpha=0.85)
                v = row["validation"]
                axes[2].scatter(v["charge_relative_l2"] * 100,
                                v["occupancy_relative_rms"] * 100, color=color)
            x = np.mean([r["validation"]["charge_relative_l2"] for r in members]) * 100
            y = np.mean([r["validation"]["occupancy_relative_rms"] for r in members]) * 100
            angle = 2 * np.pi * index / len(ranking)
            # Separate the nearly coincident capacity ablations in the screen.
            offset = {"baseline": (32, 0), "layers5": (0, 40),
                      "width64": (35, 40), "occ1": (-15, -25),
                      "width32": (30, -10), "radial16": (45, 0),
                      "modes10": (-15, -10), "lr_high": (30, -10)}.get(
                          arm, (12 * np.cos(angle), 12 * np.sin(angle)))
            axes[2].annotate(str(index + 1), (x, y),
                             xytext=offset,
                             textcoords="offset points", fontsize=8,
                             ha="center", va="center",
                             arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color})
        axes[0].set(xlabel="Epoch", ylabel="Physical (rho, m) relative L2 (%)", title="Validation field")
        axes[0].set_yscale("log")
        axes[1].set(xlabel="Epoch", ylabel="Occupancy relative RMS (%)", title="Validation PAW records")
        axes[2].set(xlabel="Charge relative L2 (%)", ylabel="Occupancy relative RMS (%)",
                    title="Restored checkpoints; lower is better")
        for panel in axes:
            panel.grid(alpha=0.2)
        axes[0].legend(fontsize=8, ncol=2)
        for extension in ("pdf", "png"):
            figure.savefig(directory / f"benchmark.{extension}", dpi=160)
        plt.close(figure)
        print(directory / "table.md")


if __name__ == "__main__":
    main()
