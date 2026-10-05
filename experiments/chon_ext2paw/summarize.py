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
    args = parser.parse_args()
    for directory in args.directories:
        rows = [json.loads(p.read_text()) for p in sorted(directory.glob("*/metrics.json"))]
        if not rows:
            continue
        groups = defaultdict(list)
        for row in rows:
            groups[row["arm"]].append(row)
        ranking = sorted(groups, key=lambda k: np.mean(
            [r["validation"]["common_score"] for r in groups[k]]))
        lines = ["| Configuração | sementes | ρ: L2 relativo (%) | PAW: RMS (%) | perda PAW | score comum | s/época | parâmetros |",
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
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
        colours = plt.get_cmap("tab10")
        for index, arm in enumerate(ranking):
            members = groups[arm]
            colour = colours(index % 10)
            for j, row in enumerate(members):
                h = row["history"]
                axes[0].plot(h["val_epoch"], np.asarray(h["val_field_error"]) * 100,
                             color=colour, alpha=0.85, label=arm if j == 0 else None)
                axes[1].plot(h["val_epoch"], np.asarray(h["val_occupancy_error"]) * 100,
                             color=colour, alpha=0.85)
                v = row["validation"]
                axes[2].scatter(v["charge_relative_l2"] * 100,
                                v["occupancy_relative_rms"] * 100, color=colour)
            x = np.mean([r["validation"]["charge_relative_l2"] for r in members]) * 100
            y = np.mean([r["validation"]["occupancy_relative_rms"] for r in members]) * 100
            axes[2].annotate(arm, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axes[0].set(xlabel="Epoch", ylabel="Physical (rho, m) relative L2 (%)", title="Validation field")
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
