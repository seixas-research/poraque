"""Measure Fourier truncation of native validation densities and residuals.

This is a representation diagnostic, not a trained high-resolution benchmark.
Never reads the reserved test structures. Atomic high-frequency content can be
restored independently of the coarse learned residual, so both cases matter.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from poraque.fields.atomic import atomic_superposition, resolve_library
from poraque.fields.density import ChargeDensity
from poraque.fields.resample import downsample_shape, spectral_resample
from poraque.ml import FieldPairDataset


def plot(payload, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolutions = [24, 32, 48, 64]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, metric, title in zip(axes,
            ("native_charge_projection_error", "native_residual_projection_error_over_charge_norm"),
            ("Projection of native charge", "Projection of residual; native atomic content restored")):
        for row in payload["per_material"]:
            axis.plot(resolutions, [100 * row["resolutions"][str(n)][metric] for n in resolutions],
                      color="0.7", alpha=0.5, linewidth=1)
        axis.plot(resolutions, [100 * payload["mean_errors"][str(n)][metric] for n in resolutions],
                  "o-", color="C0", linewidth=2,
                  label=f"Mean of {payload['structures']} validation structures")
        axis.set(xlabel="Maximum grid axis", ylabel="Relative to native charge norm (%)",
                 title=title, yscale="log", xticks=resolutions)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"resolution_audit.{extension}", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=12)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/chon_ext2chg/results")
    args = parser.parse_args()
    split = json.loads((args.results / "screen/split.json").read_text())
    data = FieldPairDataset(str(ROOT / "data/cache/chon/res24_potcar_spin"), "ext2chg", cache=False)
    candidates = sorted([(len(data.site_records(i)["species"]), r.identifier)
                         for i, r in enumerate(data.materials) if r.identifier in split["validation"]])
    # Cover atom-count quantiles; this sample is descriptive, not random.
    chosen = [candidates[i][1] for i in np.unique(np.linspace(
        0, len(candidates) - 1, args.sample).round().astype(int))]
    library = resolve_library(str(ROOT / "data/CHON/isolated_atoms"),
                              cache=str(ROOT / "data/cache/chon"))
    rows = []
    for name in chosen:
        density = ChargeDensity.read(str(ROOT / "data/CHON" / name / "CHGCAR.gz"))
        native = density.data
        atomic = atomic_superposition(density.structure, density.grid, library).data
        residual = native - atomic
        denominator = np.linalg.norm(native)
        row = {"id": name, "atoms": len(density.structure), "native_shape": list(native.shape),
               "native_atomic_baseline_error": float(np.linalg.norm(residual) / denominator),
               "resolutions": {}}
        for resolution in (24, 32, 48, 64):
            shape = downsample_shape(native.shape, target_max=resolution)
            truncated = spectral_resample(spectral_resample(native, shape), native.shape)
            residual_truncated = spectral_resample(spectral_resample(residual, shape), native.shape)
            row["resolutions"][str(resolution)] = {
                "shape": list(shape),
                "native_charge_projection_error": float(np.linalg.norm(native - truncated) / denominator),
                "native_residual_projection_error_over_charge_norm": float(
                    np.linalg.norm(residual - residual_truncated) / denominator)}
        rows.append(row)
        print(name, row["resolutions"]["24"], flush=True)
    summary = {str(n): {metric: float(np.mean([r["resolutions"][str(n)][metric] for r in rows]))
                        for metric in ("native_charge_projection_error",
                                       "native_residual_projection_error_over_charge_norm")}
               for n in (24, 32, 48, 64)}
    payload = {"sampling": "validation atom-count quantiles; descriptive sample",
               "structures": len(rows), "mean_errors": summary, "per_material": rows,
               "interpretation": "Projection errors, not model errors. The residual projection uses native atomic content; it assumes a perfect coarse residual and is not a measured inference result."}
    (args.results / "resolution_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot(payload, args.results)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
