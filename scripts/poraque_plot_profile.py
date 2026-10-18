"""Plot validated Poraque profiling CSV without pandas (Matplotlib only)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_report(csv_path: Path, output: Path) -> None:
    """Plot size dependence, exact Python/C speedup, and strong/weak efficiency.

    Different MPI k-group layouts and thread counts remain separate curves.
    Strong efficiency references C at one rank/one thread; weak efficiency
    references one rank at the same thread count. Empty references are omitted.
    """
    with csv_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('CSV contains no validated measurements')
    fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    strong = [r for r in rows if r['mode']=='strong']
    for threads,ranks,rpk in sorted({(r['threads'],r['ranks'],r['ranks_per_k']) for r in strong}):
        group = sorted([r for r in strong if (r['threads'],r['ranks'],r['ranks_per_k'])==
                        (threads,ranks,rpk)],key=lambda r:int(r['n']))
        n = [int(r['n']) for r in group]
        label = f'C ranks={ranks}, threads={threads}, ranks/k={rpk}'
        axes[0,0].errorbar(n,[float(r['c_total_s']) for r in group],
                           yerr=[float(r['c_total_std_s']) for r in group],marker='o',label=label)
        axes[0,1].plot(n,[float(r['total_speedup']) for r in group],'-o',label=label)
    python_rows = {(r['n'],r['nk'],r['python_threads']):r for r in strong}
    for nk,threads in sorted({(k[1],k[2]) for k in python_rows}):
        group = sorted([r for (n,k,t),r in python_rows.items() if k==nk and t==threads],key=lambda r:int(r['n']))
        axes[0,0].errorbar([int(r['n']) for r in group],[float(r['python_total_s']) for r in group],
                           yerr=[float(r['python_total_std_s']) for r in group],
                           linestyle='--',marker='x',label=f'SciPy threads={threads}, k={nk}')
    for mode,axis,field in [('strong',axes[1,0],'parallel_efficiency'),
                            ('weak',axes[1,1],'weak_efficiency')]:
        subset = [r for r in rows if r['mode']==mode and r.get(field)]
        keys = {(r['n'],r['threads'],r['ranks_per_k']) for r in subset}
        for n,threads,rpk in sorted(keys,key=lambda k:tuple(map(int,k))):
            group = sorted([r for r in subset if (r['n'],r['threads'],r['ranks_per_k'])==
                            (n,threads,rpk)],key=lambda r:int(r['ranks']))
            axis.plot([int(r['ranks'])*int(r['threads']) for r in group],
                      [float(r[field]) for r in group],'-o',label=f'N={n}, threads={threads}, ranks/k={rpk}')
        axis.axhline(1.,color='black',linestyle=':',linewidth=1)
        axis.set(xlabel='Requested MPI ranks × threads',ylabel='Efficiency',title=f'{mode.capitalize()} scaling')
    axes[0,0].set(xlabel='Basis dimension N',ylabel='Median seconds (± sample SD)',
                  title='Fixed-work k-point evaluation',xscale='log',yscale='log')
    axes[0,1].set(xlabel='Basis dimension N',ylabel='T_Python / T_C',title='Measured speedup',xscale='log')
    axes[0,1].axhline(1.,color='black',linestyle=':',linewidth=1)
    for axis in axes.ravel():
        axis.grid(alpha=.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=6)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)


def main() -> None:
    """Read a profiling CSV and write a standalone PNG/PDF/SVG figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path,default=Path('hpc_performance.png'))
    args = parser.parse_args()
    plot_report(args.csv,args.output)


if __name__=='__main__':
    main()
