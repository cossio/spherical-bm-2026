# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.2", "matplotlib==3.10.8"]
# ///
"""Reproduce Figure 8 from saved estimates, or verify fresh retained traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--summary", type=Path, default=root / "validation/production/summary.csv")
    parser.add_argument("--n256-summary", type=Path, default=root / "validation/n256/summary.csv")
    parser.add_argument("--input", type=Path, help="Optional raw trace directory; recomputes every plotted mean")
    parser.add_argument("--n256-input", type=Path, help="Optional raw trace directory for N=256")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.summary.open() as f:
        rows = list(csv.DictReader(f))
    with args.n256_summary.open() as f:
        rows += list(csv.DictReader(f))
    table = {(r["case"], int(r["N"]), r["observable"]): r for r in rows}
    assert len(table) == len(rows), "Duplicate observable records across cohorts"
    with (root / "results/saddle_predictions.toml").open("rb") as f:
        theory = tomllib.load(f)
    curve = np.genfromtxt(root / "results/saddle_curve.csv", delimiter=",", names=True)
    sizes = sorted({int(r["N"]) for r in rows})
    colors = {32: "#D55E00", 64: "#009E73", 128: "#0072B2", 256: "#CC79A7"}
    raw = {}
    saved_points = json.loads((root / "data/plot_points.json").read_text())["plotted_points"]
    saved = {(r["case"], r["N"], r["observable"]): r for r in saved_points}
    exported = []

    def point(case, n, observable, scale=1.0):
        r = table[case, n, observable]
        m = float(r["mean"])
        lo, hi = float(r["ci_low"]),float(r["ci_high"])
        directory = args.n256_input if n == 256 else args.input
        if directory is not None:
            if (case,n) not in raw:
                paths = sorted((directory / case / f"N{n}").glob("chain*/draws.csv"))
                assert len(paths) == 4, f"Expected four chains for {case}, N={n}"
                raw[case,n] = [np.genfromtxt(p, delimiter=",", names=True) for p in paths]
            actual = np.mean(np.concatenate([d[observable] for d in raw[case,n]]))
            assert abs(m-actual) < 1e-11, (case,n,observable,m,actual)
        else:
            original = saved[case,n,observable]
            assert all(abs(value-original[key]) < 1e-11 for key,value in [("mean",m),("ci_low",lo),("ci_high",hi)])
        exported.append(dict(case=case,N=n,observable=observable,mean=m,ci_low=lo,ci_high=hi,scale=scale))
        return m/scale, np.array([[m-lo],[hi-m]])/scale

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize":9,
        "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
        "pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans",
        "axes.spines.top":False,"axes.spines.right":False})
    fig, axs = plt.subplots(1,2,figsize=(5.5,2.7))
    fig.subplots_adjust(left=.115,right=.98,bottom=.195,top=.75,wspace=.42)
    cases = sorted([k for k,v in theory.items() if k != "provenance" and v["eta"] == 4],key=lambda k:theory[k]["gamma"])
    a,b = axs.flat
    a.plot(curve["gamma"],curve["lambda1"],color="black",lw=1.2)
    a.plot(curve["gamma"],curve["lambda2"],color="black",lw=1.2,ls="--")
    b.plot(curve["gamma"],curve["h_squared"],color="black",lw=1.2)
    for ax in (a,b):
        for boundary in (.5625,1.25):
            ax.axvline(boundary,color=".78",lw=.7,zorder=0)
        ax.set_xlim(.18,1.57)
        ax.set_xlabel(r"Regularization $\gamma$")
    for n in sizes:
        for case in cases:
            assert (case,n,"lambda1") in table, (case,n)
            x = theory[case]["gamma"]
            for obs,marker in [("lambda1","o"),("lambda2","s")]:
                y,e = point(case,n,obs)
                a.errorbar(x,y,yerr=e,fmt=marker,ms=3.7,mfc="white" if marker=="s" else colors[n],
                           color=colors[n],capsize=2,elinewidth=.8,zorder=4)
            y,e=point(case,n,"h_top2")
            b.errorbar(x,y,yerr=e,fmt="o",ms=3.7,color=colors[n],capsize=2,elinewidth=.8,zorder=4)
    a.set_ylabel(r"Top eigenvalues $\lambda_1,\lambda_2$")
    b.set_ylabel("Top-two generated mass")
    a.set_title(r"A) Outlier positions ($\eta=4$)",loc="left",fontweight="bold",pad=9)
    b.set_title(r"B) Condensation ($\eta=4$)",loc="left",fontweight="bold",pad=9)
    b.set_ylim(-.015,.68)

    fig.legend(handles=[Line2D([],[],color=colors[n],marker="o",ls="",label=f"N = {n}") for n in sizes]+[Line2D([],[],color="black",lw=1.2,label=r"$N\to\infty$ saddle")],
               loc="upper center",bbox_to_anchor=(.51,.985),ncol=len(sizes)+1,frameon=False,columnspacing=.8,handlelength=1.8,handletextpad=.55)
    assert len(exported) == 72 and sizes == [32,64,128,256]
    for item in exported:
        ax = b if item["observable"] == "h_top2" else a
        x = theory[item["case"]]["gamma"]
        xlo,xhi = ax.get_xlim()
        ylo,yhi = ax.get_ylim()
        assert xlo <= x <= xhi
        assert ylo <= item["ci_low"]/item["scale"] <= item["ci_high"]/item["scale"] <= yhi, item
    out=args.output/"equilibrium_comparison"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"),dpi=220)
    plt.close(fig)
    record={"plotted_points":exported,"summary_sha256":hashlib.sha256(args.summary.read_bytes()).hexdigest(),
            "n256_summary_sha256":hashlib.sha256(args.n256_summary.read_bytes()).hexdigest(),
            "mean_check":"each mean checked against supplied raw draws for its cohort, or its saved publication estimate when raw draws are absent",
            "raw_verified_sizes":sorted({n for _,n in raw}),
            "clipping_check":"all plotted means and confidence-interval endpoints lie within their axes",
            "pdf_sha256":hashlib.sha256(out.with_suffix(".pdf").read_bytes()).hexdigest()}
    (args.output/"comparison_data_validation.json").write_text(json.dumps(record,indent=2)+"\n")
    print(f"Exported {out}; checked {len(exported)} points; raw-verified sizes: {record['raw_verified_sizes']}")

if __name__ == "__main__":
    main()
