# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2"]
# ///
"""Render the four camera-ready RBM figures from the compact release data."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbm-release-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator
import numpy as np
from scipy.stats import t, t as student_t
ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, PURPLE = "#0072B2", "#D55E00", "#9467bd"
COLORS = {"forward_kl": BLUE, "reverse_kl": ORANGE, "q_data1": PURPLE}
def read(path): return json.loads(Path(path).read_text())


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def point_line(ax, x, summaries, color, marker="o", label=None):
    mean = np.array([s["mean"] for s in summaries])
    bounds = np.array([s["interval_95"] for s in summaries]).T
    ax.errorbar(x, mean, yerr=np.vstack([mean-bounds[0], bounds[1]-mean]), color=color,
                linestyle="--", linewidth=1.4, marker=marker, markersize=5, markerfacecolor="white",
                markeredgewidth=1.25, elinewidth=1.0, capsize=3, label=label, zorder=3)


def stats(values):
    a = np.asarray(values, dtype=float)
    if a.shape[0] != 8 or not np.isfinite(a).all():
        raise ValueError("Expected eight finite dataset observations")
    mean = a.mean(axis=0)
    half = student_t.ppf(.975, 7) * a.std(axis=0, ddof=1) / np.sqrt(8)
    return {"n_datasets": 8, "mean": mean.tolist(),
            "ci95": [(mean - half).tolist(), (mean + half).tolist()]}


def plot_eta(report):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=.115, right=.985, bottom=.28, top=.87, wspace=.42)
    for ax, direction, title in zip(axes, ("forward", "reverse"),
                                    ("A) Forward predictive KL", "B) Reverse predictive KL")):
        ax.set_title(title, loc="left", pad=10)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("KL (nats per spin)")
        ax.set_xlim(.04, 6.)
        ax.xaxis.set_major_locator(FixedLocator([.05, .2, 1, 5]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axvline(1, color=".7", linestyle=":", linewidth=1, zorder=0)
        for curve, color, marker in zip(report["curves"], (BLUE, ORANGE), ("o", "s")):
            point_line(ax, [p["eta"] for p in curve["points"]],
                             [p[direction] for p in curve["points"]], color, marker)
        ax.margins(y=.14)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor="white", linestyle="--",
                      label=rf"$\gamma={curve['gamma']:g}$")
               for curve, color, marker in zip(report["curves"], (BLUE, ORANGE), ("o", "s"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=2,
               frameon=False, columnspacing=1.5, handlelength=1.7)
    return fig


def plot_gamma_base(data, pilot=False):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))
    fig.subplots_adjust(left=.115, right=.985, bottom=.33, top=.80 if pilot else .87, wspace=.42)
    x = [row["gamma"] for row in data["points"]]
    for ax, title in zip(axes, ["A) Individual-model reverse KL", "B) Generated covariance"]):
        ax.set_title(title, loc="left", pad=10)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_xlim(.65, 1.6e4)
        ax.xaxis.set_major_locator(FixedLocator([1., 10., 100., 1000., 10000.]))
        ax.set_xticklabels(["1", "10", r"$10^2$", r"$10^3$", r"$10^4$"])
        ax.xaxis.set_minor_locator(NullLocator())
    point_line(axes[0], x, [row["reverse_kl"] for row in data["points"]], ORANGE, label=r"$\eta=0.02$")
    axes[0].axhline(data["uniform_reverse_kl"], color=".3", linestyle=":", linewidth=1, label="Uniform model")
    axes[0].set_ylabel("KL (nats per spin)")
    axes[0].set_ylim(.22, max(.68, max(row["reverse_kl"]["interval_95"][1] for row in data["points"]) + .015))
    point_line(axes[1], x, [row["covariance_lambda1_over_N"] for row in data["points"]], PURPLE, label=r"Mean $\lambda_1(C_W)/N$")
    point_line(axes[1], x, [row["teacher_q1"] for row in data["points"]], BLUE, marker="s", label=r"Teacher-mode variance $q_1$")
    axes[1].set_ylabel("Normalized variance")
    axes[1].set_ylim(0, .9)
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(-.08, -.34), frameon=False, handlelength=1.6, handletextpad=.6)
    for point in data["points"]:
        for name, ax in (("reverse_kl", axes[0]), ("covariance_lambda1_over_N", axes[1]), ("teacher_q1", axes[1])):
            if not point[name]["sampling_checks_pass"]:
                ax.scatter([point["gamma"]], [point[name]["mean"]], marker="x", color="#B2182B", s=24, linewidths=1.2, zorder=5)
    if data["flagged_points"]:
        fig.text(.99, .01, "× flagged diagnostics", ha="right", fontsize=8, color="#B2182B")
    if pilot:
        fig.text(.5, .98, "PILOT: incomplete gamma grid", ha="center", va="top", fontsize=9)
    return fig


def map_curve(path, model):
    data = read(path)
    x = np.asarray(data["gamma"], dtype=float)
    y = np.asarray(data["reverse_kl_mean"], dtype=float)
    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (x > 0).all() and (np.diff(x) > 0).all()
    bounds = None
    if model == "rbm":
        assert data["validation"]["passed"]
        assert len(data["validation"]["datasets"]) == 8
        values = np.asarray(data["reverse_kl_by_dataset"], dtype=float)
        assert values.shape == (8, len(x)) and np.isfinite(values).all()
        assert np.allclose(values.mean(axis=0), y, atol=1e-13, rtol=0)
        half = t.ppf(.975, 7) * values.std(axis=0, ddof=1) / np.sqrt(8)
        bounds = np.column_stack([y-half, y+half])
        assert np.allclose(bounds, data["reverse_kl_interval95"], atol=1e-13, rtol=0)
    return data, x, y, bounds


def plot_rbm(path):
    data, x, y, bounds = map_curve(path, "rbm")
    posterior = read(ROOT / "data/gamma.json")
    fig = plot_gamma_base(posterior)
    ax = fig.axes[0]
    ax.plot(x, y, color="black", linewidth=1.4, label="Best-found MAP", zorder=4)
    ax.fill_between(x, bounds[:, 0], bounds[:, 1], color="black", alpha=.12, zorder=1)
    ax.set_ylim(0, .68)
    ax.set_xlim(x[0]/1.5, x[-1]*1.6)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1., 1e2, 1e4]))
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-2}$", "1", r"$10^2$", r"$10^4$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(loc="upper left", bbox_to_anchor=(-.08, -.34), frameon=False,
              handlelength=1.6, handletextpad=.6)
    fig.set_size_inches(5.5, 3.25)
    fig.subplots_adjust(bottom=.37)
    return fig


def plot_beta(arrays, provenance, args):
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "lines.linewidth": 1.4})
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.7))
    fig.subplots_adjust(left=.11, right=.985, top=.925, bottom=.23, wspace=.39, hspace=.66)
    betas, gammas = arrays["betas"], arrays["gammas"]
    shown = (betas >= args.beta_min) & (betas <= args.beta_max)
    if shown.sum() < 3:
        raise ValueError("Displayed range contains too few evaluated points")
    beta1 = int(np.flatnonzero(betas == 1.)[0])
    reference = provenance["teacher_data_mode_reference"]["mean"]
    selections, plotted = [], []
    for row, gamma in enumerate((.1, 30.)):
        gi = int(np.flatnonzero(gammas == gamma)[0])
        forward = arrays["forward_kl"][:, gi]
        selected = int(np.argmin(forward.mean(axis=0)))
        optimum = float(betas[selected])
        if not args.beta_min < optimum < args.beta_max:
            raise ValueError("Selected beta is outside the plotted range")
        for col, ax in enumerate(axes[row]):
            ax.set_xscale("log")
            ax.set_xlim(args.beta_min, args.beta_max)
            ax.xaxis.set_major_locator(FixedLocator([.1, .3, 1., 3.]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(r"$\beta$", labelpad=2)
            ax.axvline(1., color=".65", linewidth=.8, zorder=0)
            ax.axvline(optimum, color=COLORS["forward_kl"], linestyle=":", linewidth=1., zorder=1)
            letter = chr(ord("A") + 2 * row + col)
            title = "KL loss" if col == 0 else "Mode expression"
            ax.set_title(rf"{letter}) {title}, $\gamma={gamma:g}$", loc="left", pad=9)
            ax.tick_params(length=3, width=.8)
        quantities = {}
        for name in ("forward_kl", "reverse_kl", "q_data1"):
            ax = axes[row, 1 if name == "q_data1" else 0]
            summary = stats(arrays[name][:, gi, shown])
            mean = np.asarray(summary["mean"])
            low, high = np.asarray(summary["ci95"])
            ax.plot(betas[shown], mean, color=COLORS[name])
            ax.fill_between(betas[shown], low, high, color=COLORS[name], alpha=.15, linewidth=0)
            quantities[name] = summary
        axes[row, 0].set_ylabel("KL (nats / spin)", labelpad=3)
        axes[row, 0].set_ylim(bottom=0.)
        axes[row, 0].yaxis.set_major_locator(MaxNLocator(4))
        axes[row, 1].axhline(reference, color=".25", linestyle="--", linewidth=1.)
        axes[row, 1].set_ylabel(r"Variance / $N$", labelpad=3)
        axes[row, 1].set_ylim(0., .95)
        axes[row, 1].set_yticks([0., .3, .6, .9])
        selected_stats = {"gamma": gamma, "common_forward_grid_beta": optimum,
                          "selection_grid_points": len(betas), "selected_at_grid_edge": selected in (0, len(betas)-1),
                          "forward_kl_gain": stats(forward[:, beta1] - forward[:, selected]),
                          "reverse_kl_gain_at_forward_selection": stats(arrays["reverse_kl"][:, gi, beta1] - arrays["reverse_kl"][:, gi, selected])}
        for name in ("q_data1", "q_teacher1", "covariance_error"):
            values = arrays[name][:, gi]
            selected_stats[name] = {"beta1": stats(values[:, beta1]), "selected_beta": stats(values[:, selected]),
                                    "paired_change": stats(values[:, selected] - values[:, beta1])}
        selections.append(selected_stats)
        plotted.append({"gamma": gamma, "betas": betas[shown].tolist(), "quantities": quantities})
    fig.legend(handles=[Line2D([], [], color=COLORS["forward_kl"], label="Forward KL"),
                        Line2D([], [], color=COLORS["reverse_kl"], label="Reverse KL"),
                        Line2D([], [], color=COLORS["forward_kl"], linestyle=":", linewidth=1., label="Minimum forward KL")],
               loc="lower left", bbox_to_anchor=(.105, .015), frameon=False,
               handlelength=1.6, handletextpad=.6, borderaxespad=0.)
    fig.legend(handles=[Line2D([], [], color=COLORS["q_data1"], label="Leading data mode"),
                        Line2D([], [], color=".25", linestyle="--", label="Teacher reference")],
               loc="lower left", bbox_to_anchor=(.61, .055), frameon=False,
               handlelength=1.6, handletextpad=.6, borderaxespad=0.)
    return fig


ARMS = [('gibbs|p=0.0625', 'Gibbs $\\nu_{\\mathrm{eff}}=3.125$', '#0072B2', '-'), ('gibbs|p=0.25', 'Gibbs $\\nu_{\\mathrm{eff}}=12.5$', '#D55E00', '-'), ('gibbs|p=1', 'Gibbs $\\nu_{\\mathrm{eff}}=50$', '#009E73', '-'), ('iid|p=1', 'Equilibrium iid', '#6B6B6B', '--'), ('exact|p=0', 'Exact gradient', '#000000', ':')]
PANELS = [('lam', 'A) Weight growth', '$\\lambda_1(WW^\\top)$', 'Leading eigenvalue of raw WW^T; no division by N, not an exact visible Ising coupling'), ('u_data', 'B) Mode alignment', 'Squared overlap', 'Squared overlap |c1^T u1(W)|^2 of the leading left singular vector of W and the unit leading eigenvector of uncentered D^T D/K'), ('fwd', 'C) Forward KL', 'KL (nats / spin)', 'KL(P_teacher || P_W)/N, exact enumeration at beta=1'), ('rev', 'D) Reverse KL', 'KL (nats / spin)', 'KL(P_W || P_teacher)/N, exact enumeration at beta=1')]


def plot_nu():
    with np.load(ROOT / "data/nu.npz") as z:
        times = z["time"]
        curves = {}
        for field, *_ in PANELS:
            curves[field] = {}
            for arm, *_ in ARMS:
                key = field+"__"+arm.replace("|","_").replace("=","_").replace(".","_")
                a = z[key]
                mean = a.mean(axis=0)
                half = student_t.ppf(.975,7)*a.std(axis=0,ddof=1)/np.sqrt(8)
                curves[field][arm] = {"mean":mean, "ci":np.array([mean-half,mean+half])}
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
        "font.size":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
        "path.simplify":False})
    fig,axes=plt.subplots(2,2,figsize=(5.5,4.7))
    fig.subplots_adjust(left=.115,right=.965,bottom=.20,top=.93,wspace=.40,hspace=.65)
    plotted={}
    for ax,(field,title,ylabel,definition) in zip(axes.flat,PANELS):
        ax.set_title(title,loc="left",fontsize=9,pad=10)
        ax.set_xlabel("Optimization time")
        ax.set_ylabel(ylabel)
        ax.set_xscale("symlog",linthresh=1.,linscale=1.,base=10)
        ax.set_xlim(0,3000)
        ax.xaxis.set_major_locator(FixedLocator([0,1,10,100,3000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(4))
        for arm,_,color,_ in ARMS:
            ci=curves[field][arm]["ci"]
            ax.fill_between(times,ci[0],ci[1],color=color,alpha=.10,linewidth=0)
        plotted[field]={"definition":definition,"arms":{}}
        for arm,label,color,linestyle in ARMS:
            mean=curves[field][arm]["mean"]
            line,=ax.plot(times,mean,color=color,ls=linestyle,lw=1.45 if arm!="exact|p=0" else 1.2,label=label)
            np.testing.assert_array_equal(line.get_xdata(),times)
            np.testing.assert_array_equal(line.get_ydata(),mean)
            plotted[field]["arms"][arm]={"color":color,"linestyle":linestyle,


                "initial_mean":float(mean[0]),"terminal_mean":float(mean[-1])}
        ax.margins(y=.05)
    handles=[Line2D([],[],color=color,ls=style,lw=1.45,label=label) for _,label,color,style in ARMS]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,
               frameon=False,fontsize=8,columnspacing=1.3,handlelength=1.7)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROOT/"generated")
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    for name in ["eta","gamma","beta","nu"]:
        plt.rcdefaults()
        if name=="eta":fig=plot_eta(read(ROOT/"data/eta.json"))
        elif name=="gamma":fig=plot_rbm(ROOT/"map/curve.json")
        elif name=="beta":
            with np.load(ROOT/"data/beta.npz") as z: arrays={k:z[k] for k in z.files}
            fig=plot_beta(arrays,read(ROOT/"data/beta.json"),SimpleNamespace(beta_min=.1,beta_max=3.))
        else:fig=plot_nu()
        fig.savefig(args.outdir/f"small_rbm_{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(args.outdir/f"small_rbm_{name}.png",dpi=200 if name in ["eta","gamma"] else 240)
        plt.close(fig)
        print(args.outdir/f"small_rbm_{name}.pdf")
if __name__=="__main__":main()
