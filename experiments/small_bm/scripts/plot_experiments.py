# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "matplotlib>=3.10"]
# ///
"""Vector figures for the independently reconstructed BM Appendix experiments."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reproduced"
COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#6B6B6B"]


def save(fig,name):
    path=OUT; path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path/f"{name}.pdf")
    fig.savefig(path/f"{name}.png",dpi=160)
    plt.close(fig)


def posterior():
    d=np.load(ROOT/"data/posterior.npz")
    x=d["etas"]; mean=d["mean"]; ci=d["ci99_halfwidth"]
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.8))
    for a,title in zip(axes,["A) Forward predictive KL","B) Reverse predictive KL"]):
        a.set_title(title,loc="left",fontsize=9,pad=10)
        a.set_xscale("log"); a.set_xlabel(r"$\eta$")
        a.xaxis.set_major_locator(FixedLocator([.05,.2,1,5,20]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
        a.xaxis.set_minor_formatter(NullFormatter())
        a.axvline(1,color=".65",ls=":",zorder=0)
        a.set_ylabel("KL (nats)")
    for g,gamma in enumerate(d["gammas"]):
        for j in range(2):
            a=axes[j]
            a.plot(x,mean[g,:,j],color=COLORS[g],marker=".",lw=1.6)
            a.fill_between(x,mean[g,:,j]-ci[g,:,j],mean[g,:,j]+ci[g,:,j],color=COLORS[g],alpha=.15)
            idx=mean[g,:,j].argmin()
            a.plot(x[idx],mean[g,idx,j],"o",mfc="white",mec=COLORS[g],ms=7)
    handles=[Line2D([],[],color=COLORS[g],label=rf"$\gamma={gamma:g}$")
             for g,gamma in enumerate(d["gammas"])]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,
               frameon=False,fontsize=8,columnspacing=1.2,handlelength=1.7)
    fig.subplots_adjust(left=.115,right=.985,bottom=.28,top=.87,wspace=.37)
    save(fig,"posterior_tempering")


def dynamics():
    d=np.load(ROOT/"data/dynamics.npz")
    metadata=json.loads((ROOT/"data/dynamics_metadata.json").read_text())
    # Match the mean-loss update to dJ/dt = K/(2N) * (c_data-c_model-N*gamma*J/K).
    learning_rate=Fraction(str(metadata["learning_rate"]))
    learning_dt=2*metadata["n"]*learning_rate/metadata["k"]
    nu_eff=[int(m)/(metadata["n"]*learning_dt) for m in d["single_site_updates"]]
    x=d["checkpoints"]; mean=d["mean"]; ci=d["ci95_halfwidth"]
    fig,axes=plt.subplots(2,2,figsize=(5.5,4.7))
    for j,(a,title,ylabel) in enumerate(zip(axes.flat,
        ["A) Coupling growth","B) Mode alignment","C) Forward KL","D) Reverse KL"],
        ["Leading eigenvalue","Squared overlap","KL (nats)","KL (nats)"])):
        a.set_title(title,loc="left",fontsize=9,pad=10)
        a.set_xlabel("Coupling updates"); a.set_ylabel(ylabel)
        for r,rate in enumerate(d["single_site_updates"]):
            a.plot(x,mean[r,:,j],color=COLORS[r],lw=1.6)
            a.fill_between(x,mean[r,:,j]-ci[r,:,j],mean[r,:,j]+ci[r,:,j],color=COLORS[r],alpha=.10)
            if j>=2:
                idx=mean[r,:,j].argmin()
                a.plot(x[idx],mean[r,idx,j],"o",ms=5,mfc="white",mec=COLORS[r])
        a.axhline(d["map_metrics"][:,j].mean(),color="black",ls=":",lw=1.2)
        a.set_xlim(0,x[-1])
        if j==1:
            a.set_ylim(.90,1.01)
    handles=[Line2D([],[],color=c,label=rf"$\nu_{{\mathrm{{eff}}}}={rate}$")
             for c,rate in zip(COLORS,nu_eff)]
    handles.append(Line2D([],[],color="black",ls=":",label="Exact MAP"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,
               frameon=False,fontsize=8,columnspacing=1.8,handlelength=1.7)
    fig.subplots_adjust(left=.115,right=.97,bottom=.20,top=.93,wspace=.40,hspace=.65)
    save(fig,"persistent_training")


def gamma():
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.titlesize":9, "ps.fonttype":42})
    d=np.load(ROOT/"data/gamma_final.npz")
    x=d["gamma"]; mean=d["mean"]; halfwidth=d["mc_ci95_halfwidth"]
    fig,axes=plt.subplots(1,2,figsize=(5.5,3.2))
    for a,title in zip(axes,["A) Reverse KL","B) Generated order"]):
        a.set_title(title,loc="left",fontsize=9,pad=10)
        a.set_xscale("log"); a.set_xlabel(r"$\gamma$")
        a.xaxis.set_minor_formatter(NullFormatter())
    for i,eta in enumerate(d["eta"]):
        axes[0].plot(x,mean[i,:,0],color=COLORS[i],label=rf"$\eta={eta:g}$")
        axes[0].fill_between(x,mean[i,:,0]-halfwidth[i,:,0],mean[i,:,0]+halfwidth[i,:,0],color=COLORS[i],alpha=.12)
    idx=int(np.flatnonzero(np.isclose(d["eta"],1))[0])
    display=x>=1e-6
    xd=x[display]
    uniform=mean[idx,-1,0]
    axes[0].axhline(uniform,color=".25",ls=":",lw=1)
    axes[0].set_ylabel("KL (nats)")
    for j,label in [(3,"Retrieval"),(4,"Squared overlap"),(5,"Mode variance / N")]:
        axes[1].plot(xd,mean[idx,display,j],color=COLORS[j-3],label=label)
        axes[1].fill_between(xd,mean[idx,display,j]-halfwidth[idx,display,j],mean[idx,display,j]+halfwidth[idx,display,j],color=COLORS[j-3],alpha=.12)
    for a in axes[1:]:
        a.axvline(1,color=".6",ls=":",lw=.8)
        a.set_xlim(1e-6,1e6)
        a.xaxis.set_major_locator(FixedLocator([1e-6,1,1e6]))
    axes[0].xaxis.set_major_locator(FixedLocator([1e-10,1e-2,1e6]))
    axes[1].set_ylabel("Generated order")
    for a in axes:
        a.legend(loc="upper left",bbox_to_anchor=(-.10,-.29),fontsize=8,frameon=False,
                 handlelength=1.3,handletextpad=.5)
    fig.subplots_adjust(left=.115,right=.965,top=.87,bottom=.43,wspace=.40)
    curve=json.loads((ROOT/"data/curve.json").read_text())
    axes[0].plot(curve["gamma"],curve["reverse_kl_mean"],color="black",linestyle="--",linewidth=1.5,label="MAP",zorder=5)
    axes[0].legend(loc="upper left",bbox_to_anchor=(-.10,-.29),fontsize=8,ncol=2,columnspacing=1,frameon=False,handlelength=1.3,handletextpad=.5)
    save(fig,"regularization_double_descent")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",choices=["posterior","dynamics","gamma","all"])
    parser.add_argument("--outdir",type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.outdir
    plt.rcParams.update({"font.size":9,"axes.labelsize":9,"xtick.labelsize":8,
                         "ytick.labelsize":8,"legend.fontsize":8,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    if args.experiment in ["posterior","all"]:
        posterior()
    if args.experiment in ["dynamics","all"]:
        dynamics()
    if args.experiment in ["gamma","all"]:
        gamma()
