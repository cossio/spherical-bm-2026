# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "matplotlib>=3.10"]
# ///
"""Render independently reproduced Appendix temperature tuning."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    d=np.load(root/"data/beta.npz")
    beta,mean,ci=d["betas"],d["mean"],d["ci99"]
    plt.rcParams.update({"font.size":9,"axes.labelsize":9,"xtick.labelsize":8,
                         "ytick.labelsize":8,"legend.fontsize":8,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.8))
    colors=["#0072B2","#D55E00","#9467bd","#8c564b"]
    for a,title in zip(axes,["A) KL loss","B) Mode expression"]):
        a.set_xscale("log"); a.set_xlabel(r"$\beta$")
        a.xaxis.set_major_locator(FixedLocator([.2,1,5]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{x:g}"))
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_title(title,loc="left",fontsize=9,pad=10)
        a.axvline(1,color=".65",lw=.8,zorder=0)
    for j,label in enumerate(["Forward KL","Reverse KL"]):
        axes[0].plot(beta,mean[:,0,j],color=colors[j],label=label)
        axes[0].fill_between(beta,mean[:,0,j]-ci[:,0,j],mean[:,0,j]+ci[:,0,j],color=colors[j],alpha=.15)
        opt=d["continuous_optima"][j]
        axes[0].axvline(opt,color=colors[j],ls=":" ,lw=1)
    for j,label in [(2,"Mode variance"),(3,"Covariance gap")]:
        axes[1].plot(beta,mean[:,0,j],color=colors[j],label=label)
        axes[1].fill_between(beta,mean[:,0,j]-ci[:,0,j],mean[:,0,j]+ci[:,0,j],color=colors[j],alpha=.15)
    axes[1].axhline(d["teacher_variance"].mean(),color=".25",ls="--",label="Teacher variance")
    axes[0].set_ylabel("KL (nats)")
    axes[1].set_ylabel(r"Variance / $N$")
    for a in axes:
        a.legend(loc="upper left",bbox_to_anchor=(-.10,-.29),fontsize=8,
                 frameon=False,handlelength=1.3,handletextpad=.5)
    fig.subplots_adjust(left=.095,right=.99,top=.86,bottom=.38,wspace=.40)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,default=root/"reproduced")
    out=parser.parse_args().outdir; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"generation_temperature.pdf")
    fig.savefig(out/"generation_temperature.png",dpi=160)


if __name__=="__main__":
    main()
