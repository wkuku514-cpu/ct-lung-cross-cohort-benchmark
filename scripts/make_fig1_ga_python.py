"""Reproducibly build Fig. 1 and the graphical abstract from verified study counts.

AI-assisted code generated for the manuscript revision; all numerical content is
read from the frozen manuscript sources and reviewed by the authors.
"""
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(r"C:\Users\kuku\Desktop\cancer-imaging-ct\paper_v1")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
})

BLUE = "#0072B2"; ORANGE = "#E69F00"; GREEN = "#009E73"
VERM = "#D55E00"; SKY = "#56B4E9"; GREY = "#F5F5F5"; DARK = "#333333"

def box(ax, xy, wh, text, fc=GREY, ec=DARK, size=8, weight="normal"):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                       facecolor=fc, edgecolor=ec, linewidth=0.75)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size,
            color=DARK, weight=weight, linespacing=1.15)
    return p

def arrow(ax, a, b, color=DARK, rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                linewidth=1.25, color=color,
                                connectionstyle=f"arc3,rad={rad}", zorder=10))

def save(fig, stem, dpi=300):
    fig.savefig(OUT/f"{stem}.pdf")
    fig.savefig(OUT/f"{stem}.png", dpi=dpi,
                facecolor="white")
    plt.close(fig)

def make_fig1():
    fig, ax = plt.subplots(figsize=(180/25.4, 118/25.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.985, "a", weight="bold", fontsize=9, va="top")
    ax.text(0.50, 0.97, "Study flow and evaluation design", ha="center",
            va="top", weight="bold", fontsize=9)

    # Source collections
    sources = [("LUNG1\nn=422", BLUE), ("Radiogenomics\nn=211", ORANGE),
               ("LungCT-Diagnosis\nn=61", SKY), ("QIN-LUNG-CT\nn=47", GREEN),
               ("LUAD-CT-Survival\nn=40", VERM)]
    xs = [0.02, .215, .41, .605, .80]
    targets = [.42, .46, .50, .54, .58]
    for x, (label, col), target_x in zip(xs, sources, targets):
        box(ax, (x, .80), (.175, .105), label, fc=GREY, ec=DARK, size=7.5, weight="bold")
        arrow(ax, (x+.0875, .80), (target_x, .71), color=col)

    box(ax, (.37, .61), (.26, .10), "781 imaging entries", fc=GREY, ec=DARK, weight="bold")
    arrow(ax, (.50, .61), (.50, .53))
    box(ax, (.34, .44), (.32, .09), "741 distinct original patient IDs", fc=GREY, ec=DARK, weight="bold")
    arrow(ax, (.50, .44), (.50, .34))
    box(ax, (.35, .25), (.30, .09), "633 with survival time and event", fc=GREY, ec=DARK, weight="bold")

    # Split to modelling roles
    arrow(ax, (.43, .25), (.24, .17), color=BLUE)
    arrow(ax, (.57, .25), (.76, .17), color=ORANGE)
    box(ax, (.08, .07), (.32, .10), "Development\nLUNG1 (n=422)", fc=GREY, ec=DARK, weight="bold")
    box(ax, (.60, .07), (.32, .10), "Target-cohort evaluation\nRadiogenomics (n=211)", fc=GREY, ec=DARK, weight="bold")
    ax.text(.50, .015,
            "Transductive ComBat used unlabeled target features; this is target-domain adaptation, not untouched external validation.",
            ha="center", va="bottom", fontsize=7.0, color="#555555")
    fig.subplots_adjust(0,0,1,1)
    save(fig, "Fig1", 300)

def make_ga():
    # Exact requested canvas: 1328 x 531 px (width x height).
    dpi = 100
    fig = plt.figure(figsize=(13.28, 5.31), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(.02,.95,"Cross-cohort CT survival modelling",fontsize=18,weight="bold",va="top",color=DARK)

    # Left quarter: scale and cohorts
    ax.text(.12,.84,"DATA",ha="center",weight="bold",fontsize=11,color=BLUE)
    box(ax,(.025,.62),(.19,.14),"5 public CT collections\n781 imaging entries",fc="#EAF4FA",ec=BLUE,size=10,weight="bold")
    box(ax,(.025,.42),(.19,.14),"741 original patient IDs\n633 survival eligible",fc="white",ec=DARK,size=9)
    box(ax,(.025,.22),(.19,.14),"Development 422\nTarget cohort 211",fc="#FFF4DF",ec=ORANGE,size=9)

    # Centre half: actual Fig 2 PCA coordinates
    pca = pd.read_csv(ROOT/"figures"/"source_data"/"fig2_pca_coordinates.csv")
    before = pca[pca["panel"].astype(str).eq("a")]
    after = pca[pca["panel"].astype(str).eq("b")]
    for data, left, title in [(before,.285,"Before ComBat"),(after,.515,"After ComBat")]:
        iax = fig.add_axes([left,.29,.19,.46])
        for cohort, marker, col in [("LUNG1","o",BLUE),("Radiogenomics","^",ORANGE)]:
            d=data[data.cohort.eq(cohort)]
            iax.scatter(d.PC1,d.PC2,s=8,alpha=.48,c=col,marker=marker,edgecolors="none")
        iax.axhline(0,color="#DDDDDD",lw=.6); iax.axvline(0,color="#DDDDDD",lw=.6)
        iax.set_title(title,fontsize=10,weight="bold",pad=3)
        iax.set_xlabel("PC1",fontsize=8); iax.set_ylabel("PC2",fontsize=8)
        iax.tick_params(labelsize=7,width=.75,length=2)
        iax.spines[['top','right']].set_visible(False)
    ax.text(.495,.84,"MARGINAL SHIFT",ha="center",weight="bold",fontsize=11,color=BLUE)
    ax.text(.495,.20,"Radiomics mean |d|  0.755 → 0.012",ha="center",fontsize=10,weight="bold")

    # Right quarter: three result layers
    ax.text(.855,.84,"WHAT CHANGED?",ha="center",weight="bold",fontsize=11,color=VERM)
    cards=[("Orientation repaired","slope −0.327 → +0.815",GREEN),
           ("Calibration improved","Brier −40% to −52%",SKY),
           ("Discrimination unchanged","C-index 0.507 (0.431–0.580)",VERM)]
    for y,(head,sub,col) in zip([.64,.45,.26],cards):
        box(ax,(.755,y),(.205,.13),f"{head}\n{sub}",fc="white",ec=col,size=9,weight="bold")

    # Footer banner
    banner=FancyBboxPatch((.015,.035),.97,.09,boxstyle="round,pad=0.008,rounding_size=.012",
                          facecolor=DARK,edgecolor=DARK,linewidth=.75)
    ax.add_patch(banner)
    ax.text(.50,.08,"ComBat corrects orientation, not discrimination",ha="center",va="center",
            color="white",fontsize=15,weight="bold")
    fig.savefig(OUT/"Graphical_Abstract.pdf",dpi=dpi,facecolor="white")
    fig.savefig(OUT/"Graphical_Abstract.png",dpi=dpi,facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    make_fig1(); make_ga()
    print("Created Fig1 and Graphical Abstract")
