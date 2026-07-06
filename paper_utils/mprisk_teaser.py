#!/usr/bin/env python3
"""
MPRisk teaser generator.

Outputs:
    mprisk_teaser.svg      -- full composed 3-panel teaser
    mprisk_panel_a.svg     -- panel (a) alone (for draw.io composition)
    mprisk_panel_b.svg     -- panel (b) alone
    mprisk_panel_c.svg     -- panel (c) alone

Only needs: numpy, matplotlib.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# True  -> text stays editable in draw.io / Inkscape (may reflow slightly)
# False -> text converted to paths (pixel-exact, not editable)
EDITABLE_TEXT = True
plt.rcParams["svg.fonttype"] = "none" if EDITABLE_TEXT else "path"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# palette (consistent with HolUE teasers)
CLASS_STROKE = ["#6C8EBF", "#82B366", "#D79B00", "#9673A6"]
UNKNOWN_PINK = "#F8CECC"
BOUNDARY_YEL = "#B09500"
GRAY = "#595959"
RISK_COLORS = {"FA": "#CC0000", "ID": "#D79B00", "FR": "#6C8EBF", "NS": "#9673A6"}

SYM_MISID = "\u25c6"  # black diamond  (misidentification)
SYM_FA = "\u2605"  # black star     (false acceptance)
SYM_FR = "\u00d7"  # times          (false rejection, corrupted known)
HALO = [pe.withStroke(linewidth=3, foreground="white")]


# ----------------------------------------------------------------------------
# Panel (a): embedding space, mixed prior, three error types
# ----------------------------------------------------------------------------
def panel_a(ax):
    centers = np.array([[3.2, 4.6], [5.0, 4.9], [4.2, 2.9], [6.3, 3.3]])
    s = 0.55  # class "spread" (plays the role of 1/sqrt(kappa_g))
    r_thr = 1.15  # accept/reject threshold distance

    xs = np.linspace(1.3, 9.2, 620)
    ys = np.linspace(0.6, 6.7, 480)
    X, Y = np.meshgrid(xs, ys)
    F = np.stack(
        [np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s)) for cx, cy in centers]
    )
    Fu = np.full_like(X, np.exp(-(r_thr**2) / (2 * s * s)))
    stack = np.concatenate([F, Fu[None]], axis=0)
    P = stack / stack.sum(0)
    H = -(P * np.log(P + 1e-12)).sum(0)
    reject = stack.argmax(0) == 4

    # ambiguity shading (entropy of p(c|z))
    ax.imshow(
        H,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        cmap="Blues",
        alpha=0.55,
        vmin=0,
        vmax=H.max(),
        aspect="auto",
        zorder=0,
    )

    # continuous-unknown (reject) region tinted pink
    pink = np.zeros(X.shape + (4,))
    pink[reject] = matplotlib.colors.to_rgba(UNKNOWN_PINK, alpha=0.40)
    ax.imshow(
        pink,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        aspect="auto",
        zorder=1,
    )

    # accept/reject boundary (threshold tau)
    ax.contour(
        X,
        Y,
        F.max(0) - Fu,
        levels=[0.0],
        colors=BOUNDARY_YEL,
        linestyles="--",
        linewidths=2.0,
        zorder=3,
    )
    # class-class boundaries inside accept region
    known_lab = F.argmax(0).astype(float)
    known_lab[reject] = np.nan
    ax.contour(
        X,
        Y,
        known_lab,
        levels=[0.5, 1.5, 2.5],
        colors=BOUNDARY_YEL,
        linestyles="--",
        linewidths=1.6,
        zorder=3,
    )

    # samples + prototypes
    rng = np.random.default_rng(3)
    for k, (cx, cy) in enumerate(centers):
        pts = np.column_stack([cx, cy]) + 0.34 * rng.standard_normal((6, 2))
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            marker="s",
            s=34,
            color=CLASS_STROKE[k],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(
            [cx],
            [cy],
            marker="o",
            s=110,
            color=CLASS_STROKE[k],
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
        )
        if k == 0:  # one high-quality sample: tight vMF cloud
            gx, gy = pts[0]
            ax.add_patch(
                Ellipse(
                    (gx, gy),
                    0.42,
                    0.42,
                    facecolor="#82B366",
                    alpha=0.35,
                    edgecolor="none",
                    zorder=4,
                )
            )
            ax.annotate(
                r"high $\kappa_{\mathbf{x}}$",
                (gx, gy),
                xytext=(gx - 1.55, gy + 0.75),
                fontsize=10,
                color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8),
            )

    # --- three error types -------------------------------------------------
    # misidentification: between two classes
    dpos = (centers[0] + centers[1]) / 2 + np.array([0.0, -0.10])
    ax.text(
        *dpos,
        SYM_MISID,
        fontsize=21,
        ha="center",
        va="center",
        color="black",
        path_effects=HALO,
        zorder=8,
    )
    # false acceptance: unknown probe just inside the accept boundary
    spos = centers[3] + 1.05 * np.array([0.80, -0.60])
    ax.text(
        *spos,
        SYM_FA,
        fontsize=21,
        ha="center",
        va="center",
        color="black",
        path_effects=HALO,
        zorder=8,
    )
    # false rejection: known sample drifted into the reject region
    xpos = np.array([7.45, 5.35])
    ax.add_patch(
        Ellipse(
            xpos,
            2.0,
            1.6,
            angle=15,
            facecolor="#82B366",
            alpha=0.28,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        *xpos,
        SYM_FR,
        fontsize=25,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        path_effects=HALO,
        zorder=8,
    )
    ax.add_patch(
        FancyArrowPatch(
            centers[1] + np.array([0.25, 0.15]),
            xpos - np.array([0.55, 0.20]),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="-|>",
            mutation_scale=13,
            color=GRAY,
            lw=1.3,
            zorder=7,
        )
    )
    ax.text(
        6.05,
        5.85,
        "corrupted sample\ndrifts out",
        fontsize=9.5,
        color=GRAY,
        ha="center",
        style="italic",
    )
    ax.text(
        8.35,
        4.55,
        r"low $\kappa_{\mathbf{x}}$"
        + "\n"
        + r"$\Rightarrow$ high $\mathcal{N}_0(\mathbf{x})$",
        fontsize=10.5,
        color="#7B5EA7",
        ha="center",
    )

    # region / threshold labels
    ax.text(
        2.05,
        1.15,
        "reject region:\ncontinuous unknown\n" + r"$c\in(K,\,K{+}1]$",
        fontsize=10.5,
        color="#B85450",
    )
    ax.text(
        6.55,
        1.05,
        r"$\tau$ (accept/reject)",
        fontsize=10.5,
        color=BOUNDARY_YEL,
        fontweight="bold",
    )

    # symbol key
    ax.text(
        0.01,
        -0.045,
        f"{SYM_MISID} misidentification    {SYM_FA} false acceptance"
        f"    {SYM_FR} false rejection (low-quality known)",
        transform=ax.transAxes,
        fontsize=10,
        color=GRAY,
        va="top",
    )

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
    ax.set_title(
        "(a) Mixed-prior Bayesian posterior", fontsize=13, fontweight="bold", pad=8
    )


# ----------------------------------------------------------------------------
# Panel (b): KL divergence != decision risk (counterexample)
# ----------------------------------------------------------------------------
def build_panel_b(fig, spec):
    cont = fig.add_subplot(spec)
    cont.axis("off")
    cont.set_title(
        "(b) Information gain " + r"$\neq$" + " decision risk",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    gs = spec.subgridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.35)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    axT.axis("off")

    labels = [r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", "unk"]
    pA = [0.48, 0.48, 0.01, 0.01, 0.02]  # split between two classes
    pB = [0.60, 0.10, 0.10, 0.10, 0.10]  # one dominant + diffuse tail

    for ax, p, sym, kl, risk, verdict, vcol in [
        (axA, pA, SYM_MISID, 1.39, 0.52, "KL: looks safe", "#2E7D32"),
        (axB, pB, "B", 0.75, 0.40, "KL: looks risky", "#C62828"),
    ]:
        bars = ax.bar(range(5), p, color="#6C8EBF", width=0.7)
        bars[-1].set_color(UNKNOWN_PINK)
        bars[-1].set_edgecolor("#B85450")
        ax.set_ylim(0, 0.78)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.set_title(f"probe {sym}", fontsize=11.5)
        ax.text(
            0.5,
            0.97,
            rf"$\mathrm{{KL}}\!=\!{kl:.2f}$"
            + "\n"
            + rf"$1{{-}}\max\,p\!=\!{risk:.2f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.text(
            0.5,
            -0.34,
            verdict,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color=vcol,
            style="italic",
        )

    axT.text(
        0.5,
        0.92,
        "KL ranking:   B  more uncertain than  " + SYM_MISID,
        ha="center",
        fontsize=11.5,
        color=GRAY,
    )
    axT.text(
        0.5,
        0.60,
        "True decision risk:   " + SYM_MISID + "  (0.52)   >   B  (0.40)",
        ha="center",
        fontsize=11.5,
        color="black",
    )
    axT.text(
        0.5,
        0.26,
        "ordering inverted!",
        ha="center",
        fontsize=13.5,
        color="#CC0000",
        fontweight="bold",
    )
    axT.text(
        0.5,
        -0.06,
        "empirical (IJB-C): inversion rate 0.55, Spearman \u22120.03",
        ha="center",
        fontsize=9.5,
        color=GRAY,
        style="italic",
    )
    return [axA, axB, axT]


# ----------------------------------------------------------------------------
# Panel (c): MPRisk decision-conditioned decomposition
# ----------------------------------------------------------------------------
def panel_c(ax):
    rows = [
        (SYM_FR + "  rejected (low quality)", {"FR": 0.22, "NS": 0.62}),
        (SYM_MISID + "  accepted, wrong ID", {"FA": 0.14, "ID": 0.66}),
        (SYM_FA + "  accepted, unknown", {"FA": 0.72, "ID": 0.18}),
    ]
    order = ["FA", "ID", "FR", "NS"]
    for y, (label, risks) in enumerate(rows):
        left = 0.0
        for comp in order:
            v = risks.get(comp, 0.0)
            if v > 0:
                ax.barh(
                    y,
                    v,
                    left=left,
                    height=0.52,
                    color=RISK_COLORS[comp],
                    edgecolor="white",
                    lw=0.8,
                )
                if v > 0.10:
                    ax.text(
                        left + v / 2,
                        y,
                        rf"$r_{{\mathrm{{{comp}}}}}$",
                        ha="center",
                        va="center",
                        fontsize=10.5,
                        color="white",
                        fontweight="bold",
                    )
                left += v
        ax.text(-0.03, y, label, ha="right", va="center", fontsize=11)

    # decision-conditioning annotations
    ax.text(
        0.92,
        2.0,
        "rejection risks " + r"$\equiv 0$" + "\n(accepted)",
        fontsize=9,
        color=GRAY,
        va="center",
    )
    ax.text(
        0.88,
        0.0,
        r"$P_0\!\approx\!1$ but diffuse:"
        + "\n"
        + r"$r_{\mathrm{NS}}=P_0\,\mathcal{N}_0$ catches it",
        fontsize=9,
        color="#7B5EA7",
        va="center",
    )

    ax.set_xlim(0, 1.32)
    ax.set_ylim(-1.85, 2.65)
    ax.axis("off")

    # legend
    handles = [
        Patch(facecolor=RISK_COLORS[c], label=rf"$r_{{\mathrm{{{c}}}}}$") for c in order
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=1.1,
        columnspacing=0.9,
        bbox_to_anchor=(1.02, 1.13),
    )

    # score formula
    ax.text(
        0.5,
        -0.72,
        r"$u_{\lambda}(\mathbf{x})=\lambda_{\mathrm{FA}}r_{\mathrm{FA}}"
        r"+\lambda_{\mathrm{ID}}r_{\mathrm{ID}}"
        r"+\lambda_{\mathrm{FR}}r_{\mathrm{FR}}"
        r"+\lambda_{\mathrm{NS}}r_{\mathrm{NS}}$",
        ha="center",
        fontsize=13,
        transform=ax.transData,
    )
    ax.text(
        0.5,
        -1.15,
        r"$\lambda$ tuned on validation for the target PRR@FPIR",
        ha="center",
        fontsize=9.5,
        color=GRAY,
        style="italic",
    )

    # result badge
    ax.add_patch(
        FancyBboxPatch(
            (0.03, -1.80),
            0.94,
            0.42,
            boxstyle="round,pad=0.06",
            facecolor="#FFF2CC",
            edgecolor="#D6B656",
        )
    )
    ax.text(
        0.5,
        -1.59,
        "IJB-C @ FPIR 0.2:  PRR 0.91  vs  0.73 (HolUE)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    ax.set_title(
        "(c) MPRisk: decision-conditioned risk", fontsize=13, fontweight="bold", pad=8
    )


# ----------------------------------------------------------------------------
# Compose full teaser
# ----------------------------------------------------------------------------
def make_full_teaser(fname="mprisk_teaser.svg"):
    fig = plt.figure(figsize=(17.0, 5.6))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.0, 1.1],
        left=0.02,
        right=0.99,
        top=0.90,
        bottom=0.10,
        wspace=0.30,
    )
    axa = fig.add_subplot(gs[0])
    panel_a(axa)
    build_panel_b(fig, gs[1])
    axc = fig.add_subplot(gs[2])
    panel_c(axc)

    # story arrows between panels (figure coordinates)
    for x0, x1, label in [
        (0.395, 0.425, "summarize\nposterior?"),
        (0.695, 0.725, "replace KL with\nexpected decision loss"),
    ]:
        fig.add_artist(
            FancyArrowPatch(
                (x0, 0.52),
                (x1, 0.52),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=22,
                color="#666666",
                lw=3,
            )
        )
        fig.text(
            (x0 + x1) / 2,
            0.585,
            label,
            ha="center",
            fontsize=9,
            color=GRAY,
            style="italic",
        )

    fig.savefig(fname)
    plt.close(fig)
    print(f"wrote {fname}")


def make_individual_panels():
    fig = plt.figure(figsize=(6.8, 5.4))
    panel_a(fig.add_axes([0.02, 0.08, 0.96, 0.82]))
    fig.savefig("mprisk_panel_a.svg")
    plt.close(fig)

    fig = plt.figure(figsize=(5.2, 5.4))
    build_panel_b(
        fig, fig.add_gridspec(1, 1, left=0.05, right=0.97, top=0.88, bottom=0.08)[0]
    )
    fig.savefig("mprisk_panel_b.svg")
    plt.close(fig)

    fig = plt.figure(figsize=(6.0, 5.4))
    panel_c(fig.add_axes([0.20, 0.05, 0.78, 0.85]))
    fig.savefig("mprisk_panel_c.svg")
    plt.close(fig)
    print("wrote mprisk_panel_a.svg, mprisk_panel_b.svg, mprisk_panel_c.svg")


if __name__ == "__main__":
    make_full_teaser()
    make_individual_panels()
