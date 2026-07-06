#!/usr/bin/env python3
"""
MPRisk teaser generator (fixed text layout):
  - panel (a): matplotlib image embedded as PNG inside the .drawio
  - panels (b), (c), arrows: native editable draw.io shapes
  - formulas: INLINE LaTeX \( ... \)  (display math $$...$$ broke layout)

Outputs:
  mprisk_teaser.drawio
  mprisk_panel_a.png
"""

import base64
import io
import xml.sax.saxutils as su

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, FancyArrowPatch

# ======================================================================
# palette
# ======================================================================
CLS_LIGHT = ["#DAE8FC", "#D5E8D4", "#FFE6CC", "#E1D5E7"]
CLS_DARK = ["#6C8EBF", "#82B366", "#D79B00", "#9673A6"]
PINK, PINK_D = "#F8CECC", "#B85450"
YEL, GRAY = "#B09500", "#595959"
R = {"FA": "#CC0000", "ID": "#D79B00", "FR": "#6C8EBF", "NS": "#9673A6"}
DIAMOND, STAR, CROSS = "\u25c6", "\u2605", "\u2715"
HALO = [pe.withStroke(linewidth=3.5, foreground="white")]


# ======================================================================
# 1. Panel (a) in matplotlib -> PNG bytes
# ======================================================================
def render_panel_a(dpi=220):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig, ax = plt.subplots(figsize=(6.6, 5.35))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.09)

    centers = np.array([[3.2, 4.6], [5.0, 4.9], [4.2, 2.9], [6.3, 3.3]])
    s = 0.55
    r_thr = 1.15

    xs = np.linspace(1.2, 9.3, 640)
    ys = np.linspace(0.5, 6.8, 500)
    X, Y = np.meshgrid(xs, ys)
    F = np.stack(
        [np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s)) for cx, cy in centers]
    )
    Fu = np.full_like(X, np.exp(-(r_thr**2) / (2 * s * s)))
    stack = np.concatenate([F, Fu[None]], axis=0)
    P = stack / stack.sum(0)
    H = -(P * np.log(P + 1e-12)).sum(0)
    reject = stack.argmax(0) == 4

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
    pink = np.zeros(X.shape + (4,))
    pink[reject] = matplotlib.colors.to_rgba(PINK, alpha=0.42)
    ax.imshow(
        pink,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        aspect="auto",
        zorder=1,
    )

    ax.contour(
        X,
        Y,
        F.max(0) - Fu,
        levels=[0.0],
        colors=YEL,
        linestyles="--",
        linewidths=2.2,
        zorder=3,
    )
    lab = F.argmax(0).astype(float)
    lab[reject] = np.nan
    ax.contour(
        X,
        Y,
        lab,
        levels=[0.5, 1.5, 2.5],
        colors=YEL,
        linestyles="--",
        linewidths=1.6,
        zorder=3,
    )

    rng = np.random.default_rng(3)
    for k, (cx, cy) in enumerate(centers):
        ax.add_patch(
            Ellipse(
                (cx, cy),
                2.5 * s * 2,
                2.0 * s * 2,
                facecolor=CLS_LIGHT[k],
                alpha=0.55,
                edgecolor="none",
                zorder=2,
            )
        )
        pts = np.array([cx, cy]) + 0.32 * rng.standard_normal((6, 2))
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            marker="s",
            s=38,
            color=CLS_DARK[k],
            edgecolor="white",
            lw=0.7,
            zorder=5,
        )
        ax.scatter(
            [cx],
            [cy],
            marker="o",
            s=130,
            color=CLS_DARK[k],
            edgecolor="black",
            lw=1.1,
            zorder=6,
        )
        if k == 0:
            gx, gy = pts[0]
            ax.add_patch(
                Ellipse(
                    (gx, gy),
                    0.42,
                    0.42,
                    facecolor="#82B366",
                    alpha=0.4,
                    edgecolor="none",
                    zorder=4,
                )
            )
            ax.annotate(
                r"high $\kappa_{\mathbf{x}}$",
                (gx, gy),
                xytext=(gx - 1.7, gy + 0.8),
                fontsize=11.5,
                color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9),
            )

    dpos = (centers[0] + centers[1]) / 2 + np.array([0.0, -0.1])
    ax.text(
        *dpos,
        DIAMOND,
        fontsize=23,
        ha="center",
        va="center",
        path_effects=HALO,
        zorder=8,
    )
    spos = centers[3] + np.array([0.86, -0.66])
    ax.text(
        *spos, STAR, fontsize=23, ha="center", va="center", path_effects=HALO, zorder=8
    )

    xpos = np.array([7.55, 5.35])
    ax.add_patch(
        Ellipse(
            xpos,
            2.1,
            1.65,
            angle=15,
            facecolor="#82B366",
            alpha=0.30,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        *xpos,
        CROSS,
        fontsize=26,
        fontweight="bold",
        ha="center",
        va="center",
        path_effects=HALO,
        zorder=8,
    )
    ax.add_patch(
        FancyArrowPatch(
            centers[1] + [0.25, 0.15],
            xpos - [0.55, 0.20],
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="-|>",
            mutation_scale=14,
            color="#666666",
            lw=1.4,
            zorder=7,
        )
    )
    ax.text(
        6.0,
        6.05,
        "corrupted known sample\ndrifts out",
        fontsize=10.5,
        color=GRAY,
        ha="center",
        style="italic",
    )
    ax.text(
        8.35,
        4.42,
        r"low $\kappa_{\mathbf{x}}$"
        "\n"
        r"$\Rightarrow$ high $\mathcal{N}_0(\mathbf{x})$",
        fontsize=12,
        color="#7B5EA7",
        ha="center",
    )

    ax.text(
        1.95,
        1.25,
        "reject region:\ncontinuous unknown\n" + r"$c\in(K,\,K{+}1]$",
        fontsize=11.5,
        color=PINK_D,
    )
    ax.text(
        6.35,
        1.05,
        r"$\tau$ (accept / reject)",
        fontsize=12,
        color=YEL,
        fontweight="bold",
    )
    ax.text(
        0.01,
        -0.035,
        f"{DIAMOND} misidentification    {STAR} false acceptance    "
        f"{CROSS} false rejection (low-quality known)    "
        "\u25a0 sample   \u25cf prototype",
        transform=ax.transAxes,
        fontsize=9.5,
        color=GRAY,
        va="top",
    )

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    png = buf.getvalue()
    with open("mprisk_panel_a.png", "wb") as f:
        f.write(png)
    return png


# ======================================================================
# 2. draw.io XML builder (math enabled)
# ======================================================================
class DrawioDoc:
    def __init__(self, w=1700, h=620):
        self.cells, self._i, self.w, self.h = [], 1, w, h

    def _nid(self):
        self._i += 1
        return f"c{self._i}"

    def group(self, x, y, w, h):
        g = self._nid()
        self.cells.append(
            f'<mxCell id="{g}" style="group" vertex="1" connectable="0" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>'
        )
        return g

    def node(self, style, x, y, w, h, value="", parent="1"):
        n = self._nid()
        v = su.escape(value, {'"': "&quot;"})
        self.cells.append(
            f'<mxCell id="{n}" value="{v}" style="{style}" vertex="1" '
            f'parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>'
        )
        return n

    def edge(self, style, p0, p1, parent="1", points=None):
        e = self._nid()
        wp = ""
        if points:
            wp = (
                '<Array as="points">'
                + "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in points)
                + "</Array>"
            )
        self.cells.append(
            f'<mxCell id="{e}" style="{style}" edge="1" parent="{parent}">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{p0[0]}" y="{p0[1]}" as="sourcePoint"/>'
            f'<mxPoint x="{p1[0]}" y="{p1[1]}" as="targetPoint"/>{wp}'
            f"</mxGeometry></mxCell>"
        )
        return e

    def save(self, fname):
        xml = (
            '<mxfile host="app.diagrams.net" type="device">'
            '<diagram id="mprisk" name="MPRisk teaser">'
            f'<mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" '
            f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
            f'page="1" pageScale="1" pageWidth="{self.w}" '
            f'pageHeight="{self.h}" math="1" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            + "".join(self.cells)
            + "</root></mxGraphModel></diagram></mxfile>"
        )
        with open(fname, "w", encoding="utf-8") as f:
            f.write(xml)
        print("wrote", fname)


# ---- style helpers ---------------------------------------------------
def TXT(
    size=12, color="#000000", bold=False, italic=False, align="center", valign="middle"
):
    fs = (1 if bold else 0) + (2 if italic else 0)
    return (
        f"text;html=1;align={align};verticalAlign={valign};"
        f"whiteSpace=wrap;fontSize={size};fontColor={color};"
        f"fontStyle={fs};"
    )


def RECT(
    fill,
    stroke="none",
    opacity=None,
    rounded=False,
    sw=1,
    font_color="#FFFFFF",
    font_size=11,
    bold=True,
):
    s = (
        f"rounded={'1' if rounded else '0'};whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
        f"fontColor={font_color};fontSize={font_size};"
        f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;"
    )
    if opacity is not None:
        s += f"opacity={opacity};"
    return s


# ======================================================================
# 3. Panels  (all math is INLINE:  \( ... \) )
# ======================================================================
def panel_a(d, png_bytes):
    g = d.group(20, 40, 540, 510)
    d.node(RECT("none", "#CCCCCC", rounded=True), 0, 0, 540, 510, parent=g)
    d.node(
        TXT(16, bold=True),
        0,
        4,
        540,
        26,
        r"(a) Mixed-prior Bayesian posterior \(p(c\mid\mathbf{x})\)",
        parent=g,
    )
    b64 = base64.b64encode(png_bytes).decode()
    d.node(
        f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition="
        f"bottom;verticalAlign=top;image=data:image/png,{b64};",
        10,
        34,
        520,
        421,
        parent=g,
    )
    d.node(
        TXT(11, GRAY),
        10,
        460,
        520,
        40,
        r"\(P_i(\mathbf{x})=\int_{\mathbb{S}^{d-1}} "
        r"p(c{=}i\mid\mathbf{z})\,p(\mathbf{z}\mid\mathbf{x})\,"
        r"d\mathbf{z},\quad P_0+\sum_i P_i = 1\)",
        parent=g,
    )


def panel_b(d):
    g = d.group(610, 40, 420, 510)
    d.node(RECT("none", "#CCCCCC", rounded=True), 0, 0, 420, 510, parent=g)
    d.node(
        TXT(16, bold=True),
        0,
        4,
        420,
        26,
        r"(b) information gain \(\neq\) decision risk",
        parent=g,
    )

    charts = [
        (
            20,
            f"probe {DIAMOND}",
            [0.48, 0.48, 0.02, 0.01, 0.01],
            r"\(\mathrm{KL}=0.73\) (lower)",
            r"\(1-\max_c p = 0.52\) (higher!)",
            "KL ranks it SAFER \u2717",
            "#CC0000",
        ),
        (
            240,
            "probe B",
            [0.60, 0.38, 0.01, 0.005, 0.005],
            r"\(\mathrm{KL}=0.84\) (higher)",
            r"\(1-\max_c p = 0.40\) (lower)",
            "KL ranks it riskier",
            GRAY,
        ),
    ]
    labels = [r"\(c_1\)", r"\(c_2\)", r"\(c_3\)", r"\(c_4\)", "unk"]
    base_y, hmax = 250, 145
    for x0, title, p, kl, risk, verdict, vcol in charts:
        d.node(TXT(12, bold=True), x0, 40, 160, 18, title, parent=g)
        d.node(
            TXT(10, GRAY, align="left"),
            x0 - 8,
            62,
            180,
            40,
            f"{kl}<br>{risk}",
            parent=g,
        )
        for i, pi in enumerate(p):
            h = max(2, round(pi * hmax))
            fill = PINK if i == 4 else "#6C8EBF"
            stroke = PINK_D if i == 4 else "none"
            d.node(RECT(fill, stroke), x0 + i * 30, base_y - h, 22, h, parent=g)
            d.node(
                TXT(9, GRAY), x0 + i * 30 - 4, base_y + 4, 30, 16, labels[i], parent=g
            )
        d.edge(
            "endArrow=none;strokeColor=#666666;strokeWidth=1;html=1;",
            (x0 - 5, base_y),
            (x0 + 150, base_y),
            parent=g,
        )
        d.node(TXT(10, vcol, italic=True), x0 - 10, 274, 180, 18, verdict, parent=g)

    d.node(
        TXT(12, GRAY),
        10,
        306,
        400,
        20,
        rf"KL ordering: {DIAMOND} \((0.73)\) &lt; B \((0.84)\)",
        parent=g,
    )
    d.node(
        TXT(12),
        10,
        330,
        400,
        20,
        rf"decision risk: {DIAMOND} \((0.52)\) &gt; B \((0.40)\)",
        parent=g,
    )
    d.node(
        TXT(15, "#CC0000", bold=True),
        10,
        358,
        400,
        24,
        "risk ordering inverted!",
        parent=g,
    )
    d.node(
        TXT(10, GRAY, italic=True),
        10,
        386,
        400,
        20,
        r"empirically on IJB-C: inversion rate \(0.55\), " r"Spearman \(-0.03\)",
        parent=g,
    )
    d.node(
        TXT(10, GRAY),
        10,
        412,
        400,
        60,
        r"\(D_{\mathrm{KL}}(p(c\mid\mathbf{x})\,\|\,p(c))\) measures "
        r"information gain, not the expected loss "
        r"\(1-\max_a p(a\mid\mathbf{x})\) of the OSR decision",
        parent=g,
    )


def panel_c(d):
    g = d.group(1080, 40, 580, 510)
    d.node(RECT("none", "#CCCCCC", rounded=True), 0, 0, 580, 510, parent=g)
    d.node(
        TXT(16, bold=True),
        0,
        4,
        580,
        26,
        "(c) MPRisk: decision-conditioned risk",
        parent=g,
    )

    for i, comp in enumerate(["FA", "ID", "FR", "NS"]):
        x = 110 + i * 100
        d.node(RECT(R[comp]), x, 44, 14, 14, parent=g)
        d.node(
            TXT(11, align="left"),
            x + 18,
            42,
            76,
            18,
            rf"\(r_{{\mathrm{{{comp}}}}}\)",
            parent=g,
        )

    scale, x_bar, bh = 280, 210, 34
    rows = [
        (f"{STAR}&nbsp; accepted, unknown", [("FA", 0.72), ("ID", 0.18)], 82),
        (f"{DIAMOND}&nbsp; accepted, wrong ID", [("FA", 0.14), ("ID", 0.66)], 142),
        (f"{CROSS}&nbsp; rejected (low quality)", [("FR", 0.22), ("NS", 0.62)], 202),
    ]
    for label, segs, y in rows:
        d.node(TXT(11, align="right"), 0, y, 200, bh, label, parent=g)
        x = x_bar
        for comp, v in segs:
            w = round(v * scale)
            val = rf"\(r_{{\mathrm{{{comp}}}}}\)" if w > 45 else ""
            d.node(
                RECT(R[comp], "#FFFFFF", sw=1, font_size=11), x, y, w, bh, val, parent=g
            )
            x += w

    d.node(
        TXT(9, GRAY, align="left"),
        482,
        92,
        96,
        70,
        r"rejection risks \(\equiv 0\) when accepted",
        parent=g,
    )
    d.node(
        TXT(10, "#7B5EA7", align="left"),
        210,
        244,
        360,
        40,
        r"\(P_0\approx 1\) but diffuse \(\Rightarrow\;"
        r"r_{\mathrm{NS}}=P_0\,\mathcal{N}_0(\mathbf{x})\) "
        "flags the suspicious reject",
        parent=g,
    )

    d.node(
        TXT(15),
        10,
        306,
        560,
        30,
        r"\(u_{\lambda}(\mathbf{x})=\lambda_{\mathrm{FA}}r_{\mathrm{FA}}"
        r"+\lambda_{\mathrm{ID}}r_{\mathrm{ID}}"
        r"+\lambda_{\mathrm{FR}}r_{\mathrm{FR}}"
        r"+\lambda_{\mathrm{NS}}r_{\mathrm{NS}}\)",
        parent=g,
    )
    d.node(
        TXT(10, GRAY, italic=True),
        10,
        342,
        560,
        24,
        r"\(\lambda^{\star}=\arg\max_{\lambda}\,"
        r"\mathrm{PRR}^{F_1}_{\mathrm{val}}(u_\lambda)\) "
        "&nbsp;(tuned for the target operating point)",
        parent=g,
    )

    d.node(
        RECT("#FFF2CC", "#D6B656", rounded=True, font_color="#000000", font_size=12),
        90,
        390,
        400,
        44,
        r"IJB-C @ FPIR \(0.2\): PRR <b>0.91</b> (MPRisk) " r"vs \(0.73\) (HolUE)",
        parent=g,
    )


def connectors(d):
    arrow = "endArrow=block;strokeColor=#666666;strokeWidth=3;html=1;"
    d.edge(arrow, (565, 300), (605, 300))
    d.node(
        TXT(10, GRAY, italic=True),
        505,
        246,
        160,
        40,
        r"summarize \(p(c\mid\mathbf{x})\) ?",
    )
    d.edge(arrow, (1033, 300), (1075, 300))
    d.node(
        TXT(10, GRAY, italic=True),
        968,
        240,
        176,
        48,
        "replace KL with<br>expected decision loss",
    )


# ======================================================================
if __name__ == "__main__":
    png = render_panel_a()
    doc = DrawioDoc()
    panel_a(doc, png)
    panel_b(doc)
    panel_c(doc)
    connectors(doc)
    doc.save("mprisk_teaser.drawio")
