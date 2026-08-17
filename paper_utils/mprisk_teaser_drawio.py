#!/usr/bin/env python3
"""
MPRisk teaser generator (theory-only, style-matched to previous paper's teaser):
  - panel (a): matplotlib -> SVG, embedded as SVG inside .drawio
      * field = equal-cost MPRisk (decision risk of the taken decision),
        NOT entropy: u(x) = 1 - max_a p(a|x)
      * white long-dash decision boundaries, class colors / marker conventions
        from the previous teaser; BuGn SCF blob only on the low-kappa FR probe
      * posteriors & risk components computed at the three error probes are
        RETURNED and reused in panel (c)  -> panels are numerically consistent
  - panels (b), (c), arrows: native editable draw.io shapes
  - formulas: INLINE LaTeX \( ... \)

Outputs:
  mprisk_teaser.drawio
  mprisk_panel_a.svg
"""

import base64
import io
import xml.sax.saxutils as su

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

# ======================================================================
# palette  (single source of truth, matched to the previous teaser)
# ======================================================================
CLASS_COLORS = ["#7f7f7f", "#ff7f0e", "#8c564b", "#0eb451"]   # gallery classes
OOG, OOG_D = "#e377c2", "#ffffff"                             # out-of-gallery
EDGE = "DarkSlateGrey"
GRAY = "#595959"
# risk components inherit the error-marker colors of the previous teaser
R = {"FA": "#e377c2", "ID": "#ff7f0e", "FR": "#7f7f7f", "NS": "#9673A6"}
DIAMOND, STAR, CROSS = "\u25c6", "\u2605", "\u2715"
HALO = [pe.withStroke(linewidth=1.5, foreground="white")]


# ======================================================================
# 1. Panel (a): MPRisk field + probes -> SVG bytes + computed risk values
#    model identical to create_uncertainty_image(fig2):
#    kappa=1, beta=0.5, value_range=8 (M=16)
# ======================================================================
def render_panel_a():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    font_size = 15
    means = np.array([[-3.0, 0.0], [-1.8, 1.8], [3.0, 0.0], [3.15, 4.4]])
    K = means.shape[0]
    kappa, beta, M = 1.0, 0.5, 16.0

    fa_pt = np.array([3.72, -1.07])       # false accept   (unknown, accepted)
    id_pt = np.array([-2.508, 0.719])     # misidentification (between classes)
    fr_pt = np.array([-0.77, -3.70])      # false reject   (corrupted known)
    oog_pt = np.array([0.14, -1.95])      # correctly rejected unknown
    fr_var = 1                          # low-quality SCF variance (low kappa_x)

    # ---- mixed posterior helper (K vMF-analogs + uniform unknown) ----
    def posterior(px, py):
        d2 = np.stack([(px - mx) ** 2 + (py - my) ** 2 for mx, my in means])
        lik = (1 - beta) / K * kappa / (2 * np.pi) * np.exp(-kappa * d2)
        unk = np.full_like(np.asarray(px, dtype=float), beta / M**2)
        stack = np.concatenate([lik, unk[None]], axis=0)
        return stack / stack.sum(0)                     # shape (K+1, ...)

    fig, ax = plt.subplots(figsize=(6.44, 5.52))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    xs = np.linspace(-7.0, 7.0, 300)
    ys = np.linspace(-5.0, 7.0, 300)
    X, Y = np.meshgrid(xs, ys)
    P = posterior(X, Y)
    reject = P.argmax(0) == K

    # ---- (1) MPRisk field: risk of the decision taken at each point ----
    # u(x) = 1 - max_a p(a|x)  (Chow's conditional risk; NS ~ 0 for the
    # high-quality mean embeddings the field represents)
    U = 1.0 - P.max(0)
    ax.contourf(X, Y, U, levels=20, cmap="Blues", zorder=0)

    # reject region tint: continuous-unknown component (MPRisk idea)
    tint = np.zeros(X.shape + (4,))
    tint[reject] = matplotlib.colors.to_rgba(OOG_D, alpha=0.15)
    ax.imshow(
        tint, extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower", aspect="auto", zorder=1,
    )

    # ---- (2) decision boundaries: white long-dash (previous style) ----
    d2 = np.stack([(X - mx) ** 2 + (Y - my) ** 2 for mx, my in means])
    lik = (1 - beta) / K * kappa / (2 * np.pi) * np.exp(-kappa * d2)
    unk = np.full_like(X, beta / M**2)
    stack = np.concatenate([lik, unk[None]], axis=0)
    for i in range(K):
        b = stack[i] - np.delete(stack, i, axis=0).max(0)
        ax.contour(
            X, Y, b, levels=[0.0],
            colors="white", linewidths=2.0, linestyles=[(0, (8, 4))], zorder=3,
        )

    # ---- (3) SCF blob ONLY on the low-quality (low-kappa) FR probe ----
    scf = np.exp(-((X - fr_pt[0]) ** 2 + (Y - fr_pt[1]) ** 2) / (2 * fr_var)) / (
        2 * np.pi * fr_var
    )
    lv = np.linspace(0.12 * scf.max(), scf.max(), 8)
    ax.contourf(X, Y, scf, levels=lv, cmap="BuGn", alpha=0.8, zorder=2)

    # ---- (4) class centers and samples (no blob on high-kappa samples) ----
    rng = np.random.default_rng(4)
    hi_pt = None
    for k, (cx, cy) in enumerate(means):
        pts = np.array([cx, cy]) + 0.2 * rng.standard_normal((3, 2))
        ax.scatter(
            pts[:, 0], pts[:, 1], marker="s", s=48,
            color=CLASS_COLORS[k], edgecolor=EDGE, lw=1.0, zorder=5,
        )
        ax.scatter(
            [cx], [cy], marker="o", s=110,
            color=CLASS_COLORS[k], edgecolor=EDGE, lw=1.0, zorder=6,
        )
        if k == 1:
            hi_pt = pts[0]
    ax.annotate(
        r"high $\kappa_{\mathbf{x}}$",
        hi_pt, xytext=(hi_pt[0] - 2.9, hi_pt[1] + 1.4),
        fontsize=font_size, color=GRAY,
        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9),
    )
    ax.text(
        3.35, -0.42, r"$\mu_c$", fontsize=font_size, color=GRAY,
        ha="left", va="center", path_effects=HALO, zorder=7,
    )

    # ---- (5) out-of-gallery sample: white square, dark edge ----
    ax.scatter(
        [oog_pt[0]], [oog_pt[1]], marker="s", s=48,
        color=OOG, edgecolor=EDGE, lw=1.0, zorder=5,
    )

    # ---- (6) error markers, previous-teaser convention + risk tags ----
    def err(pt, mk, sz, comp, dx, dy):
        ax.scatter(
            [pt[0]], [pt[1]], marker=mk, s=sz,
            color=R[comp], edgecolor=EDGE, lw=0.7, zorder=8,
        )
        ax.text(
            pt[0] + dx, pt[1] + dy, comp,
            fontsize=font_size, fontweight="bold", color=R[comp],
            ha="left", va="center", path_effects=HALO, zorder=8,
        )

    err(fa_pt, "*", 340, "FA", 0.35, -0.42)
    err(id_pt, "D", 130, "ID", 0.32, 0.42)
    err(fr_pt, "X", 170, "FR", 0.45, 0.42)

    # ---- (7) MPRisk-specific annotations ----
    ax.add_patch(
        FancyArrowPatch(
            means[0] + [0.35, -0.45], fr_pt + [-0.55, 0.95],
            connectionstyle="arc3,rad=0.3", arrowstyle="-|>",
            mutation_scale=14, color="#666666", lw=1.4, zorder=7,
        )
    )
    ax.text(
        -4.9, -3.0, "corrupted known\nsample drifts out",
        fontsize=font_size, color=GRAY, ha="center", style="italic",
    )
    ax.text(
        1.15, -4.55,
        r"low $\kappa_{\mathbf{x}}$"
        "\n"
        r"$\Rightarrow$ high $\mathcal{N}_0(\mathbf{x})$",
        fontsize=font_size, color=R["NS"], ha="left", va="center",
        path_effects=HALO, zorder=8,
    )
    # ax.text(
    #     -6.8, 6.6,
    #     "reject region:\ncontinuous unknown\n" + r"$c\in(K,\,K{+}1]$",
    #     fontsize=11.5, color=OOG_D, va="top",
    # )
    ax.text(
        3.6, 6.4, r"risk field $u(\mathbf{x})$",
        fontsize=font_size, color="#2A5B8C", ha="left", va="top",
        path_effects=HALO,
    )

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")

    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    svg = buf.getvalue()
    with open("mprisk_panel_a.svg", "wb") as f:
        f.write(svg)

    # ================================================================
    # compute risk components at the three probes -> reused in panel (c)
    # ================================================================
    def components(pt):
        p = posterior(np.array([pt[0]]), np.array([pt[1]]))[:, 0]
        p0, pcls = p[K], p[:K]
        if pcls.max() > p0:                     # accepted as chat = argmax
            chat = int(pcls.argmax())
            r_fa = p0
            r_id = pcls.sum() - pcls[chat]
            return {"segs": [("FA", r_fa), ("ID", r_id)], "u": r_fa + r_id}
        return None                             # rejected -> handled below

    # rejected FR probe: r_FR = 1 - P0 ; r_NS = P0 * N0(x) with N0 computed
    # numerically from the unknown-identity posterior induced by the SCF blob
    def components_rejected(pt, var):
        p = posterior(np.array([pt[0]]), np.array([pt[1]]))[:, 0]
        p0 = p[K]
        m = stack.sum(0)                        # marginal m(z) on the grid
        scf_g = np.exp(
            -((X - pt[0]) ** 2 + (Y - pt[1]) ** 2) / (2 * var)
        ) / (2 * np.pi * var)
        w = unk * scf_g / m                     # ∝ p(U | x, c unknown)
        dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
        area = (xs[-1] - xs[0]) * (ys[-1] - ys[0])
        p_u = w / (w.sum() * dA)
        n0 = 1.0 / (area * (p_u**2).sum() * dA)  # normalized 1/collision
        r_fr, r_ns = 1.0 - p0, p0 * n0
        return {"segs": [("FR", r_fr), ("NS", r_ns)], "u": r_fr + r_ns, "N0": n0}

    vals = {
        "star": components(fa_pt),
        "diamond": components(id_pt),
        "cross": components_rejected(fr_pt, fr_var),
    }
    print("panel (a) risk values:", {
        k: {"u": round(v["u"], 3),
            "segs": [(c, round(x, 3)) for c, x in v["segs"]]}
        for k, v in vals.items()
    })
    return svg, vals


# ======================================================================
# 2. draw.io XML builder (math enabled)
# ======================================================================
class DrawioDoc:
    def __init__(self, w=1270, h=620):
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
def panel_a(d, svg_bytes):
    g = d.group(20, 40, 540, 510)
    d.node(RECT("none", "#CCCCCC", rounded=True), 0, 0, 540, 510, parent=g)
    d.node(
        TXT(16, bold=True),
        0, 4, 540, 26,
        r"(a) mixed-prior posterior \(p(c\mid\mathbf{x})\) "
        r"and risk field \(u(\mathbf{x})\)",
        parent=g,
    )
    b64 = base64.b64encode(svg_bytes).decode()
    d.node(
        f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition="
        f"bottom;verticalAlign=top;image=data:image/svg+xml,{b64};",
        10, 34, 520, 446,
        parent=g,
    )
    d.node(
        TXT(11, GRAY),
        10, 482, 520, 26,
        r"\(u(\mathbf{x}) = 1-\max_a p(a\mid\mathbf{x})\): "
        r"risk peaks on decision boundaries, vanishes in the deep reject region",
        parent=g,
    )


def panel_b(d, vals):
    g = d.group(610, 40, 620, 510)
    d.node(RECT("none", "#CCCCCC", rounded=True), 0, 0, 620, 510, parent=g)
    d.node(
        TXT(16, bold=True),
        0, 4, 620, 26,
        "(b) MPRisk: from information gain to decision risk",
        parent=g,
    )

    # ---- strip 1: KL counterexample (compact) --------------------------
    bar_fill = CLASS_COLORS + [OOG]
    bar_stroke = ["none"] * 4 + [OOG_D]
    charts = [
        (15, "A: two-way split", [0.48, 0.48, 0.02, 0.01, 0.01]),
        (125, "B: mildly diffuse", [0.55, 0.13, 0.12, 0.11, 0.09]),
    ]
    base_y, hmax = 132, 72
    for x0, title, p in charts:
        d.node(TXT(10, bold=True), x0 - 15, 38, 120, 16, title, parent=g)
        for i, pi in enumerate(p):
            h = max(2, round(pi * hmax))
            d.node(
                RECT(bar_fill[i], bar_stroke[i]),
                x0 + i * 18, base_y - h, 14, h, parent=g,
            )
        d.edge(
            "endArrow=none;strokeColor=#666666;strokeWidth=1;html=1;",
            (x0 - 4, base_y), (x0 + 94, base_y), parent=g,
        )
        d.node(
            TXT(9, GRAY), x0 - 15, base_y + 2, 120, 14,
            r"\(c_1\ c_2\ c_3\ c_4\) unk", parent=g,
        )

    d.node(
        TXT(10, GRAY, align="left"),
        228, 36, 384, 32,
        r"\(D_{\mathrm{KL}}(p\,\|\,p_{\mathrm{unif}})=\log N-H(p)\): "
        r"\(\mathrm{KL}_A>\mathrm{KL}_B\) \(\Rightarrow\) A looks safer",
        parent=g,
    )
    d.node(
        TXT(10, align="left"),
        228, 70, 384, 32,
        r"decision risk \(1-\max_c p\): A is a coin flip, B is confident "
        r"\(\Rightarrow\) A is riskier",
        parent=g,
    )
    d.node(
        TXT(10.5, "#CC0000", bold=True, align="left"),
        228, 104, 384, 44,
        r"orderings invert for \(N\geq 3\) \(\Rightarrow\) KL features need "
        "a supervised nonlinear calibrator (HolUE); "
        "risk-aligned features do not",
        parent=g,
    )
    d.edge(
        "endArrow=none;strokeColor=#DDDDDD;strokeWidth=1;html=1;",
        (10, 158), (610, 158), parent=g,
    )

    # ---- strip 2: decision-conditioned components (values from panel a) --
    legend = [
        ("FA", r"\(r_{\mathrm{FA}}{=}P_0\)"),
        ("ID", r"\(r_{\mathrm{ID}}{=}\sum_{j\neq\hat c}P_j\)"),
        ("FR", r"\(r_{\mathrm{FR}}{=}1{-}P_0\)"),
        ("NS", r"\(r_{\mathrm{NS}}{=}P_0\,\mathcal{N}_0\)"),
    ]
    for i, (comp, formula) in enumerate(legend):
        x = 14 + i * 152
        d.node(RECT(R[comp]), x, 170, 13, 13, parent=g)
        d.node(TXT(10.5, align="left"), x + 17, 166, 132, 22, formula, parent=g)

    rows = [
        (f"{STAR}&nbsp; accepted, truly unknown", vals["star"], 198, ""),
        (f"{DIAMOND}&nbsp; accepted, wrong identity", vals["diamond"], 234, ""),
        (
            f"{CROSS}&nbsp; rejected, low-quality known",
            vals["cross"], 270,
            r"\(r_{\mathrm{FR}}\!\approx\!0\): only \(r_{\mathrm{NS}}\) "
            "flags the confident wrong reject",
        ),
    ]
    u_max = max(v["u"] for v in vals.values())
    scale, x_bar, bh = 210.0 / u_max, 200, 28
    for label, v, y, note in rows:
        d.node(TXT(10.5, align="right"), 0, y, 192, bh, label, parent=g)
        x = x_bar
        for comp, r in v["segs"]:
            w = round(r * scale)
            if w < 3:
                continue
            txt = rf"\(r_{{\mathrm{{{comp}}}}}\)" if w > 42 else ""
            d.node(
                RECT(R[comp], "#FFFFFF", sw=1, font_size=10), x, y, w, bh, txt,
                parent=g,
            )
            x += w
        d.node(
            TXT(9.5, GRAY, align="left"),
            x + 5, y + 5, 70, 18,
            rf"\(u\approx{v['u']:.2f}\)",
            parent=g,
        )
        if note:
            d.node(
                TXT(9, R["NS"], align="left", italic=True),
                x + 75, y + 1, 620 - (x + 80), bh - 2, note, parent=g,
            )

    # ---- strip 3: combination rule ---------------------------------------
    d.node(
        TXT(14),
        10, 314, 600, 28,
        r"\(u_{\lambda}(\mathbf{x})=\lambda_{\mathrm{FA}}r_{\mathrm{FA}}"
        r"+\lambda_{\mathrm{ID}}r_{\mathrm{ID}}"
        r"+\lambda_{\mathrm{FR}}r_{\mathrm{FR}}"
        r"+\lambda_{\mathrm{NS}}r_{\mathrm{NS}},"
        r"\qquad \lambda\in\mathbb{R}_{+}^{4}\)",
        parent=g,
    )
    d.node(
        TXT(10, GRAY),
        10, 346, 600, 36,
        r"risks are decision-gated (reject risks \(\equiv 0\) under "
        r"acceptance, and vice versa); \(\lambda\equiv 1\) recovers "
        r"Chow's conditional risk \(1-\max_a p(a\mid\mathbf{x})\)",
        parent=g,
    )
    d.node(
        TXT(10, GRAY, italic=True),
        10, 386, 600, 20,
        "four nonnegative cost weights tuned on validation "
        "&mdash; no supervised calibration network",
        parent=g,
    )

    # ---- strip 4: takeaway ------------------------------------------------
    # d.node(
    #     RECT("#FFF2CC", "#D6B656", rounded=True, font_color="#000000",
    #          font_size=11.5),
    #     80, 418, 460, 46,
    #     "each component is monotone in the probability of its error "
    #     r"mechanism \(\Rightarrow\) a linear rule suffices",
    #     parent=g,
    # )



def connectors(d):
    arrow = "endArrow=block;strokeColor=#666666;strokeWidth=3;html=1;"
    d.edge(arrow, (565, 300), (605, 300))
    d.node(
        TXT(10, GRAY, italic=True),
        498, 244, 176, 48,
        "score the taken decision<br>by its expected loss",
    )

# ======================================================================
if __name__ == "__main__":
    svg, vals = render_panel_a()
    doc = DrawioDoc()
    panel_a(doc, svg)
    panel_b(doc, vals)
    connectors(doc)
    doc.save("mprisk_teaser.drawio")