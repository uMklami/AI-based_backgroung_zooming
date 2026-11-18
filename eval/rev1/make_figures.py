"""
Figures for the revision, built from saved experiment outputs.

    python eval/rev1/make_figures.py fig3|fig4|fig5|fig7|all

  fig3  challenging foregrounds (one row per Table 9 condition)
  fig4  output error vs. mask F1 on SynthZoom, per perturbation type
  fig5  composition close-ups: alpha, defringe, Poisson, PCT-Net
  fig7  unselected DUTS-TE results: 6 random + 4 from the lowest F1 decile
Figures are written to BG-Zooming/media/rev1_*.png next to the paper.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, DATA, Models, core, duts_sample, set_items, load_rgb,  # noqa: E402
                    load_gray, cap_size)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_MEDIA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(RESULTS))),
                           "..", "BG-Zooming", "media")
PAPER_MEDIA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "..", "BG-Zooming", "media"))
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def tile(img, h=180):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)


def gray3(a):
    return np.dstack([a] * 3)


def grid(rows, col_titles, out, row_titles=None, h=180):
    """rows: list of lists of RGB arrays (same count per row)."""
    rows = [[tile(c, h) for c in r] for r in rows]
    widths = [max(r[j].shape[1] for r in rows) for j in range(len(rows[0]))]
    pad = 4
    W = sum(widths) + pad * (len(widths) + 1)
    top = 22
    H = top + len(rows) * (h + pad) + pad
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        x = pad
        for j, c in enumerate(r):
            y = top + i * (h + pad)
            canvas[y:y + c.shape[0], x:x + c.shape[1]] = c
            x += widths[j] + pad
    fig = plt.figure(figsize=(W / 150, H / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    x = pad
    for j, t in enumerate(col_titles):
        ax.text(x + widths[j] / 2, 14, t, ha="center", va="center", fontsize=7)
        x += widths[j] + pad
    if row_titles:
        for i, t in enumerate(row_titles):
            ax.text(6, top + i * (h + pad) + 12, t, ha="left", va="top", fontsize=7,
                    color="white", bbox=dict(facecolor="black", alpha=0.6, pad=1.5, lw=0))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def e1_row(setname, item, z_crop=True):
    """Input, GT matte, predicted matte, inpainted bg, composite, boundary close-up."""
    rgb = load_rgb(item["image"])
    gt = load_gray(item["mask"]) if item.get("mask") else None
    rgb, gt = cap_size(rgb, gt)
    name = item.get("name", os.path.basename(item["image"]))
    od = os.path.join(RESULTS, "outputs", setname)
    A = load_gray(os.path.join(od, name + "_alpha.png"))
    C = load_rgb(os.path.join(od, name + "_comp.jpg"))
    bgp = os.path.join(od, name + "_bg.jpg")
    B = load_rgb(bgp) if os.path.exists(bgp) else np.zeros_like(rgb)
    ys, xs = np.nonzero(A > 127)
    if len(ys):
        # close-up on the lower subject boundary
        cy, cx = int(np.percentile(ys, 85)), int(xs.mean())
        s = max(40, int(0.15 * max(rgb.shape[:2])))
        y0, x0 = max(0, cy - s), max(0, cx - s)
        crop = C[y0:y0 + 2 * s, x0:x0 + 2 * s]
    else:
        crop = C
    return [rgb, gray3(gt) if gt is not None else np.zeros_like(rgb), gray3(A), B, C, crop]


def fig3():
    e = {k: pd.read_csv(os.path.join(RESULTS, f"e1_{k}.csv"))
         for k in ("dis_te", "aim500", "cod10k", "duts")}
    tags = json.load(open(os.path.join(os.path.dirname(__file__), "tags_duts_cast_shadow.json")))
    multi = pd.read_csv(os.path.join(RESULTS, "attr_multi.csv"))

    def median_item(df, setname, mask=None):
        d = df if mask is None else df[mask]
        d = d.iloc[(d.f1 - d.f1.median()).abs().argsort()[:1]]
        nm = d.name.iloc[0]
        items = duts_sample() if setname == "duts" else set_items(setname)
        return next(it for it in items if it.get("name", os.path.basename(it["image"])) == nm)

    rows, titles = [], []
    rows.append(e1_row("dis_te", median_item(e["dis_te"], "dis_te")));
    titles.append("thin structures (DIS-TE)")
    a = e["aim500"]
    rows.append(e1_row("aim500", median_item(a, "aim500", a.type == "STM")));
    titles.append("semi-transparent (AIM-500 STM)")
    mnames = set(multi[(multi.set == "DUTS-TE") & (multi.multi == 1)].name)
    d = e["duts"]
    rows.append(e1_row("duts", median_item(d, "duts", d.name.isin(mnames))));
    titles.append("multiple objects (DUTS-TE)")
    rows.append(e1_row("duts", median_item(d, "duts", d.name.isin(tags["names"]))));
    titles.append("cast shadow (DUTS-TE)")
    rows.append(e1_row("cod10k", median_item(e["cod10k"], "cod10k")));
    titles.append("ambiguous boundary (COD10K)")
    grid(rows, ["input", "ground-truth matte", "predicted matte", "inpainted background",
                "composite (1.8x)", "close-up"], os.path.join(PAPER_MEDIA, "rev1_fig3_challenge.png"),
         titles, h=150)


def fig7():
    d = pd.read_csv(os.path.join(RESULTS, "e1_duts.csv"))
    items = {it["name"]: it for it in duts_sample()}
    rng = np.random.default_rng(0)
    rand = list(rng.choice(d.name.values, 6, replace=False))
    low = d[d.f1 <= d.f1.quantile(0.1)].sort_values("f1")
    worst = list(low.name.values[np.linspace(0, len(low) - 1, 4).astype(int)])
    rows = [e1_row("duts", items[n])[:5] for n in rand + worst]
    titles = [f"random, F1={d.set_index('name').f1[n]:.2f}" for n in rand] + \
             [f"lowest decile, F1={d.set_index('name').f1[n]:.2f}" for n in worst]
    grid(rows, ["input", "ground-truth matte", "predicted matte", "inpainted background",
                "composite (1.8x)"], os.path.join(PAPER_MEDIA, "rev1_fig7_unselected.png"), titles, h=130)


def fig4():
    d = pd.read_csv(os.path.join(RESULTS, "e5_perturb.csv"))
    g = d.groupby(["kind", "level"])[["f1", "lpips", "bnd_lpips", "ghost"]].mean().reset_index()
    base = g[g.kind == "none"].iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    labels = {"erode": "erosion", "dilate": "dilation", "noise": "boundary noise", "drop": "part deletion"}
    for kind, mk in zip(["erode", "dilate", "noise", "drop"], "osD^"):
        s = g[g.kind == kind].sort_values("f1")
        for ax, col in zip(axs, ["lpips", "bnd_lpips", "ghost"]):
            ax.plot(s.f1, s[col], marker=mk, ms=3.5, lw=1, label=labels[kind])
    for ax, col, t in zip(axs, ["lpips", "bnd_lpips", "ghost"],
                          ["full-frame LPIPS vs. $C^*$", "boundary-band LPIPS", "ghost ratio"]):
        ax.axhline(base[col], color="k", lw=0.6, ls="--")
        ax.set_xlabel("mask F1")
        ax.set_title(t)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    out = os.path.join(PAPER_MEDIA, "rev1_fig4_sensitivity.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig5():
    """Boundary close-ups of the four composition strategies on SynthZoom."""
    from e5_synth_ablations import samples, load_sample, pctnet
    d = pd.read_csv(os.path.join(RESULTS, "e5_comp.csv"))
    piv = d.pivot_table(index="id", columns="method", values="bnd_lpips")
    gap = (piv["alpha"] - piv["pctnet"]).sort_values()
    picks = [gap.index[len(gap) // 2], gap.index[-1], gap.index[int(len(gap) * 0.25)]]
    m = Models()
    rows, titles = [], []
    for rec, dd in samples("s1020"):
        if rec["id"] not in picks:
            continue
        s = load_sample(dd)
        I, A = s["I"], s["A"]
        h, w = A.shape
        cfg = core.Config()
        B = core.reconstruct_background(I, core.build_inpaint_mask(A, None, cfg), m.lama, cfg)
        Bz = core.magnify(B, core.anchored_window(h, w, 2.0, core.contact_point(A)), m.upscaler, cfg)
        outs = [core.composite(I, A, Bz, core.Config(composition=c)) for c in ("alpha", "defringe", "poisson")]
        outs.append(pctnet(outs[0], A.astype(np.float64) / 255.0))
        ys, xs = np.nonzero(A > 127)
        cy, cx = int(np.percentile(ys, 60)), int(xs.min()) + 10
        sz = int(0.12 * max(h, w))
        y0, x0 = max(0, cy - sz), max(0, cx - sz)
        sl = (slice(y0, y0 + 2 * sz), slice(x0, x0 + 2 * sz))
        rows.append([s["gt"]] + [o[sl] for o in outs] + [s["gt"][sl]])
        titles.append(rec["id"])
    grid(rows, ["true output $C^*$ (full)", "alpha", "defringed (ours)", "Poisson", "PCT-Net",
                "true output (crop)"], os.path.join(PAPER_MEDIA, "rev1_fig5_composition.png"), titles, h=170)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for f in (["fig3", "fig4", "fig5", "fig7"] if which == "all" else [which]):
        globals()[f]()
