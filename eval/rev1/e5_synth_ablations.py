"""
Ablations on SynthZoom (Section 5.5) -> Tables 10 (right), 12-17, Figs. 4-5.

    python eval/rev1/e5_synth_ablations.py mask|zoom|stage|order|comp|res|perturb [--limit N]

Unless stated otherwise the ground-truth matte is used, so that the
ablated stage is the only thing that changes (the effect of matte errors
is measured separately by `perturb` and by the automatic-extractor row of
`stage`). Colours of the preserved subject are the input pixels.

Regions (all at output resolution):
  hole      D_{r0}(subject): the default inpainting mask, i.e. every
            background pixel that has to be synthesised (evaluated on the
            reconstructed background B against the true background)
  boundary  +-max(3, 0.005 d) px band around the true subject edge
            (evaluated on the composite C against the true output)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, EXT, Models, core, lpips, full_psnr, full_ssim,  # noqa: E402
                    region_scores, boundary_band, band_width, f1_iou, delta_e00,
                    telea_inpainter, write_csv, seed_all, accel_device, sync, PeakMem,
                    load_rgb, load_gray)

SZ = os.path.join(DATA, "synthzoom")
Z = 2.0


def samples(split, limit=0):
    with open(os.path.join(SZ, "meta.json")) as f:
        meta = json.load(f)
    if limit:
        meta = meta[:limit]
    for rec in meta:
        d = os.path.join(SZ, split, rec["id"])
        yield rec, d


def load_sample(d, z=Z):
    s = dict(I=load_rgb(os.path.join(d, "input.png")),
             A=load_gray(os.path.join(d, "alpha.png")),
             bg=load_rgb(os.path.join(d, "bg.png")))
    if z:
        s["bgz"] = load_rgb(os.path.join(d, f"bgz_{z:g}.png"))
        s["gt"] = load_rgb(os.path.join(d, f"gt_{z:g}.png"))
    return s


def auto_matte(m, d, I):
    """Automatic (Tracer-B7 + FBA) matte and colours for a SynthZoom input,
    cached next to the sample."""
    pa, pf = os.path.join(d, "auto_alpha.png"), os.path.join(d, "auto_fg.png")
    if not (os.path.exists(pa) and os.path.exists(pf)):
        fg = m.extractor.extract_array(I)
        Image.fromarray(fg[:, :, 3]).save(pa)
        Image.fromarray(fg[:, :, :3]).save(pf)
    return load_gray(pa), load_rgb(pf)


def hole_region(A, cfg=core.Config()):
    return core.build_inpaint_mask(A, None, cfg) > 0


def ghost_ratio(B, bg, I, A):
    """Share of hidden-background pixels (true subject area) whose
    reconstructed colour is closer to the subject than to the true
    background, i.e. subject colour smeared into the background."""
    S = A > 127
    if not S.any():
        return np.nan
    b, t, f = (x[S].astype(np.float32) for x in (B, bg, I))
    return float((np.linalg.norm(b - f, axis=1) < np.linalg.norm(b - t, axis=1)).mean())


def evaluate(s, out, drift=0.0):
    I, A, bg, gt = s["I"], s["A"], s["bg"], s["gt"]
    h, w = A.shape
    hole = hole_region(A)
    band = boundary_band(A, band_width(h, w))
    hs = region_scores(out["B"], bg, hole)
    bs = region_scores(out["C"], gt, band)
    return dict(psnr=full_psnr(out["C"], gt), ssim=full_ssim(out["C"], gt),
                lpips=lpips(out["C"], gt),
                hole_psnr=hs["psnr"], hole_lpips=hs["lpips"],
                bnd_psnr=bs["psnr"], bnd_lpips=bs["lpips"],
                ghost=ghost_ratio(out["B"], bg, I, A), drift_px=drift)


def run(m, s, A, cfg, inpainter=None, fix_anchor=True):
    """Run the edit. With fix_anchor the crop window uses the true contact
    point, so a different matte changes the image only through the matte
    itself (inpainting mask, composite edge) and not through a shifted
    window; the anchor displacement is recorded separately."""
    t = time.perf_counter()
    pa = core.contact_point(s["A"]) if fix_anchor else None
    out = core.run_pipeline(s["I"], s["I"], A, Z, inpainter or m.lama, m.upscaler,
                            cfg, return_all=True, anchor_point=pa)
    sync()
    out["t"] = time.perf_counter() - t
    return out


# ---------------------------------------------------------------- mask
def exp_mask(m, limit):
    h0 = None
    configs = []
    for f in (0, 0.5, 1, 2, 3):
        configs.append((f"silhouette_r{f:g}", dict(dil_factor=f)))
    configs += [("silhouette_fixed15", dict(fixed=15)),
                ("hull_r1", dict(dil_factor=1, shape="hull")),
                ("bbox_r1", dict(dil_factor=1, shape="bbox")),
                ("feather_half", dict(dil_factor=1, feather=11)),
                ("feather_double", dict(dil_factor=1, feather=41))]
    rows = []
    for rec, d in samples("s1020", limit):
        s = load_sample(d)
        h, w = s["A"].shape
        r0 = core.dilation_size(h, w, core.Config())
        for name, c in configs:
            cfg = core.Config()
            if "fixed" in c:
                cfg.dilation_fixed = c["fixed"]
            else:
                cfg.dilation_fixed = max(1, int(round(c["dil_factor"] * r0)))
            cfg.mask_shape = c.get("shape", "silhouette")
            cfg.feather_ksize = c.get("feather", 21)
            t = time.perf_counter()
            M = core.build_inpaint_mask(s["A"], None, cfg)
            B = core.reconstruct_background(s["I"], M, m.lama, cfg)
            t_inp = time.perf_counter() - t
            p = core.contact_point(s["A"])
            Bz = core.magnify(B, core.anchored_window(h, w, Z, p), m.upscaler, cfg)
            C = core.composite(s["I"], s["A"], Bz, cfg)
            r = dict(id=rec["id"], config=name, r0=r0, r=cfg.dilation_fixed,
                     t_inp_ms=t_inp * 1000, **evaluate(s, dict(B=B, C=C)))
            rows.append(r)
        print(rec["id"], flush=True)
    write_csv(rows, "e5_mask.csv")


# ---------------------------------------------------------------- zoom
def exp_zoom(m, limit):
    from e0_build_synthzoom import Z_SWEEP  # noqa
    rows = []
    modes = [("lanczos", "lanczos", False), ("sr", "sr", False),
             ("fixed09", "fixed", True), ("adaptive", "adaptive", True)]
    for rec, d in samples("s510", limit):
        bg = load_rgb(os.path.join(d, "bg.png"))
        A = load_gray(os.path.join(d, "alpha.png"))
        h, w = A.shape
        p = core.contact_point(A)
        for z in Z_SWEEP:
            ref = load_rgb(os.path.join(d, f"bgz_{z:g}.png"))
            win = core.anchored_window(h, w, z, p)
            x0, y0, cw, ch = win
            crop = bg[y0:y0 + ch, x0:x0 + cw]
            for name, mode, sh in modes:
                t = time.perf_counter()
                out = m.upscaler.upscale(crop, w, h, mode=mode, unsharp=sh, fixed=0.9)
                sync()
                rows.append(dict(id=rec["id"], z=z, mode=name,
                                 t_ms=(time.perf_counter() - t) * 1000,
                                 psnr=full_psnr(out, ref), lpips=lpips(out, ref)))
        print(rec["id"], flush=True)
    write_csv(rows, "e5_zoom.csv")


# ---------------------------------------------------------------- stage
_sam = None


def mobilesam_mask(img, box):
    global _sam
    if _sam is None:
        from mobile_sam import sam_model_registry, SamPredictor
        from common import PROJECT
        sam = sam_model_registry["vit_t"](checkpoint=os.path.join(PROJECT, "weights", "mobile_sam.pt"))
        sam.eval()
        _sam = SamPredictor(sam)
    _sam.set_image(img)
    masks, _, _ = _sam.predict(box=np.array(box, np.float32), multimask_output=False)
    return masks[0].astype(np.uint8) * 255


def exp_stage(m, limit, which=None):
    from inpainters import SDInpainter
    rows = []
    sd = None
    names = which.split(",") if which else None
    for rec, d in samples("s1020", limit):
        s = load_sample(d)
        h, w = s["A"].shape
        p = core.contact_point(s["A"])
        auto = None
        cfgs = [
            ("full", core.Config(), None, s["A"]),
            ("fixed_dilation_15", core.Config(dilation_fixed=15), None, s["A"]),
            ("no_blend_back", core.Config(blend_back=False), None, s["A"]),
            ("telea", core.Config(), telea_inpainter, s["A"]),
            ("sd2", core.Config(), "sd", s["A"]),
            ("center_anchor", core.Config(anchor="center"), None, s["A"]),
            ("lanczos_only", core.Config(sr_mode="lanczos"), None, s["A"]),
            ("sr_only", core.Config(sr_mode="sr"), None, s["A"]),
            ("no_unsharp", core.Config(unsharp=False), None, s["A"]),
            ("no_defringe", core.Config(composition="alpha"), None, s["A"]),
            ("auto_matte", core.Config(), None, "auto"),
            ("mobilesam_raw", core.Config(), None, "sam_raw"),
            ("mobilesam_refined", core.Config(), None, "sam_ref"),
        ]
        sam_m = None
        for name, cfg, inp, A in cfgs:
            if names and name not in names:
                continue
            if isinstance(A, str):
                if A == "auto":
                    if auto is None:
                        auto = auto_matte(m, d, s["I"])[0]
                    A = auto
                else:
                    if sam_m is None:
                        ys, xs = np.nonzero(s["A"] > 127)
                        sam_m = mobilesam_mask(s["I"], (xs.min(), ys.min(), xs.max(), ys.max()))
                    A = sam_m if A == "sam_raw" else core.refine_mask_edges(s["I"], sam_m)
            if inp == "sd":
                if sd is None:
                    sd = SDInpainter()
                inp = sd
            out = run(m, s, A, cfg, inp)
            drift = core.contact_drift(h, w, Z, p) if cfg.anchor == "center" else 0.0
            if A is not s["A"]:
                pp = core.contact_point(A)
                drift = float(np.hypot(p[0] - pp[0], p[1] - pp[1])) * Z if pp else np.nan
            r = dict(id=rec["id"], config=name, t_ms=out["t"] * 1000, **evaluate(s, out, drift))
            if A is not s["A"]:
                r["f1"] = f1_iou(A, s["A"])[0]
            rows.append(r)
        print(rec["id"], flush=True)
        write_csv(rows, f"e5_stage{'_' + which.replace(',', '_') if which else ''}.csv")


# ---------------------------------------------------------------- order
def exp_order(m, limit):
    rows = []
    dev = accel_device()
    for rec, d in samples("s1020", limit):
        bg = load_rgb(os.path.join(d, "bg.png"))
        ref = load_rgb(os.path.join(d, "bgz_2.png"))
        A = load_gray(os.path.join(d, "alpha.png"))
        h, w = A.shape
        win = core.anchored_window(h, w, Z, core.contact_point(A))
        for order in "ABCD":
            cfg = core.Config(sr_order=order)
            with PeakMem(dev) as pm:
                t = time.perf_counter()
                out = core.magnify(bg, win, m.upscaler, cfg)
                sync()
                dt = time.perf_counter() - t
            rows.append(dict(id=rec["id"], order=order, t_ms=dt * 1000, mem_gb=pm.gb,
                             psnr=full_psnr(out, ref), lpips=lpips(out, ref)))
        print(rec["id"], flush=True)
    write_csv(rows, "e5_order.csv")


# ---------------------------------------------------------------- comp
_pct = None


def pctnet(comp, mask01):
    global _pct
    if _pct is None:
        import torch
        sys.path.insert(0, os.path.join(EXT, "pctnet"))
        cwd = os.getcwd()
        os.chdir(os.path.join(EXT, "pctnet"))
        from iharm.inference.predictor import Predictor
        from iharm.inference.utils import load_model
        from iharm.mconfigs import ALL_MCONFIGS
        mdl = load_model("ViT_pct", "pretrained_models/PCTNet_ViT.pth", verbose=False)
        prm = ALL_MCONFIGS["ViT_pct"]["params"]
        _pct = Predictor(mdl, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         use_attn=prm["use_attn"],
                         mean=prm["input_normalization"]["mean"],
                         std=prm["input_normalization"]["std"])
        os.chdir(cwd)
    _, full = _pct.predict(cv2.resize(comp, (256, 256)), comp,
                           cv2.resize(mask01, (256, 256)), mask01)
    return np.clip(full, 0, 255).astype(np.uint8)


def exp_comp(m, limit):
    rows = []
    for rec, d in samples("s1020", limit):
        s = load_sample(d)
        I, A = s["I"], s["A"]
        h, w = A.shape
        cfg = core.Config()
        M = core.build_inpaint_mask(A, None, cfg)
        B = core.reconstruct_background(I, M, m.lama, cfg)
        Bz = core.magnify(B, core.anchored_window(h, w, Z, core.contact_point(A)), m.upscaler, cfg)
        inner = A > 242
        for name in ("alpha", "defringe", "poisson", "pctnet"):
            t = time.perf_counter()
            if name == "pctnet":
                base = core.composite(I, A, Bz, core.Config(composition="alpha"))
                C = pctnet(base, A.astype(np.float64) / 255.0)
            else:
                C = core.composite(I, A, Bz, core.Config(composition=name))
            dt = time.perf_counter() - t
            ev = evaluate(s, dict(B=B, C=C))
            rows.append(dict(id=rec["id"], method=name, t_ms=dt * 1000,
                             de00=delta_e00(C, I, inner), **ev))
        print(rec["id"], flush=True)
    write_csv(rows, "e5_comp.csv")


def exp_comp_auto(m, limit):
    """Table 16, second block: the same strategies with the automatic
    matte and extractor colours, i.e. the situation defringing is for."""
    rows = []
    for rec, d in samples("s1020", limit):
        s = load_sample(d)
        I, Agt = s["I"], s["A"]
        A, F = auto_matte(m, d, I)
        h, w = A.shape
        cfg = core.Config()
        pg = core.contact_point(Agt)
        B = core.reconstruct_background(I, core.build_inpaint_mask(A, None, cfg), m.lama, cfg)
        Bz = core.magnify(B, core.anchored_window(h, w, Z, pg), m.upscaler, cfg)
        inner = Agt > 242
        for name in ("alpha", "defringe", "poisson", "pctnet"):
            t = time.perf_counter()
            if name == "pctnet":
                base = core.composite(F, A, Bz, core.Config(composition="alpha"))
                C = pctnet(base, A.astype(np.float64) / 255.0)
            else:
                C = core.composite(F, A, Bz, core.Config(composition=name))
            dt = time.perf_counter() - t
            ev = evaluate(s, dict(B=B, C=C))
            rows.append(dict(id=rec["id"], method=name, t_ms=dt * 1000,
                             de00=delta_e00(C, I, inner), f1=f1_iou(A, Agt)[0], **ev))
        print(rec["id"], flush=True)
    write_csv(rows, "e5_comp_auto.csv")


# ---------------------------------------------------------------- res
def exp_res(m, limit):
    rows = []
    for rec, d in samples("s2040", limit):
        s = load_sample(d, z=None)
        I, A, bg = s["I"], s["A"], s["bg"]
        hole = hole_region(A)
        M = core.build_inpaint_mask(A, None, core.Config())
        for L in (512, 768, 1024, 1536, 0):
            cfg = core.Config(work_res=L)
            t = time.perf_counter()
            B = core.reconstruct_background(I, M, m.lama, cfg)
            dt = time.perf_counter() - t
            hs = region_scores(B, bg, hole)
            rows.append(dict(id=rec["id"], L=L or max(I.shape[:2]), t_ms=dt * 1000,
                             hole_psnr=hs["psnr"], hole_lpips=hs["lpips"],
                             full_lpips=lpips(B, bg)))
        print(rec["id"], flush=True)
    write_csv(rows, "e5_res.csv")


# ---------------------------------------------------------------- perturb
def perturb(A, kind, level, rng):
    """Degrade the soft ground-truth alpha while keeping it soft, so that
    only the matte error (not the loss of fractional edges) is measured."""
    a = A.copy()
    if kind == "erode":
        return cv2.erode(a, np.ones((level, level), np.uint8))
    if kind == "dilate":
        return cv2.dilate(a, np.ones((level, level), np.uint8))
    if kind == "noise":
        dx = cv2.GaussianBlur(rng.standard_normal(a.shape).astype(np.float32), (0, 0), 8)
        dy = cv2.GaussianBlur(rng.standard_normal(a.shape).astype(np.float32), (0, 0), 8)
        dx *= level / (np.abs(dx).max() + 1e-6)
        dy *= level / (np.abs(dy).max() + 1e-6)
        yy, xx = np.mgrid[:a.shape[0], :a.shape[1]].astype(np.float32)
        return np.clip(cv2.remap(a.astype(np.float32), xx + dx, yy + dy,
                                 cv2.INTER_LINEAR), 0, 255).astype(np.uint8)
    if kind == "drop":
        ys, xs = np.nonzero(a > 127)
        if len(ys) == 0:
            return a
        k = rng.integers(len(ys))
        d2 = (ys - ys[k]) ** 2 + (xs - xs[k]) ** 2
        order = np.argsort(d2)[:int(level * len(ys))]
        cut = np.zeros(a.shape, np.uint8)
        cut[ys[order], xs[order]] = 255
        cut = cv2.dilate(cut, np.ones((3, 3), np.uint8))      # take the soft rim too
        a[cut > 0] = 0
        return a
    raise ValueError(kind)


PERTURB = [("erode", 2), ("erode", 4), ("erode", 8), ("erode", 16),
           ("dilate", 2), ("dilate", 4), ("dilate", 8), ("dilate", 16),
           ("noise", 4), ("noise", 8), ("noise", 16),
           ("drop", 0.1), ("drop", 0.2), ("drop", 0.3)]


def exp_perturb(m, limit):
    rows = []
    rng = np.random.default_rng(42)
    for rec, d in samples("s1020", limit):
        s = load_sample(d)
        for kind, level in [("none", 0)] + PERTURB:
            A = s["A"] if kind == "none" else perturb(s["A"], kind, level, rng)
            out = run(m, s, A, core.Config())
            ev = evaluate(s, out)
            pa, pp = core.contact_point(s["A"]), core.contact_point(A)
            shift = float(np.hypot(pa[0] - pp[0], pa[1] - pp[1])) if pp else np.nan
            rows.append(dict(id=rec["id"], kind=kind, level=level,
                             f1=f1_iou(A, s["A"])[0], anchor_shift_px=shift, **ev))
        print(rec["id"], flush=True)
        write_csv(rows, "e5_perturb.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="stage: comma-separated configs")
    a = ap.parse_args()
    seed_all()
    mm = Models()
    if a.exp == "stage":
        exp_stage(mm, a.limit, a.only)
    else:
        globals()["exp_" + a.exp](mm, a.limit)
