"""
Array-level implementation of the selective background zoom pipeline.

Every function here corresponds to an equation in Section 3.2 of the
paper, and both the web app (app.py) and the evaluation scripts
(eval/) call these functions, so the numbers reported in the paper come
from exactly the code that users run.

    Eq. 1-2   refine_mask_edges      guided-filter refinement of prompt masks
    Eq. 3     fuse_mattes            alpha = (1 - R) * max(alpha_auto, alpha_user)
    Eq. 4     build_inpaint_mask     M = D_r(1[alpha > tau] OR R)
    Eq. 5     reconstruct_background blend-back inpainting at <= 1024 px
    Eq. 6     rect_to_window         zoom factor and window from a rectangle
    Eq. 7     contact_point          ground-contact anchor p
    Eq. 8     anchored_window        o = p (1 - 1/z)
    Eq. 10-11 SmartUpscaler          adaptive SR/Lanczos blend + unsharp
    Eq. 12    defringe               alpha~ = G * E(alpha)
    Eq. 13    composite              C = alpha~ F + (1 - alpha~) B_z

The Config switches exist only for the ablations in Section 5.5; the
defaults are the configuration described in the paper.
"""
from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from PIL import Image

TAU = 0.5            # matte binarisation threshold (Eq. 4)
MAX_ZOOM = 4.0


@dataclass
class Config:
    # Eq. 4: r = max(dilation_min, ceil(dilation_scale * d))
    dilation_scale: float = 0.02
    dilation_min: int = 15
    dilation_fixed: int = 0            # >0: fixed dilation in px (ablation)
    mask_shape: str = "silhouette"     # silhouette | hull | bbox
    # Eq. 5: blend-back feather (21x21 Gaussian, sigma = 3.5 px)
    feather_ksize: int = 21
    blend_back: bool = True
    work_res: int = 1024               # inpainting working resolution L
    # Eq. 7-8
    anchor: str = "contact"            # contact | center
    # Eq. 10-11
    sr_mode: str = "adaptive"          # adaptive | lanczos | sr | fixed
    sr_fixed_weight: float = 0.9
    unsharp: bool = True
    sr_order: str = "A"                # A | B | C | D (Table 15)
    # Eq. 12-13
    composition: str = "defringe"      # alpha | defringe | poisson
    extra: dict = field(default_factory=dict)


def diag(h, w):
    return math.sqrt(h * h + w * w)


# ------------------------------------------------------------ Eq. 1-2
def guided_filter(guide, src, radius, eps=1e-4):
    """Grey-guide guided filter (He et al.) via box filters."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, ksize)
    mean_p = cv2.boxFilter(src, -1, ksize)
    corr_ip = cv2.boxFilter(guide * src, -1, ksize)
    corr_ii = cv2.boxFilter(guide * guide, -1, ksize)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, ksize) * guide + cv2.boxFilter(b, -1, ksize)


def refine_mask_edges(rgb, mask_u8):
    """Binary prompt mask K -> soft matte alpha_user (Eqs. 1-2).
    Erosion e_m = max(1, floor(d/1200)) px, window radius ceil(0.02 d)."""
    h, w = mask_u8.shape[:2]
    d = diag(h, w)
    er = max(1, int(d / 1200))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))
    m = cv2.erode(mask_u8, k, iterations=1)
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    src = m.astype(np.float32) / 255.0
    radius = int(math.ceil(0.02 * d))
    q = guided_filter(guide, src, radius, 1e-4)
    out = np.clip((q - 0.2) / 0.6, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ------------------------------------------------------------ Eq. 3
def fuse_mattes(alpha_auto_u8, alpha_user_u8=None, remove_bin=None):
    alpha = alpha_auto_u8.copy()
    if alpha_user_u8 is not None:
        alpha = np.maximum(alpha, alpha_user_u8)
    if remove_bin is not None:
        alpha[remove_bin] = 0
    return alpha


# ------------------------------------------------------------ Eq. 4
def dilation_size(h, w, cfg):
    if cfg.dilation_fixed > 0:
        r = cfg.dilation_fixed
    else:
        r = max(cfg.dilation_min, int(math.ceil(cfg.dilation_scale * diag(h, w))))
    return r | 1 if r > 0 else 0          # odd so the element is centred


def build_inpaint_mask(alpha_u8, remove_bin=None, cfg=Config()):
    h, w = alpha_u8.shape[:2]
    m = alpha_u8 > int(TAU * 255)
    if remove_bin is not None:
        m = m | remove_bin
    m = m.astype(np.uint8) * 255
    if cfg.mask_shape == "hull" and m.any():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(cnts))
        m = np.zeros_like(m)
        cv2.fillConvexPoly(m, hull, 255)
    elif cfg.mask_shape == "bbox" and m.any():
        ys, xs = np.nonzero(m)
        m = np.zeros_like(m)
        m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
    r = dilation_size(h, w, cfg)
    if r > 1:
        m = cv2.dilate(m, np.ones((r, r), np.uint8), iterations=1)
    return m


# ------------------------------------------------------------ Eq. 5
def reconstruct_background(rgb, mask_u8, inpainter, cfg=Config()):
    """B = (1-g) I + g up(Phi(down I, down M)), g = G * M."""
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb)
    long_side = max(w, h)
    L = cfg.work_res if cfg.work_res else long_side
    if long_side > L:
        s = L / long_side
        small = (max(1, round(w * s)), max(1, round(h * s)))
        inp = inpainter(img.resize(small, Image.LANCZOS),
                        Image.fromarray(mask_u8).resize(small, Image.NEAREST))
        inp = inp.crop((0, 0, small[0], small[1]))
        inp_full = np.array(inp.resize((w, h), Image.LANCZOS))
    else:
        inp = inpainter(img, Image.fromarray(mask_u8)).crop((0, 0, w, h))
        inp_full = np.array(inp)
    if not cfg.blend_back:
        return inp_full
    k = cfg.feather_ksize | 1
    g = cv2.GaussianBlur(mask_u8, (k, k), 0).astype(np.float32)[..., None] / 255.0
    out = rgb.astype(np.float32) * (1 - g) + inp_full.astype(np.float32) * g
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ Eq. 6-8
def contact_point(alpha_u8):
    """Eq. 7: centroid of foreground pixels in the lowest 5% of the
    subject's height, computed on the largest connected component so a
    stray matte fragment cannot move the anchor. Returns (x, y) or None
    for an empty matte."""
    fgm = (alpha_u8 > int(TAU * 255)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fgm, connectivity=8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == big)
    y_min, y_max = ys.min(), ys.max()
    band = ys >= y_max - 0.05 * (y_max - y_min)
    return float(xs[band].mean()), float(ys[band].mean())


def anchored_window(h, w, z, p=None):
    """Eq. 8: o = p (1 - 1/z), s = (W/z, H/z); centre anchor if p is None.
    Returns integer (x0, y0, cw, ch)."""
    cw, ch = max(int(w / z), 1), max(int(h / z), 1)
    if p is None:
        ox, oy = (w - cw) / 2.0, (h - ch) / 2.0
    else:
        ox, oy = p[0] * (1 - 1 / z), p[1] * (1 - 1 / z)
    x0 = int(round(min(max(ox, 0), w - cw)))
    y0 = int(round(min(max(oy, 0), h - ch)))
    return x0, y0, cw, ch


def rect_to_window(h, w, rect):
    """Eq. 6: z = clip(1/max(q_w, q_h), 1, 4); window centred on Q."""
    qw, qh = max(float(rect['w']), 1e-6), max(float(rect['h']), 1e-6)
    z = min(max(1.0 / max(qw, qh), 1.0), MAX_ZOOM)
    cx = (float(rect['x']) + qw / 2) * w
    cy = (float(rect['y']) + qh / 2) * h
    cw, ch = max(int(w / z), 1), max(int(h / z), 1)
    x0 = int(round(min(max(cx - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), h - ch)))
    return z, (x0, y0, cw, ch)


def contact_drift(h, w, z, p):
    """Eq. 9 in output pixels: how far the ground under p moves when the
    crop is centre-anchored instead of contact-anchored."""
    c = np.array([w / 2.0, h / 2.0])
    return float(np.linalg.norm((np.array(p) - c) * (1 - 1 / z)) * z)


# ------------------------------------------------------------ Eq. 10-11
def magnify(B, window, upscaler, cfg=Config()):
    """Crop the window from the full-resolution background and upscale it
    to the output size. The default order (A) crops first, so the only
    upsampling is the super-resolution of the crop (Section 3.6)."""
    h, w = B.shape[:2]
    x0, y0, cw, ch = window
    crop = B[y0:y0 + ch, x0:x0 + cw]
    order = cfg.sr_order
    if order == "B":
        return upscaler.upscale(crop, w, h, mode="lanczos", unsharp=cfg.unsharp)
    if order == "C":                     # SR the full frame, then crop
        big = upscaler.sr4(B)
        crop4 = big[y0 * 4:(y0 + ch) * 4, x0 * 4:(x0 + cw) * 4]
        ai = cv2.resize(crop4, (w, h), interpolation=cv2.INTER_AREA)
        return upscaler.finish(ai, crop, w, h, mode=cfg.sr_mode, unsharp=cfg.unsharp,
                               fixed=cfg.sr_fixed_weight)
    if order == "D":                     # interpolate first, then SR
        lan = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        ai = cv2.resize(upscaler.sr4(lan), (w, h), interpolation=cv2.INTER_AREA)
        return upscaler.finish(ai, crop, w, h, mode=cfg.sr_mode, unsharp=cfg.unsharp,
                               fixed=cfg.sr_fixed_weight)
    return upscaler.upscale(crop, w, h, mode=cfg.sr_mode, unsharp=cfg.unsharp,
                            fixed=cfg.sr_fixed_weight)


# ------------------------------------------------------------ Eq. 12-13
def defringe(alpha_u8):
    """Erode by e = clip(round(d/1400), 1, 3) iterations of a 3x3 ellipse,
    then feather with a Gaussian of kernel size ~0.004 d (odd, >= 3)."""
    h, w = alpha_u8.shape[:2]
    d = diag(h, w)
    iters = max(1, min(3, int(round(d / 1400))))
    bk = max(3, int(d * 0.004)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a = cv2.erode(alpha_u8, k, iterations=iters)
    return cv2.GaussianBlur(a, (bk, bk), 0)


def composite(F_rgb, alpha_u8, Bz, cfg=Config()):
    if cfg.composition == "poisson":
        m = (alpha_u8 > 127).astype(np.uint8) * 255
        if not m.any():
            return Bz.copy()
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # seamlessClone places the patch by the centre of its bounding box
        src = np.zeros_like(Bz)
        src[m > 0] = F_rgb[m > 0]
        center = ((x0 + x1) // 2, (y0 + y1) // 2)
        patch_src, patch_m = src[y0:y1, x0:x1], m[y0:y1, x0:x1]
        try:
            out = cv2.seamlessClone(cv2.cvtColor(patch_src, cv2.COLOR_RGB2BGR),
                                    cv2.cvtColor(Bz, cv2.COLOR_RGB2BGR),
                                    patch_m, center, cv2.NORMAL_CLONE)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        except cv2.error:
            pass                          # mask touching the border: fall back
    a_u8 = defringe(alpha_u8) if cfg.composition == "defringe" else alpha_u8
    a = a_u8.astype(np.float32)[..., None] / 255.0
    out = F_rgb.astype(np.float32) * a + Bz.astype(np.float32) * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ whole edit
def run_pipeline(rgb, F_rgb, alpha_u8, z, inpainter, upscaler, cfg=Config(),
                 remove_bin=None, rect=None, return_all=False, anchor_point=None):
    """Algorithm 1, steps 6-13, on arrays. `alpha_u8` is the fused matte.
    `anchor_point` overrides the contact point (evaluation only)."""
    h, w = rgb.shape[:2]
    M = build_inpaint_mask(alpha_u8, remove_bin, cfg)
    B = reconstruct_background(rgb, M, inpainter, cfg)
    if rect is not None:
        z, window = rect_to_window(h, w, rect)
        p = None
    else:
        p = contact_point(alpha_u8) if cfg.anchor == "contact" else None
        if anchor_point is not None and cfg.anchor == "contact":
            p = anchor_point
        window = anchored_window(h, w, z, p)
    Bz = magnify(B, window, upscaler, cfg)
    C = composite(F_rgb, alpha_u8, Bz, cfg)
    if return_all:
        return dict(C=C, M=M, B=B, Bz=Bz, window=window, p=p, z=z)
    return C
