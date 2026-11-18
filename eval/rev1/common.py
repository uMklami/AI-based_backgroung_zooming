"""
Shared utilities for the revision experiments (Revision 1, The Visual Computer).

Data, external repositories and model weights live outside the project
folder (default ~/selzoom_rev, override with SELZOOM_REV):
    data/DUTS-TE, data/DIV2K_valid_HR, data/sets/<name>/{images,masks}
    ext/GFM, ext/pctnet
All experiments use fixed seeds and write one CSV row per image to
eval/rev1/results/.
"""
import json
import math
import os
import random
import sys
import time
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT)

import pipeline_core as core                                   # noqa: E402

REV = os.environ.get("SELZOOM_REV", os.path.expanduser("~/selzoom_rev"))
DATA = os.path.join(REV, "data")
EXT = os.path.join(REV, "ext")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
MAX_SIDE = 1024


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def accel_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev=None):
    dev = dev or accel_device()
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


# ------------------------------------------------------------------ data
def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def load_gray(p):
    return np.array(Image.open(p).convert("L"))


def cap_size(rgb, gt=None, max_side=MAX_SIDE):
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    if gt is not None and gt.shape[:2] != rgb.shape[:2]:
        gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    return rgb, gt


def duts_sample(n=500, seed=SEED):
    """Same 500-image sample as the original submission (eval/run_eval.py)."""
    img_dir = os.path.join(DATA, "DUTS-TE", "DUTS-TE-Image")
    msk_dir = os.path.join(DATA, "DUTS-TE", "DUTS-TE-Mask")
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = random.Random(seed)
    rng.shuffle(names)
    out = []
    for nm in names[:n]:
        out.append(dict(name=nm, image=os.path.join(img_dir, nm),
                        mask=os.path.join(msk_dir, os.path.splitext(nm)[0] + ".png")))
    return out


def set_items(name):
    """Items of a prepared set in data/sets/<name>/ (see prep_data.py)."""
    root = os.path.join(DATA, "sets", name)
    meta = os.path.join(root, "meta.json")
    with open(meta) as f:
        items = json.load(f)
    for it in items:
        it["image"] = os.path.join(root, "images", it["file"])
        if it.get("mask_file"):
            it["mask"] = os.path.join(root, "masks", it["mask_file"])
    return items


# ------------------------------------------------------------------ metrics
_lp = None


def lpips_net():
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net="alex", verbose=False).eval().to(accel_device())
    return _lp


def lpips(a, b):
    """LPIPS (AlexNet) between two uint8 RGB images of equal size."""
    net = lpips_net()

    dev = next(net.parameters()).device

    def t(x):
        return (torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1).unsqueeze(0).to(dev))
    with torch.no_grad():
        return float(net(t(a), t(b)).item())


def region_crop(mask, pad=4):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    return (max(ys.min() - pad, 0), min(ys.max() + 1 + pad, h),
            max(xs.min() - pad, 0), min(xs.max() + 1 + pad, w))


def region_scores(out, ref, mask):
    """PSNR/SSIM/LPIPS of `out` vs `ref` restricted to `mask`: pixels
    outside the mask are copied from ref, then the mask's bounding box is
    compared (standard practice for region LPIPS)."""
    bb = region_crop(mask)
    if bb is None:
        return dict(psnr=np.nan, ssim=np.nan, lpips=np.nan)
    y0, y1, x0, x1 = bb
    o = np.where(mask[..., None], out, ref)[y0:y1, x0:x1]
    r = ref[y0:y1, x0:x1]
    if min(o.shape[:2]) < 8:
        o = cv2.resize(o, (max(8, o.shape[1]), max(8, o.shape[0])))
        r = cv2.resize(r, (o.shape[1], o.shape[0]))
    m = mask[y0:y1, x0:x1]
    mse = ((o.astype(np.float64) - r) ** 2)[m].mean() if m.any() else np.nan
    ps = 10 * math.log10(255 ** 2 / mse) if mse and mse > 0 else 99.0
    return dict(psnr=ps, ssim=full_ssim(o, r), lpips=lpips(o, r))


def full_psnr(a, b):
    return float(sk_psnr(b, a, data_range=255))


def full_ssim(a, b):
    win = min(7, min(a.shape[:2]) // 2 * 2 - 1)
    if win < 3:
        return np.nan
    return float(sk_ssim(b, a, channel_axis=2, data_range=255, win_size=win))


def f1_iou(pred_u8, gt_u8, t=127):
    p, g = pred_u8 > t, gt_u8 > t
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    denom = p.sum() + g.sum()
    iou = inter / union if union else 1.0
    f1 = 2 * inter / denom if denom else 1.0
    return float(f1), float(iou)


def leak_ratio(gt_u8, M_u8):
    g = gt_u8 > 127
    return float((g & (M_u8 == 0)).sum() / max(g.sum(), 1))


def carry_over_ratio(pred_u8, gt_u8):
    p, g = pred_u8 > 127, gt_u8 > 127
    return float((p & ~g).sum() / max(p.sum(), 1))


def boundary_band(gt_u8, w):
    g = (gt_u8 > 127).astype(np.uint8)
    k = np.ones((2 * w + 1, 2 * w + 1), np.uint8)
    return (cv2.dilate(g, k) - cv2.erode(g, k)) > 0


def band_width(h, w):
    return max(3, int(round(0.005 * core.diag(h, w))))


def delta_e00(a_rgb, b_rgb, mask):
    from skimage.color import rgb2lab, deltaE_ciede2000
    if not mask.any():
        return np.nan
    la = rgb2lab(a_rgb[mask][None].astype(np.float64) / 255.0)
    lb = rgb2lab(b_rgb[mask][None].astype(np.float64) / 255.0)
    return float(deltaE_ciede2000(la, lb).mean())


# matting metrics (standard definitions, evaluated over the whole image as
# in the AIM-500 protocol; alpha in [0, 1])
def matting_metrics(pred, gt):
    from scipy.ndimage import gaussian_filter
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    diff = pred - gt
    sad = np.abs(diff).sum() / 1000.0
    mse = (diff ** 2).mean()

    def grad(x):
        gx = gaussian_filter(x, 1.4, order=[0, 1])
        gy = gaussian_filter(x, 1.4, order=[1, 0])
        return np.sqrt(gx ** 2 + gy ** 2)
    g = ((grad(pred) - grad(gt)) ** 2).sum() / 1000.0
    return dict(sad=sad, mse=mse, grad=g, conn=connectivity(pred, gt))


def connectivity(pred, gt, step=0.1):
    thresh_steps = np.arange(0, 1 + step, step)
    l_map = np.full(pred.shape, -1.0)
    for i in range(1, len(thresh_steps)):
        pred_t = pred >= thresh_steps[i]
        gt_t = gt >= thresh_steps[i]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (pred_t & gt_t).astype(np.uint8), connectivity=4)
        if n > 1:
            big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            omega = lab == big
        else:
            omega = np.zeros_like(pred_t)
        flag = (l_map == -1) & ~omega
        l_map[flag] = thresh_steps[i - 1]
    l_map[l_map == -1] = 1
    d_pred = pred - l_map
    d_gt = gt - l_map
    phi_p = 1 - d_pred * (d_pred >= 0.15)
    phi_g = 1 - d_gt * (d_gt >= 0.15)
    return float(np.abs(phi_p - phi_g).sum() / 1000.0)


# ------------------------------------------------------------------ hole
def find_bg_hole(gt_bin, frac=0.12, min_frac=0.02, tries=200, rng=None):
    """Synthetic-hole protocol of the original submission (unchanged)."""
    h, w = gt_bin.shape
    rng = rng or random
    f = frac
    while f >= min_frac:
        area = h * w * f
        rh = int((area * h / w) ** 0.5)
        rw = int(area / max(rh, 1))
        rh, rw = min(rh, h - 2), min(rw, w - 2)
        if rh > 4 and rw > 4:
            for _ in range(tries):
                y = rng.randint(0, h - rh)
                x = rng.randint(0, w - rw)
                if gt_bin[y:y + rh, x:x + rw].sum() == 0:
                    return y, x, rh, rw, f
        f /= 2
    return None


# ------------------------------------------------------------------ models
class Models:
    """Lazy holder for the pipeline models shared by all experiments."""

    def __init__(self, sr_device=None):
        self._ex = None
        self._lama = None
        self._up = None
        self.sr_device = sr_device or accel_device()

    @property
    def extractor(self):
        if self._ex is None:
            from ExtracForeground import ForegroundExtractor
            # CarveKit runs on CUDA when present; its FBA stage fails on MPS
            self._ex = ForegroundExtractor(device="cuda" if torch.cuda.is_available() else "cpu")
        return self._ex

    @property
    def lama(self):
        if self._lama is None:
            from inpainters import LamaInpainter
            self._lama = LamaInpainter()
        return self._lama

    @property
    def upscaler(self):
        if self._up is None:
            os.environ.setdefault("SR_DEVICE", self.sr_device)
            from upscaler import SmartUpscaler
            self._up = SmartUpscaler()
        return self._up


def telea_inpainter(image, mask):
    rgb = np.array(image)
    m = np.array(mask)
    out = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ------------------------------------------------------------------ timing
class PeakMem:
    """Peak accelerator memory during a block. On MPS there is no peak
    counter, so the driver allocation is sampled every 5 ms."""

    def __init__(self, dev):
        self.dev = dev
        self.peak = 0
        self._stop = False

    def __enter__(self):
        if self.dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif self.dev == "mps":
            torch.mps.empty_cache()
            self.base = torch.mps.driver_allocated_memory()
            self.peak = self.base
            self._t = threading.Thread(target=self._poll, daemon=True)
            self._t.start()
        return self

    def _poll(self):
        while not self._stop:
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
            time.sleep(0.005)

    def __exit__(self, *a):
        if self.dev == "cuda":
            self.peak = torch.cuda.max_memory_allocated()
        elif self.dev == "mps":
            self._stop = True
            self._t.join()
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())

    @property
    def gb(self):
        return self.peak / 2 ** 30


def timed(fn, *a, dev=None, **kw):
    dev = dev or accel_device()
    sync(dev)
    t = time.perf_counter()
    out = fn(*a, **kw)
    sync(dev)
    return out, (time.perf_counter() - t) * 1000.0


def write_csv(rows, name):
    import pandas as pd
    p = os.path.join(RESULTS, name)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p
