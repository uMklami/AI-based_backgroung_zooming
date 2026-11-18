"""
E0: build the SynthZoom benchmark (Section 4.1).

    python eval/rev1/e0_build_synthzoom.py --n 200

Each sample pastes an AIM-500 foreground (ground-truth alpha; colours
estimated with the multilevel foreground estimator of pymatting) onto a
DIV2K validation background (2040 px long side). Because the background
is known at twice the output resolution or more, the true zoomed output
exists:

    I   = a F + (1 - a) down(B*)
    C*z = a F + (1 - a) down(crop_z(B*))

where crop_z is the contact-anchored window of Eq. 8 computed from the
true matte, mapped to the high-resolution background.

Three splits share the same composites (and scale with them):
  s1020 : output long side 1020 px, z = 2            (true detail up to 2x)
  s510  : output long side 510 px,  z in Z_SWEEP     (true detail up to 4x)
  s2040 : output long side 2040 px, no zoom          (working-resolution study)
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

# torch is deliberately not imported here: its OpenMP runtime and numba's
# (used by pymatting) crash when loaded in the same process on macOS.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import pipeline_core as core  # noqa: E402

DATA = os.path.join(os.environ.get("SELZOOM_REV", os.path.expanduser("~/selzoom_rev")), "data")
SEED = 42


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def load_gray(p):
    return np.array(Image.open(p).convert("L"))


def set_items(name):
    root = os.path.join(DATA, "sets", name)
    with open(os.path.join(root, "meta.json")) as f:
        items = json.load(f)
    for it in items:
        it["image"] = os.path.join(root, "images", it["file"])
        it["mask"] = os.path.join(root, "masks", it["mask_file"])
    return items

OUT = os.path.join(DATA, "synthzoom")
Z_SWEEP = [1.25, 1.5, 1.8, 2.5, 3.0, 4.0]


def fg_pool():
    items = [it for it in set_items("aim500") if it["type"] in ("SO", "STM")]
    return items


def cut_foreground(it, max_side=1024):
    """Tight crop of an AIM-500 foreground with its alpha and estimated F."""
    from pymatting import estimate_foreground_ml
    rgb = load_rgb(it["image"])
    a = load_gray(it["mask"])
    ys, xs = np.nonzero(a > 10)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb, a = rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]
    s = min(1.0, max_side / max(rgb.shape[:2]))
    if s < 1:
        size = (int(rgb.shape[1] * s), int(rgb.shape[0] * s))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    F = estimate_foreground_ml(rgb / 255.0, a / 255.0)
    return (np.clip(F, 0, 1) * 255).astype(np.uint8), a


def compose(F, A, B):
    a = A.astype(np.float32)[..., None] / 255.0
    return np.clip(F * a + B * (1 - a), 0, 255).astype(np.uint8)


def place(Fc, Ac, H, W, rng):
    """Scale the foreground to 10-30% of the frame area and put its base in
    the lower third. Returns full-frame (F, alpha) at H x W."""
    area_frac = rng.uniform(0.10, 0.30)
    cover = (Ac > 127).sum() / Ac.size                      # fill of its bbox
    k = np.sqrt(area_frac * H * W / (cover * Ac.shape[0] * Ac.shape[1]))
    k = min(k, 0.95 * H / Ac.shape[0], 0.95 * W / Ac.shape[1])
    fh, fw = max(8, int(Ac.shape[0] * k)), max(8, int(Ac.shape[1] * k))
    Fr = cv2.resize(Fc, (fw, fh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    Ar = cv2.resize(Ac, (fw, fh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    y1 = rng.randint(max(fh, int(2 * H / 3)), H)            # base row (exclusive)
    x0 = rng.randint(0, W - fw)
    F = np.zeros((H, W, 3), np.uint8)
    A = np.zeros((H, W), np.uint8)
    F[y1 - fh:y1, x0:x0 + fw] = Fr
    A[y1 - fh:y1, x0:x0 + fw] = Ar
    return F, A


def true_zoom(B_hr, H, W, z, p):
    """Crop of the high-resolution background at the Eq. 8 window."""
    x0, y0, cw, ch = core.anchored_window(H, W, z, p)
    sx, sy = B_hr.shape[1] / W, B_hr.shape[0] / H
    X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
    CW, CH = int(round(cw * sx)), int(round(ch * sy))
    crop = B_hr[Y0:Y0 + CH, X0:X0 + CW]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


def resize_long(img, L, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = L / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(SEED)
    bgs = sorted(os.path.join(DATA, "DIV2K_valid_HR", f)
                 for f in os.listdir(os.path.join(DATA, "DIV2K_valid_HR")) if f.endswith(".png"))
    fgs = fg_pool()
    rng.shuffle(fgs)
    for sp in ("s1020", "s510", "s2040"):
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)
    meta = []
    for i in range(args.n):
        it = fgs[i % len(fgs)]
        bpath = bgs[i % len(bgs)]
        B_hr = load_rgb(bpath)
        Fc, Ac = cut_foreground(it)
        # placement is decided once, in normalised coordinates, on the 2040 split
        H2, W2 = resize_long(B_hr, 2040).shape[:2]
        F2, A2 = place(Fc, Ac, H2, W2, rng)
        rec = dict(id=f"sz{i:03d}", fg=it["name"], fg_type=it["type"],
                   bg=os.path.basename(bpath), area=float((A2 > 127).mean()))
        for sp, L in (("s1020", 1020), ("s510", 510), ("s2040", 2040)):
            d = os.path.join(OUT, sp, rec["id"])
            os.makedirs(d, exist_ok=True)
            B = resize_long(B_hr, L)
            H, W = B.shape[:2]
            F = cv2.resize(F2, (W, H), interpolation=cv2.INTER_AREA)
            A = cv2.resize(A2, (W, H), interpolation=cv2.INTER_AREA)
            I = compose(F, A, B)
            Image.fromarray(I).save(os.path.join(d, "input.png"))
            Image.fromarray(A).save(os.path.join(d, "alpha.png"))
            Image.fromarray(F).save(os.path.join(d, "fg.png"))
            Image.fromarray(B).save(os.path.join(d, "bg.png"))
            p = core.contact_point(A)
            zs = [2.0] if sp == "s1020" else (Z_SWEEP if sp == "s510" else [])
            for z in zs:
                Bz = true_zoom(B_hr, H, W, z, p)
                Image.fromarray(Bz).save(os.path.join(d, f"bgz_{z:g}.png"))
                Image.fromarray(compose(F, A, Bz)).save(os.path.join(d, f"gt_{z:g}.png"))
            rec[f"p_{sp}"] = list(p)
        meta.append(rec)
        print(i, rec["id"], rec["fg"], rec["bg"], round(rec["area"], 3), flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
