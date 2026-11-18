"""
Regenerate the Fig. 2 qualitative strips with the CURRENT pipeline.

Usage (from the project root, after eval deps are ready):
    python eval/make_figures.py --data /path/to/DUTS-TE --out fig2

Produces one labeled strip per image (fig2/fig2_a.png ... fig2_j.png):
  Original | GT mask | Predicted mask | Inpainted BG | Zoom GT (1.8x) |
  Zoom output (1.8x) | Final composite
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ExtracForeground import ForegroundExtractor          # noqa: E402
from inpainters import get_inpainter                      # noqa: E402
from upscaler import SmartUpscaler                        # noqa: E402

# The 15 best-performing images from the 500-image evaluation, ranked by a
# combined score of segmentation F1, inpainting fidelity (PSNR/LPIPS), and
# perceptual zoom quality (see eval_results.csv)
DEFAULT_IDS = [
    "ILSVRC2012_test_00000473", "ILSVRC2013_test_00006260",
    "ILSVRC2013_test_00002923", "ILSVRC2013_test_00004661",
    "ILSVRC2012_test_00026259", "ILSVRC2012_test_00001563",
    "ILSVRC2012_test_00022882", "ILSVRC2012_test_00060735",
    "ILSVRC2012_test_00017104", "sun_aiezcuylgnnrseoz",
    "ILSVRC2012_test_00059436", "sun_adgykrlymnfhzhqa",
    "sun_birvjjkruobbrzse", "sun_aoxdtxbwjjnlbhgj",
    "ILSVRC2012_test_00053904",
]

PANEL_H = 300          # panel height in px
GAP = 6                # white gap between panels
LABEL_H = 26           # label bar height

LABELS = ["Original", "GT mask", "Predicted mask", "Inpainted BG",
          "Zoom GT (1.8x)", "Zoom output (1.8x)", "Final composite"]


def panel(img_rgb, label):
    h, w = img_rgb.shape[:2]
    pw = int(w * PANEL_H / h)
    p = cv2.resize(img_rgb, (pw, PANEL_H), interpolation=cv2.INTER_AREA)
    bar = np.full((LABEL_H, pw, 3), 255, np.uint8)
    cv2.putText(bar, label, (4, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, p])


def strip(panels):
    h = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)
            p = np.vstack([p, pad])
        out.append(p)
        out.append(np.full((h, GAP, 3), 255, np.uint8))
    return np.hstack(out[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="fig2")
    ap.add_argument("--zoom", type=float, default=1.8)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.data, "DUTS-TE-Image")
    msk_dir = os.path.join(args.data, "DUTS-TE-Mask")

    extractor = ForegroundExtractor()
    inpainter = get_inpainter()
    ups = SmartUpscaler()

    for letter, iid in zip("abcdefghijklmno", args.ids):
        ipath = os.path.join(img_dir, iid + ".jpg")
        mpath = os.path.join(msk_dir, iid + ".png")
        if not (os.path.exists(ipath) and os.path.exists(mpath)):
            print(f"({letter}) {iid}: not found, skipping")
            continue

        orig = np.array(Image.open(ipath).convert("RGB"))
        h0, w0 = orig.shape[:2]
        if max(h0, w0) > args.max_side:
            s = args.max_side / max(h0, w0)
            orig = cv2.resize(orig, (int(w0 * s), int(h0 * s)),
                              interpolation=cv2.INTER_AREA)
        h, w = orig.shape[:2]
        gt = np.array(Image.open(mpath).convert("L"))
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        # current pipeline
        tmp = os.path.join(args.out, "_tmp.png")
        Image.fromarray(orig).save(tmp)
        fg = extractor.extract_foreground(tmp)
        if fg.shape[:2] != (h, w):
            fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LANCZOS4)
        pred = ((fg[:, :, 3] > 127).astype(np.uint8)) * 255

        diag = (h * h + w * w) ** 0.5
        k = max(15, int(diag * 0.02)) | 1
        dil = cv2.dilate(pred, np.ones((k, k), np.uint8))
        bg = np.array(inpainter(Image.fromarray(orig), Image.fromarray(dil)))

        # subject-anchored crop: keep the ground line under the subject
        ch, cw = int(h / args.zoom), int(w / args.zoom)
        ys, xs = np.nonzero(pred > 127)
        if len(ys):
            y_low = float(np.percentile(ys, 99))
            span = max(float(ys.max() - ys.min()), 1.0)
            band = ys >= (y_low - 0.05 * span)
            cx, cy = float(xs[band].mean()), y_low
        else:
            cx, cy = w / 2.0, h / 2.0
        # crop origin so the contact point maps to itself: origin = p(1-1/z)
        sx = max(0, min(int(round(cx * (1.0 - 1.0 / args.zoom))), w - cw))
        sy = max(0, min(int(round(cy * (1.0 - 1.0 / args.zoom))), h - ch))
        bz = ups.upscale(bg[sy:sy + ch, sx:sx + cw], w, h)
        zoom_gt = cv2.resize(orig[sy:sy + ch, sx:sx + cw], (w, h),
                             interpolation=cv2.INTER_LANCZOS4)

        alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
        comp = (fg[:, :, :3] * alpha + bz * (1 - alpha)).astype(np.uint8)

        gt3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
        pr3 = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        panels = [panel(x, l) for x, l in zip(
            [orig, gt3, pr3, bg, zoom_gt, bz, comp], LABELS)]
        out = strip(panels)
        path = os.path.join(args.out, f"fig2_{letter}.png")
        Image.fromarray(out).save(path)
        print(f"({letter}) {iid} -> {path}  {out.shape[1]}x{out.shape[0]}")

    if os.path.exists(os.path.join(args.out, "_tmp.png")):
        os.remove(os.path.join(args.out, "_tmp.png"))
    print("Done. Zip the folder and download it.")


if __name__ == "__main__":
    main()
