"""
Batch evaluation for the selective background zoom pipeline on DUTS-TE.

Usage (from the project root):
    python eval/run_eval.py --data /path/to/DUTS-TE --n 500 --zoom 1.8

Expects the standard DUTS-TE layout:
    DUTS-TE/DUTS-TE-Image/*.jpg      (inputs)
    DUTS-TE/DUTS-TE-Mask/*.png       (ground-truth saliency masks)

Metrics (all written to eval_results.csv, means printed at the end):
  1. Segmentation: mIoU, F1 of the extractor's alpha vs. ground truth.
  2. Inpainting (synthetic-hole protocol): a rectangular hole covering
     ~12% of the image is placed entirely in KNOWN background (zero GT
     coverage), inpainted, and compared to the true pixels — a rigorous
     protocol with real ground truth, unlike scoring the occluded region.
  3. Zoom (downscale–upscale protocol): the background is downscaled by
     the zoom factor and upscaled back by (a) Lanczos and (b) the AI
     upscaler; each is compared to the original — an honest reference,
     and a built-in ablation.
  4. Composite deviation: full-frame LPIPS/SSIM/PSNR of the final output
     vs. the unedited original (measures intended change, reported for
     transparency).
  5. Runtime per stage.
"""
import argparse
import csv
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ExtracForeground import ForegroundExtractor          # noqa: E402
from inpainters import get_inpainter                      # noqa: E402
from upscaler import SmartUpscaler                        # noqa: E402


# ---------------------------------------------------------------- helpers

def load_lpips():
    try:
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False)
        if torch.cuda.is_available():
            net = net.cuda()
        return net
    except Exception as e:
        print(f"LPIPS unavailable ({e}); LPIPS columns will be empty.")
        return None


def lpips_score(net, a_rgb, b_rgb):
    if net is None:
        return ""
    def t(x):
        x = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x.cuda() if torch.cuda.is_available() else x
    with torch.no_grad():
        return float(net(t(a_rgb), t(b_rgb)).item())


def seg_scores(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = inter / union if union else 1.0
    denom = pred_bin.sum() + gt_bin.sum()
    f1 = 2 * inter / denom if denom else 1.0
    return iou, f1


def find_bg_hole(gt_bin, frac=0.12, min_frac=0.02, tries=200, rng=None):
    """Random rectangle with zero GT overlap covering ~frac of the image.
    If the subject is large, the target fraction shrinks (halving down to
    min_frac) until a placement exists. Returns (y, x, h, w, frac) or None."""
    h, w = gt_bin.shape
    rng = rng or random
    f = frac
    while f >= min_frac:
        area = h * w * f
        hw_ratio = h / w
        rh = int((area * hw_ratio) ** 0.5)
        rw = int(area / max(rh, 1))
        rh, rw = min(rh, h - 2), min(rw, w - 2)
        if rh > 4 and rw > 4:
            for _ in range(tries):
                y = rng.randint(0, h - rh)
                x = rng.randint(0, w - rw)
                if gt_bin[y:y + rh, x:x + rw].sum() == 0:
                    return y, x, rh, rw, f
        f /= 2
    return None  # subject fills the frame at every size


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DUTS-TE root directory")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--zoom", type=float, default=1.8)
    ap.add_argument("--max-side", type=int, default=1024,
                    help="resize inputs so the long side is at most this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    img_dir = os.path.join(args.data, "DUTS-TE-Image")
    msk_dir = os.path.join(args.data, "DUTS-TE-Mask")
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = random.Random(args.seed)
    rng.shuffle(names)
    names = names[:args.n]
    print(f"Evaluating {len(names)} images | zoom {args.zoom}x | "
          f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    extractor = ForegroundExtractor()
    inpainter = get_inpainter()
    ups = SmartUpscaler()
    lp = load_lpips()

    cols = ["image", "iou", "f1",
            "inp_psnr", "inp_ssim", "inp_lpips",
            "zoom_lanczos_psnr", "zoom_lanczos_lpips",
            "zoom_ai_psnr", "zoom_ai_lpips",
            "comp_psnr", "comp_ssim", "comp_lpips",
            "t_extract", "t_inpaint", "t_zoom", "t_total"]
    rows = []

    for i, name in enumerate(names, 1):
        try:
            ipath = os.path.join(img_dir, name)
            mpath = os.path.join(msk_dir, os.path.splitext(name)[0] + ".png")
            if not os.path.exists(mpath):
                continue

            orig = np.array(Image.open(ipath).convert("RGB"))
            gt = np.array(Image.open(mpath).convert("L"))
            h0, w0 = orig.shape[:2]
            if max(h0, w0) > args.max_side:
                s = args.max_side / max(h0, w0)
                orig = cv2.resize(orig, (int(w0 * s), int(h0 * s)),
                                  interpolation=cv2.INTER_AREA)
            h, w = orig.shape[:2]
            if gt.shape != (h, w):
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_bin = gt > 127

            row = {"image": name}
            t_all = time.time()

            # ---- 1. segmentation
            t0 = time.time()
            tmp = os.path.join(os.path.dirname(args.out) or ".", "_eval_tmp.png")
            Image.fromarray(orig).save(tmp)
            fg = extractor.extract_foreground(tmp)
            if fg.shape[:2] != (h, w):
                fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LANCZOS4)
            pred_bin = fg[:, :, 3] > 127
            row["t_extract"] = round(time.time() - t0, 2)
            iou, f1 = seg_scores(pred_bin, gt_bin)
            row["iou"], row["f1"] = round(iou, 4), round(f1, 4)

            # ---- 2. inpainting: synthetic hole in known background
            hole = find_bg_hole(gt_bin, rng=rng)
            if hole:
                y, x, rh, rw, hole_frac = hole
                hole_mask = np.zeros((h, w), np.uint8)
                hole_mask[y:y + rh, x:x + rw] = 255
                t0 = time.time()
                inp = np.array(inpainter(Image.fromarray(orig),
                                         Image.fromarray(hole_mask)))
                row["t_inpaint"] = round(time.time() - t0, 2)
                a = inp[y:y + rh, x:x + rw]
                b = orig[y:y + rh, x:x + rw]
                row["inp_psnr"] = round(sk_psnr(b, a, data_range=255), 2)
                row["inp_ssim"] = round(sk_ssim(b, a, channel_axis=2,
                                                data_range=255), 4)
                row["inp_lpips"] = (round(lpips_score(lp, a, b), 4)
                                    if lp else "")

            # ---- 3. zoom ablation: downscale -> upscale vs original
            t0 = time.time()
            small = cv2.resize(orig, (int(w / args.zoom), int(h / args.zoom)),
                               interpolation=cv2.INTER_AREA)
            lan = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
            ai = ups.upscale(small, w, h)
            row["t_zoom"] = round(time.time() - t0, 2)
            row["zoom_lanczos_psnr"] = round(sk_psnr(orig, lan, data_range=255), 2)
            row["zoom_ai_psnr"] = round(sk_psnr(orig, ai, data_range=255), 2)
            if lp:
                row["zoom_lanczos_lpips"] = round(lpips_score(lp, lan, orig), 4)
                row["zoom_ai_lpips"] = round(lpips_score(lp, ai, orig), 4)

            # ---- 4. composite deviation (full automatic pipeline)
            full_mask = (fg[:, :, 3] > 127).astype(np.uint8) * 255
            diag = (h * h + w * w) ** 0.5
            k = max(15, int(diag * 0.02)) | 1
            full_dil = cv2.dilate(full_mask, np.ones((k, k), np.uint8))
            bg = np.array(inpainter(Image.fromarray(orig),
                                    Image.fromarray(full_dil)))
            ch, cw = int(h / args.zoom), int(w / args.zoom)
            sy, sx = (h - ch) // 2, (w - cw) // 2
            bz = ups.upscale(bg[sy:sy + ch, sx:sx + cw], w, h)
            alpha = (fg[:, :, 3:4].astype(np.float32)) / 255.0
            comp = (fg[:, :, :3] * alpha + bz * (1 - alpha)).astype(np.uint8)
            row["comp_psnr"] = round(sk_psnr(orig, comp, data_range=255), 2)
            row["comp_ssim"] = round(sk_ssim(orig, comp, channel_axis=2,
                                             data_range=255), 4)
            row["comp_lpips"] = (round(lpips_score(lp, comp, orig), 4)
                                 if lp else "")

            row["t_total"] = round(time.time() - t_all, 2)
            rows.append(row)
            print(f"[{i}/{len(names)}] {name}  F1={row['f1']}  "
                  f"t={row['t_total']}s")
        except Exception as e:
            print(f"[{i}/{len(names)}] {name} FAILED: {e}")

    # ---- write CSV + means
    with open(args.out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({c: r.get(c, "") for c in cols})
        means = {"image": f"MEAN (n={len(rows)})"}
        for c in cols[1:]:
            vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
            if vals:
                means[c] = round(sum(vals) / len(vals), 4)
        wcsv.writerow({c: means.get(c, "") for c in cols})

    print("\n===== SUMMARY =====")
    for c in cols[1:]:
        if c in means:
            print(f"{c:22s} {means[c]}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
