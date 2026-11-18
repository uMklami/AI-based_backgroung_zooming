"""
Prepare the evaluation sets used in the revision (Section 4.1).

    python eval/rev1/prep_data.py [aim500 dis cod10k exdark wilddepth]

Sources (all public):
  AIM-500, DIS5K test, COD10K test  - Hugging Face mirrors (nobg/*) as parquet
  ExDark                            - original-resolution mirror (fuhao7i/ExDark)
  WildDepth                         - Zenodo records (RGB videos, CC-BY-4.0)
Every set is written to data/sets/<name>/{images,masks}/ with a meta.json.
Sampling is seeded (SEED = 42) so the sets are reproducible.
"""
import io
import json
import os
import random
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SEED  # noqa: E402

HF = os.path.join(DATA, "hf")
SETS = os.path.join(DATA, "sets")


def out_dirs(name):
    root = os.path.join(SETS, name)
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    return root


def save_meta(root, items):
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(items, f, indent=1)
    print(f"{os.path.basename(root)}: {len(items)} items")


def pq_rows(path, columns=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns)
    return t.to_pylist()


def img_from(cell):
    return Image.open(io.BytesIO(cell["bytes"]))


def prep_aim500():
    root = out_dirs("aim500")
    items = []
    for r in pq_rows(os.path.join(HF, "aim500.parquet")):
        nm = r["image_name"]
        img_from(r["image"]).convert("RGB").save(os.path.join(root, "images", nm + ".png"))
        img_from(r["mask"]).convert("L").save(os.path.join(root, "masks", nm + ".png"))
        items.append(dict(file=nm + ".png", mask_file=nm + ".png", name=nm,
                          type=r["type"], category=r["category"],
                          source="AIM-500", alpha=True))
    save_meta(root, items)


def prep_dis(per_subset=100):
    root = out_dirs("dis_te")
    rng = random.Random(SEED)
    items = []
    for sub, fn in [("TE1", "dis_te1.parquet"), ("TE2", "dis_te2.parquet"),
                    ("TE3", "dis_te3a.parquet"), ("TE4", "dis_te4a.parquet")]:
        rows = pq_rows(os.path.join(HF, fn))
        idx = sorted(rng.sample(range(len(rows)), min(per_subset, len(rows))))
        for i in idx:
            r = rows[i]
            nm = os.path.splitext(os.path.basename(r["image_name"]))[0]
            img_from(r["image"]).convert("RGB").save(os.path.join(root, "images", nm + ".jpg"), quality=95)
            img_from(r["label"]).convert("L").save(os.path.join(root, "masks", nm + ".png"))
            items.append(dict(file=nm + ".jpg", mask_file=nm + ".png", name=nm,
                              subset=sub, source="DIS-TE"))
    save_meta(root, items)


def prep_cod10k(n=300):
    root = out_dirs("cod10k")
    rows = pq_rows(os.path.join(HF, "cod10k_test0.parquet"))
    rng = random.Random(SEED)
    keep = []
    for i, r in enumerate(rows):
        m = np.array(img_from(r["mask"]).convert("L"))
        frac = (m > 127).mean()
        if 0.005 < frac < 0.9:          # camouflaged object present
            keep.append(i)
    idx = sorted(rng.sample(keep, min(n, len(keep))))
    items = []
    for i in idx:
        r = rows[i]
        nm = f"cod10k_{i:04d}"
        img_from(r["image"]).convert("RGB").save(os.path.join(root, "images", nm + ".jpg"), quality=95)
        img_from(r["mask"]).convert("L").save(os.path.join(root, "masks", nm + ".png"))
        items.append(dict(file=nm + ".jpg", mask_file=nm + ".png", name=nm, source="COD10K"))
    save_meta(root, items)


EXDARK_BASE = "https://huggingface.co/datasets/fuhao7i/ExDark/resolve/main/JPEGImages/"


def prep_exdark(n=150):
    """Low-light images with one dominant annotated object (largest box
    covering 5-60% of the frame and at least twice the next box)."""
    root = out_dirs("exdark")
    with urllib.request.urlopen(EXDARK_BASE + "test.json") as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    boxes = {}
    for a in d["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a)
    cand = []
    for im in d["images"]:
        bs = sorted(boxes.get(im["id"], []), key=lambda a: -a["bbox"][2] * a["bbox"][3])
        if not bs:
            continue
        A = im["width"] * im["height"]
        a0 = bs[0]["bbox"][2] * bs[0]["bbox"][3] / A
        a1 = bs[1]["bbox"][2] * bs[1]["bbox"][3] / A if len(bs) > 1 else 0
        if 0.05 <= a0 <= 0.6 and a0 >= 2 * a1:
            cand.append((im, bs[0]))
    rng = random.Random(SEED)
    rng.shuffle(cand)
    items = []
    for im, b in cand:
        if len(items) >= n:
            break
        stem = os.path.splitext(im["file_name"])[0]
        data = None
        for ext in (".jpg", ".png", ".JPG", ".jpeg", ".PNG", ".JPEG"):
            try:
                with urllib.request.urlopen(EXDARK_BASE + "IMGS/" + stem + ext) as f:
                    data = f.read()
                break
            except Exception:
                continue
        if data is None:
            continue
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (im["width"], im["height"]):
            continue
        img.save(os.path.join(root, "images", stem + ".jpg"), quality=95)
        items.append(dict(file=stem + ".jpg", mask_file=None, name=stem,
                          category=cats[b["category_id"]], bbox=b["bbox"],
                          source="ExDark"))
    save_meta(root, items)


# WildDepth sequences: two to four per site, spanning the species present
WILDDEPTH = [
    ("19371340", "giraffe_1_video.mp4"), ("19371340", "donkey_2_video.mp4"),
    ("19371340", "goat_4_video.mp4"), ("19371340", "giraffe_3_video.mp4"),
    ("19372084", "zebra_4_video.mp4"), ("19372084", "warthog_4_video.mp4"),
    ("19372084", "zebra_5_video.mp4"), ("19372084", "warthog_3_video.mp4"),
    ("19152060", "July08_graffe_1_video.mp4"), ("19152060", "July08_Ankle_cow_1_video.mp4"),
    ("19152060", "July08_cheetah_1_video.mp4"), ("19152060", "July07_zebra_2_video.mp4"),
    ("19152060", "July08_camel_1_video.mp4"), ("19154614", "July08_lion_2_video.mp4"),
    ("19154614", "July08_lioness_1_video.mp4"), ("19154614", "July08_tiger_1_video.mp4"),
]


def prep_wilddepth(frames_per_video=10, min_gap_s=2.0):
    root = out_dirs("wilddepth")
    vid_dir = os.path.join(DATA, "wilddepth_videos")
    os.makedirs(vid_dir, exist_ok=True)
    items = []
    for rec, fn in WILDDEPTH:
        vp = os.path.join(vid_dir, fn)
        if not os.path.exists(vp):
            url = f"https://zenodo.org/api/records/{rec}/files/{fn}/content"
            urllib.request.urlretrieve(url, vp)
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        gap = int(min_gap_s * fps)
        k = min(frames_per_video, max(1, n // max(gap, 1)))
        idx = np.linspace(0, n - 1, k + 2)[1:-1].astype(int)   # skip first/last
        species = fn.replace("_video.mp4", "").split("_")
        species = [s for s in species if not s.isdigit() and not s.startswith("July")]
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            if fr.shape[1] > 3000:          # RGB | LiDAR | overlay: keep RGB
                fr = fr[:, :fr.shape[1] // 3]
            stem = f"{fn.replace('_video.mp4', '')}_f{i:05d}"
            cv2.imwrite(os.path.join(root, "images", stem + ".jpg"), fr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            items.append(dict(file=stem + ".jpg", mask_file=None, name=stem,
                              species="_".join(species), record=rec,
                              frame=int(i), fps=fps, source="WildDepth",
                              note="left RGB panel of the WildDepth visualisation video"))
        cap.release()
    save_meta(root, items)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["aim500", "dis", "cod10k", "exdark", "wilddepth"]
    for t in todo:
        globals()["prep_" + t]()
