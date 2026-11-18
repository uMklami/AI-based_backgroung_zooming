"""
One-off setup for the revision experiments on a new machine (Windows,
Linux or macOS). Downloads datasets and external baselines into
SELZOOM_REV (default ~/selzoom_rev), prepares the evaluation sets and
builds the SynthZoom benchmark. Safe to re-run: finished steps are skipped.

    python eval/rev1/setup_rev1.py

Disk: about 6 GB (plus ~5 GB of model weights cached by Hugging Face).
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
REV = os.environ.get("SELZOOM_REV", os.path.expanduser("~/selzoom_rev"))
DATA = os.path.join(REV, "data")
EXT = os.path.join(REV, "ext")

HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
PARQUETS = [
    ("nobg/AIM-500", "data/test-00000-of-00001.parquet", "aim500.parquet"),
    ("nobg/DIS5K", "data/DIS_TE1-00000-of-00001-f9bce3f0fddcae93.parquet", "dis_te1.parquet"),
    ("nobg/DIS5K", "data/DIS_TE2-00000-of-00001-3bc3b5747aca882c.parquet", "dis_te2.parquet"),
    ("nobg/DIS5K", "data/DIS_TE3-00000-of-00002-a1eaca5f23c66c61.parquet", "dis_te3a.parquet"),
    ("nobg/DIS5K", "data/DIS_TE4-00000-of-00002-4c2d885b5b32c3b6.parquet", "dis_te4a.parquet"),
    ("nobg/COD10K", "data/test-00000-of-00002.parquet", "cod10k_test0.parquet"),
]


def log(*a):
    print("[setup]", *a, flush=True)


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    log("downloading", url)
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dst)
    return dst


def unzip(zp, dst_dir, marker):
    if os.path.exists(os.path.join(dst_dir, marker)):
        return
    log("unzipping", os.path.basename(zp))
    with zipfile.ZipFile(zp) as z:
        z.extractall(dst_dir)


def github_zip(repo, branch, dst):
    """Download a GitHub repository as a zip (no git needed)."""
    if os.path.isdir(dst):
        return
    url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    log("downloading", url)
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    top = z.namelist()[0].split("/")[0]
    z.extractall(EXT)
    os.replace(os.path.join(EXT, top), dst)


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(EXT, exist_ok=True)

    # ---- DUTS-TE and DIV2K validation
    zp = fetch("http://saliencydetection.net/duts/download/DUTS-TE.zip",
               os.path.join(DATA, "DUTS-TE.zip"))
    unzip(zp, DATA, "DUTS-TE")
    zp = fetch("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
               os.path.join(DATA, "DIV2K_valid_HR.zip"))
    unzip(zp, DATA, "DIV2K_valid_HR")

    # ---- AIM-500, DIS-TE, COD10K (Hugging Face parquet mirrors)
    for repo, path, out in PARQUETS:
        fetch(HF.format(repo, path), os.path.join(DATA, "hf", out))

    # ---- GFM (code + AM-2k weights) and PCT-Net (code + weights)
    github_zip("JizhiziLi/GFM", "master", os.path.join(EXT, "GFM"))
    w = os.path.join(EXT, "GFM", "models", "pretrained", "gfm_r2b_tt.zip")
    if not os.path.exists(w):
        os.makedirs(os.path.dirname(w), exist_ok=True)
        import gdown
        gdown.download(id="1Y8dgOprcPWdUgHUPSdue0lkFAUVvW10Q", output=w, quiet=False)
    util = os.path.join(EXT, "GFM", "core", "util.py")
    s = open(util).read()
    if "fusion_sigmoid.cuda()" in s:          # make GFM device-agnostic
        open(util, "w").write(s.replace("fusion_sigmoid = fusion_sigmoid.cuda()",
                                        "fusion_sigmoid = fusion_sigmoid.to(glance_sigmoid.device)"))
    github_zip("rakutentech/PCT-Net-Image-Harmonization", "main", os.path.join(EXT, "pctnet"))

    # ---- evaluation sets (AIM-500, DIS-TE, COD10K, ExDark, WildDepth)
    sets = os.path.join(DATA, "sets")
    todo = [s for s in ("aim500", "dis", "cod10k", "exdark", "wilddepth")
            if not os.path.exists(os.path.join(sets, {"dis": "dis_te"}.get(s, s), "meta.json"))]
    if todo:
        subprocess.check_call([sys.executable, os.path.join(HERE, "prep_data.py")] + todo,
                              cwd=PROJECT)
    # WildDepth videos show RGB | LiDAR | overlay side by side: keep the RGB panel
    wd = os.path.join(sets, "wilddepth", "images")
    if os.path.isdir(wd):
        from PIL import Image
        for f in os.listdir(wd):
            p = os.path.join(wd, f)
            im = Image.open(p)
            if im.size[0] > 3000:
                im.crop((0, 0, im.size[0] // 3, im.size[1])).save(p, quality=95)

    # ---- SynthZoom (separate process: torch and pymatting/numba clash)
    if not os.path.exists(os.path.join(DATA, "synthzoom", "meta.json")):
        subprocess.check_call([sys.executable, os.path.join(HERE, "e0_build_synthzoom.py"),
                               "--n", "200"], cwd=PROJECT)
    log("done. Data in", DATA)


if __name__ == "__main__":
    main()
