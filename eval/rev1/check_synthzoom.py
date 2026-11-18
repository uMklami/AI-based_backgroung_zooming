"""Compare a freshly built SynthZoom with the one used on the original
machine (results/synthzoom_checksums.json). Exact matches mean the
already-finished SynthZoom ablations and new ones use identical data;
otherwise the mean absolute pixel difference is reported."""
import hashlib, json, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(os.environ.get("SELZOOM_REV", os.path.expanduser("~/selzoom_rev")), "data", "synthzoom")
ref = json.load(open(os.path.join(HERE, "results", "synthzoom_checksums.json")))
exact = 0; diffs = []
for k, (md5, mean) in ref.items():
    a = np.array(Image.open(os.path.join(root, k)))
    if hashlib.md5(a.tobytes()).hexdigest() == md5:
        exact += 1
    else:
        diffs.append(abs(float(a.mean()) - mean))
print(f"{exact}/{len(ref)} files identical; max |mean difference| of the rest: "
      f"{max(diffs) if diffs else 0:.4f} grey levels")
