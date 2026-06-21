
import re
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose")
IMG_IN = BASE / "dataset" / "train" / "images"
LBL_IN = BASE / "dataset" / "train" / "labels"
IMG_OUT = BASE / "dataset" / "train_aug" / "images"

KP_ACTIVOS = [10, 11, 12, 13, 14, 15]  
pat = re.compile(r"^(.*)_(jpg|jpeg|png)_bg\d+$")


def kpts_px(label_path, w, h):
    vals = label_path.read_text().split()
    arr = np.array(vals[5:53], dtype=float).reshape(16, 3)
    out = {}
    for i in KP_ACTIVOS:
        x, y, v = arr[i]
        out[i] = (int(round(x * w)), int(round(y * h)), int(v))
    return out


tot_vis = preservados = 0
for out_img in sorted(IMG_OUT.glob("*.jpg")):
    m = pat.match(out_img.stem)
    stem, ext = m.group(1), m.group(2)
    orig = IMG_IN / f"{stem}.{ext}"
    lbl = LBL_IN / f"{stem}.txt"

    o = np.array(Image.open(orig).convert("RGB"))
    g = np.array(Image.open(out_img).convert("RGB"))
    h, w = o.shape[:2]
    kp = kpts_px(lbl, w, h)

    msg = []
    for i, (x, y, v) in kp.items():
        if v < 1:
            continue
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        
        win_o = o[max(0, y-1):y+2, max(0, x-1):x+2].astype(int)
        win_g = g[max(0, y-1):y+2, max(0, x-1):x+2].astype(int)
        diff = np.abs(win_o - win_g).sum(axis=2).min()
        tot_vis += 1
        ok = diff <= 8  
        preservados += ok
        msg.append(f"KP{i}:{'OK' if ok else 'FONDO!'}(d={diff})")
    print(f"{out_img.name[:38]:38s} {' '.join(msg)}")

print("-" * 60)
print(f"Keypoints visibles: {tot_vis} | preservados (sobre dron): {preservados} "
      f"({100*preservados/max(tot_vis,1):.1f}%)")
