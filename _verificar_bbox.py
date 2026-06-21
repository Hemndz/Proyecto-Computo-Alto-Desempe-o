
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

tot_vis = dentro = 0
for out_img in sorted(IMG_OUT.glob("*.jpg")):
    m = pat.match(out_img.stem)
    stem, ext = m.group(1), m.group(2)
    orig = IMG_IN / f"{stem}.{ext}"
    lbl = LBL_IN / f"{stem}.txt"
    o = np.array(Image.open(orig).convert("RGB")).astype(int)
    g = np.array(Image.open(out_img).convert("RGB")).astype(int)
    h, w = o.shape[:2]

    lineas = [l.split() for l in lbl.read_text().splitlines() if len(l.split()) >= 53]
    
    frac_cambio = float((np.abs(o - g).sum(axis=2) > 40).mean())

    msg = []
    for p in lineas:
        cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        bx1, by1, bx2, by2 = cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2
        arr = np.array(p[5:53], dtype=float).reshape(16, 3)
        for i in KP_ACTIVOS:
            x, y, v = arr[i]
            if v < 1:
                continue
            tot_vis += 1
            ok = (bx1 <= x <= bx2) and (by1 <= y <= by2)
            dentro += ok
            if not ok:
                msg.append(f"KP{i}:FUERA")
    estado = "todos dentro del bbox" if not msg else " ".join(msg)
    print(f"{out_img.name[:38]:38s} cambio_fondo={frac_cambio*100:4.0f}%  {estado}")

print("-" * 64)
print(f"Keypoints visibles: {tot_vis} | dentro del bbox (preservados): {dentro} "
      f"({100*dentro/max(tot_vis,1):.1f}%)")
