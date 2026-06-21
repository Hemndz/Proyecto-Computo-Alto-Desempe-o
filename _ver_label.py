
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose")
stem = "frame_0253_jpg.rf.h4l19t5dzMcJ95LlvVLC"
img = Image.open(BASE / "dataset" / "train" / "images" / f"{stem}.jpg").convert("RGB")
vals = (BASE / "dataset" / "train" / "labels" / f"{stem}.txt").read_text().split()
w, h = img.size
d = ImageDraw.Draw(img)

cx, cy, bw, bh = [float(v) for v in vals[1:5]]
x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

arr = np.array(vals[5:53], dtype=float).reshape(16, 3)
colores = {10: (0, 255, 0), 15: (0, 128, 255)}
for i in range(16):
    x, y, v = arr[i]
    if v < 1:
        continue
    px, py = x * w, y * h
    c = colores.get(i, (255, 255, 0))
    r = 7 if i in colores else 4
    d.ellipse([px - r, py - r, px + r, py + r], fill=c)

img.save(BASE / "_label_vis.jpg", quality=92)
print("guardado _label_vis.jpg  | bbox area frac:", round(bw * bh, 4))
print("KP10 centro (verde), KP15 frente (azul)")
