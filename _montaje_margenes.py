
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont

BASE = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose")
IMG_IN = BASE / "dataset" / "train" / "images"
LBL_IN = BASE / "dataset" / "train" / "labels"
FONDOS = Path(r"C:\Users\benja\OneDrive\Escritorio\Fondos")

STEMS = [
    "frame_0253_jpg.rf.h4l19t5dzMcJ95LlvVLC",
    "frame_0258_jpg.rf.1SSLHO8mkF6uRzzEsDom",
]
MARGENES = [0.0, 0.10, 0.20, 0.30]
FEATHER = 6
TILE_W, TILE_H = 320, 240


fondo_path = sorted(p for p in FONDOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[10]


def cubrir(fp, w, h):
    with Image.open(fp) as f:
        bg = ImageOps.exif_transpose(f).convert("RGB")
    bw, bh = bg.size
    s = max(w / bw, h / bh)
    bg = bg.resize((max(w, int(np.ceil(bw*s))), max(h, int(np.ceil(bh*s)))), Image.LANCZOS)
    l, t = (bg.width - w)//2, (bg.height - h)//2
    return bg.crop((l, t, l + w, t + h))


def bboxes(lbl, w, h, margen):
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = float(p[1])*w, float(p[2])*h, float(p[3])*w, float(p[4])*h
        ex, ey = bw*(1+margen), bh*(1+margen)
        out.append((max(0, cx-ex/2), max(0, cy-ey/2), min(w, cx+ex/2), min(h, cy+ey/2)))
    return out


def componer(orig, lbl, margen):
    w, h = orig.size
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for b in bboxes(lbl, w, h, margen):
        d.rectangle(b, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(FEATHER))
    return Image.composite(orig, cubrir(fondo_path, w, h), m)


cols = 1 + len(MARGENES)
montaje = Image.new("RGB", (cols*TILE_W, len(STEMS)*TILE_H), (30, 30, 30))
draw = ImageDraw.Draw(montaje)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()


def etiqueta(x, y, txt):
    draw.rectangle([x, y, x+TILE_W, y+22], fill=(0, 0, 0))
    draw.text((x+5, y+3), txt, fill=(255, 255, 0), font=font)


for r, stem in enumerate(STEMS):
    orig = Image.open(IMG_IN / f"{stem}.jpg").convert("RGB")
    lbl = LBL_IN / f"{stem}.txt"
    w, h = orig.size

    
    ref = orig.copy()
    dd = ImageDraw.Draw(ref)
    for b in bboxes(lbl, w, h, 0.0):
        dd.rectangle(b, outline=(255, 0, 0), width=3)
    tile = ref.resize((TILE_W, TILE_H))
    montaje.paste(tile, (0, r*TILE_H))
    etiqueta(0, r*TILE_H, "original + bbox")

    for c, mg in enumerate(MARGENES, start=1):
        comp = componer(orig, lbl, mg).resize((TILE_W, TILE_H))
        montaje.paste(comp, (c*TILE_W, r*TILE_H))
        etiqueta(c*TILE_W, r*TILE_H, f"margen {int(mg*100)}%")

montaje.save(BASE / "_montaje_margenes.jpg", quality=92)
print("guardado _montaje_margenes.jpg  | fondo fijo:", fondo_path.name)
