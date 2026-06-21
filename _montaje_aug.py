
from pathlib import Path
from PIL import Image

BASE = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose")
ORIG = BASE / "dataset" / "train" / "images"
AUG  = BASE / "dataset" / "train_aug" / "images"
OUT  = BASE / "_montaje_aug.jpg"

STEMS = [
    "frame_0253_jpg.rf.h4l19t5dzMcJ95LlvVLC",
    "frame_0256_jpg.rf.OcccR6hPx88r7wmswcsY",
    "frame_0258_jpg.rf.1SSLHO8mkF6uRzzEsDom",
]
TW, TH = 320, 240

cols = 4  
rows = len(STEMS)
canvas = Image.new("RGB", (cols * TW, rows * TH), (20, 20, 20))

for r, stem in enumerate(STEMS):
    orig_path = ORIG / f"{stem}.jpg"
    if orig_path.exists():
        img = Image.open(orig_path).convert("RGB").resize((TW, TH))
        canvas.paste(img, (0, r * TH))
    for c, idx in enumerate([0, 1, 2]):
        aug_path = AUG / f"{stem}_jpg_bg{idx}.jpg"
        if aug_path.exists():
            img = Image.open(aug_path).convert("RGB").resize((TW, TH))
            canvas.paste(img, ((c + 1) * TW, r * TH))

from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    font = ImageFont.load_default()

headers = ["Original", "bg0", "bg1", "bg2"]
for c, h in enumerate(headers):
    draw.rectangle([c * TW, 0, (c + 1) * TW, 22], fill=(0, 0, 0, 180))
    draw.text((c * TW + 5, 3), h, fill=(255, 220, 0), font=font)

canvas.save(OUT, "JPEG", quality=90)
print(f"Guardado: {OUT}")
