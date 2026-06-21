import filecmp
import re
from pathlib import Path

from PIL import Image

BASE = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose")
IMG_IN = BASE / "dataset" / "train" / "images"
LBL_IN = BASE / "dataset" / "train" / "labels"
IMG_OUT = BASE / "dataset" / "train_aug" / "images"
LBL_OUT = BASE / "dataset" / "train_aug" / "labels"


pat = re.compile(r"^(.*)_(jpg|jpeg|png)_bg\d+$")

ok_dim = ok_lbl = total = 0
for out_img in sorted(IMG_OUT.glob("*.jpg")):
    m = pat.match(out_img.stem)
    if not m:
        print("NO MATCH:", out_img.name)
        continue
    stem, ext = m.group(1), m.group(2)
    orig_img = IMG_IN / f"{stem}.{ext}"
    orig_lbl = LBL_IN / f"{stem}.txt"
    out_lbl = LBL_OUT / (out_img.stem + ".txt")
    total += 1

    wo, ho = Image.open(orig_img).size
    wn, hn = Image.open(out_img).size
    dim_ok = (wo, ho) == (wn, hn)
    lbl_ok = filecmp.cmp(orig_lbl, out_lbl, shallow=False)
    ok_dim += dim_ok
    ok_lbl += lbl_ok
    print(f"{out_img.name:45s} dim {wn}x{hn} (orig {wo}x{ho}) {'OK' if dim_ok else 'MISMATCH'} | label {'IDENTICA' if lbl_ok else 'DIFERENTE'}")

print("-" * 60)
print(f"Total: {total} | dim OK: {ok_dim}/{total} | label identica: {ok_lbl}/{total}")

muestras = sorted(IMG_OUT.glob("*.jpg"))
if muestras:
    print("MUESTRA:", muestras[0])
    print("MUESTRA:", muestras[len(muestras)//2])
