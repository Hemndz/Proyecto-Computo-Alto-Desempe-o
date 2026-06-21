

from __future__ import annotations

import argparse
import random
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

BASE = Path(r"C:/Users/benja/OneDrive/Escritorio/CAAD PROYECT/dron_pose")
IMAGES_IN = BASE / "dataset" / "train" / "images"
LABELS_IN = BASE / "dataset" / "train" / "labels"
FONDOS_DIR = Path(r"C:/Users/benja/OneDrive/Escritorio/Fondos")
IMAGES_OUT = BASE / "dataset" / "train_aug" / "images"
LABELS_OUT = BASE / "dataset" / "train_aug" / "labels"

IMG_EXTS = (".jpg", ".jpeg", ".png")
BG_EXTS = (".jpg", ".jpeg", ".png")

SEED = 42
JPEG_QUALITY = 95
LOG_EVERY = 50

MARGEN_BBOX_DEFAULT = 0.15  
FEATHER_PX_DEFAULT = 6      


def cubrir_fondo(fondo_path: Path, w: int, h: int) -> Image.Image:
    
    with Image.open(fondo_path) as bg_file:
        fondo = ImageOps.exif_transpose(bg_file).convert("RGB")
    bw, bh = fondo.size
    escala = max(w / bw, h / bh)
    new_w = max(w, int(np.ceil(bw * escala)))
    new_h = max(h, int(np.ceil(bh * escala)))
    fondo = fondo.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    fondo = fondo.crop((left, top, left + w, top + h))
    if fondo.size != (w, h):
        raise RuntimeError(f"Fondo {fondo.size} != objetivo {(w, h)}")
    return fondo


def iluminacion_aleatoria(img_rgb: Image.Image, rng: random.Random) -> Image.Image:
    
    if img_rgb.mode != "RGB":
        img_rgb = img_rgb.convert("RGB")
    img = ImageEnhance.Brightness(img_rgb).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    r_factor = rng.uniform(0.9, 1.1)
    b_factor = rng.uniform(0.9, 1.1)
    arr = np.array(img, dtype=np.float32)
    arr[..., 0] *= r_factor
    arr[..., 2] *= b_factor
    np.clip(arr, 0, 255, out=arr)
    return Image.fromarray(np.rint(arr).astype(np.uint8), mode="RGB")


def leer_bboxes(label_path: Path, w: int, h: int, margen: float) -> list[tuple[int, int, int, int]]:
    
    cajas = []
    for linea in label_path.read_text().splitlines():
        p = linea.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(p[1]) * w, float(p[2]) * h,
                          float(p[3]) * w, float(p[4]) * h)
        ex, ey = bw * (1 + margen), bh * (1 + margen)
        x1 = int(max(0, round(cx - ex / 2)))
        y1 = int(max(0, round(cy - ey / 2)))
        x2 = int(min(w, round(cx + ex / 2)))
        y2 = int(min(h, round(cy + ey / 2)))
        if x2 > x1 and y2 > y1:
            cajas.append((x1, y1, x2, y2))
    return cajas


def procesar_imagen(img_path, label_path, fondos, n_variantes, rng, margen, feather) -> int:
    stem = img_path.stem
    ext_tag = img_path.suffix.lstrip(".").lower()

    with Image.open(img_path) as im:
        orientation = im.getexif().get(0x0112)
        if orientation not in (None, 1):
            raise RuntimeError(f"EXIF Orientation={orientation} no soportada")
        original = im.convert("RGB")
    w, h = original.size

    cajas = leer_bboxes(label_path, w, h, margen)
    if not cajas:
        raise RuntimeError("etiqueta sin bbox valido")

    
    
    mascara = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mascara)
    for (x1, y1, x2, y2) in cajas:
        md.rectangle([x1, y1, x2, y2], fill=255)
    mascara = mascara.filter(ImageFilter.GaussianBlur(feather))

    if n_variantes <= len(fondos):
        bg_choices = rng.sample(fondos, n_variantes)
    else:
        bg_choices = [rng.choice(fondos) for _ in range(n_variantes)]

    generadas = 0
    for i in range(n_variantes):
        try:
            fondo = cubrir_fondo(bg_choices[i], w, h)
            
            compuesta = Image.composite(original, fondo, mascara)
            if compuesta.size != (w, h):
                raise RuntimeError(f"compuesta {compuesta.size} != {(w, h)}")
            final = iluminacion_aleatoria(compuesta, rng)
            if final.size != (w, h):
                raise RuntimeError(f"final {final.size} != {(w, h)}")
            out_img = IMAGES_OUT / f"{stem}_{ext_tag}_bg{i}.jpg"
            out_lbl = LABELS_OUT / f"{stem}_{ext_tag}_bg{i}.txt"
            final.save(out_img, "JPEG", quality=JPEG_QUALITY)
            shutil.copy(label_path, out_lbl)
            generadas += 1
        except Exception as exc:  
            print(f"  [WARN] variante {i} de '{stem}' fallo: {exc}", flush=True)
            continue
    return generadas


def limpiar_salida() -> int:
    borrados = 0
    for carpeta, patrones in ((IMAGES_OUT, ("*.jpg", "*.jpeg")), (LABELS_OUT, ("*.txt",))):
        if carpeta.is_dir():
            for patron in patrones:
                for f in carpeta.glob(patron):
                    try:
                        f.unlink()
                        borrados += 1
                    except OSError as exc:
                        print(f"  [WARN] no se pudo borrar {f}: {exc}", flush=True)
    return borrados


def main() -> None:
    ap = argparse.ArgumentParser(description="Domain Randomization por bbox (etiquetas intactas).")
    ap.add_argument("--variantes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--margen", type=float, default=MARGEN_BBOX_DEFAULT,
                    help="expansion del bbox por lado (ej. 0.10 = 10%%). Default 0.15.")
    ap.add_argument("--feather", type=int, default=FEATHER_PX_DEFAULT,
                    help="radio de suavizado del borde en px. Default 6.")
    args = ap.parse_args()
    if args.variantes < 1:
        print("ERROR: --variantes debe ser >= 1.")
        sys.exit(1)
    if args.margen < 0:
        print("ERROR: --margen debe ser >= 0.")
        sys.exit(1)

    rng = random.Random(SEED)

    for d, nombre in ((IMAGES_IN, "imagenes"), (LABELS_IN, "etiquetas"), (FONDOS_DIR, "fondos")):
        if not d.is_dir():
            print(f"ERROR: no existe el directorio de {nombre}: {d}")
            sys.exit(1)
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.mkdir(parents=True, exist_ok=True)

    if args.clean:
        print(f"[INFO] --clean: {limpiar_salida()} archivos previos borrados.")
    else:
        previos = len(list(IMAGES_OUT.glob("*.jpg")))
        if previos:
            print(f"[WARN] {previos} imagenes ya existen en salida (usa --clean para limpiar).", flush=True)

    fondos = sorted(p for p in FONDOS_DIR.iterdir() if p.is_file() and p.suffix.lower() in BG_EXTS)
    imagenes = sorted(p for p in IMAGES_IN.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not fondos:
        print(f"ERROR: sin fondos en {FONDOS_DIR}"); sys.exit(1)
    if not imagenes:
        print(f"ERROR: sin imagenes en {IMAGES_IN}"); sys.exit(1)
    if args.limit and args.limit > 0:
        imagenes = imagenes[: args.limit]

    print("=" * 70)
    print("AUGMENTACION - Domain Randomization por BBOX (etiquetas intactas)")
    print("=" * 70)
    print(f"Imagenes a procesar : {len(imagenes)}")
    print(f"Fondos disponibles  : {len(fondos)}")
    print(f"Variantes / imagen  : {args.variantes}")
    print(f"Margen bbox         : {args.margen:.0%}  | feather: {args.feather}px")
    print("-" * 70)

    n_ok = n_var = n_fallos = n_sin_label = 0
    fallidas = []
    for idx, img_path in enumerate(imagenes, start=1):
        label_path = LABELS_IN / f"{img_path.stem}.txt"
        if not label_path.is_file():
            n_sin_label += 1
            print(f"[SKIP] sin etiqueta: {img_path.name}")
            continue
        try:
            n_var += procesar_imagen(img_path, label_path, fondos, args.variantes,
                                     rng, args.margen, args.feather)
            n_ok += 1
        except Exception as exc:  
            n_fallos += 1
            fallidas.append(img_path.name)
            print(f"[FALLO] {img_path.name}: {exc}", flush=True)
            traceback.print_exc()
        if idx % LOG_EVERY == 0:
            print(f"  progreso: {idx}/{len(imagenes)} | {n_var} variantes | {n_fallos} fallos", flush=True)

    print("-" * 70)
    print("RESUMEN FINAL")
    print(f"  Originales procesadas con exito : {n_ok}")
    print(f"  Originales sin etiqueta (skip)  : {n_sin_label}")
    print(f"  Originales con fallo            : {n_fallos}")
    print(f"  Variantes sinteticas generadas  : {n_var}")
    if fallidas:
        print(f"  ABORTADAS ({len(fallidas)}): " + ", ".join(fallidas))
    print("=" * 70)


if __name__ == "__main__":
    main()
