


from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps




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
REMBG_MODEL = "u2net"





MIN_COBERTURA_ALPHA = 0.005





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
    return fondo.convert("RGBA")





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





def quitar_fondo(original_rgb: Image.Image, session, remove_fn) -> Image.Image:
    
    w, h = original_rgb.size
    cutout = remove_fn(original_rgb, session=session, post_process_mask=True)

    
    if not isinstance(cutout, Image.Image):
        cutout = Image.open(io.BytesIO(cutout))
    cutout = cutout.convert("RGBA")

    
    
    if cutout.size != (w, h):
        lienzo = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        lienzo.paste(cutout, (0, 0))
        cutout = lienzo

    
    
    
    arr = np.array(cutout)
    a = arr[..., 3]
    arr[..., 3] = np.where(a > 128, 255, 0).astype(np.uint8)
    cutout = Image.fromarray(arr, mode="RGBA")

    
    frac = float((arr[..., 3] > 0).mean())
    if frac < MIN_COBERTURA_ALPHA:
        raise RuntimeError(
            f"rembg dejo mascara casi vacia (cobertura={frac:.4f} < "
            f"{MIN_COBERTURA_ALPHA}); se omite la imagen para no invalidar la etiqueta"
        )

    if cutout.size != (w, h):
        raise RuntimeError(f"Cutout {cutout.size} != original {(w, h)}")
    return cutout





def procesar_imagen(
    img_path: Path,
    label_path: Path,
    fondos: list[Path],
    session,
    remove_fn,
    n_variantes: int,
    rng: random.Random,
) -> int:
    
    stem = img_path.stem
    
    
    ext_tag = img_path.suffix.lstrip(".").lower()

    
    with Image.open(img_path) as im:
        
        
        
        
        
        orientation = im.getexif().get(0x0112)  
        if orientation not in (None, 1):
            raise RuntimeError(
                f"imagen con EXIF Orientation={orientation} no soportada "
                f"(re-exportar sin EXIF para no desincronizar la etiqueta)"
            )
        
        
        original_rgb = im.convert("RGB")
    w, h = original_rgb.size

    
    
    dron_rgba = quitar_fondo(original_rgb, session, remove_fn)
    if dron_rgba.size != (w, h):
        raise RuntimeError(f"Cutout {dron_rgba.size} != original {(w, h)}")

    
    
    if n_variantes <= len(fondos):
        bg_choices = rng.sample(fondos, n_variantes)
    else:
        bg_choices = [rng.choice(fondos) for _ in range(n_variantes)]

    generadas = 0
    for i in range(n_variantes):
        try:
            
            fondo_rgba = cubrir_fondo(bg_choices[i], w, h)  

            
            compuesta = Image.alpha_composite(fondo_rgba, dron_rgba)

            
            
            if compuesta.size != (w, h):
                raise RuntimeError(
                    f"Dimension compuesta {compuesta.size} != original {(w, h)}"
                )

            
            compuesta_rgb = compuesta.convert("RGB")
            final = iluminacion_aleatoria(compuesta_rgb, rng)

            
            if final.size != (w, h):
                raise RuntimeError(
                    f"Dimension final {final.size} != original {(w, h)}"
                )

            
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
    for carpeta, patrones in (
        (IMAGES_OUT, ("*.jpg", "*.jpeg")),
        (LABELS_OUT, ("*.txt",)),
    ):
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
    parser = argparse.ArgumentParser(
        description="Domain Randomization nivel 3 (cambio de fondo + iluminacion) "
        "para YOLOv8-Pose de dron. Solo fotometrico: etiquetas se copian intactas."
    )
    parser.add_argument(
        "--variantes",
        type=int,
        default=3,
        help="Numero de variantes sinteticas por imagen original (default: 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Procesar solo las primeras N imagenes (0 o ausente = todas). "
        "Util para prueba de humo.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Vaciar las carpetas de salida (*.jpg/*.jpeg/*.txt) antes de "
        "empezar, para evitar mezclar corridas (huerfanos al bajar --variantes).",
    )
    args = parser.parse_args()

    if args.variantes < 1:
        print("ERROR: --variantes debe ser >= 1.")
        sys.exit(1)

    
    
    rng = random.Random(SEED)

    
    try:
        from rembg import new_session, remove
    except Exception as exc:  
        print("ERROR: no se pudo importar rembg / su backend onnxruntime.")
        print(f"       Detalle: {exc}")
        print('       Instala con: .venv/Scripts/python.exe -m pip install "rembg[cpu]" onnxruntime')
        sys.exit(1)

    
    if not IMAGES_IN.is_dir():
        print(f"ERROR: no existe el directorio de imagenes: {IMAGES_IN}")
        sys.exit(1)
    if not LABELS_IN.is_dir():
        print(f"ERROR: no existe el directorio de etiquetas: {LABELS_IN}")
        sys.exit(1)
    if not FONDOS_DIR.is_dir():
        print(f"ERROR: no existe el directorio de fondos: {FONDOS_DIR}")
        sys.exit(1)

    
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.mkdir(parents=True, exist_ok=True)

    
    if args.clean:
        borrados = limpiar_salida()
        print(f"[INFO] --clean: {borrados} archivos previos borrados en salida.")
    else:
        previos = len(list(IMAGES_OUT.glob("*.jpg")))
        if previos:
            print(
                f"[WARN] {previos} imagenes ya existen en {IMAGES_OUT}. "
                f"Se sobrescribiran por nombre; archivos huerfanos de corridas "
                f"previas permaneceran. Usa --clean para empezar limpio.",
                flush=True,
            )

    
    fondos = sorted(
        p for p in FONDOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in BG_EXTS
    )
    if not fondos:
        print(f"ERROR: no se encontraron fondos en {FONDOS_DIR}")
        sys.exit(1)

    
    imagenes = sorted(
        p for p in IMAGES_IN.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not imagenes:
        print(f"ERROR: no se encontraron imagenes en {IMAGES_IN}")
        sys.exit(1)

    
    
    stems_vistos: dict[str, str] = {}
    for p in imagenes:
        if p.stem in stems_vistos:
            print(
                f"[WARN] stem duplicado: '{p.name}' y '{stems_vistos[p.stem]}' "
                f"comparten stem; la salida los distingue por extension.",
                flush=True,
            )
        else:
            stems_vistos[p.stem] = p.name

    if args.limit and args.limit > 0:
        imagenes = imagenes[: args.limit]

    print("=" * 70)
    print("AUGMENTACION - Domain Randomization nivel 3 (cambio de fondo + luz)")
    print("=" * 70)
    print(f"Imagenes a procesar : {len(imagenes)}")
    print(f"Fondos disponibles  : {len(fondos)}")
    print(f"Variantes / imagen  : {args.variantes}")
    print(f"Salida imagenes     : {IMAGES_OUT}")
    print(f"Salida etiquetas    : {LABELS_OUT}")
    print("-" * 70)

    
    
    
    print(f"[INFO] Creando sesion rembg ('{REMBG_MODEL}')...", flush=True)
    try:
        session = new_session(REMBG_MODEL)
    except Exception as exc:  
        print(f"ERROR: no se pudo crear la sesion rembg ('{REMBG_MODEL}').")
        print(f"       Detalle: {exc}")
        print("       Revisa la conexion (descarga del modelo ~170MB) o pre-descarga el .onnx.")
        sys.exit(1)

    n_originales_ok = 0
    n_variantes_total = 0
    n_fallos = 0
    n_sin_label = 0
    fallidas: list[str] = []  

    for idx, img_path in enumerate(imagenes, start=1):
        label_path = LABELS_IN / f"{img_path.stem}.txt"

        
        if not label_path.is_file():
            n_sin_label += 1
            print(f"[SKIP] sin etiqueta: {img_path.name}")
            continue

        try:
            gen = procesar_imagen(
                img_path,
                label_path,
                fondos,
                session,
                remove,
                args.variantes,
                rng,
            )
            n_originales_ok += 1
            n_variantes_total += gen
        except Exception as exc:  
            n_fallos += 1
            fallidas.append(img_path.name)
            print(f"[FALLO] {img_path.name}: {exc}", flush=True)
            traceback.print_exc()

        if idx % LOG_EVERY == 0:
            print(
                f"  progreso: {idx}/{len(imagenes)} originales | "
                f"{n_variantes_total} variantes generadas | "
                f"{n_fallos} fallos",
                flush=True,
            )

    print("-" * 70)
    print("RESUMEN FINAL")
    print(f"  Originales procesadas con exito : {n_originales_ok}")
    print(f"  Originales sin etiqueta (skip)  : {n_sin_label}")
    print(f"  Originales con fallo            : {n_fallos}")
    print(f"  Variantes sinteticas generadas  : {n_variantes_total}")
    print(f"  Salida imagenes                 : {IMAGES_OUT}")
    print(f"  Salida etiquetas                : {LABELS_OUT}")
    if fallidas:
        print("-" * 70)
        print(f"  IMAGENES ABORTADAS ({len(fallidas)}) - revisar manualmente:")
        for nombre in fallidas:
            print(f"    - {nombre}")
    print("=" * 70)


if __name__ == "__main__":
    main()
