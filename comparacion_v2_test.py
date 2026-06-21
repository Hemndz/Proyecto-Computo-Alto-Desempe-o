

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).parent
DATA_YAML = BASE / "dataset" / "data.yaml"
TEST_IMAGES = BASE / "dataset" / "test" / "images"
TEST_LABELS = BASE / "dataset" / "test" / "labels"

KP_CENTRO = 10  
KP_FRENTE = 15  
CONF_MIN = 0.5

MODELOS = {
    "yolov8n-v2 (nano dataset ampliado)": BASE / "runs" / "dron_pose_v2_nano" / "weights" / "best.pt",
    "yolov8s-v2 (small dataset ampliado)": BASE / "runs" / "dron_pose_v2" / "weights" / "best.pt",
}


def angulo_desde_kpts(cx, cy, fx, fy):
    return np.degrees(np.arctan2(-(fy - cy), fx - cx)) % 360.0


def angulo_gt(label_path, img_w, img_h):
    
    valores = label_path.read_text().split()
    kpts = np.array(valores[5:53], dtype=float).reshape(16, 3)
    c, f = kpts[KP_CENTRO], kpts[KP_FRENTE]
    if c[2] < 2 or f[2] < 2:
        return None
    return angulo_desde_kpts(c[0] * img_w, c[1] * img_h, f[0] * img_w, f[1] * img_h)


def error_circular(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main():
    imagenes = sorted(TEST_IMAGES.glob("*.jpg"))
    print(f"Test set: {len(imagenes)} imagenes\n")

    for nombre, pesos in MODELOS.items():
        print(f"========== {nombre} ==========")
        model = YOLO(str(pesos))

        m = model.val(data=str(DATA_YAML), split="test", device=0, verbose=False)
        print(f"Box  mAP50: {m.box.map50:.4f} | mAP50-95: {m.box.map:.4f} | "
              f"P: {m.box.mp:.4f} | R: {m.box.mr:.4f}")
        print(f"Pose mAP50: {m.pose.map50:.4f} | mAP50-95: {m.pose.map:.4f} | "
              f"P: {m.pose.mp:.4f} | R: {m.pose.mr:.4f}")

        errores, sin_deteccion, kpt_baja_conf = [], 0, 0
        for img_path in imagenes:
            label_path = TEST_LABELS / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            w, h = Image.open(img_path).size
            gt = angulo_gt(label_path, w, h)
            if gt is None:
                continue

            res = model.predict(str(img_path), device=0, verbose=False)[0]
            if res.keypoints is None or len(res.keypoints) == 0:
                sin_deteccion += 1
                continue
            kpts = res.keypoints.data.cpu().numpy()[0]
            (cx, cy, cc), (fx, fy, cf) = kpts[KP_CENTRO], kpts[KP_FRENTE]
            if cc < CONF_MIN or cf < CONF_MIN:
                kpt_baja_conf += 1
                continue
            pred = angulo_desde_kpts(cx, cy, fx, fy)
            errores.append(abs(error_circular(pred, gt)))

        errores = np.array(errores)
        n_eval = len(errores)
        print(f"\nAngulo Centro->Frente vs ground truth ({n_eval} imagenes evaluables):")
        if n_eval:
            print(f"  error medio   : {errores.mean():6.2f} deg")
            print(f"  error mediano : {np.median(errores):6.2f} deg")
            print(f"  error p95     : {np.percentile(errores, 95):6.2f} deg")
            print(f"  error maximo  : {errores.max():6.2f} deg")
            print(f"  dentro de 10 deg: {(errores <= 10).mean() * 100:.1f}%")
        print(f"  sin deteccion        : {sin_deteccion}")
        print(f"  kpt conf < {CONF_MIN}       : {kpt_baja_conf}")
        print()


if __name__ == "__main__":
    main()
