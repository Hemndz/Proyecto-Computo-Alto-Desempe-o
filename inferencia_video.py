

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose")

MODELOS = {
    "v2-nano": BASE / "runs" / "dron_pose_v2_nano" / "weights" / "best.pt",
    "v3":      BASE / "runs" / "dron_pose_v3"      / "weights" / "best.pt",
    "v4":      BASE / "runs" / "dron_pose_v4"      / "weights" / "best.pt",
}

KP_CENTRO = 10
KP_FRENTE = 15
CONF_MIN  = 0.5
KP_CONF_MIN = 0.4


COLOR_BOX    = (0, 220, 0)
COLOR_KP     = (0, 200, 255)
COLOR_CENTRO = (0, 80, 255)
COLOR_FRENTE = (255, 80, 0)
COLOR_FLECHA = (255, 255, 0)
COLOR_TEXTO  = (255, 255, 255)
COLOR_BG     = (0, 0, 0)

KP_ACTIVOS = list(range(10, 16))  


def angulo_desde_kpts(cx, cy, fx, fy) -> float:
    return math.degrees(math.atan2(-(fy - cy), fx - cx)) % 360.0


def dibujar_flecha(frame, cx, cy, angulo_deg, longitud=60):
    rad = math.radians(angulo_deg)
    ex = int(cx + longitud * math.cos(rad))
    ey = int(cy - longitud * math.sin(rad))
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey),
                    COLOR_FLECHA, 3, tipLength=0.35)


def texto_con_fondo(frame, texto, x, y, escala=0.65, grosor=1):
    (tw, th), bl = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + bl), COLOR_BG, -1)
    cv2.putText(frame, texto, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, escala, COLOR_TEXTO, grosor, cv2.LINE_AA)


def procesar_video(video_path: Path, modelo_key: str, salida: Path, conf: float):
    pesos = MODELOS[modelo_key]
    if not pesos.exists():
        print(f"ERROR: no existe {pesos}")
        return

    print(f"Modelo : {modelo_key}  ({pesos})")
    print(f"Video  : {video_path}")

    model = YOLO(str(pesos))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: no se pudo abrir el video.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolucion: {width}x{height} @ {fps:.1f}fps  |  {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(salida), fourcc, fps, (width, height))

    n_frames = 0
    n_det = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1

        results = model.predict(frame, device=0, verbose=False, conf=conf)[0]

        if results.boxes is not None and len(results.boxes):
            for i, box in enumerate(results.boxes):
                conf_det = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

                angulo = None
                if results.keypoints is not None and i < len(results.keypoints):
                    kpts = results.keypoints.data.cpu().numpy()[i]  

                    
                    for idx in KP_ACTIVOS:
                        kx, ky, kc = kpts[idx]
                        if kc >= KP_CONF_MIN:
                            color = COLOR_CENTRO if idx == KP_CENTRO else (
                                    COLOR_FRENTE if idx == KP_FRENTE else COLOR_KP)
                            cv2.circle(frame, (int(kx), int(ky)), 5, color, -1)

                    
                    cx_k, cy_k, cc = kpts[KP_CENTRO]
                    fx_k, fy_k, fc = kpts[KP_FRENTE]
                    if cc >= KP_CONF_MIN and fc >= KP_CONF_MIN:
                        angulo = angulo_desde_kpts(cx_k, cy_k, fx_k, fy_k)
                        dibujar_flecha(frame, cx_k, cy_k, angulo)
                        texto_con_fondo(frame, f"{angulo:.1f} deg",
                                        x1, max(y1 - 8, 18))
                        n_det += 1

        if n_frames % 100 == 0:
            print(f"  frame {n_frames}/{total}", flush=True)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nListo: {n_frames} frames procesados, {n_det} con orientacion detectada.")
    print(f"Video anotado: {salida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="v2-nano", choices=list(MODELOS))
    ap.add_argument("--video",  default=r"C:\Users\benja\Downloads\Drone_det.mp4")
    ap.add_argument("--conf",   type=float, default=CONF_MIN)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video no encontrado: {video_path}")
        return

    salida = BASE / f"Drone_det_{args.modelo}_anotado.mp4"
    procesar_video(video_path, args.modelo, salida, args.conf)


if __name__ == "__main__":
    main()
