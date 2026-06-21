

from __future__ import annotations

import argparse
import math
import time
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

KP_CENTRO   = 10
KP_FRENTE   = 15
KP_ACTIVOS  = list(range(10, 16))
CONF_MIN    = 0.5
KP_CONF_MIN = 0.4

COLOR_BOX    = (0, 220, 0)
COLOR_KP     = (0, 200, 255)
COLOR_CENTRO = (0, 80, 255)
COLOR_FRENTE = (255, 80, 0)
COLOR_FLECHA = (0, 255, 255)
COLOR_TEXTO  = (255, 255, 255)
COLOR_BG     = (0, 0, 0)


def angulo_desde_kpts(cx, cy, fx, fy) -> float:
    return math.degrees(math.atan2(-(fy - cy), fx - cx)) % 360.0


def dibujar_flecha(frame, cx, cy, angulo_deg, longitud=70):
    rad = math.radians(angulo_deg)
    ex = int(cx + longitud * math.cos(rad))
    ey = int(cy - longitud * math.sin(rad))
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey),
                    COLOR_FLECHA, 3, tipLength=0.35)


def texto_bg(frame, texto, x, y, escala=0.65, grosor=1, color=COLOR_TEXTO):
    (tw, th), bl = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + bl), COLOR_BG, -1)
    cv2.putText(frame, texto, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="v2-nano", choices=list(MODELOS))
    ap.add_argument("--video",  default=r"C:\Users\benja\Downloads\Drone_det.mp4")
    ap.add_argument("--camara", type=int, default=None,
                    help="Indice de camara (ej. 0). Sobreescribe --video.")
    ap.add_argument("--conf",   type=float, default=CONF_MIN)
    ap.add_argument("--pausa",  action="store_true",
                    help="Empieza en pausa (barra espaciadora para continuar).")
    args = ap.parse_args()

    pesos = MODELOS[args.modelo]
    if not pesos.exists():
        print(f"ERROR: no existe {pesos}")
        return

    model = YOLO(str(pesos))

    fuente = args.camara if args.camara is not None else args.video
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la fuente: {fuente}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    es_camara = args.camara is not None

    titulo = f"Drone Pose — {args.modelo}  |  Q/ESC=salir  SPACE=pausa"
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)

    pausado   = args.pausa
    n_frames  = 0
    fps_inf   = 0.0
    t_prev    = time.perf_counter()

    print(f"Modelo: {args.modelo}   |   Fuente: {fuente}")
    print("Controles: Q/ESC = salir  |  SPACE = pausa/continuar  |  -> = avanzar frame")

    while True:
        if not pausado or es_camara:
            ok, frame = cap.read()
            if not ok:
                if not es_camara:
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            n_frames += 1

            t0 = time.perf_counter()
            results = model.predict(frame, device=0, verbose=False, conf=args.conf)[0]
            fps_inf = 0.7 * fps_inf + 0.3 / max(time.perf_counter() - t0, 1e-6)

            if results.boxes is not None and len(results.boxes):
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

                    if results.keypoints is not None and i < len(results.keypoints):
                        kpts = results.keypoints.data.cpu().numpy()[i]

                        for idx in KP_ACTIVOS:
                            kx, ky, kc = kpts[idx]
                            if kc >= KP_CONF_MIN:
                                color = (COLOR_CENTRO if idx == KP_CENTRO else
                                         COLOR_FRENTE if idx == KP_FRENTE else COLOR_KP)
                                cv2.circle(frame, (int(kx), int(ky)), 5, color, -1)

                        cx_k, cy_k, cc = kpts[KP_CENTRO]
                        fx_k, fy_k, fc = kpts[KP_FRENTE]
                        if cc >= KP_CONF_MIN and fc >= KP_CONF_MIN:
                            ang = angulo_desde_kpts(cx_k, cy_k, fx_k, fy_k)
                            dibujar_flecha(frame, cx_k, cy_k, ang)
                            texto_bg(frame, f"{ang:.1f} deg", x1, max(y1 - 8, 18),
                                     color=(0, 255, 255))

            
            texto_bg(frame, f"FPS inf: {fps_inf:.1f}", 8, 22, escala=0.6)
            if not es_camara:
                texto_bg(frame, f"Frame {n_frames}/{total}", 8, 46, escala=0.55)
            if pausado:
                texto_bg(frame, "PAUSADO", frame.shape[1]//2 - 40, 28,
                         escala=0.8, color=(0, 200, 255))

        cv2.imshow(titulo, frame)

        
        delay = max(1, int(1000 / fps_video) - 5) if not es_camara else 1
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   
            break
        elif key == ord(' '):
            pausado = not pausado
        elif key == 83 and pausado:            
            ok, frame = cap.read()
            if ok:
                n_frames += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Cerrado. {n_frames} frames mostrados.")


if __name__ == "__main__":
    main()
