

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).parent
PESOS_DEFAULT = BASE / "runs" / "preliminar_50ep" / "weights" / "best.pt"

KP_CENTRO = 10  
KP_FRENTE = 15  
CONF_KPT_MIN = 0.5    
UMBRAL_SALTO = 60.0   
FRAMES_CONFIRMAR = 5  
ALPHA = 0.35          


class FiltroAngular:
    

    def __init__(self):
        self.vec = None          
        self.candidato = None    
        self.n_candidato = 0

    @property
    def angulo(self):
        if self.vec is None:
            return None
        return math.degrees(math.atan2(self.vec[1], self.vec[0])) % 360.0

    @staticmethod
    def _a_vec(angulo):
        rad = math.radians(angulo)
        return np.array([math.cos(rad), math.sin(rad)])

    @staticmethod
    def _dif(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def update(self, angulo_crudo):
        if angulo_crudo is None:
            return self.angulo  

        if self.vec is None:
            self.vec = self._a_vec(angulo_crudo)
            return self.angulo

        if self._dif(angulo_crudo, self.angulo) > UMBRAL_SALTO:
            
            if (self.candidato is not None
                    and self._dif(angulo_crudo, self.candidato) < UMBRAL_SALTO):
                self.n_candidato += 1
            else:
                self.candidato, self.n_candidato = angulo_crudo, 1
            if self.n_candidato >= FRAMES_CONFIRMAR:
                
                self.vec = self._a_vec(angulo_crudo)
                self.candidato, self.n_candidato = None, 0
            return self.angulo

        self.candidato, self.n_candidato = None, 0
        self.vec = (1 - ALPHA) * self.vec + ALPHA * self._a_vec(angulo_crudo)
        self.vec /= np.linalg.norm(self.vec)
        return self.angulo


def procesar(source, pesos, mostrar=False):
    model = YOLO(str(pesos))

    es_camara = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if es_camara else str(source))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = "camara" if es_camara else Path(source).stem
    out_video = BASE / f"{stem}_orientacion.mp4"
    out_csv = BASE / f"{stem}_orientacion.csv"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    filtro = FiltroAngular()
    filas = []
    n_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, device=0, verbose=False)[0]
        crudo, cc, cf = None, 0.0, 0.0
        centro = frente = None

        if res.keypoints is not None and len(res.keypoints) > 0:
            kpts = res.keypoints.data.cpu().numpy()[0]
            cx, cy, cc = kpts[KP_CENTRO]
            fx, fy, cf = kpts[KP_FRENTE]
            if cc >= CONF_KPT_MIN and cf >= CONF_KPT_MIN:
                crudo = math.degrees(math.atan2(-(fy - cy), fx - cx)) % 360.0
                centro, frente = (int(cx), int(cy)), (int(fx), int(fy))

        filtrado = filtro.update(crudo)

        
        if centro is not None:
            cv2.arrowedLine(frame, centro, frente, (0, 0, 255), 3, tipLength=0.25)
            cv2.circle(frame, centro, 5, (0, 255, 0), -1)
        if filtrado is not None:
            
            origen = centro if centro is not None else (80, h - 80)
            rad = math.radians(filtrado)
            punta = (int(origen[0] + 70 * math.cos(rad)),
                     int(origen[1] - 70 * math.sin(rad)))
            cv2.arrowedLine(frame, origen, punta, (255, 200, 0), 2, tipLength=0.3)
            cv2.putText(frame, f"{filtrado:5.1f} deg", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)
        if crudo is None:
            cv2.putText(frame, "sin medicion", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)
        filas.append([n_frame, round(n_frame / fps, 3),
                      None if crudo is None else round(crudo, 2),
                      None if filtrado is None else round(filtrado, 2),
                      round(float(cc), 3), round(float(cf), 3)])

        if mostrar:
            cv2.imshow("Orientacion dron", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n_frame += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t_seg", "angulo_crudo", "angulo_filtrado",
                     "conf_centro", "conf_frente"])
        wr.writerows(filas)

    medidos = sum(1 for r in filas if r[2] is not None)
    print(f"Frames procesados : {n_frame}")
    print(f"Con medicion      : {medidos} ({100 * medidos / max(n_frame, 1):.1f}%)")
    print(f"Video anotado     : {out_video}")
    print(f"CSV de angulos    : {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Orientacion de dron en video")
    ap.add_argument("--source", required=True,
                    help="ruta a video (mp4/avi) o indice de camara (0, 1, ...)")
    ap.add_argument("--pesos", default=str(PESOS_DEFAULT),
                    help="ruta a los pesos .pt (default: nano preliminar)")
    ap.add_argument("--mostrar", action="store_true",
                    help="mostrar ventana en vivo (q para salir)")
    args = ap.parse_args()
    procesar(args.source, args.pesos, args.mostrar)
