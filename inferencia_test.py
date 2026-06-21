
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "final_yolov8s" / "weights" / "best.pt"
TEST_IMAGES = BASE / "dataset" / "test" / "images"
OUT_DIR = BASE / "runs" / "inferencia_final"


KP_CENTRO = 10  
KP_FRENTE = 15  

model = YOLO(str(WEIGHTS))

imagenes = sorted(TEST_IMAGES.glob("*.jpg"))[:5]
if len(imagenes) < 5:
    imagenes = sorted(TEST_IMAGES.iterdir())[:5]

resultados = model.predict(
    [str(p) for p in imagenes],
    device=0,
    save=True,
    project=str(OUT_DIR.parent),
    name=OUT_DIR.name,
    exist_ok=True,
)

print("\n===== ORIENTACION (vector Centro -> Frente) =====")
for img_path, res in zip(imagenes, resultados):
    if res.keypoints is None or len(res.keypoints) == 0:
        print(f"{img_path.name}: sin detecciones")
        continue
    for i, kpts in enumerate(res.keypoints.data.cpu().numpy()):
        cx, cy, conf_c = kpts[KP_CENTRO]
        fx, fy, conf_f = kpts[KP_FRENTE]
        if conf_c < 0.5 or conf_f < 0.5:
            print(f"{img_path.name} det {i}: centro/frente con baja confianza "
                  f"(centro={conf_c:.2f}, frente={conf_f:.2f})")
            continue
        dx = fx - cx
        dy = fy - cy
        angulo = np.degrees(np.arctan2(-dy, dx)) % 360.0
        print(f"{img_path.name} det {i}: angulo = {angulo:6.1f} deg  "
              f"(centro=({cx:.0f},{cy:.0f}), frente=({fx:.0f},{fy:.0f}), "
              f"conf c/f = {conf_c:.2f}/{conf_f:.2f})")

print(f"\nVisualizaciones guardadas en: {OUT_DIR}")
