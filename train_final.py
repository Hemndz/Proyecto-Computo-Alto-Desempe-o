

from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).parent
DATA_YAML = BASE / "dataset" / "data.yaml"


def main():
    model = YOLO("yolov8s-pose.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        patience=30,
        flipud=0.0,
        fliplr=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        project=str(BASE / "runs"),
        name="final_yolov8s",
        exist_ok=True,
        workers=2,  
    )

    metrics = model.val(data=str(DATA_YAML), device=0)
    print("\n===== METRICAS FINALES (valid) =====")
    print(f"Box  mAP50    : {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95 : {metrics.box.map:.4f}")
    print(f"Box  precision: {metrics.box.mp:.4f}")
    print(f"Box  recall   : {metrics.box.mr:.4f}")
    print(f"Pose mAP50    : {metrics.pose.map50:.4f}")
    print(f"Pose mAP50-95 : {metrics.pose.map:.4f}")
    print(f"Pose precision: {metrics.pose.mp:.4f}")
    print(f"Pose recall   : {metrics.pose.mr:.4f}")


if __name__ == "__main__":
    main()
