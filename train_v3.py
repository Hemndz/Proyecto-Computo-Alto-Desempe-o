

from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).parent
DATA_YAML = BASE / "dataset" / "data.yaml"
PESOS_PREVIOS = BASE / "runs" / "dron_pose_v2_nano" / "weights" / "best.pt"


def main():
    model = YOLO(str(PESOS_PREVIOS))

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=25,
        project=str(BASE / "runs"),
        name="dron_pose_v3",
        exist_ok=True,
        workers=2,
        optimizer="SGD",
        lr0=0.0005,         

        
        flipud=0.0,
        fliplr=0.0,

        
        hsv_h=0.3,          
        hsv_s=0.9,          
        hsv_v=0.5,          
        degrees=30.0,       
        translate=0.15,     
        scale=0.6,          
        erasing=0.3,        

        
        copy_paste=0.5,     
        mosaic=1.0,         
    )

    metrics = model.val(data=str(DATA_YAML), device=0)
    print("\n===== METRICAS FINALES v3 (valid) =====")
    print(f"Box  mAP50    : {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95 : {metrics.box.map:.4f}")
    print(f"Box  precision: {metrics.box.mp:.4f}")
    print(f"Box  recall   : {metrics.box.mr:.4f}")
    print(f"Pose mAP50    : {metrics.pose.map50:.4f}")
    print(f"Pose mAP50-95 : {metrics.pose.map:.4f}")
    print(f"Pose precision: {metrics.pose.mp:.4f}")
    print(f"Pose recall   : {metrics.pose.mr:.4f}")
    print("\n===== METRICAS FINALES v3 =====")


if __name__ == "__main__":
    main()
