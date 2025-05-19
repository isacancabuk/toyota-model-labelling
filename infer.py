# infer.py
from ultralytics import YOLO
from pathlib import Path

def infer(
    model_path: str = "runs/train/4runner-corolla4/weights/best.pt",
    source_dir: str = "images/test",
    save_dir: str = "runs/infer",
    run_name: str = "4runner-corolla-infer",
    conf_thresh: float = 0.7  # <<< eşiği %70’e çekiyoruz
):
    model = YOLO(model_path)
    results = model.predict(
        source=source_dir,
        save=True,
        project=save_dir,
        name=run_name,
        conf=conf_thresh,      # <<< burada
        iou=0.5                # istenirse NMS IOU eşiğini de ayarla
    )

    for r in results:
        img_name = Path(r.path).name
        boxes   = r.boxes.xyxy.cpu().numpy()
        scores  = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Sadece güveni eşiğin üstünde olanları yazdır
        filtered = [(cls,box,score) for cls,box,score in zip(classes,boxes,scores) if score >= conf_thresh]
        print(f"{img_name}: {len(filtered)} adet tespit (conf ≥ {conf_thresh})")
        for cls, box, score in filtered:
            print(f"  - Sınıf: {cls}, Kutu: {box.tolist()}, Güven: {score:.2f}")

if __name__ == "__main__":
    infer()
