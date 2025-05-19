import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# train.py
from ultralytics import YOLO

def main():
    # 1) Önceden eğitilmiş ağı yükle
    model = YOLO("yolov8s.pt")

    # 2) Eğitim parametrelerini burada ver
    model.train(
        data="data.yaml",     # dataset yapılandırma dosyan
        epochs=20,            # kaç epoch çalışsın
        imgsz=640,            # girdi boyutu
        batch=8,             # batch size
        device="cuda",
        project="runs/train", # sonuçların kaydedileceği klasör
        name="final"       # alt klasör ismi
    )

if __name__ == "__main__":
    main()
