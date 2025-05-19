# infer_video.py
import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
from collections import Counter

def main(weights, source, output, conf_thresh=0.25):
    # Modeli yükle
    model = YOLO(weights)

    # Video okuyucu ve yazıcı ayarları
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(output, fourcc, fps, (w, h))

    counts = Counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Tahmin yap
        results = model.predict(source=[frame], conf=conf_thresh, verbose=False)
        r = results[0]

        # Her tespit için sayaç artır
        for cls in r.boxes.cls.cpu().numpy().astype(int):
            counts[model.names[cls]] += 1

        # Görseli kutucuklarla çiz ve kaydet
        annotated = r.plot()  # OpenCV formatında numpy array döner
        out.write(annotated)

        # (İsteğe bağlı) her 100 karede bir durum yaz
        if frame_idx % 100 == 0:
            print(f"[{frame_idx}] şu ana kadar sayılar: {dict(counts)}")

    cap.release()
    out.release()

    print("\n=== SONUÇ ===")
    for name, cnt in counts.items():
        print(f"{name}: {cnt} adet tespit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="runs/train/your-run-name/weights/best.pt")
    parser.add_argument("--source", type=str, default="input.mp4",
                        help="Tahmin yapılacak video dosyası")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Kaydedilecek anotasyonlu video")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    args = parser.parse_args()
    main(args.weights, args.source, args.output, conf_thresh=args.conf)
