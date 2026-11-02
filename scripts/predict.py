"""Run pitch detection on individual images or a folder; write annotated images and CSV."""
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Image path or directory')
    p.add_argument('--weights', default=str(ROOT / 'outputs/training/weights/best.pt'))
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--imgsz', type=int, default=640)
    a = p.parse_args()
    out = ROOT / 'outputs/predictions'
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    rows = []
    for result in model.predict(source=a.source, conf=a.conf, imgsz=a.imgsz, stream=True,
                                save=True, project=str(ROOT / 'outputs'), name='annotated', exist_ok=True):
        if result.boxes is None:
            continue
        for xyxy, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                     result.boxes.conf.cpu().tolist(),
                                     result.boxes.cls.cpu().tolist()):
            rows.append({'image': Path(result.path).name, 'class': model.names[int(cls)],
                         'confidence': round(score, 5), 'x1': round(xyxy[0], 1),
                         'y1': round(xyxy[1], 1), 'x2': round(xyxy[2], 1),
                         'y2': round(xyxy[3], 1)})
    with (out / 'detections.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image','class','confidence','x1','y1','x2','y2'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} detections. CSV: {out / "detections.csv"}')
    print('Annotated images: outputs/annotated/')

if __name__ == '__main__':
    main()
