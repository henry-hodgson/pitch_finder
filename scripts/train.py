"""Train a YOLO detector from labelled aerial imagery."""
from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--model', default='yolo26n.pt', help='Pretrained YOLO detection model')
    args = parser.parse_args()
    for split in ('train', 'val'):
        images = list((ROOT / 'data/images' / split).glob('*.*'))
        if not images:
            parser.error(f'No images in data/images/{split}. See README for labelling instructions.')
    model = YOLO(args.model)
    model.train(data=str(ROOT / 'dataset.yaml'), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, project=str(ROOT / 'outputs'),
                name='training', exist_ok=True)
    print('Trained weights:', ROOT / 'outputs/training/weights/best.pt')

if __name__ == '__main__':
    main()
