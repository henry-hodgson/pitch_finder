"""Detect pitches in georeferenced tile images, export WGS84 GeoJSON boxes.
Requires PNG files and matching JSON sidecars from tile_geotiff.py.
"""
import argparse
import json
from pathlib import Path
from affine import Affine
from rasterio.warp import transform as reproj
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tiles', default='data/unlabelled')
    p.add_argument('--weights', default=str(ROOT / 'outputs/training/weights/best.pt'))
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--output', default=str(ROOT / 'outputs/pitches.geojson'))
    args = p.parse_args()
    folder = Path(args.tiles)
    files = sorted(f for f in folder.glob('*.png') if f.with_suffix('.json').exists())
    if not files:
        p.error('No georeferenced PNG/JSON tile pairs found. Run tile_geotiff.py.')
    model = YOLO(args.weights)
    features = []
    for path in files:
        meta = json.loads(path.with_suffix('.json').read_text())
        affine = Affine(*meta['transform'])
        result = model.predict(str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        for xyxy, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = xyxy
            pixels = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
            xs, ys = zip(*(affine * xy for xy in pixels))
            lons, lats = reproj(meta['crs'], 'EPSG:4326', list(xs), list(ys))
            features.append({'type':'Feature', 'geometry':{'type':'Polygon',
                             'coordinates':[[list(pt) for pt in zip(lons,lats)]]},
                             'properties':{'confidence':round(score,5), 'tile':path.name}})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'type':'FeatureCollection','features':features}, indent=2))
    print(f'{len(features)} candidate detections saved to {output}')
    print('Note: overlapping tiles can produce duplicates; review and deduplicate before counting.')

if __name__ == '__main__':
    main()
