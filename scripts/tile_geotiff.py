"""Split a licensed georeferenced RGB GeoTIFF into model-ready image chips.

Preserves tile georeferencing in a JSON metadata sidecar for later geo-export.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window, transform as window_transform


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Licensed RGB GeoTIFF')
    p.add_argument('--output', default='data/unlabelled')
    p.add_argument('--size', type=int, default=640)
    p.add_argument('--stride', type=int, default=512)
    args = p.parse_args()
    if args.stride < 1 or args.size < 1 or args.stride > args.size:
        p.error('Require 1 <= stride <= size')
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with rasterio.open(args.input) as src:
        if src.count < 3:
            p.error('Expected GeoTIFF with at least three RGB bands')
        if not src.crs:
            p.error('GeoTIFF must have a coordinate reference system')
        stem = Path(args.input).stem
        for top in range(0, max(src.height - args.size + 1, 1), args.stride):
            for left in range(0, max(src.width - args.size + 1, 1), args.stride):
                if left + args.size > src.width or top + args.size > src.height:
                    continue
                window = Window(left, top, args.size, args.size)
                rgb = src.read([1, 2, 3], window=window)
                if rgb.dtype != np.uint8:
                    raise ValueError('RGB bands must be uint8; pre-process 16-bit imagery first')
                rgb = np.moveaxis(rgb, 0, -1)
                filename = f'{stem}_{top}_{left}.png'
                Image.fromarray(rgb).save(dst / filename)
                tr = window_transform(window, src.transform)
                (dst / f'{Path(filename).stem}.json').write_text(json.dumps({
                    'source': str(Path(args.input).resolve()), 'crs': src.crs.to_string(),
                    'transform': list(tr)[:6], 'width': args.size, 'height': args.size,
                    'top': top, 'left': left
                }, indent=2))
                n += 1
    print(f'Created {n} image chips and georeferencing JSON files in {dst}')

if __name__ == '__main__':
    main()
