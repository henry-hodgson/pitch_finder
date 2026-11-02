"""Download mapped London soccer pitches from OpenStreetMap as review references.

These are NOT verified training labels. Data © OpenStreetMap contributors (ODbL).
Requires internet access; public Overpass servers may rate limit requests.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
# Broad rectangle covering Greater London; not an administrative boundary.
SOUTH, WEST, NORTH, EAST = 51.28, -0.52, 51.70, 0.34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(ROOT / 'data/london_osm_pitches.geojson'))
    ap.add_argument('--endpoint', default='https://overpass.kumi.systems/api/interpreter')
    args = ap.parse_args()
    bbox = f'{SOUTH},{WEST},{NORTH},{EAST}'
    query = f'''[out:json][timeout:180];
    (nwr["leisure"="pitch"]["sport"="soccer"]({bbox}););
    out center;'''
    req = Request(args.endpoint, data=urlencode({'data': query}).encode(),
                  headers={'User-Agent': 'london-football-pitch-detector-research/0.1'})
    with urlopen(req, timeout=210) as response:
        elems = json.load(response)['elements']
    features = []
    for e in elems:
        center = e.get('center', e)
        if 'lon' not in center or 'lat' not in center:
            continue
        features.append({'type':'Feature',
                         'geometry':{'type':'Point','coordinates':[center['lon'],center['lat']]},
                         'properties':{'osm_type':e['type'],'osm_id':e['id'],
                                       'name':e.get('tags',{}).get('name',''),
                                       'source':'OpenStreetMap (ODbL); unverified reference'}})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'type':'FeatureCollection', 'features':features}, indent=2))
    print(f'Exported {len(features)} OSM reference points to {target}')
    print('These are not verified bounding-box labels or a complete list of pitches.')

if __name__ == '__main__':
    main()
