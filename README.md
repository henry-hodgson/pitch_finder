# Football Pitch Detector — aerial imagery ML project

Train a custom YOLO object detector to find association-football pitches in aerial imagery. It supports image prediction, CSV exports, RGB GeoTIFF tiling and candidate GeoJSON exports.

## Important: imagery rights
Do **not** bulk-download Google Maps satellite tiles or use Google Maps imagery to train, validate or test ML models unless you have explicit permission. Google Maps Platform terms restrict this. Use imagery licensed for your intended ML use. [OpenAerialMap](https://openaerialmap.org/) provides openly licensed aerial imagery (check metadata and attribution for each source). Alternatively use aerial imagery you own or have permission to use. Do not assume ordinary OpenStreetMap basemap tiles are aerial images or free of usage restrictions.

## 1. Install

Requires Python 3.10+ and an internet connection for package installation and initial model weights. From this project folder:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

GPU helps but isn't required. Ultralytics may have AGPL/commercial licensing requirements; check their terms for deployment.

## 2. Get and annotate images

Use legally obtained aerial images with pitches and without pitches. Start with ~100–300 images/chips covering different locations, zoom levels, surfaces, and lighting. Keep geographically separated train and validation areas to avoid overly optimistic scores.

Use a bounding-box annotation tool that exports **YOLO detection** labels (e.g. CVAT or Label Studio). Define exactly one class: `football_pitch`, index `0`.

Data format:
```
data/
  images/train/example1.png
  images/val/example2.png
  labels/train/example1.txt
  labels/val/example2.txt
```
Each label file contains one line per pitch:
```
0 0.50 0.50 0.30 0.18
```
The columns are `class_id x_center y_center width height`, all normalized to 0..1. An image without any pitches should have an empty matching `.txt` file. Label the full pitch playing area, not surrounding sports ground or stands. For images with partial pitches, choose a consistent annotation policy.

**Important:** This project does not include a trained football-pitch model or any real imagery. A generic pretrained YOLO checkpoint is only the starting point and is not a reliable pitch detector until fine-tuned on labelled aerial imagery.

## 3. Train

```bash
python scripts/train.py --epochs 50 --imgsz 640 --batch 8
```
Weights appear at `outputs/training/weights/best.pt`. Compare validation precision, recall and mAP metrics and manually inspect false positives (parks, tennis courts, fields, car parks).

## 4. Predict on normal aerial images

```bash
python scripts/predict.py --source path/to/aerial_images --weights outputs/training/weights/best.pt
```
Creates `outputs/annotated/` and `outputs/predictions/detections.csv`. The CSV coordinates are image pixels, not latitude/longitude.

## 5. Optional: search a georeferenced aerial GeoTIFF

Download a permitted high-resolution *RGB uint8* GeoTIFF from an imagery provider. Aerial imagery with sufficient spatial resolution to see each pitch is needed.

```bash
python scripts/tile_geotiff.py --input path/to/licensed_ortho.tif --output data/unlabelled --size 640 --stride 512
python scripts/predict_geo.py --tiles data/unlabelled --weights outputs/training/weights/best.pt --output outputs/pitches.geojson
```
`pitches.geojson` contains bounding-box polygons in WGS84 coordinates, loadable in QGIS or a map viewer. Because of overlapping tiles and model false positives, these are **candidate** locations: deduplicate and review them. This prototype does not promise high accuracy or exhaustive city-scale detection.

## Good research questions

- How do grass, 3G and five-a-side pitches affect recall?
- Does training across cities generalize better than training in one area?
- Does adding hard negatives (parks, rugby pitches, tennis courts) improve precision?
- How does detection quality vary by image resolution?

## Possible improvements

- Add automated spatial deduplication (NMS or geospatial clustering).
- Annotate pitch **segmentation masks** for precise boundaries, instead of boxes.
- Add a lightweight Streamlit review interface and map.
- Compare predictions against OpenStreetMap `leisure=pitch` / `sport=soccer` features, while being careful about OSM data licensing and incomplete ground truth.

## Project reference links

- https://docs.ultralytics.com/usage/python/
- https://openaerialmap.org/
- https://docs.openaerialmap.org/browser/user-guide/
- https://cloud.google.com/maps-platform/terms/

## London-specific pilot

**Scope:** Greater London, starting with Hackney Marshes, Regent's Park and Wimbledon Common. The `london.json` file records a broad London search rectangle (lat 51.28–51.70, lon -0.52–0.34); this is **not** the exact Greater London administrative boundary.

1. Find high-resolution aerial orthophotos that explicitly permit machine-learning training and derivative outputs. Check each dataset's individual licence; do not assume a London Datastore entry or an aerial-view service grants these rights. For possible sources, inspect [OpenAerialMap](https://openaerialmap.org/) or discuss permission with a data provider. [Aerial Digimap](https://digimap.edina.ac.uk/aerial) has GB photography but access and permitted uses depend on its licence.
2. Start with **Hackney Marshes**: manually obtain permitted georeferenced RGB GeoTIFF imagery; run `python scripts/tile_geotiff.py --input path/to/hackney.tif --output data/unlabelled/hackney --size 640 --stride 512`. Tiles are for labelling or inference, not automatically labelled examples.
3. Label about 100–300 diverse chips with one pitch bounding box per pitch (including empty chips and hard negatives). Use separate areas for training and validation to avoid leakage. Put images and YOLO labels in `data/images/{train,val}` and `data/labels/{train,val}`.
4. Train: `python scripts/train.py --epochs 50 --imgsz 640 --batch 8`. This requires labelled images; the package does **not** include them.
5. Detect on any permitted georeferenced image: `python scripts/tile_geotiff.py --input path/to/london.tif --output data/unlabelled --size 640 --stride 512` then `python scripts/predict_geo.py --tiles data/unlabelled --weights outputs/training/weights/best.pt --output outputs/london_candidates.geojson`.
6. Optionally query existing OpenStreetMap pitch *locations* as independent references: `python scripts/london_osm_reference.py`. It writes `data/london_osm_pitches.geojson` using the public Overpass API; this step requires internet access and may be rate-limited. OSM locations are **not image ground truth** and not proof that a pitch exists today; comply with ODbL and attribute OpenStreetMap contributors.

**Milestone:** Evaluate on a held-out London area, review candidate GeoJSON in QGIS, then enlarge coverage borough by borough. Adjacent inference tiles can detect the same pitch twice, so deduplicate before counting. Also filter predictions against an authoritative Greater London boundary before claiming results are London-only.
