"""
Convert Mountain Fire KML perimeter → GeoJSON.
Uses only standard library (xml + json) — no geopandas needed.
Output file can be loaded with geopandas.read_file(), viewed in QGIS,
or uploaded to Derecho alongside the plotting scripts.
"""
import xml.etree.ElementTree as ET
import json
from pathlib import Path

kml_path = Path(r"/home/voyager-sbarc/duine/sundowners/mountain_fire/perimeter/CA-VNC-MOUNTAIN-N4OY_110620241928Z_PERIM_221257090_10431_acres (1).kml")
out_path = kml_path.with_name("mountain_fire_perimeter.geojson")

ns = {'kml': 'http://www.opengis.net/kml/2.2'}
tree = ET.parse(kml_path)
root = tree.getroot()

features = []
for pm in root.findall('.//kml:Placemark', ns):
    coords_el = pm.find('.//kml:coordinates', ns)
    if coords_el is None:
        continue
    pts = []
    for triple in coords_el.text.strip().split():
        parts = triple.split(',')
        pts.append([float(parts[0]), float(parts[1])])
    if len(pts) >= 3:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [pts]},
            "properties": {}
        })

geojson = {"type": "FeatureCollection", "features": features}

with open(out_path, 'w') as f:
    json.dump(geojson, f, indent=2)

print(f"Saved {len(features)} polygons → {out_path}")
