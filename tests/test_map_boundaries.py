"""
Tests for Survey of India (SOI) boundary compliance and geospatial data integrity.
"""
from pathlib import Path
import json


def test_india_soi_geojson_exists_and_valid():
    soi_path = Path("data/geojson/india_soi_simplified.geojson")
    assert soi_path.exists(), "Official SOI GeoJSON file missing!"

    with open(soi_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("type") == "FeatureCollection"
    assert len(data.get("features", [])) > 0

    # Extract coordinate bounds
    lons, lats = [], []
    for feat in data["features"]:
        geom = feat["geometry"]

        def extract(coords):
            if isinstance(coords[0], (int, float)):
                lons.append(coords[0])
                lats.append(coords[1])
            else:
                for sub in coords:
                    extract(sub)

        extract(geom["coordinates"])

    # Verify official Survey of India territorial coverage
    # Northernmost: Indira Col, Ladakh/J&K (~37.1N)
    assert max(lats) >= 37.0, f"Northern border truncated! Max Lat: {max(lats)}"
    # Southernmost: Indira Point, Great Nicobar (~6.8N)
    assert min(lats) <= 7.0, f"Southern islands omitted! Min Lat: {min(lats)}"
    # Westernmost: Guhar Moti, Gujarat (~68.2E)
    assert min(lons) <= 68.5, f"Western border incorrect! Min Lon: {min(lons)}"
    # Easternmost: Kibithu, Arunachal Pradesh (~97.4E)
    assert max(lons) >= 97.0, f"Eastern border incorrect! Max Lon: {max(lons)}"


def test_india_states_soi_geojson_coverage():
    states_path = Path("data/geojson/india_states_soi.geojson")
    assert states_path.exists(), "SOI States GeoJSON missing!"

    with open(states_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("type") == "FeatureCollection"
    assert len(data.get("features", [])) >= 35

    state_names = {
        f["properties"].get("st_nm", "").lower()
        for f in data["features"]
    }

    # Critical sovereign territories check
    assert any("jammu" in s or "kashmir" in s for s in state_names), "Jammu and Kashmir missing from states!"
    assert any("ladakh" in s for s in state_names), "Ladakh missing from states!"
    assert any("arunachal" in s or "arunanchal" in s for s in state_names), "Arunachal Pradesh missing from states!"

    # Coastal cyclone risk zones check
    coastal_states = [f for f in data["features"] if f["properties"].get("is_coastal")]
    assert len(coastal_states) >= 9, "Key coastal states not identified for cyclone landfall analysis!"
