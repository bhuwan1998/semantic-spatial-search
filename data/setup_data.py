"""
Data Setup Script - Download Adelaide OSM data and build a GeoPackage.

Uses osmnx to fetch data from the Overpass API and writes multiple layers
(points, lines, polygons, multipolygons) into a single GeoPackage file.

Usage:
    python data/setup_data.py
"""

import os
import sys
import warnings
import geopandas as gpd
import osmnx as ox

warnings.filterwarnings("ignore", category=FutureWarning)

# Output path
GPKG_PATH = os.path.join(os.path.dirname(__file__), "adelaide_osm.gpkg")
PLACE = "Adelaide, South Australia, Australia"

# Layer definitions: (layer_name, osm_tags, description)
LAYER_DEFS = [
    # --- POINTS ---
    (
        "schools",
        {"amenity": "school"},
        "Schools (Point/Polygon)",
    ),
    (
        "hospitals",
        {"amenity": "hospital"},
        "Hospitals (Point/Polygon)",
    ),
    (
        "restaurants",
        {"amenity": ["restaurant", "cafe", "fast_food", "pub", "bar"]},
        "Restaurants, cafes, pubs (Point/Polygon)",
    ),
    (
        "pharmacies",
        {"amenity": "pharmacy"},
        "Pharmacies (Point/Polygon)",
    ),
    # --- LINESTRINGS ---
    (
        "roads",
        {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]},
        "Road network (LineString)",
    ),
    (
        "waterways",
        {"waterway": ["river", "stream", "canal"]},
        "Rivers, streams, canals (LineString)",
    ),
    (
        "railways",
        {"railway": "rail"},
        "Railway lines (LineString)",
    ),
    # --- POLYGONS ---
    (
        "parks",
        {"leisure": ["park", "garden", "nature_reserve"]},
        "Parks and gardens (Polygon)",
    ),
    (
        "buildings",
        {"building": True},
        "Buildings (Polygon) - sampled",
    ),
    (
        "landuse",
        {"landuse": True},
        "Land use zones (Polygon)",
    ),
    (
        "natural",
        {"natural": ["water", "wood", "scrub", "wetland", "grassland"]},
        "Natural features - water bodies, woods (Polygon)",
    ),
    # --- MULTIPOLYGONS ---
    (
        "boundaries",
        {"boundary": "administrative", "admin_level": ["6", "8"]},
        "Administrative boundaries - LGA and suburb level (MultiPolygon)",
    ),
]


def download_layer(layer_name: str, tags: dict, description: str) -> gpd.GeoDataFrame | None:
    """Download a single OSM layer for Adelaide."""
    print(f"  Downloading {layer_name}: {description}...")
    try:
        gdf = ox.features_from_place(PLACE, tags=tags)
        if gdf.empty:
            print(f"    -> No features found for {layer_name}, skipping.")
            return None

        # Keep only useful columns + geometry
        keep_cols = ["geometry"]
        for col in ["name", "amenity", "highway", "waterway", "railway",
                     "leisure", "building", "landuse", "natural", "boundary",
                     "admin_level", "cuisine", "opening_hours", "addr:street",
                     "addr:housenumber", "operator", "surface", "lanes",
                     "maxspeed", "oneway"]:
            if col in gdf.columns:
                keep_cols.append(col)

        gdf = gdf[keep_cols].copy()

        # For buildings, take a sample to keep file size reasonable
        if layer_name == "buildings" and len(gdf) > 5000:
            gdf = gdf.sample(n=5000, random_state=42)
            print(f"    -> Sampled 5000 buildings from {len(gdf)} total.")

        # Ensure CRS is WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        # Reset index to avoid MultiIndex issues from osmnx
        gdf = gdf.reset_index(drop=True)

        print(f"    -> {len(gdf)} features, geometry types: {gdf.geom_type.unique().tolist()}")
        return gdf

    except Exception as e:
        print(f"    -> ERROR: {e}")
        return None


def build_geopackage():
    """Download all layers and write to a single GeoPackage."""
    print(f"Building GeoPackage: {GPKG_PATH}")
    print(f"Place: {PLACE}")
    print(f"Layers to download: {len(LAYER_DEFS)}")
    print("=" * 60)

    # Remove existing file
    if os.path.exists(GPKG_PATH):
        os.remove(GPKG_PATH)
        print(f"Removed existing {GPKG_PATH}")

    successful = 0
    for layer_name, tags, description in LAYER_DEFS:
        gdf = download_layer(layer_name, tags, description)
        if gdf is not None and not gdf.empty:
            try:
                gdf.to_file(GPKG_PATH, layer=layer_name, driver="GPKG")
                successful += 1
                print(f"    -> Written to layer '{layer_name}'")
            except Exception as e:
                print(f"    -> WRITE ERROR for '{layer_name}': {e}")

    print("=" * 60)
    print(f"Done! {successful}/{len(LAYER_DEFS)} layers written to {GPKG_PATH}")

    # Print file size
    if os.path.exists(GPKG_PATH):
        size_mb = os.path.getsize(GPKG_PATH) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

    # Print layer summary
    print("\nLayer summary:")
    import pyogrio
    for layer_name in pyogrio.list_layers(GPKG_PATH)[:, 0]:
        gdf = gpd.read_file(GPKG_PATH, layer=layer_name)
        geom_types = gdf.geom_type.unique().tolist()
        print(f"  {layer_name:15s} -> {len(gdf):6d} features  {geom_types}")


if __name__ == "__main__":
    build_geopackage()
