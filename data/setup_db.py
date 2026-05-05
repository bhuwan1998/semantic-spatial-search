"""
Data Setup Script - Download Adelaide OSM data and load into PostgreSQL + PostGIS.

Phases:
  1. Verify extensions (PostGIS, pgvector, AGE) — created by docker/initdb.sql
  2. Download 12 OSM layers with osmnx and write to PostGIS tables
  3. Embed table descriptions with nomic-embed-text and store in pgvector
  4. Build the Apache AGE property graph (nodes + NEAR/WITHIN/CONNECTED_TO edges)

Usage:
    python data/setup_db.py
    python data/setup_db.py --skip-download   # re-embed + rebuild graph only
    python data/setup_db.py --skip-graph       # skip AGE graph build
"""

import argparse
import os
import sys
import warnings

import geopandas as gpd
import ollama
import osmnx as ox
import psycopg

warnings.filterwarnings("ignore", category=FutureWarning)

# osmnx graph simplification on large road/boundary layers can hit Python's
# default recursion limit of 1000; raise it to handle Adelaide-scale datasets.
sys.setrecursionlimit(10000)

# ─────────────────────────── constants ──────────────────────────────────────

PLACE = "Adelaide, South Australia, Australia"

# (table_name, osm_tags, natural-language description for RAG embeddings)
LAYER_DEFS = [
    (
        "osm_schools",
        {"amenity": "school"},
        "Schools and educational institutions in Adelaide. "
        "Contains point and polygon geometries for primary schools, high schools, "
        "colleges. Key columns: name, amenity, addr:street.",
    ),
    (
        "osm_hospitals",
        {"amenity": "hospital"},
        "Hospitals and medical centres in Adelaide. "
        "Contains point and polygon geometries. Key columns: name, amenity, operator.",
    ),
    (
        "osm_restaurants",
        {"amenity": ["restaurant", "cafe", "fast_food", "pub", "bar"]},
        "Restaurants, cafes, pubs, bars and fast food outlets in Adelaide. "
        "Key columns: name, amenity, cuisine, addr:street.",
    ),
    (
        "osm_pharmacies",
        {"amenity": "pharmacy"},
        "Pharmacies and chemists in Adelaide. Key columns: name, amenity, operator.",
    ),
    (
        "osm_roads",
        {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]},
        "Road network in Adelaide including motorways, primary, secondary and residential roads. "
        "LineString geometries. Key columns: name, highway, surface, lanes, maxspeed.",
    ),
    (
        "osm_waterways",
        {"waterway": ["river", "stream", "canal"]},
        "Rivers, streams and canals in Adelaide. LineString geometries. "
        "Key columns: name, waterway. Includes River Torrens.",
    ),
    (
        "osm_railways",
        {"railway": "rail"},
        "Railway lines in Adelaide. LineString geometries. Key columns: name, railway.",
    ),
    (
        "osm_parks",
        {"leisure": ["park", "garden", "nature_reserve"]},
        "Parks, gardens and nature reserves in Adelaide. Polygon geometries. "
        "Key columns: name, leisure.",
    ),
    (
        "osm_buildings",
        {"building": True},
        "Buildings in Adelaide (sampled). Polygon geometries. "
        "Key columns: name, building, addr:street.",
    ),
    (
        "osm_landuse",
        {"landuse": True},
        "Land use zones in Adelaide such as residential, commercial, industrial, "
        "retail, recreation. Polygon geometries. Key columns: name, landuse.",
    ),
    (
        "osm_natural",
        {"natural": ["water", "wood", "scrub", "wetland", "grassland"]},
        "Natural features in Adelaide: water bodies, woods, scrubland, wetlands. "
        "Polygon geometries. Key columns: name, natural.",
    ),
    (
        "osm_boundaries",
        {"boundary": "administrative", "admin_level": ["6", "8"]},
        "Administrative boundaries for Adelaide: LGA and suburb level. "
        "MultiPolygon geometries. Key columns: name, boundary, admin_level. "
        "Used for suburb-level spatial joins.",
    ),
]

# Columns to keep per layer
KEEP_COLS_BASE = [
    "geometry", "name", "amenity", "highway", "waterway", "railway",
    "leisure", "building", "landuse", "natural", "boundary", "admin_level",
    "cuisine", "opening_hours", "addr:street", "addr:housenumber",
    "operator", "surface", "lanes", "maxspeed", "oneway",
]

# Column → PostGIS table column mapping
TABLE_COLUMNS = {
    "osm_schools":     ["name", "amenity", "addr:street", "addr:housenumber", "operator", "opening_hours"],
    "osm_hospitals":   ["name", "amenity", "addr:street", "addr:housenumber", "operator", "opening_hours"],
    "osm_restaurants": ["name", "amenity", "cuisine", "addr:street", "addr:housenumber", "opening_hours"],
    "osm_pharmacies":  ["name", "amenity", "addr:street", "addr:housenumber", "operator", "opening_hours"],
    "osm_roads":       ["name", "highway", "surface", "lanes", "maxspeed", "oneway"],
    "osm_waterways":   ["name", "waterway"],
    "osm_railways":    ["name", "railway"],
    "osm_parks":       ["name", "leisure"],
    "osm_buildings":   ["name", "building", "addr:street", "addr:housenumber"],
    "osm_landuse":     ["name", "landuse"],
    "osm_natural":     ["name", "natural"],
    "osm_boundaries":  ["name", "boundary", "admin_level"],
}

# Node label per table for AGE graph
AGE_LABELS = {
    "osm_schools":     "School",
    "osm_hospitals":   "Hospital",
    "osm_restaurants": "Restaurant",
    "osm_pharmacies":  "Pharmacy",
    "osm_roads":       "Road",
    "osm_waterways":   "Waterway",
    "osm_railways":    "Railway",
    "osm_parks":       "Park",
    "osm_buildings":   "Building",
    "osm_landuse":     "Landuse",
    "osm_natural":     "NaturalFeature",
    "osm_boundaries":  "Boundary",
}

NEAR_DISTANCE_M = 500   # metres for NEAR edges
AGE_GRAPH_NAME  = "osm_spatial"

# ─────────────────────────── helpers ────────────────────────────────────────

def _dsn() -> str:
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    dbname   = os.getenv("DB_NAME", "geospatial")
    user     = os.getenv("DB_USER", "geo")
    password = os.getenv("DB_PASSWORD", "geo")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(_dsn())
    conn.autocommit = True
    return conn


def _safe_val(gdf: gpd.GeoDataFrame, col: str, idx: int):
    """Return scalar value or None, handling missing columns gracefully."""
    if col not in gdf.columns:
        return None
    v = gdf.iloc[idx][col]
    # pandas NA / numpy nan → None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v) if v is not None else None


# ─────────────────────────── phase 1: verify extensions ─────────────────────

def verify_extensions(conn: psycopg.Connection) -> None:
    print("Phase 1: Verifying database extensions...")
    with conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('postgis','vector','age')")
        found = {r[0] for r in cur.fetchall()}
    for ext in ("postgis", "vector", "age"):
        status = "OK" if ext in found else "MISSING"
        print(f"  {ext}: {status}")
    if "postgis" not in found:
        raise RuntimeError("PostGIS extension not found. Check docker/initdb.sql ran correctly.")
    print("  Extensions OK.\n")


# ─────────────────────────── phase 2: download & load OSM data ──────────────

def download_layer(layer_name: str, tags: dict, description: str) -> gpd.GeoDataFrame | None:
    print(f"  Downloading {layer_name}...")
    try:
        gdf = ox.features_from_place(PLACE, tags=tags)
        if gdf.empty:
            print(f"    -> No features found, skipping.")
            return None

        # osmnx returns a GeoDataFrame with MultiIndex (element_type, osmid).
        # Flatten to a plain index and keep only the columns we need.
        gdf = gdf.reset_index(drop=True)
        keep = ["geometry"] + [c for c in KEEP_COLS_BASE if c in gdf.columns and c != "geometry"]
        gdf = gdf[keep].copy()
        # Re-pin geometry column (geopandas 1.x can lose the active geometry
        # column reference after column filtering, causing infinite recursion
        # in .crs via the geometry property).
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf._geometry_column_name not in gdf.columns:
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

        if layer_name == "osm_buildings" and len(gdf) > 5000:
            gdf = gdf.sample(n=5000, random_state=42)

        crs = gdf.geometry.values.crs
        if crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        gdf = gdf.reset_index(drop=True)
        print(f"    -> {len(gdf)} features")
        return gdf
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    -> ERROR: {e}")
        return None


def load_layer(conn: psycopg.Connection, table_name: str, gdf: gpd.GeoDataFrame) -> int:
    """Insert a GeoDataFrame into the corresponding PostGIS table."""
    attr_cols = TABLE_COLUMNS[table_name]
    loaded = 0

    # Truncate existing data
    with conn.cursor() as cur:
        cur.execute(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE')

    for i in range(len(gdf)):
        row = gdf.iloc[i]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        vals = {col: _safe_val(gdf, col, i) for col in attr_cols}

        # Build dynamic INSERT
        col_list = ", ".join(f'"{c}"' for c in attr_cols) + ", geometry"
        placeholder_list = ", ".join(["%s"] * len(attr_cols)) + ", ST_SetSRID(ST_GeomFromText(%s), 4326)"
        sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholder_list})'

        params = [vals[c] for c in attr_cols] + [geom.wkt]

        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            loaded += 1
        except psycopg.Error:
            conn.rollback()

    return loaded


def run_data_load(conn: psycopg.Connection) -> None:
    print("Phase 2: Downloading and loading OSM data...")
    for table_name, tags, description in LAYER_DEFS:
        gdf = download_layer(table_name, tags, description)
        if gdf is not None and not gdf.empty:
            n = load_layer(conn, table_name, gdf)
            print(f"    -> {n} rows inserted into {table_name}")
        else:
            print(f"    -> Skipped {table_name}")
    print("  Data load complete.\n")


# ─────────────────────────── phase 3: embed table descriptions ──────────────

def embed_text(text: str, model: str) -> list[float]:
    """Call Ollama embedding API and return a vector."""
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def run_embedding_index(conn: psycopg.Connection) -> None:
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    print(f"Phase 3: Building pgvector schema embeddings (model: {embedding_model})...")

    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE table_descriptions RESTART IDENTITY")

    for table_name, _tags, description in LAYER_DEFS:
        try:
            vec = embed_text(description, embedding_model)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO table_descriptions (table_name, description, embedding) "
                    "VALUES (%s, %s, %s::vector) "
                    "ON CONFLICT (table_name) DO UPDATE SET description = EXCLUDED.description, embedding = EXCLUDED.embedding",
                    (table_name, description, vec),
                )
            print(f"  Embedded {table_name}")
        except Exception as e:
            print(f"  WARNING: Could not embed {table_name}: {e}")

    print("  Embedding index complete.\n")


# ─────────────────────────── phase 4: build AGE graph ───────────────────────

def _age_exec(conn: psycopg.Connection, cypher: str, graph: str = AGE_GRAPH_NAME) -> None:
    """Execute a Cypher statement via AGE's cypher() SQL function."""
    sql = f"SELECT * FROM cypher('{graph}', $$ {cypher} $$) AS (v agtype);"
    with conn.cursor() as cur:
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        cur.execute(sql)


def run_graph_build(conn: psycopg.Connection) -> None:
    print("Phase 4: Building Apache AGE property graph...")

    with conn.cursor() as cur:
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")

        # Drop and recreate graph
        try:
            cur.execute(f"SELECT drop_graph('{AGE_GRAPH_NAME}', true);")
        except psycopg.Error:
            conn.rollback()
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")

        cur.execute(f"SELECT create_graph('{AGE_GRAPH_NAME}');")

    print("  Graph created.")

    # Create nodes for each layer
    total_nodes = 0
    for table_name, _tags, _desc in LAYER_DEFS:
        label = AGE_LABELS[table_name]
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            # Get features with suburb via spatial join on boundaries
            cur.execute(f"""
                SELECT f.id, f.osm_id, f.name,
                       ST_AsText(ST_Centroid(f.geometry)) AS centroid_wkt,
                       b.name AS suburb
                FROM "{table_name}" f
                LEFT JOIN osm_boundaries b
                  ON ST_Within(ST_Centroid(f.geometry), b.geometry)
                  AND b.admin_level IN ('8', '10')
                WHERE f.geometry IS NOT NULL
                LIMIT 5000
            """)
            rows = cur.fetchall()

        for row in rows:
            fid      = row[0]
            osm_id   = row[1] or 0
            name     = (row[2] or "").replace("'", "\\'")
            suburb   = (row[4] or "").replace("'", "\\'")
            layer    = table_name.replace("osm_", "")
            try:
                with conn.cursor() as cur:
                    cur.execute("LOAD 'age';")
                    cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                    sql = (
                        f"SELECT * FROM cypher('{AGE_GRAPH_NAME}', $$ "
                        f"CREATE (:{label} {{fid: {fid}, osm_id: {osm_id}, "
                        f"name: '{name}', layer: '{layer}', suburb: '{suburb}'}}) "
                        f"$$) AS (v agtype);"
                    )
                    cur.execute(sql)
                total_nodes += 1
            except psycopg.Error:
                conn.rollback()

        print(f"  Nodes created for {table_name} (label: {label})")

    print(f"  Total nodes: {total_nodes}")

    # WITHIN edges: all feature layers inside suburb-level admin boundaries
    print("  Building WITHIN edges...")
    within_layers = [
        "osm_schools", "osm_hospitals", "osm_restaurants", "osm_parks",
        "osm_pharmacies", "osm_waterways", "osm_railways",
        "osm_buildings", "osm_landuse", "osm_natural",
    ]
    total_within = 0
    for table_name in within_layers:
        label = AGE_LABELS[table_name]
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT f.id AS fid, b.name AS boundary_name
                    FROM "{table_name}" f
                    JOIN osm_boundaries b
                      ON ST_Within(ST_Centroid(f.geometry), b.geometry)
                      AND b.admin_level = '8'
                    WHERE f.geometry IS NOT NULL
                    LIMIT 2000
                """)
                pairs = cur.fetchall()

            for pair in pairs:
                fid   = pair[0]
                bname = (pair[1] or "").replace("'", "\\'")
                try:
                    with conn.cursor() as cur:
                        cur.execute("LOAD 'age';")
                        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                        sql = (
                            f"SELECT * FROM cypher('{AGE_GRAPH_NAME}', $$ "
                            f"MATCH (a:{label} {{fid: {fid}}}), (b:Boundary {{name: '{bname}'}}) "
                            f"CREATE (a)-[:WITHIN {{boundary_name: '{bname}'}}]->(b) "
                            f"$$) AS (v agtype);"
                        )
                        cur.execute(sql)
                    total_within += 1
                except psycopg.Error:
                    conn.rollback()
        except psycopg.Error as e:
            print(f"    WARNING: WITHIN edges for {table_name}: {e}")
            conn.rollback()

    print(f"  WITHIN edges created: {total_within}")

    # NEAR edges: cross-layer pairs within NEAR_DISTANCE_M metres
    print("  Building NEAR edges...")
    near_pairs = [
        ("osm_schools",     "osm_parks",       "School",     "Park"),
        ("osm_schools",     "osm_hospitals",   "School",     "Hospital"),
        ("osm_restaurants", "osm_parks",       "Restaurant", "Park"),
        ("osm_hospitals",   "osm_pharmacies",  "Hospital",   "Pharmacy"),
        ("osm_pharmacies",  "osm_hospitals",   "Pharmacy",   "Hospital"),
        ("osm_restaurants", "osm_restaurants", "Restaurant", "Restaurant"),
        ("osm_schools",     "osm_restaurants", "School",     "Restaurant"),
        ("osm_parks",       "osm_waterways",   "Park",       "Waterway"),
    ]
    total_near = 0
    for ta, tb, la, lb in near_pairs:
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT a.id AS aid, b.id AS bid,
                           ROUND(ST_Distance(a.geometry::geography, b.geometry::geography)) AS dist_m
                    FROM "{ta}" a
                    JOIN "{tb}" b
                      ON ST_DWithin(a.geometry::geography, b.geometry::geography, %s)
                    LIMIT 5000
                """, (NEAR_DISTANCE_M,))
                pairs = cur.fetchall()

            for pair in pairs:
                aid    = pair[0]
                bid    = pair[1]
                dist_m = int(pair[2])
                try:
                    with conn.cursor() as cur:
                        cur.execute("LOAD 'age';")
                        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                        sql = (
                            f"SELECT * FROM cypher('{AGE_GRAPH_NAME}', $$ "
                            f"MATCH (a:{la} {{fid: {aid}}}), (b:{lb} {{fid: {bid}}}) "
                            f"CREATE (a)-[:NEAR {{distance_m: {dist_m}}}]->(b) "
                            f"$$) AS (v agtype);"
                        )
                        cur.execute(sql)
                    total_near += 1
                except psycopg.Error:
                    conn.rollback()
        except psycopg.Error as e:
            print(f"    WARNING: NEAR edges {ta}↔{tb}: {e}")
            conn.rollback()

    print(f"  NEAR edges created: {total_near}")
    print("  Graph build complete.\n")


# ─────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Set up the geospatial PostGIS database.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OSM download; only re-embed and rebuild graph.")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip Apache AGE graph build.")
    args = parser.parse_args()

    sys.setrecursionlimit(10000)
    print(f"Connecting to PostgreSQL: {os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}")
    conn = _connect()

    verify_extensions(conn)

    if not args.skip_download:
        run_data_load(conn)

    run_embedding_index(conn)

    if not args.skip_graph:
        run_graph_build(conn)

    conn.close()
    print("Setup complete!")


if __name__ == "__main__":
    import threading
    # Run in a thread with a larger stack (64 MB) to avoid C-extension stack
    # overflows (GEOS/shapely polygon processing) on macOS where ulimit -s
    # cannot be raised beyond the OS hard limit.
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=main)
    t.start()
    t.join()
