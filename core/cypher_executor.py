"""
Cypher Executor.

Executes an openCypher query against the Apache AGE `osm_spatial` graph and
back-joins geometry from PostGIS so results can be rendered on the map.

Pipeline
────────
  execute_cypher(cypher) → CypherResult
    │
    ├─ _run_cypher(cypher)
    │    └─ AGE cypher() SQL wrapper → raw agtype rows
    │
    ├─ _parse_agtype_rows(rows, columns)
    │    └─ parse each agtype value → Python scalar
    │
    ├─ _enrich_with_geometry(parsed_rows)
    │    └─ collect all fid values from result
    │       → batch SELECT geometry FROM osm_all_mat WHERE id IN (...)
    │       → merge geometry back as GeoJSON string per row
    │
    └─ build GeoDataFrame if any geometry present → CypherResult

The `fid` column(s) in the Cypher RETURN clause are the hook used to join
PostGIS geometry.  The Cypher generator always returns at least one `*_fid`
column for this purpose.

CypherResult mirrors QueryResult so the rest of the pipeline (analyst,
build_map_html, table rendering) works without modification.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass (mirrors core.executor.QueryResult interface)
# ---------------------------------------------------------------------------

@dataclass
class CypherResult:
    """Result from a Cypher query, optionally enriched with PostGIS geometry."""
    gdf: object | None = None           # GeoDataFrame | None
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    has_geometry: bool = False
    error: str | None = None
    raw_rows: list[dict] | None = None
    cypher: str = ""                    # the Cypher that was executed


# ---------------------------------------------------------------------------
# AGE execution helper
# ---------------------------------------------------------------------------

def _run_cypher(cypher: str) -> tuple[list, list[str]]:
    """
    Execute a Cypher query via AGE's cypher() SQL function.

    Returns (rows, column_names).
    Raises RuntimeError on failure.
    """
    from core.db import get_conn

    # Extract RETURN clause column aliases to name the result columns.
    # Pattern: RETURN expr AS alias, ... [ORDER BY ...] [LIMIT ...]
    column_names = _extract_return_columns(cypher)

    # Build the AS clause for AGE's cypher() — each column must be declared.
    # AGE requires: SELECT * FROM cypher(...) AS (col1 agtype, col2 agtype, ...)
    if column_names:
        as_clause = ", ".join(f"{c} agtype" for c in column_names)
    else:
        as_clause = "result agtype"

    sql = (
        f"SELECT * FROM cypher('osm_spatial', $$ {cypher} $$) "
        f"AS ({as_clause});"
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            cur.execute(sql)
            rows = cur.fetchall()

    return rows, column_names


def _extract_return_columns(cypher: str) -> list[str]:
    """
    Extract column aliases from the RETURN clause of a Cypher query.

    Handles:
      RETURN a.name AS school, b.fid AS fid, r.distance_m AS distance_m
      RETURN DISTINCT labels(b)[0] AS label, b.name AS name
    """
    # Find the RETURN clause (everything after RETURN up to ORDER BY / LIMIT / end)
    m = re.search(
        r"\bRETURN\b\s+(?:DISTINCT\s+)?(.+?)(?:\s+(?:ORDER\s+BY|LIMIT|SKIP)\b|$)",
        cypher,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ["result"]

    return_body = m.group(1).strip()
    # Split on commas not inside parentheses
    parts = _split_return_items(return_body)

    columns = []
    for part in parts:
        part = part.strip()
        # "expr AS alias" → alias
        as_match = re.search(r"\bAS\s+(\w+)\s*$", part, re.IGNORECASE)
        if as_match:
            columns.append(as_match.group(1))
        else:
            # No alias: use the last word or a generated name
            last_word = re.sub(r"[^a-zA-Z0-9_]", "_", part.split(".")[-1])
            columns.append(last_word or f"col_{len(columns)}")

    return columns if columns else ["result"]


def _split_return_items(s: str) -> list[str]:
    """Split a RETURN clause on commas, ignoring commas inside parentheses/brackets."""
    items: list[str] = []
    depth = 0
    current = ""
    for ch in s:
        if ch in "([{":
            depth += 1
            current += ch
        elif ch in ")]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        items.append(current)
    return items


# ---------------------------------------------------------------------------
# agtype value parser
# ---------------------------------------------------------------------------

def _parse_agtype_value(val: object) -> object:
    """
    Convert a single agtype value (returned as a Python string by psycopg3)
    to a native Python type.
    """
    if val is None:
        return None
    s = str(val).strip()
    # Quoted string: "Adelaide CBD"
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    # Numeric
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    # Boolean
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    # Null
    if s.lower() in ("null", "none"):
        return None
    # JSON / list / map
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: return as string (may be a complex agtype like vertex/edge repr)
    return s


def _parse_agtype_rows(rows: list, columns: list[str]) -> list[dict]:
    """Convert raw agtype rows to a list of plain Python dicts.

    psycopg3 with a dict row factory returns rows as dicts keyed by column name,
    not tuples. Support both dict rows and sequence rows defensively.
    """
    result = []
    for row in rows:
        parsed: dict = {}
        for i, col in enumerate(columns):
            if isinstance(row, dict):
                val = row.get(col)
            else:
                val = row[i] if i < len(row) else None
            parsed[col] = _parse_agtype_value(val)
        result.append(parsed)
    return result


# ---------------------------------------------------------------------------
# PostGIS geometry back-join
# ---------------------------------------------------------------------------

# Map AGE node labels to osm_all_mat layer values.
# osm_all_mat.layer is the string used in the source pipeline.
_LABEL_TO_LAYER: dict[str, str] = {
    "School":         "schools",
    "Hospital":       "hospitals",
    "Restaurant":     "restaurants",
    "Pharmacy":       "pharmacies",
    "Road":           "roads",
    "Waterway":       "waterways",
    "Railway":        "railways",
    "Park":           "parks",
    "Building":       "buildings",
    "Landuse":        "landuse",
    "NaturalFeature": "natural",
    "Boundary":       "boundaries",
}

# Columns in the result that hold fid values (generated by cypher_generator.py).
# Naming convention from few-shots: school_fid, park_fid, fid1, fid2, start_fid, etc.
_FID_COL_PATTERN = re.compile(r"_fid\d*$|^fid\d*$", re.IGNORECASE)

# Derive the layer name from a *_fid column name.
# e.g. "school_fid" → "schools",  "park_fid" → "parks",  "fid" / "fid1" → None
_COL_TO_LAYER: dict[str, str] = {
    f"{label.lower()}_fid": layer
    for label, layer in _LABEL_TO_LAYER.items()
}
# also handle plurals already in key: school_fid→schools, park_fid→parks, etc.


def _layer_from_col(col: str) -> str | None:
    """
    Infer the osm_all_mat layer string from a fid column name.
    e.g. 'school_fid' → 'schools', 'park_fid' → 'parks'.
    Returns None for generic columns like 'fid', 'fid1', 'start_fid', 'end_fid'.
    """
    col_lower = col.lower()
    # Try direct lookup first (school_fid, park_fid, …)
    for label, layer in _LABEL_TO_LAYER.items():
        if col_lower.startswith(label.lower()):
            return layer
    return None


def _collect_fid_layer_pairs(rows: list[dict]) -> list[tuple[int, str | None]]:
    """
    Collect (fid, layer_or_None) pairs from *_fid columns across all rows.
    Returns deduplicated list preserving order.
    """
    seen: set[tuple[int, str | None]] = set()
    result: list[tuple[int, str | None]] = []
    for row in rows:
        for col, val in row.items():
            if _FID_COL_PATTERN.search(col) and isinstance(val, (int, float)):
                fid = int(val)
                layer = _layer_from_col(col)
                pair = (fid, layer)
                if pair not in seen:
                    seen.add(pair)
                    result.append(pair)
    return result


def _fetch_geometries(fid_layer_pairs: list[tuple[int, str | None]]) -> dict[tuple[int, str | None], str]:
    """
    Batch-fetch GeoJSON point centroids from osm_all_mat.

    Because fid (id) is NOT unique across layers in osm_all_mat, we must filter
    by (id, layer) when the layer is known.  For generic fid columns (layer=None)
    we fall back to returning the centroid of whichever row comes first for that id.

    Returns {(fid, layer_or_None): geojson_point_str}.
    """
    if not fid_layer_pairs:
        return {}
    from core.db import get_conn

    # Split into layer-known and layer-unknown
    with_layer    = [(fid, layer) for fid, layer in fid_layer_pairs if layer]
    without_layer = [(fid, None)  for fid, layer in fid_layer_pairs if not layer]

    geom_map: dict[tuple[int, str | None], str] = {}
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Layer-specific fetch — exact match avoids cross-layer collisions.
                if with_layer:
                    for fid, layer in with_layer:
                        cur.execute(
                            "SELECT ST_AsGeoJSON(ST_Centroid(geometry)) AS geojson "
                            "FROM osm_all_mat WHERE id = %s AND layer = %s LIMIT 1",
                            (fid, layer),
                        )
                        row = cur.fetchone()
                        if row:
                            gj = row["geojson"] if isinstance(row, dict) else row[0]
                            if gj:
                                geom_map[(fid, layer)] = gj

                # Generic fetch — no layer filter, take first match.
                if without_layer:
                    ids = ", ".join(str(f) for f, _ in without_layer)
                    cur.execute(
                        f"SELECT DISTINCT ON (id) id, ST_AsGeoJSON(ST_Centroid(geometry)) AS geojson "
                        f"FROM osm_all_mat WHERE id IN ({ids})"
                    )
                    for row in cur.fetchall():
                        fid_val = row["id"]    if isinstance(row, dict) else row[0]
                        gj      = row["geojson"] if isinstance(row, dict) else row[1]
                        if gj:
                            geom_map[(int(fid_val), None)] = gj
    except Exception as exc:
        logger.warning("Geometry back-join failed: %s", exc)
    return geom_map


def _enrich_with_geometry(rows: list[dict]) -> tuple[list[dict], bool]:
    """
    For each result row build a GeoJSON LineString connecting the ordered
    node centroids (school → park → school, etc.).

    If a row has only a single fid column, a Point is returned instead.
    Returns (enriched_rows, any_geometry_added).
    """
    fid_layer_pairs = _collect_fid_layer_pairs(rows)
    if not fid_layer_pairs:
        return rows, False

    geom_map = _fetch_geometries(fid_layer_pairs)
    if not geom_map:
        return rows, False

    has_geom = False
    for row in rows:
        # Collect ordered fid columns present in this row (preserving dict insertion order)
        ordered_fid_cols = [
            col for col in row
            if _FID_COL_PATTERN.search(col) and isinstance(row[col], (int, float))
        ]
        points: list[list[float]] = []
        for col in ordered_fid_cols:
            fid   = int(row[col])
            layer = _layer_from_col(col)
            gj    = geom_map.get((fid, layer)) or geom_map.get((fid, None))
            if gj:
                try:
                    coords = json.loads(gj)["coordinates"]  # [lon, lat] from centroid
                    points.append(coords)
                except Exception:
                    pass

        if len(points) >= 2:
            # Build a LineString connecting the path nodes in order
            row["geometry"] = json.dumps({"type": "LineString", "coordinates": points})
            has_geom = True
        elif len(points) == 1:
            row["geometry"] = json.dumps({"type": "Point", "coordinates": points[0]})
            has_geom = True

    return rows, has_geom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_cypher(cypher: str) -> CypherResult:
    """
    Execute an openCypher query and return a CypherResult.

    Steps:
      1. Run against AGE via cypher() SQL wrapper.
      2. Parse agtype values to Python types.
      3. Back-join PostGIS geometry using fid columns.
      4. Build a GeoDataFrame if geometry is present.

    On any failure, returns CypherResult(error=...) so callers can degrade gracefully.
    """
    result = CypherResult(cypher=cypher)

    # 1. Execute
    try:
        rows, columns = _run_cypher(cypher)
    except Exception as exc:
        result.error = f"Cypher execution failed: {exc}"
        logger.error("Cypher execution error: %s\nCypher: %s", exc, cypher)
        return result

    result.columns = columns
    result.row_count = len(rows)

    if not rows:
        result.raw_rows = []
        return result

    # 2. Parse agtype
    parsed = _parse_agtype_rows(rows, columns)

    # 3. Back-join geometry
    enriched, has_geom = _enrich_with_geometry(parsed)
    result.raw_rows = enriched

    # 4. Build GeoDataFrame
    if has_geom:
        try:
            import geopandas as gpd
            from shapely.geometry import shape

            geo_rows = []
            for row in enriched:
                geojson_str = row.get("geometry")
                if not geojson_str:
                    continue
                try:
                    geom = shape(json.loads(geojson_str))
                    flat = {k: v for k, v in row.items() if k != "geometry"}
                    flat["geometry"] = geom
                    geo_rows.append(flat)
                except Exception:
                    pass

            if geo_rows:
                result.gdf = gpd.GeoDataFrame(geo_rows, crs="EPSG:4326")
                result.has_geometry = True
                result.row_count = len(geo_rows)
        except Exception as exc:
            logger.warning("GeoDataFrame build failed for Cypher result: %s", exc)
            # Fall back to raw rows without geometry
            result.has_geometry = False

    return result
