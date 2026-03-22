"""
SpatiaLite Query Executor Module.

Executes validated SQL against a GeoPackage file using SpatiaLite,
and converts results into GeoDataFrames for visualization.
"""

import json
import sqlite3
from dataclasses import dataclass

import geopandas as gpd
import shapely
from shapely.geometry import shape


@dataclass
class QueryResult:
    """Container for spatial query results."""
    gdf: gpd.GeoDataFrame | None
    columns: list[str]
    row_count: int
    has_geometry: bool
    error: str | None = None
    raw_rows: list[dict] | None = None


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with SpatiaLite loaded and GPKG amphibious mode enabled."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    search_paths = [
        "mod_spatialite",
        "/opt/homebrew/lib/mod_spatialite",
        "/usr/local/lib/mod_spatialite",
    ]
    for path in search_paths:
        try:
            conn.load_extension(path)
            # Enable GPKG amphibious mode so ST_ functions work directly
            # on GeoPackage Binary geometry without manual GeomFromGPB() wrapping
            conn.execute("SELECT EnableGpkgAmphibiousMode()")
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError:
            continue
    raise RuntimeError(
        "mod_spatialite not found. Install with: brew install libspatialite"
    )


def _try_parse_geojson(value: str) -> shapely.Geometry | None:
    """Try to parse a string as GeoJSON geometry."""
    try:
        geom_dict = json.loads(value)
        if "type" in geom_dict and "coordinates" in geom_dict:
            return shape(geom_dict)
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass
    return None


def _try_parse_wkb(value: bytes) -> shapely.Geometry | None:
    """Try to parse bytes as WKB geometry."""
    try:
        return shapely.from_wkb(value)
    except Exception:
        pass
    # Try SpatiaLite blob format (skip first 43 bytes header for GPKG binary)
    try:
        return shapely.from_wkb(value, hex=False)
    except Exception:
        pass
    return None


def _try_parse_wkt(value: str) -> shapely.Geometry | None:
    """Try to parse a string as WKT geometry."""
    try:
        return shapely.from_wkt(value)
    except Exception:
        pass
    return None


def execute_query(db_path: str, sql: str) -> QueryResult:
    """
    Execute a spatial SQL query against a GeoPackage and return results.

    The function attempts to detect geometry columns in the results and
    automatically converts them to Shapely geometries for a GeoDataFrame.
    """
    try:
        conn = get_connection(db_path)
    except RuntimeError as e:
        return QueryResult(
            gdf=None, columns=[], row_count=0,
            has_geometry=False, error=str(e)
        )

    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return QueryResult(
                gdf=None, columns=columns, row_count=0,
                has_geometry=False, error=None,
                raw_rows=[]
            )

        # Convert to list of dicts
        records = [dict(zip(columns, row)) for row in rows]

        # Detect geometry columns
        geom_col = None
        geometries = []

        for col_name in columns:
            sample = records[0][col_name]

            # Check if this column contains GeoJSON strings
            if isinstance(sample, str):
                geom = _try_parse_geojson(sample)
                if geom is not None:
                    geom_col = col_name
                    break
                geom = _try_parse_wkt(sample)
                if geom is not None:
                    geom_col = col_name
                    break

            # Check if this column contains WKB blobs
            if isinstance(sample, bytes):
                geom = _try_parse_wkb(sample)
                if geom is not None:
                    geom_col = col_name
                    break

        if geom_col:
            # Parse all geometries
            for record in records:
                raw = record[geom_col]
                geom = None
                if isinstance(raw, str):
                    geom = _try_parse_geojson(raw) or _try_parse_wkt(raw)
                elif isinstance(raw, bytes):
                    geom = _try_parse_wkb(raw)
                geometries.append(geom)

            # Build GeoDataFrame
            # Remove the raw geometry column from attributes
            attr_cols = [c for c in columns if c != geom_col]
            attr_data = [{c: r[c] for c in attr_cols} for r in records]

            gdf = gpd.GeoDataFrame(
                attr_data,
                geometry=geometries,
                crs="EPSG:4326",
            )

            return QueryResult(
                gdf=gdf,
                columns=attr_cols,
                row_count=len(records),
                has_geometry=True,
                error=None,
                raw_rows=records,
            )
        else:
            # No geometry detected — return as raw data
            return QueryResult(
                gdf=None,
                columns=columns,
                row_count=len(records),
                has_geometry=False,
                error=None,
                raw_rows=records,
            )

    except sqlite3.OperationalError as e:
        return QueryResult(
            gdf=None, columns=[], row_count=0,
            has_geometry=False, error=f"SQL execution error: {e}"
        )
    except Exception as e:
        return QueryResult(
            gdf=None, columns=[], row_count=0,
            has_geometry=False, error=f"Unexpected error: {e}"
        )
    finally:
        conn.close()
