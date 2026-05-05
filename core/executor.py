"""
PostGIS Query Executor Module.

Executes validated SQL against PostgreSQL + PostGIS using psycopg3,
and converts results into GeoDataFrames for visualization.
"""

import json
from dataclasses import dataclass

import geopandas as gpd
import psycopg
import shapely
from shapely.geometry import shape

from core.db import get_conn


@dataclass
class QueryResult:
    """Container for spatial query results."""
    gdf: gpd.GeoDataFrame | None
    columns: list[str]
    row_count: int
    has_geometry: bool
    error: str | None = None
    raw_rows: list[dict] | None = None


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
    return None


def _try_parse_wkt(value: str) -> shapely.Geometry | None:
    """Try to parse a string as WKT geometry."""
    try:
        return shapely.from_wkt(value)
    except Exception:
        pass
    return None


def execute_query(sql: str) -> QueryResult:
    """
    Execute a spatial SQL query against PostGIS and return results.

    The function detects geometry columns in the result set (GeoJSON strings
    produced by ST_AsGeoJSON(), WKB bytes, or WKT strings) and automatically
    builds a GeoDataFrame.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    return QueryResult(
                        gdf=None, columns=[], row_count=0,
                        has_geometry=False, error="Query returned no description."
                    )
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()

                if not rows:
                    return QueryResult(
                        gdf=None, columns=columns, row_count=0,
                        has_geometry=False, error=None, raw_rows=[]
                    )

                # rows are dicts because we use dict_row factory
                records = list(rows)

                # Detect geometry column
                geom_col = None
                for col_name in columns:
                    sample = records[0].get(col_name)
                    if sample is None:
                        continue
                    if isinstance(sample, str):
                        if _try_parse_geojson(sample) is not None:
                            geom_col = col_name
                            break
                        if _try_parse_wkt(sample) is not None:
                            geom_col = col_name
                            break
                    if isinstance(sample, (bytes, memoryview)):
                        raw = bytes(sample)
                        if _try_parse_wkb(raw) is not None:
                            geom_col = col_name
                            break

                if geom_col:
                    geometries = []
                    for record in records:
                        raw = record.get(geom_col)
                        geom = None
                        if isinstance(raw, str):
                            geom = _try_parse_geojson(raw) or _try_parse_wkt(raw)
                        elif isinstance(raw, (bytes, memoryview)):
                            geom = _try_parse_wkb(bytes(raw))
                        geometries.append(geom)

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
                    return QueryResult(
                        gdf=None,
                        columns=columns,
                        row_count=len(records),
                        has_geometry=False,
                        error=None,
                        raw_rows=records,
                    )

    except psycopg.Error as e:
        return QueryResult(
            gdf=None, columns=[], row_count=0,
            has_geometry=False, error=f"SQL execution error: {e}"
        )
    except Exception as e:
        return QueryResult(
            gdf=None, columns=[], row_count=0,
            has_geometry=False, error=f"Unexpected error: {e}"
        )
