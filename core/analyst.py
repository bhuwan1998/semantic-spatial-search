"""
Spatial Analysis Agent Module.

After a query executes, this module:
  1. Computes descriptive statistics directly from the GeoDataFrame
  2. Calls generate_analysis() (LLM) to produce a natural-language summary
  3. Suggests 3 contextual follow-up queries

Returns an AnalysisResult dataclass consumed by app.py.
"""

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np

from core.executor import QueryResult
from core.llm import generate_analysis


@dataclass
class AnalysisResult:
    """Container for spatial analysis output."""
    summary: str                      # 2-3 sentence LLM interpretation
    stats: dict                       # computed statistics dict
    followups: list[str] = field(default_factory=list)  # 3 suggested next queries


def _compute_stats(result: QueryResult) -> dict:
    """Compute descriptive statistics from a QueryResult."""
    stats: dict = {"count": result.row_count}

    if not result.has_geometry or result.gdf is None or result.gdf.empty:
        # Non-spatial result — basic attribute stats
        if result.raw_rows:
            stats["columns"] = result.columns
        return stats

    gdf = result.gdf

    # Geometry type breakdown
    geom_types = gdf.geometry.geom_type.value_counts().to_dict()
    stats["geometry_types"] = geom_types

    # Sample names
    if "name" in gdf.columns:
        names = gdf["name"].dropna().tolist()
        stats["sample_names"] = names[:10]
        stats["unique_names"] = int(gdf["name"].nunique())

    # Distance stats (if distance column present)
    dist_cols = [c for c in gdf.columns if "distance" in c.lower() or c == "dist_m"]
    if dist_cols:
        col = dist_cols[0]
        vals = gdf[col].dropna()
        if not vals.empty:
            stats["distance_stats"] = {
                "min_m":  float(np.round(vals.min(), 1)),
                "max_m":  float(np.round(vals.max(), 1)),
                "avg_m":  float(np.round(vals.mean(), 1)),
            }

    # Area stats for polygons
    polygon_types = {"Polygon", "MultiPolygon"}
    has_polygon = any(t in polygon_types for t in geom_types)
    if has_polygon and "area_sq_meters" in gdf.columns:
        areas = gdf["area_sq_meters"].dropna()
        if not areas.empty:
            stats["area_stats"] = {
                "total_m2":  float(np.round(areas.sum(), 0)),
                "largest_m2": float(np.round(areas.max(), 0)),
            }

    # Top amenity/type values
    for col in ("amenity", "highway", "waterway", "railway", "leisure",
                "landuse", "natural", "building", "cuisine"):
        if col in gdf.columns:
            top = gdf[col].dropna().value_counts().head(3).to_dict()
            if top:
                stats[f"top_{col}"] = top
            break  # Only report first matching column

    return stats


def _generate_followups(user_query: str, stats: dict) -> list[str]:
    """
    Generate 3 contextual follow-up query suggestions based on what was found.
    These are heuristic — the LLM analysis call produces richer ones if available.
    """
    count = stats.get("count", 0)
    geom_types = stats.get("geometry_types", {})
    sample_names = stats.get("sample_names", [])

    suggestions = []

    if count > 0 and sample_names:
        first_name = sample_names[0]
        suggestions.append(f"Show hospitals near {first_name}")

    if "Point" in geom_types or "MultiPoint" in geom_types:
        suggestions.append("How many results are within 1km of Adelaide CBD?")
    elif "Polygon" in geom_types or "MultiPolygon" in geom_types:
        suggestions.append("Which results are the largest by area?")
    elif "LineString" in geom_types or "MultiLineString" in geom_types:
        suggestions.append("Show only primary or trunk roads")

    suggestions.append("Show schools near these results")

    return suggestions[:3]


def analyse(
    user_query: str,
    sql: str,
    result: QueryResult,
) -> AnalysisResult:
    """
    Run the full analysis pipeline for a query result.

    1. Compute stats from GeoDataFrame
    2. Call LLM for natural-language summary
    3. Generate follow-up suggestions
    """
    stats = _compute_stats(result)

    # LLM summary
    summary = ""
    if result.row_count > 0:
        summary = generate_analysis(user_query, sql, stats)

    if not summary:
        # Fallback summary if LLM unavailable
        count = result.row_count
        if count == 0:
            summary = "No features were found matching your query."
        else:
            summary = f"Found {count} feature{'s' if count != 1 else ''} matching your query."

    followups = _generate_followups(user_query, stats)

    return AnalysisResult(
        summary=summary,
        stats=stats,
        followups=followups,
    )
