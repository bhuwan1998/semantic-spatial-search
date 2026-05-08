"""
Spatial Analysis Agent Module.

After a query executes, this module:
  1. Computes descriptive statistics directly from the GeoDataFrame
  2. Calls generate_analysis() (LLM) to produce a natural-language summary
     enriched with semantic spatial context from SpatialReasoner
  3. Suggests 3 contextual follow-up queries

Returns an AnalysisResult dataclass consumed by app.py.

Streaming support
-----------------
analyse_stream() is a variant of analyse() that returns the AnalysisResult
with summary="" and a separate generator `stream` attribute so app.py can
call st.write_stream(result.stream) to render the summary token-by-token.
"""

from dataclasses import dataclass, field
from typing import Generator, Optional

import geopandas as gpd
import numpy as np

from core.executor import QueryResult
from core.llm import generate_analysis, generate_analysis_stream, should_skip_analysis


@dataclass
class AnalysisResult:
    """Container for spatial analysis output."""
    summary: str                      # 2-3 sentence LLM interpretation
    stats: dict                       # computed statistics dict
    followups: list[str] = field(default_factory=list)  # 3 suggested next queries
    stream: Generator | None = None   # live token generator (None = already complete)
    spatial_intent = None             # SpatialIntent | None — set by analyse_stream/analyse


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


def _generate_followups(user_query: str, stats: dict, intent=None) -> list[str]:
    """
    Generate 3 contextual follow-up query suggestions based on what was found
    and the detected spatial intent.
    """
    count = stats.get("count", 0)
    geom_types = stats.get("geometry_types", {})
    sample_names = stats.get("sample_names", [])

    suggestions = []

    # Intent-aware follow-ups
    if intent is not None:
        primary = getattr(intent, "primary_intent", "unknown")
        directions = getattr(intent, "directions", [])
        feature_types = getattr(intent, "feature_types", [])
        named_places = getattr(intent, "named_places", [])

        if primary == "proximity_search" and named_places:
            place = named_places[0]
            suggestions.append(f"Extend the search radius — show results within 5km of {place}")
        elif primary == "containment_query" and named_places:
            place = named_places[0]
            suggestions.append(f"Compare these results with an adjacent suburb")
        elif primary == "density_ranking":
            suggestions.append("Show the same density ranking for a different feature type")
        elif primary == "gap_analysis":
            suggestions.append("Show the features that DO have nearby counterparts for comparison")
        elif primary == "directional_query" and directions:
            opp = {"north": "south", "south": "north", "east": "west", "west": "east",
                   "northeast": "southwest", "southwest": "northeast",
                   "northwest": "southeast", "southeast": "northwest"}
            opp_dir = opp.get(directions[0], "south")
            suggestions.append(f"Now show the same results {opp_dir} of the reference point")

        # Directional follow-up (add the perpendicular direction)
        if directions and len(suggestions) < 2:
            perp = {"north": "east", "south": "west", "east": "north", "west": "south"}
            perp_dir = perp.get(directions[0])
            if perp_dir and sample_names:
                suggestions.append(f"Show these features {perp_dir} of {named_places[0] if named_places else 'Adelaide CBD'}")

    # Geometry-based follow-ups
    if count > 0 and sample_names and len(suggestions) < 3:
        first_name = sample_names[0]
        suggestions.append(f"Show hospitals near {first_name}")

    if len(suggestions) < 3:
        if "Point" in geom_types or "MultiPoint" in geom_types:
            suggestions.append("How many results are within 1km of Adelaide CBD?")
        elif "Polygon" in geom_types or "MultiPolygon" in geom_types:
            suggestions.append("Which results are the largest by area?")
        elif "LineString" in geom_types or "MultiLineString" in geom_types:
            suggestions.append("Show only primary or trunk roads")

    if len(suggestions) < 3:
        suggestions.append("Show schools near these results")

    return suggestions[:3]


def analyse(
    user_query: str,
    sql: str,
    result: QueryResult,
    intent=None,
) -> AnalysisResult:
    """
    Run the full analysis pipeline for a query result.

    1. Compute stats from GeoDataFrame
    2. Call LLM for natural-language summary enriched with spatial intent context
    3. Generate follow-up suggestions (intent-aware)

    intent: optional SpatialIntent from spatial_reasoner.decompose_intent()
    """
    from core.spatial_reasoner import explain_results as _explain

    stats = _compute_stats(result)

    # Build enriched analysis prompt if intent provided
    spatial_prompt: Optional[str] = None
    if intent is not None and result.row_count > 0:
        spatial_prompt = _explain(intent, stats, user_query)

    # Skip analysis LLM for zero-row, scalar, or pure-numeric results
    summary = ""
    if result.row_count > 0 and not should_skip_analysis(result):
        summary = generate_analysis(user_query, sql, stats, spatial_prompt=spatial_prompt)

    if not summary:
        count = result.row_count
        if count == 0:
            summary = "No features were found matching your query."
        else:
            summary = f"Found {count} feature{'s' if count != 1 else ''} matching your query."

    followups = _generate_followups(user_query, stats, intent=intent)

    ar = AnalysisResult(
        summary=summary,
        stats=stats,
        followups=followups,
    )
    ar.spatial_intent = intent
    return ar


def analyse_stream(
    user_query: str,
    sql: str,
    result: QueryResult,
    intent=None,
) -> AnalysisResult:
    """
    Streaming variant of analyse().

    Returns an AnalysisResult with summary="" and stream set to a generator
    that yields text chunks from the LLM.  The caller should render the stream
    with st.write_stream() and then persist the accumulated text as the summary.

    For results where analysis is skipped (0 rows, pure aggregate), returns
    normally with summary set and stream=None so the caller can skip streaming.

    intent: optional SpatialIntent from spatial_reasoner.decompose_intent()
    """
    from core.spatial_reasoner import explain_results as _explain

    stats = _compute_stats(result)
    followups = _generate_followups(user_query, stats, intent=intent)

    if result.row_count == 0:
        ar = AnalysisResult(
            summary="No features were found matching your query.",
            stats=stats,
            followups=followups,
            stream=None,
        )
        ar.spatial_intent = intent
        return ar

    if should_skip_analysis(result):
        count = result.row_count
        ar = AnalysisResult(
            summary=f"Found {count} feature{'s' if count != 1 else ''} matching your query.",
            stats=stats,
            followups=followups,
            stream=None,
        )
        ar.spatial_intent = intent
        return ar

    # Build enriched spatial prompt if intent provided
    spatial_prompt: Optional[str] = None
    if intent is not None:
        spatial_prompt = _explain(intent, stats, user_query)

    ar = AnalysisResult(
        summary="",
        stats=stats,
        followups=followups,
        stream=generate_analysis_stream(user_query, sql, stats, spatial_prompt=spatial_prompt),
    )
    ar.spatial_intent = intent
    return ar
