"""
Spatial Concepts Ontology.

Defines the vocabulary for semantic spatial reasoning:
  - Topological relations  → PostGIS operator hints
  - Directional relations  → bearing ranges
  - Proximity tiers        → distance thresholds (metres)
  - Qualitative scales     → quantitative thresholds
  - Fuzzy concept mappings → canonical PostGIS patterns

Used by SpatialReasoner to classify user intent before SQL generation,
and to annotate result explanations after execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Proximity tiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProximityTier:
    label: str           # human label  e.g. "walking distance"
    max_m: float         # upper bound in metres
    description: str     # explanation for reasoning trace

PROXIMITY_TIERS: list[ProximityTier] = [
    ProximityTier("immediate",        50,    "immediately adjacent (same block)"),
    ProximityTier("walking distance", 800,   "comfortable walking distance (≤800 m)"),
    ProximityTier("short walk",       1500,  "short walk or cycling distance (≤1.5 km)"),
    ProximityTier("nearby",           3000,  "nearby, within a few minutes drive (≤3 km)"),
    ProximityTier("local area",       8000,  "within the broader local area (≤8 km)"),
    ProximityTier("city-wide",        30000, "city-wide scale (≤30 km)"),
]

# Default fallback when no explicit distance given
DEFAULT_PROXIMITY_M = 2000   # "near" without qualifier → 2 km


def classify_proximity(distance_m: float) -> ProximityTier:
    """Return the ProximityTier that best describes a given distance in metres."""
    for tier in PROXIMITY_TIERS:
        if distance_m <= tier.max_m:
            return tier
    return PROXIMITY_TIERS[-1]


# ---------------------------------------------------------------------------
# Qualitative size / density scales
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SizeThreshold:
    label: str
    min_m2: float
    max_m2: float | None
    description: str

PARK_SIZE_THRESHOLDS: list[SizeThreshold] = [
    SizeThreshold("tiny",   0,       500,    "pocket park or median strip"),
    SizeThreshold("small",  500,     5_000,  "small neighbourhood park"),
    SizeThreshold("medium", 5_000,   50_000, "medium suburb park"),
    SizeThreshold("large",  50_000,  500_000,"large regional park or reserve"),
    SizeThreshold("major",  500_000, None,   "major park, conservation area or national park"),
]

DENSITY_LABELS: dict[str, tuple[float, float]] = {
    # label → (min per km², max per km²)
    "sparse":   (0,    5),
    "moderate": (5,    20),
    "dense":    (20,   80),
    "very dense": (80, float("inf")),
}


def classify_park_size(area_m2: float) -> SizeThreshold:
    for t in PARK_SIZE_THRESHOLDS:
        if t.max_m2 is None or area_m2 < t.max_m2:
            return t
    return PARK_SIZE_THRESHOLDS[-1]


def classify_density(per_km2: float) -> str:
    for label, (lo, hi) in DENSITY_LABELS.items():
        if lo <= per_km2 < hi:
            return label
    return "very dense"


# ---------------------------------------------------------------------------
# Directional relations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectionalRelation:
    label: str            # "north of"
    bearing_min: float    # degrees clockwise from north
    bearing_max: float
    sql_hint: str         # PostGIS bearing expression hint

DIRECTIONS: list[DirectionalRelation] = [
    DirectionalRelation("north of",     337.5, 360,   "ST_Azimuth < 0.2 OR ST_Azimuth > 5.9"),
    DirectionalRelation("north of",     0,     22.5,  "ST_Azimuth < 0.4"),
    DirectionalRelation("northeast of", 22.5,  67.5,  "ST_Azimuth BETWEEN 0.4 AND 1.2"),
    DirectionalRelation("east of",      67.5,  112.5, "ST_Azimuth BETWEEN 1.2 AND 2.0"),
    DirectionalRelation("southeast of", 112.5, 157.5, "ST_Azimuth BETWEEN 2.0 AND 2.7"),
    DirectionalRelation("south of",     157.5, 202.5, "ST_Azimuth BETWEEN 2.7 AND 3.5"),
    DirectionalRelation("southwest of", 202.5, 247.5, "ST_Azimuth BETWEEN 3.5 AND 4.3"),
    DirectionalRelation("west of",      247.5, 292.5, "ST_Azimuth BETWEEN 4.3 AND 5.1"),
    DirectionalRelation("northwest of", 292.5, 337.5, "ST_Azimuth BETWEEN 5.1 AND 5.9"),
]

# Simplified bearing ranges for reasoning trace (canonical 8 directions)
DIRECTION_BEARING: dict[str, tuple[float, float]] = {
    "north":     (337.5, 22.5),
    "northeast": (22.5,  67.5),
    "east":      (67.5,  112.5),
    "southeast": (112.5, 157.5),
    "south":     (157.5, 202.5),
    "southwest": (202.5, 247.5),
    "west":      (247.5, 292.5),
    "northwest": (292.5, 337.5),
}

DIRECTION_KEYWORDS: dict[str, list[str]] = {
    "north":     ["north of", "north-of", "to the north", "northern", "northward"],
    "northeast": ["northeast of", "north-east of", "to the northeast"],
    "east":      ["east of", "to the east", "eastern", "eastward"],
    "southeast": ["southeast of", "south-east of", "to the southeast"],
    "south":     ["south of", "to the south", "southern", "southward"],
    "southwest": ["southwest of", "south-west of", "to the southwest"],
    "west":      ["west of", "to the west", "western", "westward"],
    "northwest": ["northwest of", "north-west of", "to the northwest"],
}


def detect_directions(text: str) -> list[str]:
    """Return canonical direction names found in text."""
    text_lower = text.lower()
    found = []
    for direction, keywords in DIRECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(direction)
    return found


# ---------------------------------------------------------------------------
# Topological relation keywords → PostGIS operator hints
# ---------------------------------------------------------------------------

TOPOLOGICAL_RELATIONS: dict[str, dict] = {
    "within": {
        "keywords": ["within", "inside", "in", "contained by", "contained in"],
        "postgis":  "ST_Within(a.geometry, b.geometry)",
        "description": "feature is fully inside another geometry",
    },
    "contains": {
        "keywords": ["contains", "encloses", "covering", "covers"],
        "postgis":  "ST_Contains(a.geometry, b.geometry)",
        "description": "feature fully contains another geometry",
    },
    "intersects": {
        "keywords": ["intersects", "crosses", "overlaps", "through"],
        "postgis":  "ST_Intersects(a.geometry, b.geometry)",
        "description": "geometries share any space",
    },
    "touches": {
        "keywords": ["touches", "borders", "adjacent to", "next to", "adjoins"],
        "postgis":  "ST_Touches(a.geometry, b.geometry)",
        "description": "geometries share a boundary but no interior",
    },
    "proximity": {
        "keywords": ["near", "close to", "within", "around", "nearby", "km of", "m of", "metres of", "meters of"],
        "postgis":  "ST_DWithin(a.geometry::geography, b.geometry::geography, dist_m)",
        "description": "features within a distance threshold",
    },
    "gap_analysis": {
        "keywords": ["no", "without", "missing", "lack", "absent", "none"],
        "postgis":  "NOT EXISTS (SELECT 1 FROM ... WHERE ST_DWithin(...))",
        "description": "features with no nearby counterpart",
    },
}


def detect_topological_relations(text: str) -> list[str]:
    """Return topological relation keys detected in text."""
    text_lower = text.lower()
    found = []
    for rel, info in TOPOLOGICAL_RELATIONS.items():
        if any(kw in text_lower for kw in info["keywords"]):
            found.append(rel)
    return found


# ---------------------------------------------------------------------------
# Qualitative descriptors → numeric hints
# ---------------------------------------------------------------------------

QUALITATIVE_DESCRIPTORS: dict[str, dict] = {
    # Size
    "large":      {"dimension": "area",    "hint": "large (area > 50,000 m²)",    "threshold_m2": 50_000},
    "small":      {"dimension": "area",    "hint": "small (area < 5,000 m²)",     "threshold_m2": 5_000},
    "big":        {"dimension": "area",    "hint": "large (area > 50,000 m²)",    "threshold_m2": 50_000},
    "tiny":       {"dimension": "area",    "hint": "tiny (area < 500 m²)",        "threshold_m2": 500},
    "huge":       {"dimension": "area",    "hint": "very large (area > 200,000 m²)", "threshold_m2": 200_000},
    # Density
    "dense":      {"dimension": "density", "hint": "dense (> 20 features/km²)",   "per_km2": 20},
    "sparse":     {"dimension": "density", "hint": "sparse (< 5 features/km²)",   "per_km2": 5},
    "crowded":    {"dimension": "density", "hint": "crowded/dense area",           "per_km2": 30},
    # Proximity qualifiers
    "walkable":   {"dimension": "distance","hint": "walkable (≤800 m)",            "dist_m": 800},
    "walking":    {"dimension": "distance","hint": "walking distance (≤800 m)",    "dist_m": 800},
    "close":      {"dimension": "distance","hint": "close (≤1500 m)",              "dist_m": 1500},
    "far":        {"dimension": "distance","hint": "far (> 5 km)",                 "dist_m": 5000},
    "isolated":   {"dimension": "distance","hint": "isolated (furthest from nearest neighbour)", "dist_m": None},
    # Quantity
    "most":       {"dimension": "count",   "hint": "highest count, ORDER BY count DESC"},
    "least":      {"dimension": "count",   "hint": "lowest count, ORDER BY count ASC"},
    "busiest":    {"dimension": "count",   "hint": "highest activity / density"},
    "quietest":   {"dimension": "count",   "hint": "lowest activity / density"},
    "popular":    {"dimension": "count",   "hint": "high count or density"},
}


def detect_qualitative_descriptors(text: str) -> list[str]:
    """Return qualitative descriptor keys found in text."""
    text_lower = text.lower()
    return [key for key in QUALITATIVE_DESCRIPTORS if key in text_lower]


# ---------------------------------------------------------------------------
# Intent type taxonomy
# ---------------------------------------------------------------------------

IntentType = Literal[
    "proximity_search",    # find X near Y
    "containment_query",   # find X in/within area Y
    "directional_query",   # find X north/east/... of Y
    "gap_analysis",        # find X with no Y nearby
    "density_ranking",     # rank areas by count/density
    "comparison",          # compare two feature types
    "attribute_filter",    # filter by non-spatial attribute
    "aggregate_stats",     # COUNT/GROUP BY / area totals
    "nearest_neighbour",   # find nearest single feature
    "topological_join",    # intersects/touches/overlaps
    "mixed",               # more than one primary intent
    "unknown",
]

INTENT_PATTERNS: dict[str, list[str]] = {
    "proximity_search":  ["near", "within", "km of", "m of", "metres", "meters", "around", "close to", "nearby"],
    "containment_query": ["in ", "inside", "within the", "contained in", "suburb", "area", "region"],
    "directional_query": ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"],
    "gap_analysis":      ["no ", "without", "missing", "no pharmacy", "no school", "no restaurant", "not exist"],
    "density_ranking":   ["density", "densest", "per km", "per square", "most per", "rank", "busiest"],
    "comparison":        ["compared to", "vs", "versus", "compare", "ratio of", "how much"],
    "attribute_filter":  ["type", "amenity", "named", "called", "where", "highway", "cuisine"],
    "aggregate_stats":   ["how many", "count", "total", "sum", "average", "largest", "biggest", "area"],
    "nearest_neighbour": ["nearest", "closest", "nearest single", "for each", "closest to"],
    "topological_join":  ["intersect", "cross", "overlap", "touch", "border"],
}


def classify_intent(text: str) -> list[str]:
    """Return a list of matching IntentType keys (ordered by confidence)."""
    text_lower = text.lower()
    matches = []
    for intent, keywords in INTENT_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            matches.append((intent, score))
    matches.sort(key=lambda x: -x[1])
    return [m[0] for m in matches] or ["unknown"]


# ---------------------------------------------------------------------------
# PostGIS operator glossary (used in reasoning trace UI)
# ---------------------------------------------------------------------------

POSTGIS_GLOSSARY: dict[str, str] = {
    "ST_DWithin":    "radius proximity filter using spatial index",
    "ST_Distance":   "exact metric distance between two geometries",
    "ST_Within":     "strict containment — A is fully inside B",
    "ST_Contains":   "strict containment — B is fully inside A",
    "ST_Intersects": "any shared space between A and B",
    "ST_Touches":    "shared boundary but no interior overlap",
    "ST_Area":       "area in square metres (cast to ::geography)",
    "ST_Length":     "length in metres (cast to ::geography)",
    "ST_Centroid":   "centre point of a geometry",
    "ST_Azimuth":    "bearing angle from A to B (radians, clockwise from north)",
    "<->":           "KNN (k-nearest-neighbour) distance operator for ORDER BY",
}
