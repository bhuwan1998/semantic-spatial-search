"""
Semantic Spatial Reasoner.

Adds a two-phase reasoning layer around the PostGIS SQL pipeline:

  PRE-QUERY  (decompose_intent)
  ─────────────────────────────
  Given a user query string, produce a SpatialIntent dataclass that:
    • classifies the primary spatial intent type
    • identifies spatial operators needed (proximity, containment, etc.)
    • detects directional qualifiers (north of, east of…)
    • detects qualitative descriptors (large, dense, walkable…)
    • extracts an explicit distance in metres (if given)
    • produces an LLM-ready context string to improve SQL generation

  POST-QUERY  (explain_results)
  ──────────────────────────────
  Given a SpatialIntent + query result stats, produce a richer spatial
  narrative than the generic analysis prompt — one that references the
  identified concepts explicitly.

The module works in two modes:
  1. Fast rule-based   – uses regex/keyword matching in spatial_concepts.py.
     Always runs; no LLM call needed.
  2. LLM-enhanced      – an optional lightweight Ollama call to produce a
     structured decomposition.  Falls back gracefully to rule-based if the
     LLM is unavailable or slow.
"""

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.spatial_concepts import (
    classify_intent,
    detect_directions,
    detect_topological_relations,
    detect_qualitative_descriptors,
    classify_proximity,
    DEFAULT_PROXIMITY_M,
    TOPOLOGICAL_RELATIONS,
    QUALITATIVE_DESCRIPTORS,
    DIRECTION_KEYWORDS,
    POSTGIS_GLOSSARY,
    PROXIMITY_TIERS,
    GRAPH_INTENT_TYPES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SpatialIntent  — the core data structure
# ---------------------------------------------------------------------------

@dataclass
class SpatialIntent:
    """Structured decomposition of the spatial concepts in a user query."""

    # Primary intent classification
    primary_intent: str = "unknown"          # from IntentType taxonomy
    secondary_intents: list[str] = field(default_factory=list)

    # Spatial operators detected
    topological_relations: list[str] = field(default_factory=list)

    # Directional qualifiers
    directions: list[str] = field(default_factory=list)

    # Qualitative descriptors
    qualitative_descriptors: list[str] = field(default_factory=list)

    # Distance extracted from text (if any explicit mention)
    explicit_distance_m: Optional[float] = None

    # Proximity tier label resolved from explicit_distance_m
    proximity_tier: Optional[str] = None

    # Named entities recognised (places, landmarks)
    named_places: list[str] = field(default_factory=list)

    # Feature types identified
    feature_types: list[str] = field(default_factory=list)

    # Whether the query has directional, qualitative, or fuzzy spatial language
    has_directional: bool = False
    has_qualitative: bool = False
    has_fuzzy_proximity: bool = False    # "nearby" / "close" without explicit metres

    # PostGIS hints for SQL generation (derived from the above)
    postgis_hints: list[str] = field(default_factory=list)

    # Human-readable reasoning notes (shown in the UI trace)
    reasoning_notes: list[str] = field(default_factory=list)

    # Short context string injected into the LLM SQL generation prompt
    llm_context: str = ""

    # Whether rule-based or LLM-enhanced decomposition was used
    source: str = "rule-based"   # "rule-based" | "llm-enhanced"

    # Whether this query should be routed to the Cypher engine instead of SQL.
    # Set to True when primary_intent is in GRAPH_INTENT_TYPES.
    requires_graph: bool = False

    def to_display_dict(self) -> dict:
        """Return a UI-friendly dict for the reasoning trace panel."""
        d: dict = {}
        if self.primary_intent and self.primary_intent != "unknown":
            d["Intent"] = self.primary_intent.replace("_", " ").title()
        if self.secondary_intents:
            d["Also detected"] = ", ".join(
                i.replace("_", " ").title() for i in self.secondary_intents
            )
        if self.topological_relations:
            d["Spatial operators"] = ", ".join(
                f"{r} → `{TOPOLOGICAL_RELATIONS[r]['postgis'].split('(')[0]}`"
                for r in self.topological_relations
                if r in TOPOLOGICAL_RELATIONS
            )
        if self.directions:
            d["Directional"] = ", ".join(self.directions)
        if self.qualitative_descriptors:
            hints = []
            for k in self.qualitative_descriptors:
                info = QUALITATIVE_DESCRIPTORS.get(k, {})
                hints.append(info.get("hint", k))
            d["Qualitative"] = " | ".join(hints)
        if self.explicit_distance_m is not None:
            d["Distance"] = f"{self.explicit_distance_m:,.0f} m ({self.proximity_tier})"
        elif self.has_fuzzy_proximity:
            d["Proximity"] = f"fuzzy (default {DEFAULT_PROXIMITY_M:,} m — '{self.proximity_tier}')"
        if self.named_places:
            d["Named places"] = ", ".join(self.named_places)
        if self.feature_types:
            d["Feature types"] = ", ".join(self.feature_types)
        if self.postgis_hints:
            d["PostGIS hints"] = " | ".join(self.postgis_hints)
        if self.reasoning_notes:
            d["Notes"] = " · ".join(self.reasoning_notes)
        d["Engine"] = "Apache AGE (Cypher)" if self.requires_graph else "PostGIS (SQL)"
        d["Source"] = self.source
        return d


# ---------------------------------------------------------------------------
# Distance extraction
# ---------------------------------------------------------------------------

_DISTANCE_PATTERNS = [
    # "2km", "2 km", "2.5 km", "2 kilometres"
    (r"(\d+(?:\.\d+)?)\s*km\b", 1000),
    (r"(\d+(?:\.\d+)?)\s*kilometre[s]?\b", 1000),
    (r"(\d+(?:\.\d+)?)\s*kilometer[s]?\b", 1000),
    # "500m", "500 metres", "500 meters"
    (r"(\d+(?:\.\d+)?)\s*m\b(?!i)", 1),
    (r"(\d+(?:\.\d+)?)\s*metre[s]?\b", 1),
    (r"(\d+(?:\.\d+)?)\s*meter[s]?\b", 1),
]


def _extract_distance(text: str) -> Optional[float]:
    """Extract the first explicit distance mention and return metres, or None."""
    for pattern, multiplier in _DISTANCE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return float(m.group(1)) * multiplier
    return None


# ---------------------------------------------------------------------------
# Named entity / feature type extraction (lightweight, keyword-based)
# ---------------------------------------------------------------------------

_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "school":     ["school", "schools"],
    "hospital":   ["hospital", "hospitals", "clinic", "medical centre"],
    "restaurant": ["restaurant", "restaurants", "cafe", "cafes", "food", "dining", "eatery"],
    "pharmacy":   ["pharmacy", "pharmacies", "chemist"],
    "park":       ["park", "parks", "reserve", "garden", "green space"],
    "road":       ["road", "roads", "street", "highway", "motorway", "lane"],
    "waterway":   ["waterway", "waterways", "river", "creek", "stream", "lake", "pond"],
    "railway":    ["railway", "rail", "train", "station", "tram"],
    "building":   ["building", "buildings", "structure"],
    "landuse":    ["landuse", "land use", "residential", "industrial", "commercial"],
    "boundary":   ["suburb", "suburbs", "boundary", "boundaries", "lga", "region"],
}

# Place/landmark indicators (precede a proper noun)
_PLACE_INDICATORS = [
    "near", "around", "in", "of", "at", "from", "close to",
    "within", "north of", "south of", "east of", "west of",
    "northeast of", "northwest of", "southeast of", "southwest of",
]


def _detect_feature_types(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        ftype for ftype, keywords in _FEATURE_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]


def _extract_named_places(text: str) -> list[str]:
    """
    Very lightweight proper-noun extraction: capitalised words that follow
    place indicator prepositions.  No NER model needed.
    """
    places = []
    for indicator in _PLACE_INDICATORS:
        pattern = rf"\b{re.escape(indicator)}\s+([A-Z][a-zA-Z\s]{{2,30}}?)(?:\s+(?:and|or|with|,|\.|$)|\s*$)"
        for m in re.finditer(pattern, text):
            candidate = m.group(1).strip().rstrip(",.")
            if len(candidate) > 2 and candidate not in places:
                places.append(candidate)
    return places[:6]  # cap at 6


# ---------------------------------------------------------------------------
# PostGIS hint builder
# ---------------------------------------------------------------------------

def _build_postgis_hints(intent: SpatialIntent) -> list[str]:
    hints = []
    if "proximity" in intent.topological_relations or intent.primary_intent == "proximity_search":
        dist = intent.explicit_distance_m or DEFAULT_PROXIMITY_M
        hints.append(f"ST_DWithin(a::geography, b::geography, {dist:.0f})")
    if "within" in intent.topological_relations or intent.primary_intent == "containment_query":
        hints.append("ST_Within(feature.geometry, boundary.geometry)")
    if "intersects" in intent.topological_relations or intent.primary_intent == "topological_join":
        hints.append("ST_Intersects(a.geometry, b.geometry)")
    if "gap_analysis" in intent.topological_relations or intent.primary_intent == "gap_analysis":
        hints.append("NOT EXISTS (SELECT 1 FROM ... WHERE ST_DWithin(...))")
    if intent.primary_intent == "nearest_neighbour":
        hints.append("ORDER BY geometry <-> ref_point LIMIT N")
    if intent.primary_intent == "density_ranking":
        hints.append("COUNT(*) / ST_Area(geometry::geography) * 1e6 AS per_km2")
    for desc in intent.qualitative_descriptors:
        info = QUALITATIVE_DESCRIPTORS.get(desc, {})
        if info.get("dimension") == "area" and "threshold_m2" in info:
            hints.append(f"ST_Area(geometry::geography) {'>' if desc in ('large','big','huge') else '<'} {info['threshold_m2']}")
    return hints


# ---------------------------------------------------------------------------
# LLM-enhanced decomposition (optional)
# ---------------------------------------------------------------------------

_LLM_DECOMPOSE_SYSTEM = """\
You are a spatial reasoning assistant. Given a natural-language geospatial query,
decompose it into a structured JSON object with EXACTLY these keys:

{
  "primary_intent": "<one of: proximity_search | containment_query | directional_query | gap_analysis | density_ranking | comparison | attribute_filter | aggregate_stats | nearest_neighbour | topological_join | graph_traversal | path_query | cluster_pattern | mixed | unknown>",
  "secondary_intents": ["<intent>", ...],
  "topological_relations": ["<within|contains|intersects|touches|proximity|gap_analysis>", ...],
  "directions": ["<north|south|east|west|northeast|northwest|southeast|southwest>", ...],
  "qualitative_descriptors": ["<large|small|dense|sparse|walkable|close|far|isolated|most|least|busiest|quietest>", ...],
  "explicit_distance_m": <number or null>,
  "named_places": ["<place name>", ...],
  "feature_types": ["<school|hospital|restaurant|pharmacy|park|road|waterway|railway|building|landuse|boundary>", ...],
  "reasoning_notes": ["<short reasoning observation>", ...]
}

Intent guide for graph-engine intents:
  graph_traversal : multi-hop reachability via NEAR edges (e.g. "reachable in 2 hops", "connected via", "transitively")
  path_query      : shortest path between two named features (e.g. "shortest path between X and Y")
  cluster_pattern : subgraph where multiple features are mutually near (e.g. "cluster of school, hospital and park")

Rules:
- Output ONLY valid JSON. No markdown, no explanation.
- If a field is empty, use [] or null as appropriate.
- reasoning_notes should be 1-3 short bullet observations about the spatial challenge in the query.
"""


def _llm_decompose(query: str, model: str) -> Optional[dict]:
    """
    Call the local Ollama model for structured spatial decomposition.
    Returns a parsed dict or None on any failure.
    """
    try:
        import ollama
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_DECOMPOSE_SYSTEM},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0, "num_predict": 512, "num_ctx": 2048},
            stream=False,
        )
        raw = response["message"]["content"].strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        return json.loads(raw)
    except Exception as exc:
        logger.debug("LLM spatial decomposition failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API: decompose_intent
# ---------------------------------------------------------------------------

def decompose_intent(
    query: str,
    use_llm: bool = True,
    model: str | None = None,
) -> SpatialIntent:
    """
    Decompose a natural-language geospatial query into a SpatialIntent.

    1. Always runs rule-based extraction first (instant, no LLM needed).
    2. If use_llm=True, attempts a lightweight Ollama call for structured
       decomposition and merges the result over the rule-based baseline.

    Returns a SpatialIntent regardless of LLM availability.
    """
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

    # ── Rule-based baseline ──────────────────────────────────────────────────
    intents        = classify_intent(query)
    primary        = intents[0] if intents else "unknown"
    secondary      = intents[1:4]
    topo_rels      = detect_topological_relations(query)
    directions     = detect_directions(query)
    qualitative    = detect_qualitative_descriptors(query)
    distance_m     = _extract_distance(query)
    feature_types  = _detect_feature_types(query)
    named_places   = _extract_named_places(query)

    # Proximity tier
    prox_tier: Optional[str] = None
    has_fuzzy = False
    if distance_m is not None:
        tier = classify_proximity(distance_m)
        prox_tier = tier.label
    else:
        # Check qualitative proximity descriptors first (walking → 800m, close → 1500m)
        from core.spatial_concepts import QUALITATIVE_DESCRIPTORS as _QD
        qual_dist: Optional[float] = None
        for k in qualitative:
            info = _QD.get(k, {})
            if info.get("dimension") == "distance" and info.get("dist_m"):
                qual_dist = float(info["dist_m"])
                break
        if qual_dist is not None:
            distance_m = qual_dist
            tier = classify_proximity(distance_m)
            prox_tier = tier.label
        else:
            # Check for fuzzy proximity words
            fuzzy_words = ["near", "close", "nearby", "around", "walkable", "walking"]
            if any(w in query.lower() for w in fuzzy_words):
                has_fuzzy = True
                tier = classify_proximity(DEFAULT_PROXIMITY_M)
                prox_tier = tier.label

    # Build reasoning notes
    notes: list[str] = []
    if directions:
        notes.append(f"Directional constraint detected: {', '.join(directions)}")
    if qualitative:
        notes.append(
            "Qualitative descriptors require threshold mapping: "
            + ", ".join(qualitative)
        )
    if has_fuzzy:
        notes.append(f"Fuzzy proximity — defaulting to {DEFAULT_PROXIMITY_M:,} m")
    if "gap_analysis" in topo_rels:
        notes.append("Gap analysis: NOT EXISTS pattern recommended")
    if primary == "density_ranking":
        notes.append("Density query: normalise by suburb area in km²")
    if primary == "comparison":
        notes.append("Comparison query: UNION ALL pattern with summary stat rows")
    if primary in GRAPH_INTENT_TYPES:
        notes.append(
            f"Graph-engine query ({primary.replace('_', ' ')}) — "
            "will be executed as Cypher against Apache AGE, not PostGIS SQL."
        )

    requires_graph = primary in GRAPH_INTENT_TYPES

    intent = SpatialIntent(
        primary_intent=primary,
        secondary_intents=secondary,
        topological_relations=topo_rels,
        directions=directions,
        qualitative_descriptors=qualitative,
        explicit_distance_m=distance_m,   # may have been set from qualitative tier above
        proximity_tier=prox_tier,
        named_places=named_places,
        feature_types=feature_types,
        has_directional=bool(directions),
        has_qualitative=bool(qualitative),
        has_fuzzy_proximity=has_fuzzy,
        reasoning_notes=notes,
        source="rule-based",
        requires_graph=requires_graph,
    )
    intent.postgis_hints = _build_postgis_hints(intent)

    # ── LLM enhancement ──────────────────────────────────────────────────────
    if use_llm:
        llm_data = _llm_decompose(query, model)
        if llm_data:
            # Merge LLM fields over rule-based baseline (LLM wins on non-empty values)
            if llm_data.get("primary_intent") and llm_data["primary_intent"] != "unknown":
                intent.primary_intent = llm_data["primary_intent"]
            if llm_data.get("secondary_intents"):
                intent.secondary_intents = llm_data["secondary_intents"]
            if llm_data.get("topological_relations"):
                # Union of rule-based + LLM
                for r in llm_data["topological_relations"]:
                    if r not in intent.topological_relations:
                        intent.topological_relations.append(r)
            if llm_data.get("directions"):
                for d in llm_data["directions"]:
                    if d not in intent.directions:
                        intent.directions.append(d)
                intent.has_directional = True
            if llm_data.get("qualitative_descriptors"):
                for q in llm_data["qualitative_descriptors"]:
                    if q not in intent.qualitative_descriptors:
                        intent.qualitative_descriptors.append(q)
                intent.has_qualitative = True
            if llm_data.get("explicit_distance_m") is not None and intent.explicit_distance_m is None:
                intent.explicit_distance_m = float(llm_data["explicit_distance_m"])
                intent.proximity_tier = classify_proximity(intent.explicit_distance_m).label
            if llm_data.get("named_places"):
                for p in llm_data["named_places"]:
                    if p not in intent.named_places:
                        intent.named_places.append(p)
            if llm_data.get("feature_types"):
                for ft in llm_data["feature_types"]:
                    if ft not in intent.feature_types:
                        intent.feature_types.append(ft)
            if llm_data.get("reasoning_notes"):
                intent.reasoning_notes.extend(
                    n for n in llm_data["reasoning_notes"]
                    if n not in intent.reasoning_notes
                )
            intent.postgis_hints = _build_postgis_hints(intent)
            intent.source = "llm-enhanced"
            # Re-evaluate graph routing after LLM may have updated primary_intent
            intent.requires_graph = intent.primary_intent in GRAPH_INTENT_TYPES

    # ── Build LLM context string ──────────────────────────────────────────────
    intent.llm_context = _build_llm_context(intent)
    return intent


# ---------------------------------------------------------------------------
# LLM context string (injected into SQL generation prompt)
# ---------------------------------------------------------------------------

def _build_llm_context(intent: SpatialIntent) -> str:
    """
    Produce a compact context string to prepend to the user query when calling
    generate_sql().  This gives the SQL-generation LLM explicit spatial guidance.
    """
    lines: list[str] = ["[SPATIAL REASONING CONTEXT]"]

    lines.append(f"Intent: {intent.primary_intent.replace('_', ' ')}")

    if intent.explicit_distance_m is not None:
        lines.append(
            f"Distance: {intent.explicit_distance_m:,.0f} m "
            f"({intent.proximity_tier})"
        )
    elif intent.has_fuzzy_proximity and intent.proximity_tier:
        lines.append(
            f"Fuzzy proximity detected — use default {DEFAULT_PROXIMITY_M:,} m "
            f"({intent.proximity_tier})"
        )

    if intent.directions:
        lines.append(
            "Directional constraint: "
            + ", ".join(intent.directions)
            + " — use ST_Azimuth(ref_geom, feature_geom) to filter bearing"
        )

    if intent.qualitative_descriptors:
        hints = []
        for k in intent.qualitative_descriptors:
            info = QUALITATIVE_DESCRIPTORS.get(k, {})
            hints.append(info.get("hint", k))
        lines.append("Qualitative thresholds: " + "; ".join(hints))

    if intent.postgis_hints:
        lines.append("Recommended PostGIS operators: " + " | ".join(intent.postgis_hints))

    if intent.requires_graph:
        lines.append(
            "Engine: Apache AGE Cypher (graph traversal) — "
            "query will NOT use PostGIS SQL."
        )

    if intent.reasoning_notes:
        lines.append("Notes: " + " · ".join(intent.reasoning_notes))

    lines.append("[/SPATIAL REASONING CONTEXT]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API: explain_results
# ---------------------------------------------------------------------------

def explain_results(
    intent: SpatialIntent,
    stats: dict,
    user_query: str,
) -> str:
    """
    Build an enriched analysis prompt that incorporates the spatial intent
    context.  Called by analyst.py instead of the generic prompt.

    Returns the prompt string (not the LLM response — the LLM call happens
    in llm.generate_analysis_stream / generate_analysis).
    """
    count       = stats.get("count", 0)
    geom_types  = stats.get("geometry_types", {})
    names       = stats.get("sample_names", [])
    dist_stats  = stats.get("distance_stats", {})
    area_stats  = stats.get("area_stats", {})
    extra       = {k: v for k, v in stats.items()
                   if k not in ("count", "geometry_types", "sample_names",
                                "distance_stats", "area_stats")}

    lines: list[str] = [
        f'User asked: "{user_query}"',
        f"Spatial intent: {intent.primary_intent.replace('_', ' ')}",
    ]

    if intent.topological_relations:
        lines.append("Spatial relations used: " + ", ".join(intent.topological_relations))
    if intent.directions:
        lines.append("Directional context: " + ", ".join(intent.directions))
    if intent.qualitative_descriptors:
        lines.append("Qualitative context: " + ", ".join(intent.qualitative_descriptors))
    if intent.explicit_distance_m:
        tier = intent.proximity_tier or ""
        lines.append(
            f"Search radius: {intent.explicit_distance_m:,.0f} m "
            f"({tier} — {_tier_description(tier)})"
        )
    elif intent.has_fuzzy_proximity:
        lines.append(
            f"Proximity (fuzzy): default {DEFAULT_PROXIMITY_M:,} m "
            f"({intent.proximity_tier})"
        )
    if intent.named_places:
        lines.append("Reference places: " + ", ".join(intent.named_places))

    lines.append(f"\nResults: {count} feature(s) returned.")
    if geom_types:
        lines.append(f"Geometry types: {geom_types}")
    if names:
        lines.append(f"Sample names: {', '.join(str(n) for n in names[:5])}")
    if dist_stats:
        lines.append(
            f"Distance range: {dist_stats.get('min_m', '?')} m – "
            f"{dist_stats.get('max_m', '?')} m "
            f"(avg {dist_stats.get('avg_m', '?')} m)"
        )
    if area_stats:
        lines.append(
            f"Area stats: largest {area_stats.get('largest_m2', '?')} m², "
            f"total {area_stats.get('total_m2', '?')} m²"
        )
    if extra:
        lines.append(f"Additional stats: {extra}")

    lines.append("\nProvide a spatial interpretation that:")
    lines.append(
        f"1. Explains what was found in the context of the '{intent.primary_intent.replace('_', ' ')}' intent."
    )
    if intent.has_directional:
        lines.append(
            f"2. Comments on the directional aspect ({', '.join(intent.directions)}) "
            "— are the results clustered in that direction?"
        )
    elif dist_stats:
        lines.append(
            "2. Discusses the distance distribution — are results tightly clustered or spread out?"
        )
    else:
        lines.append("2. Highlights any notable spatial patterns or clusters.")
    if intent.has_qualitative:
        lines.append(
            f"3. Interprets the qualitative dimension ({', '.join(intent.qualitative_descriptors)}) "
            "against the actual result data."
        )
    else:
        lines.append("3. Notes any surprising or significant findings.")
    lines.append("Be concise, factual, and spatially specific. 2-4 sentences.")

    return "\n".join(lines)


def _tier_description(tier_label: str) -> str:
    for t in PROXIMITY_TIERS:
        if t.label == tier_label:
            return t.description
    return ""
