"""
Cypher Query Generator.

Translates natural-language graph-traversal queries into openCypher statements
that can be executed against the Apache AGE `osm_spatial` property graph.

Handles three graph-native query patterns that PostGIS SQL cannot express well:
  1. graph_traversal  — multi-hop reachability via pre-computed NEAR edges
  2. path_query       — shortest path between two named features
  3. cluster_pattern  — subgraph matching (mutually-NEAR feature clusters)

Architecture
────────────
  generate_cypher(user_query, intent, model)
    │
    ├─ _build_cypher_prompt(intent)   — schema + few-shots + rules
    │
    ├─ Ollama chat call (temperature=0, num_predict=512)
    │
    ├─ _strip_cypher_fences(raw)      — remove ```cypher … ``` if present
    │
    ├─ _validate_cypher(cypher)       — safety: read-only, no DDL/DML
    │
    └─ returns (cypher_str, error_str | None)

The generated Cypher is then executed by cypher_executor.py which also
back-joins fids to PostGIS geometry for map rendering.

Graph schema (osm_spatial)
──────────────────────────
Node labels    : School, Hospital, Restaurant, Pharmacy, Road, Waterway,
                 Railway, Park, Building, Landuse, NaturalFeature, Boundary
Node properties: fid (int), name (str), suburb (str)
Edge types     : NEAR  {distance_m: float}  — ≤500 m between specific label pairs
                 WITHIN                     — feature centroid inside boundary polygon

NEAR pairs (8): School→Park, School→Hospital, School→Restaurant,
                Restaurant→Park, Restaurant→Restaurant,
                Hospital→Pharmacy, Pharmacy→Hospital, Park→Waterway
"""

from __future__ import annotations

import os
import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.spatial_reasoner import SpatialIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph schema context (injected into every Cypher generation prompt)
# ---------------------------------------------------------------------------

_GRAPH_SCHEMA = """\
Graph name: osm_spatial  (Apache AGE openCypher)

Node labels and properties:
  School        {fid: int, name: str, suburb: str}
  Hospital      {fid: int, name: str, suburb: str}
  Restaurant    {fid: int, name: str, suburb: str}
  Pharmacy      {fid: int, name: str, suburb: str}
  Road          {fid: int, name: str, suburb: str}
  Waterway      {fid: int, name: str, suburb: str}
  Railway       {fid: int, name: str, suburb: str}
  Park          {fid: int, name: str, suburb: str}
  Building      {fid: int, name: str, suburb: str}
  Landuse       {fid: int, name: str, suburb: str}
  NaturalFeature{fid: int, name: str, suburb: str}
  Boundary      {fid: int, name: str, suburb: str}

Edge types:
  [:NEAR   {distance_m: float}]  — pre-computed ≤500 m proximity
  [:WITHIN]                       — feature centroid inside boundary polygon

NEAR edge pairs (only these combinations exist):
  (School)     -[:NEAR]-> (Park)
  (School)     -[:NEAR]-> (Hospital)
  (School)     -[:NEAR]-> (Restaurant)
  (Restaurant) -[:NEAR]-> (Park)
  (Restaurant) -[:NEAR]-> (Restaurant)
  (Hospital)   -[:NEAR]-> (Pharmacy)
  (Pharmacy)   -[:NEAR]-> (Hospital)
  (Park)       -[:NEAR]-> (Waterway)

WITHIN edges: all 10 feature labels → Boundary nodes.
"""

# ---------------------------------------------------------------------------
# Few-shot Cypher examples
# ---------------------------------------------------------------------------

_FEW_SHOTS: list[tuple[str, str]] = [
    # ── graph_traversal: 2-hop reachability ────────────────────────────────
    # AGE ORDER BY hard rule (confirmed against live DB):
    #   ORDER BY NEVER resolves aliases — not in RETURN, not in WITH.
    #   You MUST repeat the raw expression in ORDER BY.
    #   e.g.  WITH s.name AS sname  ORDER BY s.name   ← raw expr, not sname
    (
        "Find all amenities reachable within 2 hops of schools near Rundle Mall",
        """\
MATCH (s:School)-[:NEAR*1..2]->(b)
WHERE s.suburb CONTAINS 'Adelaide'
WITH labels(b)[0] AS label, b.name AS bname, b.fid AS fid
ORDER BY labels(b)[0], b.name
RETURN DISTINCT label, bname AS name, fid
LIMIT 50""",
    ),
    (
        "What parks can be reached in 2 hops from a hospital?",
        """\
MATCH (h:Hospital)-[:NEAR*1..2]->(pk:Park)
WITH h.name AS hospital, pk.name AS park, pk.fid AS fid
ORDER BY h.name, pk.name
RETURN DISTINCT hospital, park, fid
LIMIT 50""",
    ),
    (
        "Show all features connected to Norwood schools via NEAR edges",
        """\
MATCH (s:School)-[:NEAR*1..3]->(b)
WHERE s.suburb CONTAINS 'Norwood'
WITH labels(b)[0] AS type, b.name AS bname, b.fid AS fid
ORDER BY labels(b)[0], b.name
RETURN DISTINCT type, bname AS name, fid
LIMIT 50""",
    ),
    # ── path_query: shortest-hop (no shortestPath() in AGE) ───────────────
    # For path length ordering: ORDER BY length(pathvar) — the raw call.
    # LIMIT must come after RETURN, not inside WITH.
    (
        "What is the shortest path between Norwood Primary School and the nearest pharmacy?",
        """\
MATCH rpath = (s:School)-[:NEAR*1..4]-(ph:Pharmacy)
WHERE s.name CONTAINS 'Norwood Primary'
WITH s, ph, length(rpath) AS hops
ORDER BY length(rpath)
RETURN s.name AS start_name, s.fid AS school_fid,
       ph.name AS end_name, ph.fid AS pharmacy_fid,
       hops
LIMIT 1""",
    ),
    (
        "Find the shortest graph path between schools and pharmacies",
        """\
MATCH (s:School)-[:NEAR*1..3]->(ph:Pharmacy)
WITH s, ph, count(*) AS path_count
ORDER BY count(*) DESC
RETURN s.name AS from_school, s.fid AS school_fid,
       ph.name AS to_pharmacy, ph.fid AS pharmacy_fid,
       path_count
LIMIT 10""",
    ),
    # ── path via intermediate node (the benchmark query) ──────────────────
    # School→Park exists but Park→School does NOT. For "two schools via a shared park"
    # use: (s1)-[:NEAR]->(pk:Park)<-[:NEAR]-(s2) — both point INTO the park.
    (
        "What is the shortest path between two schools via a connected park?",
        """\
MATCH (s1:School)-[:NEAR]->(pk:Park)<-[:NEAR]-(s2:School)
WHERE s1.fid <> s2.fid
RETURN s1.name AS school1, s1.fid AS fid1,
       pk.name AS park, pk.fid AS park_fid,
       s2.name AS school2, s2.fid AS fid2
LIMIT 10""",
    ),
    # ── cluster_pattern: mutually-NEAR subgraph ────────────────────────────
    (
        "Find clusters where a school, hospital and park are all mutually near each other",
        """\
MATCH (s:School)-[:NEAR]->(h:Hospital),
      (s)-[:NEAR]->(pk:Park),
      (h)-[:NEAR]->(pk2:Park)
WHERE pk.fid = pk2.fid
RETURN s.name AS school, h.name AS hospital, pk.name AS park,
       s.fid AS school_fid, h.fid AS hospital_fid, pk.fid AS park_fid
LIMIT 20""",
    ),
    (
        "Find co-located groups of restaurant, park and waterway",
        """\
MATCH (r:Restaurant)-[:NEAR]->(pk:Park)-[:NEAR]->(w:Waterway)
RETURN r.name AS restaurant, pk.name AS park, w.name AS waterway,
       r.fid AS restaurant_fid, pk.fid AS park_fid, w.fid AS waterway_fid
LIMIT 20""",
    ),
    # ── within + traversal combo ───────────────────────────────────────────
    (
        "Which suburbs are transitively connected to Norwood via shared feature boundaries?",
        """\
MATCH (f)-[:WITHIN]->(b1:Boundary {name: 'Norwood'}),
      (f)-[:WITHIN]->(b2:Boundary)
WHERE b2.name <> 'Norwood'
WITH b2.name AS connected_suburb, count(f) AS shared_features
ORDER BY count(f) DESC
RETURN DISTINCT connected_suburb, shared_features
LIMIT 20""",
    ),
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are an openCypher query generator for the Apache AGE graph database `osm_spatial`.
This graph contains Adelaide (South Australia) OpenStreetMap features as nodes and
spatial relationships as edges.

{_GRAPH_SCHEMA}

RULES — follow exactly:
1. Output ONLY a valid openCypher MATCH/RETURN statement. No markdown, no explanation.
2. Never use CREATE, MERGE, DELETE, SET, DETACH, REMOVE, DROP — read-only only.
3. Always include LIMIT (max 100) at the end of the query.
4. AGE DOES NOT support: shortestPath(), allShortestPaths(), list comprehensions
   [x IN list | expr], ANY(), ALL(), NONE(), SINGLE(), EXISTS {{...}} subqueries.
   Do not use them. For path queries use variable-length edges -[:NEAR*1..4]-
   with ORDER BY length(pathvar).
5. Variable-length hops: [:NEAR*1..3] — never exceed 4 hops (performance).
6. For cluster patterns, match each edge explicitly (no variable-length paths).
7. Always RETURN node fids so geometry can be joined from PostGIS after execution.
8. Name columns clearly: label/type, name, fid — plus any edge properties needed.
9. If the query mentions a specific place name, filter by n.name CONTAINS 'Name'
   (case-sensitive, exact substring) or n.suburb CONTAINS 'Suburb'.
10. NEAR edges are directed — always use -[:NEAR]-> not <-[:NEAR]- unless the
    pair exists in reverse (e.g. Hospital→Pharmacy AND Pharmacy→Hospital both exist).
    For "two X connected via a shared Y" where only X→Y exists (not Y→X), use:
      (x1)-[:NEAR]->(y)<-[:NEAR]-(x2)  NOT  (x1)-[:NEAR]->(y)-[:NEAR]->(x2)
11. Supported list functions: nodes(p), relationships(p), labels(n), keys(n), range().
    Supported predicates: exists(n.property), exists((pattern)).
    Everything else from standard Cypher is NOT available in AGE.
12. CRITICAL — variable name collision: if you use a path variable (MATCH rpath = ...)
    do NOT reuse that same name as a node variable (rpath:Park). Use distinct names:
    path variable → use `rpath` or `mpath`; Park nodes → use `pk`.
13. CRITICAL — ORDER BY NEVER resolves aliases (confirmed against live AGE).
    This applies everywhere: after RETURN, after WITH, everywhere.
    You MUST repeat the raw expression in ORDER BY, not the alias name.
    CORRECT:   WITH s.name AS sname  ORDER BY s.name    ← raw property
    CORRECT:   WITH length(rpath) AS hops  ORDER BY length(rpath)  ← raw call
    WRONG:     WITH s.name AS sname  ORDER BY sname     ← alias, will FAIL
    WRONG:     RETURN x AS hops  ORDER BY hops           ← alias, will FAIL
    Structure for sorted queries:
      MATCH ...
      WITH raw_expr1 AS col1, raw_expr2 AS col2
      ORDER BY raw_expr1 [DESC]
      RETURN col1, col2
      LIMIT n
"""


# ---------------------------------------------------------------------------
# Safety validator
# ---------------------------------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET\s+\w|DETACH\s+DELETE|REMOVE|DROP)\b",
    re.IGNORECASE,
)

# AGE-unsupported Cypher constructs — catch before execution to allow self-correction
_AGE_UNSUPPORTED = re.compile(
    r"\b(shortestPath|allShortestPaths|ANY\s*\(|ALL\s*\(|NONE\s*\(|SINGLE\s*\()\b"
    r"|\[\s*\w+\s+IN\s+"        # list comprehension [x IN <expr> | ...]
    r"|EXISTS\s*\{",            # EXISTS { MATCH ... } subquery syntax
    re.IGNORECASE,
)


def _check_path_node_collision(cypher: str) -> str | None:
    """
    Detect the pattern  MATCH <var> = (...)-[...]->(<var>:Label)
    where the same identifier is used as both a path variable and a node variable.
    e.g.  MATCH p = (s:School)-[:NEAR]->(p:Park)  ← p is both path and node
    """
    path_vars = set(re.findall(r'\bMATCH\s+(\w+)\s*=', cypher, re.IGNORECASE))
    if not path_vars:
        return None
    # Only match labelled nodes (var:Label) to avoid false positives on length(p)
    node_vars = set(re.findall(r'\((\w+)\s*:', cypher))
    collision = path_vars & node_vars
    if collision:
        name = next(iter(collision))
        return (
            f"Variable name collision: '{name}' is used as both a path variable "
            f"(MATCH {name} = ...) and a node variable (...({name}:Label)...). "
            f"Rename the path variable to 'rpath' or 'mpath', and use 'pk' for Park nodes."
        )
    return None


def _check_orderby_alias(cypher: str) -> str | None:
    """
    AGE never resolves aliases in ORDER BY — not after RETURN, not after WITH.
    Confirmed against live DB: ORDER BY alias always fails with 'could not find rte'.
    Detect any ORDER BY that references a name introduced by AS anywhere in the query.
    """
    text = re.sub(r'\s+', ' ', cypher.strip())

    # Collect ALL aliases defined anywhere in the query via AS
    all_aliases = set(re.findall(r'\bAS\s+(\w+)', text, re.IGNORECASE))
    if not all_aliases:
        return None

    # Find every ORDER BY block and check its terms against known aliases.
    # Strip dotted property refs (n.name → just 'n') so that ORDER BY b.name
    # is not flagged when 'name' happens to be an alias elsewhere.
    for orb in re.finditer(r'\bORDER\s+BY\s+(.*?)(?=\bRETURN\b|\bLIMIT\b|\bWITH\b|$)',
                           text, re.IGNORECASE):
        order_body = orb.group(1)
        # Remove dotted accesses so 'b.name' becomes 'b', not 'b' + 'name'
        order_body_no_dots = re.sub(r'\b\w+\.\w+', '', order_body)
        order_terms = set(re.findall(r'\b([A-Za-z_]\w*)\b', order_body_no_dots))
        order_terms -= {'ASC', 'DESC', 'ASCENDING', 'DESCENDING', 'NULL'}
        bad = all_aliases & order_terms
        if bad:
            return (
                f"ORDER BY uses alias(es) {bad} — AGE never resolves aliases in ORDER BY "
                f"(confirmed: 'could not find rte'). "
                f"Repeat the raw expression instead:\n"
                f"  CORRECT: WITH length(rpath) AS hops  ORDER BY length(rpath)\n"
                f"  WRONG:   WITH length(rpath) AS hops  ORDER BY hops"
            )
    return None


def _validate_cypher(cypher: str) -> str | None:
    """
    Return an error message if the Cypher is unsafe or uses AGE-unsupported
    constructs, else None.
    """
    if _FORBIDDEN.search(cypher):
        return "Cypher contains forbidden write operation (CREATE/MERGE/DELETE/SET/REMOVE/DROP)."
    if _AGE_UNSUPPORTED.search(cypher):
        return (
            "Cypher uses constructs not supported by Apache AGE: "
            "shortestPath(), allShortestPaths(), ANY(), ALL(), NONE(), SINGLE(), "
            "EXISTS { MATCH } subqueries, or list comprehensions [x IN list | expr]. "
            "Use variable-length edges -[:NEAR*1..4]- with ORDER BY length(rpath) instead."
        )
    collision_err = _check_path_node_collision(cypher)
    if collision_err:
        return collision_err
    alias_err = _check_orderby_alias(cypher)
    if alias_err:
        return alias_err
    if ";" in cypher and cypher.index(";") < len(cypher) - 1:
        return "Multiple Cypher statements not allowed."
    return None


def _strip_cypher_fences(raw: str) -> str:
    """Remove ```cypher … ``` or ``` … ``` markdown fences."""
    raw = re.sub(r"^```(?:cypher)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw.strip())
    return raw.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cypher(
    user_query: str,
    intent: "SpatialIntent",
    model: str | None = None,
    max_retries: int = 2,
) -> tuple[str, str | None]:
    """
    Generate an openCypher query from a natural-language user query.

    Args:
        user_query:  The original NL question from the user.
        intent:      SpatialIntent produced by decompose_intent(); used to
                     add extra context (intent type, named places, feature types).
        model:       Ollama model name; defaults to $OLLAMA_MODEL env var.
        max_retries: Self-correction loop limit on validation failure.

    Returns:
        (cypher, None)       on success
        ("",    error_str)   after all retries exhausted
    """
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        import ollama as _ollama
        client = _ollama.Client(host=ollama_host)
    except Exception as exc:
        return "", f"Ollama client unavailable: {exc}"

    # Build the initial messages list
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject few-shot examples
    for user_ex, cypher_ex in _FEW_SHOTS:
        messages.append({"role": "user",      "content": user_ex})
        messages.append({"role": "assistant", "content": cypher_ex})

    # Build the enriched user message with intent context
    intent_context = _build_intent_context(intent)
    user_message = f"{intent_context}\n\n{user_query}" if intent_context else user_query
    messages.append({"role": "user", "content": user_message})

    last_error: str = "Unknown error"
    for attempt in range(max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": 0,
                    "seed": 42 + attempt,
                    "num_predict": 512,
                    "num_ctx": 16384,
                },
                stream=False,
            )
            raw = response["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Cypher generation Ollama call failed: %s", exc)
            return "", f"LLM call failed: {exc}"

        cypher = _strip_cypher_fences(raw)
        err = _validate_cypher(cypher)

        if err is None:
            logger.debug("Cypher generated (attempt %d):\n%s", attempt, cypher)
            return cypher, None

        last_error = err
        logger.warning("Cypher validation failed (attempt %d): %s", attempt, err)

        # Self-correction: append error and ask LLM to fix
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                f"The Cypher you generated is invalid: {err}\n"
                "Please rewrite it following the rules strictly. "
                "Output ONLY the corrected Cypher query."
            ),
        })

    return "", f"Cypher generation failed after {max_retries + 1} attempts: {last_error}"


def _build_intent_context(intent: "SpatialIntent") -> str:
    """
    Build a compact [GRAPH QUERY CONTEXT] block to prepend to the user message,
    giving the Cypher generator explicit hints from the spatial intent.
    """
    lines: list[str] = ["[GRAPH QUERY CONTEXT]"]
    lines.append(f"Intent: {intent.primary_intent.replace('_', ' ')}")

    if intent.feature_types:
        # Map generic feature type names to AGE node labels (capitalised)
        _LABEL_MAP = {
            "school": "School", "hospital": "Hospital",
            "restaurant": "Restaurant", "pharmacy": "Pharmacy",
            "park": "Park", "road": "Road", "waterway": "Waterway",
            "railway": "Railway", "building": "Building",
            "landuse": "Landuse", "boundary": "Boundary",
        }
        labels = [_LABEL_MAP.get(ft, ft.title()) for ft in intent.feature_types]
        lines.append(f"Node labels involved: {', '.join(labels)}")

    if intent.named_places:
        lines.append(f"Named places to filter on: {', '.join(intent.named_places)}")

    if intent.explicit_distance_m is not None:
        lines.append(
            f"Note: NEAR edges are pre-computed at ≤500 m. "
            f"User asked for {intent.explicit_distance_m:,.0f} m — "
            f"use hop count to approximate wider reach."
        )

    if intent.primary_intent == "graph_traversal":
        lines.append("Use variable-length NEAR edges: -[:NEAR*1..2]-> or -[:NEAR*1..3]->")
    elif intent.primary_intent == "path_query":
        lines.append(
            "For path queries: use MATCH p = (a)-[:NEAR*1..4]-(b) "
            "then ORDER BY length(p) LIMIT 1. "
            "Do NOT use shortestPath() — it is not supported in Apache AGE."
        )
    elif intent.primary_intent == "cluster_pattern":
        lines.append(
            "Match each edge explicitly — do NOT use variable-length paths for clusters."
        )

    lines.append("[/GRAPH QUERY CONTEXT]")
    return "\n".join(lines)
