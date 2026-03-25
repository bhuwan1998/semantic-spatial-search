"""
LLM Integration Module - Ollama-powered natural language to SpatiaLite SQL.

Translates natural language queries into SpatiaLite SQL using Llama 3.1 8B,
with schema-aware prompting, few-shot examples, and self-correction.
"""

import re

import ollama

from core.geocoder import geocode
from core.validator import SQLValidator, ValidationError


DEVICE_LOCATION_PATTERNS = [
    r"\bnear me\b",
    r"\baround me\b",
    r"\baroud me\b",
    r"\bnearby\b",
    r"\bclose to me\b",
    r"\bnext to me\b",
    r"\bby me\b",
]


SYSTEM_PROMPT = """You are a SpatiaLite SQL query generator for a GeoPackage database containing OpenStreetMap data for Adelaide, South Australia.

CRITICAL RULES:
1. Output ONLY a valid SpatiaLite SELECT query. No markdown, no explanation, no backticks, no commentary.
2. Use ONLY the tables and columns listed in the SCHEMA below.
3. All geometry columns are named "geom" and use SRID 4326 (WGS84 longitude/latitude).
4. Use SpatiaLite syntax, NOT PostGIS:
   - Use MakePoint(longitude, latitude, 4326) to create points (NOTE: longitude first, latitude second)
   - There is NO ST_DWithin function. Use: ST_Distance(a, b) < threshold
   - There is NO <-> operator. Use: ORDER BY ST_Distance(geom, point) ASC LIMIT N
   - Use ST_Distance(geom, point) for distance (returns degrees; multiply by 111320 for approximate meters)
   - For geometry output, ALWAYS use AsGeoJSON(CastAutomagic(geom)) - this is required for GeoPackage databases
   - Use ST_Area(geom) for area, ST_Length(geom) for length
   - Use ST_Buffer(geom, radius_in_degrees) for buffers
   - Use ST_Contains(polygon, point) to test containment
   - Use ST_Intersects(a, b) for intersection tests
5. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, ATTACH, or PRAGMA statements.
6. ALWAYS include a LIMIT clause (default LIMIT 100 unless user specifies a number).
7. For distance in meters, use: ST_Distance(geom, point) * 111320
8. For "nearby" queries without a specific distance, use a 5km radius: ST_Distance(geom, point) * 111320 < 5000
9. When the user says "me", "near me", "around me", misspells it as "aroud me", or says "nearby", that refers to the user's device location. I will provide those device coordinates separately. Use ONLY those provided device coordinates with MakePoint(lng, lat, 4326). NEVER guess, geocode, or substitute another location for these phrases.
10. When user mentions a place name, I will provide coordinates. Use those coordinates with MakePoint(lng, lat, 4326).
11. Always include AsGeoJSON(CastAutomagic(geom)) AS geojson in the SELECT list so results can be mapped.
12. When filtering by name, use LIKE with % wildcards for partial matching and COLLATE NOCASE for case-insensitive search.
13. Column names containing colons (like addr:street, addr:housenumber) MUST be wrapped in double quotes: "addr:street", "addr:housenumber".
14. When users ask about features on a specific street or road, filter using "addr:street" LIKE '%street_name%' COLLATE NOCASE, or match by the name column of the relevant table.
15. In ORDER BY clauses, ALWAYS repeat the full expression (e.g. ORDER BY ST_Distance(geom, MakePoint(...)) ASC). NEVER reference a SELECT alias like distance_meters in ORDER BY -- SQLite may not resolve it.

SCHEMA:
{schema_context}
"""

# Few-shot examples tuned for SpatiaLite syntax and Adelaide context
FEW_SHOT_EXAMPLES = [
    # Pattern 1: Find N nearest points
    {
        "role": "user",
        "content": "Find 5 schools near Adelaide CBD"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, ST_Distance(geom, MakePoint(138.6007, -34.9285, 4326)) * 111320 AS distance_meters, "
            "AsGeoJSON(CastAutomagic(geom)) AS geojson "
            "FROM schools "
            "WHERE name IS NOT NULL "
            "ORDER BY ST_Distance(geom, MakePoint(138.6007, -34.9285, 4326)) ASC "
            "LIMIT 5;"
        )
    },
    # Pattern 2: Proximity / radius search
    {
        "role": "user",
        "content": "Show me restaurants within 2km of Glenelg Beach"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, amenity, ST_Distance(geom, MakePoint(138.5149, -34.9818, 4326)) * 111320 AS distance_meters, "
            "AsGeoJSON(CastAutomagic(geom)) AS geojson "
            "FROM restaurants "
            "WHERE ST_Distance(geom, MakePoint(138.5149, -34.9818, 4326)) * 111320 < 2000 "
            "ORDER BY ST_Distance(geom, MakePoint(138.5149, -34.9818, 4326)) ASC "
            "LIMIT 100;"
        )
    },
    # Pattern 3: Attribute filter on linestrings
    {
        "role": "user",
        "content": "Show all primary roads"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, highway, AsGeoJSON(CastAutomagic(geom)) AS geojson "
            "FROM roads "
            "WHERE highway = 'primary' AND name IS NOT NULL "
            "LIMIT 100;"
        )
    },
    # Pattern 4: Polygon area calculation
    {
        "role": "user",
        "content": "What are the largest parks by area?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, leisure, ST_Area(geom) * 12365000000 AS area_sq_meters, "
            "AsGeoJSON(CastAutomagic(geom)) AS geojson "
            "FROM parks "
            "WHERE name IS NOT NULL "
            "ORDER BY ST_Area(geom) DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 5: Intersection between layers
    {
        "role": "user",
        "content": "Which roads intersect parks?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT r.name AS road_name, r.highway, p.name AS park_name, "
            "AsGeoJSON(CastAutomagic(r.geom)) AS geojson "
            "FROM roads r, parks p "
            "WHERE ST_Intersects(r.geom, p.geom) "
            "AND r.name IS NOT NULL AND p.name IS NOT NULL "
            "LIMIT 50;"
        )
    },
    # Pattern 6: Count / aggregate
    {
        "role": "user",
        "content": "How many restaurants are there by type?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT amenity, COUNT(*) AS count "
            "FROM restaurants "
            "GROUP BY amenity "
            "ORDER BY count DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 7: Search by name
    {
        "role": "user",
        "content": "Find parks with 'creek' in the name"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, leisure, AsGeoJSON(CastAutomagic(geom)) AS geojson "
            "FROM parks "
            "WHERE name LIKE '%creek%' COLLATE NOCASE "
            "LIMIT 50;"
        )
    },
    # Pattern 8: Buffer / containment
    {
        "role": "user",
        "content": "Show me hospitals within 3km of the University of Adelaide"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, ST_Distance(geom, MakePoint(138.6040, -34.9200, 4326)) * 111320 AS distance_meters, "
            "AsGeoJSON(CastAutomagic(geom)) AS geojson "
            "FROM hospitals "
            "WHERE ST_Distance(geom, MakePoint(138.6040, -34.9200, 4326)) * 111320 < 3000 "
            "ORDER BY ST_Distance(geom, MakePoint(138.6040, -34.9200, 4326)) ASC "
            "LIMIT 100;"
        )
    },
]


def query_requires_device_location(query: str) -> bool:
    """Return True when the query refers to the user's current location."""
    normalized_query = query.strip().lower()

    if normalized_query == "me":
        return True

    return any(
        re.search(pattern, normalized_query, re.IGNORECASE)
        for pattern in DEVICE_LOCATION_PATTERNS
    )


def _resolve_location_in_query(query: str) -> tuple[str, tuple[float, float] | None]:
    """
    Try to extract and resolve a location from the user query.
    Returns the (possibly augmented) query and resolved coordinates.
    """
    if query_requires_device_location(query):
        return query, None

    coords = None

    # Try common patterns: "near X", "around X", "in X", "close to X", "within X of Y"
    location_patterns = [
        r"(?:near|nearby|around|close to|next to|by)\s+(.+?)(?:\s*$|\s+(?:that|which|with|and))",
        r"within\s+\d+\s*(?:km|m|meters?|kilometres?|kilometers?|miles?)\s+(?:of|from)\s+(.+?)(?:\s*$|\s+(?:that|which|with|and))",
        r"(?:in|at)\s+(.+?)(?:\s*$|\s+(?:that|which|with|and))",
    ]

    for pattern in location_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            place_name = match.group(1).strip().rstrip("?.,!")
            result = geocode(place_name)
            if result:
                coords = result
                break

    return query, coords


def _build_location_context(
    coords: tuple[float, float] | None,
    *,
    source: str = "place",
) -> str:
    """Build a location context string for the prompt."""
    if coords:
        lat, lng = coords
        location_label = "device location" if source == "device" else "location reference"
        return (
            f"\nUSER LOCATION CONTEXT: The user's {location_label} is "
            f"latitude={lat}, longitude={lng}. "
            f"Use MakePoint({lng}, {lat}, 4326) for this location in the query."
        )
    return ""


def generate_sql(
    user_query: str,
    schema_context: str,
    allowed_tables: set[str],
    device_coords: tuple[float, float] | None = None,
    model: str = "llama3.1:8b",
    max_retries: int = 3,
) -> tuple[str, str | None]:
    """
    Generate a validated SpatiaLite SQL query from natural language.

    Returns (sql, error) tuple. If successful, error is None.
    If all retries fail, sql is empty and error contains the last error message.
    """
    validator = SQLValidator(allowed_tables)

    if query_requires_device_location(user_query) and device_coords is None:
        return "", (
            "This query refers to your current location. "
            "Please share your device location and try again."
        )

    # Resolve location references
    if device_coords is not None:
        query = user_query
        coords = device_coords
        location_context = _build_location_context(coords, source="device")
    else:
        query, coords = _resolve_location_in_query(user_query)
        location_context = _build_location_context(coords)

    # Build messages
    system_content = SYSTEM_PROMPT.format(schema_context=schema_context)
    if location_context:
        system_content += location_context

    messages = [
        {"role": "system", "content": system_content},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": query},
    ]

    last_error = None

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": 0,
                    "seed": 42 + attempt,
                    "num_predict": 1024,
                    "num_ctx": 8192,
                },
                stream=False,
            )
            raw_sql = response["message"]["content"]

            # Validate the generated SQL
            try:
                validated_sql = validator.validate(raw_sql)
                return validated_sql, None
            except ValidationError as e:
                last_error = str(e)
                # Feed the error back for self-correction
                messages.append({"role": "assistant", "content": raw_sql})
                messages.append({
                    "role": "user",
                    "content": (
                        f"ERROR: {last_error}. "
                        f"Please fix the query. Output ONLY valid SpatiaLite SQL, "
                        f"no markdown or explanation. "
                        f"Available tables: {allowed_tables}"
                    ),
                })

        except Exception as e:
            error_str = str(e)
            if "connection refused" in error_str.lower() or "connect" in error_str.lower():
                return "", (
                    "Cannot connect to Ollama. Make sure Ollama is running: "
                    "`ollama serve` and the model is pulled: `ollama pull llama3.1:8b`"
                )
            last_error = f"Ollama error: {error_str}"

    return "", f"Failed after {max_retries} attempts. Last error: {last_error}"
