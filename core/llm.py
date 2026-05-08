"""
LLM Integration Module - Ollama-powered natural language to PostGIS SQL.

Translates natural language queries into PostGIS SQL using a local Ollama model,
with schema-aware prompting (Hybrid RAG), few-shot examples, and self-correction.

DESIGN PRINCIPLE: All place name resolution happens INSIDE PostGIS via subqueries
against osm_boundaries and osm_all. No geocoder, no hardcoded coordinates, no
Nominatim calls at runtime. The only coordinates injected into the prompt are
device GPS coordinates (genuinely from the browser, cannot be resolved another way).
"""

import os
import re

import ollama

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


SYSTEM_PROMPT = """You are a PostGIS SQL query generator for a PostgreSQL database containing OpenStreetMap data for Adelaide, South Australia.
You are equipped with semantic spatial reasoning — you understand proximity tiers (walking distance ≤800 m, nearby ≤3 km),
directional relations (north/south/east/west via ST_Azimuth), topological relations (within, intersects, touches),
and qualitative descriptors (large park → area > 50000 m², dense → many per km², walkable → ≤800 m).
When a [SPATIAL REASONING CONTEXT] block is provided in the user message, USE it to select the correct PostGIS operators,
distance thresholds, and query patterns.

CRITICAL RULES:
1. Output ONLY a valid PostGIS SELECT query. No markdown, no explanation, no backticks, no commentary.
2. Use ONLY the tables and columns listed in the SCHEMA below.
3. All geometry columns are named "geometry" and use SRID 4326 (WGS84 longitude/latitude).
4. Use PostGIS syntax:
   - ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) to create points from known coords
   - ST_DWithin(a::geography, b::geography, distance_m) for proximity filters (uses spatial index)
   - ST_Distance(a::geography, b::geography) returns metres directly
   - ORDER BY geometry <-> point_expr LIMIT N for nearest-neighbour sort
   - ST_AsGeoJSON(geometry) AS geojson — ALWAYS include this in SELECT so results can be mapped
     Even for count/comparison queries, use UNION ALL to return the actual geometries alongside counts
   - ST_Area(geometry::geography) for area in square metres
   - ST_Length(geometry::geography) for length in metres
   - ILIKE '%term%' for case-insensitive name filtering
5. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, COPY, or EXECUTE statements.
6. ALWAYS include a LIMIT clause (default LIMIT 100 unless user specifies a number).
7. NEVER invent table names. The ONLY tables that exist are the osm_* tables listed in SCHEMA below.
   Do NOT use tables named: boundary, suburb, area, place, location, region, feature, layer, or any
   other name not prefixed with osm_. For suburb/boundary geometry always use osm_boundaries.
8. When using CTEs (WITH ... AS), you may reference the CTE name inside the main SELECT/FROM — that
   is fine. But NEVER use a bare word like "boundary" or "suburb" as a FROM target unless it is
   defined as a CTE in the same query.

PLACE NAME RESOLUTION — THIS IS THE MOST IMPORTANT RULE:
- NEVER hardcode latitude/longitude values for named places (suburbs, landmarks, beaches, parks, etc.)
- ALL place names must be resolved at query time using a PostGIS subquery against the database:
    (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%place name%' LIMIT 1)
  or for suburb/boundary lookups:
    (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%suburb name%' LIMIT 1)
- The only exception is when I explicitly provide device GPS coordinates in the prompt below.
  In that case, use ST_SetSRID(ST_MakePoint(lng, lat), 4326) with the exact provided values.

DISTANCE PATTERN FOR NAMED PLACES:
    WHERE ST_DWithin(
        f.geometry::geography,
        (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%place%' LIMIT 1)::geography,
        distance_m
    )
    ORDER BY f.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%place%' LIMIT 1)
    LIMIT N

CONTAINMENT PATTERN (features inside a suburb):
    -- Always use ORDER BY admin_level DESC to prefer the suburb polygon over larger council/region boundaries
    WHERE ST_Within(
        f.geometry,
        (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%suburb%' ORDER BY admin_level DESC LIMIT 1)
    )

COMPARISON PATTERN (count two feature types in the same suburb — geometry rows first so the map renders, count rows at end):
    SELECT 'cafe' AS feature_type, NULL AS count, ST_AsGeoJSON(geometry) AS geojson
    FROM osm_restaurants
    WHERE amenity = 'cafe'
      AND ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%suburb%' ORDER BY admin_level DESC LIMIT 1))
    UNION ALL
    SELECT 'park' AS feature_type, NULL AS count, ST_AsGeoJSON(geometry) AS geojson
    FROM osm_parks
    WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%suburb%' ORDER BY admin_level DESC LIMIT 1))
    UNION ALL
    SELECT 'cafes_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson
    FROM osm_restaurants
    WHERE amenity = 'cafe'
      AND ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%suburb%' ORDER BY admin_level DESC LIMIT 1))
    UNION ALL
    SELECT 'parks_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson
    FROM osm_parks
    WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%suburb%' ORDER BY admin_level DESC LIMIT 1))
    LIMIT 200;

SCHEMA:
{schema_context}
"""


FEW_SHOT_EXAMPLES = [
    # Pattern 1: Nearest N — place resolved via osm_all subquery
    {
        "role": "user",
        "content": "Find 5 schools near Adelaide CBD"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT s.name, "
            "ROUND(ST_Distance(s.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Adelaide CBD%' LIMIT 1)::geography)) AS distance_meters, "
            "ST_AsGeoJSON(s.geometry) AS geojson "
            "FROM osm_schools s "
            "WHERE s.name IS NOT NULL "
            "ORDER BY s.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Adelaide CBD%' LIMIT 1) "
            "LIMIT 5;"
        )
    },
    # Pattern 2: Radius search — place resolved via osm_all subquery
    {
        "role": "user",
        "content": "Show me restaurants within 2km of Glenelg Beach"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT r.name, r.amenity, r.cuisine, "
            "ROUND(ST_Distance(r.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg%' LIMIT 1)::geography)) AS distance_meters, "
            "ST_AsGeoJSON(r.geometry) AS geojson "
            "FROM osm_restaurants r "
            "WHERE ST_DWithin("
            "r.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg%' LIMIT 1)::geography, "
            "2000) "
            "ORDER BY r.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg%' LIMIT 1) "
            "LIMIT 100;"
        )
    },
    # Pattern 3: Features inside a suburb — osm_boundaries subquery
    {
        "role": "user",
        "content": "Show all pharmacies in Norwood"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT p.name, p.amenity, ST_AsGeoJSON(p.geometry) AS geojson "
            "FROM osm_pharmacies p "
            "WHERE ST_Within("
            "p.geometry, "
            "(SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)"
            ") "
            "LIMIT 100;"
        )
    },
    # Pattern 4: Attribute filter on linestrings
    {
        "role": "user",
        "content": "Show all primary roads"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, highway, ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_roads "
            "WHERE highway = 'primary' AND name IS NOT NULL "
            "LIMIT 100;"
        )
    },
    # Pattern 5: Polygon area calculation
    {
        "role": "user",
        "content": "What are the largest parks by area?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, leisure, "
            "ROUND(ST_Area(geometry::geography)) AS area_sq_meters, "
            "ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_parks "
            "WHERE name IS NOT NULL "
            "ORDER BY ST_Area(geometry::geography) DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 6: Cross-layer intersection
    {
        "role": "user",
        "content": "Which roads intersect parks?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT r.name AS road_name, r.highway, p.name AS park_name, "
            "ST_AsGeoJSON(r.geometry) AS geojson "
            "FROM osm_roads r "
            "JOIN osm_parks p ON ST_Intersects(r.geometry, p.geometry) "
            "WHERE r.name IS NOT NULL AND p.name IS NOT NULL "
            "LIMIT 50;"
        )
    },
    # Pattern 7: Count / aggregate
    {
        "role": "user",
        "content": "How many restaurants are there by type?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT amenity, COUNT(*) AS count "
            "FROM osm_restaurants "
            "GROUP BY amenity "
            "ORDER BY count DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 8: Name search
    {
        "role": "user",
        "content": "Find parks with 'creek' in the name"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, leisure, ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_parks "
            "WHERE name ILIKE '%creek%' "
            "LIMIT 50;"
        )
    },
    # Pattern 9: Near me — device GPS injected by app
    {
        "role": "user",
        "content": "Find hospitals near me"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, "
            "ROUND(ST_Distance(geometry::geography, ST_SetSRID(ST_MakePoint({lng}, {lat}), 4326)::geography)) AS distance_meters, "
            "ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_hospitals "
            "WHERE ST_DWithin(geometry::geography, ST_SetSRID(ST_MakePoint({lng}, {lat}), 4326)::geography, 5000) "
            "ORDER BY geometry <-> ST_SetSRID(ST_MakePoint({lng}, {lat}), 4326) "
            "LIMIT 20;"
        )
    },
    # Pattern 10: Compare two feature types inside a suburb — UNION ALL with geometries for mapping
    {
        "role": "user",
        "content": "How many cafes are there in Unley compared to parks"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT 'cafe' AS feature_type, NULL AS count, ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_restaurants "
            "WHERE amenity = 'cafe' "
            "AND ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Unley%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'park' AS feature_type, NULL AS count, ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_parks "
            "WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Unley%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'cafes_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson "
            "FROM osm_restaurants "
            "WHERE amenity = 'cafe' "
            "AND ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Unley%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'parks_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson "
            "FROM osm_parks "
            "WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Unley%' ORDER BY admin_level DESC LIMIT 1)) "
            "LIMIT 200;"
        )
    },
    # Pattern 11: Radius from a named landmark resolved via DB
    {
        "role": "user",
        "content": "Show me schools within 3km of the University of Adelaide"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT s.name, "
            "ROUND(ST_Distance(s.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%University of Adelaide%' LIMIT 1)::geography)) AS distance_meters, "
            "ST_AsGeoJSON(s.geometry) AS geojson "
            "FROM osm_schools s "
            "WHERE ST_DWithin("
            "s.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%University of Adelaide%' LIMIT 1)::geography, "
            "3000) "
            "ORDER BY s.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%University of Adelaide%' LIMIT 1) "
            "LIMIT 100;"
        )
    },
    # Pattern 12: Restaurant density per suburb — normalised by area (km²)
    {
        "role": "user",
        "content": "Which suburbs have the most restaurants per square kilometre?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT b.name AS suburb, "
            "COUNT(r.id) AS restaurant_count, "
            "ROUND(ST_Area(b.geometry::geography)::numeric / 1000000, 4) AS area_km2, "
            "ROUND((COUNT(r.id) / (ST_Area(b.geometry::geography) / 1000000))::numeric, 2) AS restaurants_per_km2, "
            "ST_AsGeoJSON(b.geometry) AS geojson "
            "FROM osm_boundaries b "
            "LEFT JOIN osm_restaurants r ON ST_Within(r.geometry, b.geometry) "
            "WHERE b.admin_level = '9' "
            "GROUP BY b.id, b.name, b.geometry "
            "HAVING COUNT(r.id) > 0 "
            "ORDER BY restaurants_per_km2 DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 13: Multi-type count in one suburb — schools, hospitals, pharmacies
    {
        "role": "user",
        "content": "How many schools, hospitals and pharmacies are in Norwood?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT 'school' AS feature_type, NULL AS count, ST_AsGeoJSON(s.geometry) AS geojson "
            "FROM osm_schools s "
            "WHERE ST_Within(s.geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'hospital' AS feature_type, NULL AS count, ST_AsGeoJSON(h.geometry) AS geojson "
            "FROM osm_hospitals h "
            "WHERE ST_Within(h.geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'pharmacy' AS feature_type, NULL AS count, ST_AsGeoJSON(p.geometry) AS geojson "
            "FROM osm_pharmacies p "
            "WHERE ST_Within(p.geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'schools_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson "
            "FROM osm_schools "
            "WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'hospitals_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson "
            "FROM osm_hospitals "
            "WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)) "
            "UNION ALL "
            "SELECT 'pharmacies_total' AS feature_type, COUNT(*) AS count, NULL::text AS geojson "
            "FROM osm_pharmacies "
            "WHERE ST_Within(geometry, (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1)) "
            "LIMIT 200;"
        )
    },
    # Pattern 14: Hospitals with no pharmacy within 1km — NOT EXISTS proximity gap
    {
        "role": "user",
        "content": "Which hospitals have no pharmacy within 1km?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT h.name, ST_AsGeoJSON(h.geometry) AS geojson "
            "FROM osm_hospitals h "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM osm_pharmacies p "
            "WHERE ST_DWithin(h.geometry::geography, p.geometry::geography, 1000)"
            ") "
            "AND h.name IS NOT NULL "
            "ORDER BY h.name "
            "LIMIT 50;"
        )
    },
    # Pattern 15: Landuse types inside a suburb
    {
        "role": "user",
        "content": "What landuse types are present in Glenelg?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT l.landuse, COUNT(*) AS count, "
            "ROUND(SUM(ST_Area(ST_Intersection(l.geometry, b.geometry)::geography))::numeric, 0) AS total_area_m2, "
            "ST_AsGeoJSON(l.geometry) AS geojson "
            "FROM osm_landuse l "
            "JOIN osm_boundaries b ON ST_Intersects(l.geometry, b.geometry) "
            "WHERE b.name ILIKE '%Glenelg%' AND b.admin_level = '9' "
            "GROUP BY l.landuse, l.geometry "
            "ORDER BY total_area_m2 DESC "
            "LIMIT 100;"
        )
    },
    # Pattern 16: Most isolated schools — largest minimum distance to any other school
    {
        "role": "user",
        "content": "Which schools are most isolated — furthest from any other school?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT a.name, "
            "ROUND(MIN(ST_Distance(a.geometry::geography, b.geometry::geography))::numeric) AS nearest_school_m, "
            "ST_AsGeoJSON(a.geometry) AS geojson "
            "FROM osm_schools a "
            "JOIN osm_schools b ON a.id <> b.id "
            "WHERE a.name IS NOT NULL "
            "GROUP BY a.id, a.name, a.geometry "
            "ORDER BY nearest_school_m DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 17: Largest residential landuse areas
    {
        "role": "user",
        "content": "What are the 10 largest residential landuse areas?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, landuse, "
            "ROUND(ST_Area(geometry::geography)::numeric) AS area_m2, "
            "ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_landuse "
            "WHERE landuse = 'residential' "
            "ORDER BY ST_Area(geometry::geography) DESC "
            "LIMIT 10;"
        )
    },
    # Pattern 18a-d: Green vs concrete ratio — multiple phrasing variants, same SQL
    # Returns clipped park polygons (green) and building polygons (concrete) for mapping
    # plus summary stat rows so metric cards appear. Uses UNION ALL pattern.
    {
        "role": "user",
        "content": "What is the ratio of green space to building coverage in the Adelaide CBD?"
    },
    {
        "role": "assistant",
        "content": (
            "WITH suburb AS ( "
            "SELECT geometry FROM osm_boundaries "
            "WHERE name ILIKE '%Adelaide%' AND admin_level = '9' ORDER BY admin_level DESC LIMIT 1 "
            "), "
            "stats AS ( "
            "SELECT "
            "ST_Area((SELECT geometry FROM suburb)::geography) AS total_m2, "
            "COALESCE(SUM(CASE WHEN src='park' THEN area ELSE 0 END), 0) AS green_m2, "
            "COALESCE(SUM(CASE WHEN src='building' THEN area ELSE 0 END), 0) AS concrete_m2 "
            "FROM ("
            "SELECT 'park' AS src, ST_Area(ST_Intersection(p.geometry, s.geometry)::geography) AS area "
            "FROM osm_parks p, suburb s WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS src, ST_Area(ST_Intersection(b.geometry, s.geometry)::geography) AS area "
            "FROM osm_buildings b, suburb s WHERE ST_Intersects(b.geometry, s.geometry)"
            ") areas "
            ") "
            "SELECT 'park' AS feature_type, NULL::numeric AS count, "
            "p.name, "
            "ROUND(ST_Area(ST_Intersection(p.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(p.geometry, s.geometry)) AS geojson "
            "FROM osm_parks p, suburb s, stats st "
            "WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS feature_type, NULL::numeric AS count, "
            "b.name, "
            "ROUND(ST_Area(ST_Intersection(b.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(b.geometry, s.geometry)) AS geojson "
            "FROM osm_buildings b, suburb s, stats st "
            "WHERE ST_Intersects(b.geometry, s.geometry) "
            "UNION ALL "
            "SELECT 'green_area_m2' AS feature_type, ROUND(st.green_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson "
            "FROM stats st "
            "UNION ALL "
            "SELECT 'building_area_m2' AS feature_type, ROUND(st.concrete_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson "
            "FROM stats st "
            "UNION ALL "
            "SELECT 'green_to_concrete_ratio' AS feature_type, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson "
            "FROM stats st "
            "LIMIT 500;"
        )
    },
    {
        "role": "user",
        "content": "Find the ratio of buildings to green spaces in Adelaide CBD"
    },
    {
        "role": "assistant",
        "content": (
            "WITH suburb AS ( "
            "SELECT geometry FROM osm_boundaries "
            "WHERE name ILIKE '%Adelaide%' AND admin_level = '9' ORDER BY admin_level DESC LIMIT 1 "
            "), "
            "stats AS ( "
            "SELECT "
            "ST_Area((SELECT geometry FROM suburb)::geography) AS total_m2, "
            "COALESCE(SUM(CASE WHEN src='park' THEN area ELSE 0 END), 0) AS green_m2, "
            "COALESCE(SUM(CASE WHEN src='building' THEN area ELSE 0 END), 0) AS concrete_m2 "
            "FROM ("
            "SELECT 'park' AS src, ST_Area(ST_Intersection(p.geometry, s.geometry)::geography) AS area "
            "FROM osm_parks p, suburb s WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS src, ST_Area(ST_Intersection(b.geometry, s.geometry)::geography) AS area "
            "FROM osm_buildings b, suburb s WHERE ST_Intersects(b.geometry, s.geometry)"
            ") areas "
            ") "
            "SELECT 'park' AS feature_type, NULL::numeric AS count, "
            "p.name, "
            "ROUND(ST_Area(ST_Intersection(p.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(p.geometry, s.geometry)) AS geojson "
            "FROM osm_parks p, suburb s, stats st "
            "WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS feature_type, NULL::numeric AS count, "
            "b.name, "
            "ROUND(ST_Area(ST_Intersection(b.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(b.geometry, s.geometry)) AS geojson "
            "FROM osm_buildings b, suburb s, stats st "
            "WHERE ST_Intersects(b.geometry, s.geometry) "
            "UNION ALL "
            "SELECT 'green_area_m2' AS feature_type, ROUND(st.green_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "UNION ALL "
            "SELECT 'building_area_m2' AS feature_type, ROUND(st.concrete_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "UNION ALL "
            "SELECT 'green_to_concrete_ratio' AS feature_type, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "LIMIT 500;"
        )
    },
    {
        "role": "user",
        "content": "Compare green areas vs buildings in the Adelaide CBD"
    },
    {
        "role": "assistant",
        "content": (
            "WITH suburb AS ( "
            "SELECT geometry FROM osm_boundaries "
            "WHERE name ILIKE '%Adelaide%' AND admin_level = '9' ORDER BY admin_level DESC LIMIT 1 "
            "), "
            "stats AS ( "
            "SELECT "
            "ST_Area((SELECT geometry FROM suburb)::geography) AS total_m2, "
            "COALESCE(SUM(CASE WHEN src='park' THEN area ELSE 0 END), 0) AS green_m2, "
            "COALESCE(SUM(CASE WHEN src='building' THEN area ELSE 0 END), 0) AS concrete_m2 "
            "FROM ("
            "SELECT 'park' AS src, ST_Area(ST_Intersection(p.geometry, s.geometry)::geography) AS area "
            "FROM osm_parks p, suburb s WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS src, ST_Area(ST_Intersection(b.geometry, s.geometry)::geography) AS area "
            "FROM osm_buildings b, suburb s WHERE ST_Intersects(b.geometry, s.geometry)"
            ") areas "
            ") "
            "SELECT 'park' AS feature_type, NULL::numeric AS count, "
            "p.name, "
            "ROUND(ST_Area(ST_Intersection(p.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(p.geometry, s.geometry)) AS geojson "
            "FROM osm_parks p, suburb s, stats st "
            "WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS feature_type, NULL::numeric AS count, "
            "b.name, "
            "ROUND(ST_Area(ST_Intersection(b.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(b.geometry, s.geometry)) AS geojson "
            "FROM osm_buildings b, suburb s, stats st "
            "WHERE ST_Intersects(b.geometry, s.geometry) "
            "UNION ALL "
            "SELECT 'green_area_m2' AS feature_type, ROUND(st.green_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "UNION ALL "
            "SELECT 'building_area_m2' AS feature_type, ROUND(st.concrete_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "UNION ALL "
            "SELECT 'green_to_concrete_ratio' AS feature_type, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "LIMIT 500;"
        )
    },
    {
        "role": "user",
        "content": "How much of Unley is parks vs buildings?"
    },
    {
        "role": "assistant",
        "content": (
            "WITH suburb AS ( "
            "SELECT geometry FROM osm_boundaries "
            "WHERE name ILIKE '%Unley%' AND admin_level = '9' ORDER BY admin_level DESC LIMIT 1 "
            "), "
            "stats AS ( "
            "SELECT "
            "ST_Area((SELECT geometry FROM suburb)::geography) AS total_m2, "
            "COALESCE(SUM(CASE WHEN src='park' THEN area ELSE 0 END), 0) AS green_m2, "
            "COALESCE(SUM(CASE WHEN src='building' THEN area ELSE 0 END), 0) AS concrete_m2 "
            "FROM ("
            "SELECT 'park' AS src, ST_Area(ST_Intersection(p.geometry, s.geometry)::geography) AS area "
            "FROM osm_parks p, suburb s WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS src, ST_Area(ST_Intersection(b.geometry, s.geometry)::geography) AS area "
            "FROM osm_buildings b, suburb s WHERE ST_Intersects(b.geometry, s.geometry)"
            ") areas "
            ") "
            "SELECT 'park' AS feature_type, NULL::numeric AS count, "
            "p.name, "
            "ROUND(ST_Area(ST_Intersection(p.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(p.geometry, s.geometry)) AS geojson "
            "FROM osm_parks p, suburb s, stats st "
            "WHERE ST_Intersects(p.geometry, s.geometry) "
            "AND ST_GeometryType(p.geometry) IN ('ST_Polygon','ST_MultiPolygon') "
            "UNION ALL "
            "SELECT 'building' AS feature_type, NULL::numeric AS count, "
            "b.name, "
            "ROUND(ST_Area(ST_Intersection(b.geometry, s.geometry)::geography)::numeric) AS area_m2, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS green_to_concrete_ratio, "
            "ROUND((100.0 * st.green_m2 / st.total_m2)::numeric, 2) AS green_pct, "
            "ROUND((100.0 * st.concrete_m2 / st.total_m2)::numeric, 2) AS building_pct, "
            "ST_AsGeoJSON(ST_Intersection(b.geometry, s.geometry)) AS geojson "
            "FROM osm_buildings b, suburb s, stats st "
            "WHERE ST_Intersects(b.geometry, s.geometry) "
            "UNION ALL "
            "SELECT 'green_area_m2' AS feature_type, ROUND(st.green_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "UNION ALL "
            "SELECT 'building_area_m2' AS feature_type, ROUND(st.concrete_m2::numeric) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "UNION ALL "
            "SELECT 'green_to_concrete_ratio' AS feature_type, "
            "ROUND((st.green_m2 / NULLIF(st.concrete_m2, 0))::numeric, 4) AS count, "
            "NULL, NULL, NULL, NULL, NULL, NULL::text AS geojson FROM stats st "
            "LIMIT 500;"
        )
    },
    # Pattern 19: Density variant phrasing
    {
        "role": "user",
        "content": "Rank suburbs by cafe density"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT b.name AS suburb, "
            "COUNT(r.id) AS cafe_count, "
            "ROUND(ST_Area(b.geometry::geography)::numeric / 1000000, 4) AS area_km2, "
            "ROUND((COUNT(r.id) / (ST_Area(b.geometry::geography) / 1000000))::numeric, 2) AS cafes_per_km2, "
            "ST_AsGeoJSON(b.geometry) AS geojson "
            "FROM osm_boundaries b "
            "LEFT JOIN osm_restaurants r ON ST_Within(r.geometry, b.geometry) AND r.amenity = 'cafe' "
            "WHERE b.admin_level = '9' "
            "GROUP BY b.id, b.name, b.geometry "
            "HAVING COUNT(r.id) > 0 "
            "ORDER BY cafes_per_km2 DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 20: NOT EXISTS variant — schools with no restaurant within 500m
    {
        "role": "user",
        "content": "Which schools have no restaurant within 500 metres?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT s.name, ST_AsGeoJSON(s.geometry) AS geojson "
            "FROM osm_schools s "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM osm_restaurants r "
            "WHERE ST_DWithin(s.geometry::geography, r.geometry::geography, 500)"
            ") "
            "AND s.name IS NOT NULL "
            "ORDER BY s.name "
            "LIMIT 50;"
        )
    },
    # Pattern 21: Parks near waterways
    {
        "role": "user",
        "content": "Show parks near the River Torrens"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT p.name, "
            "ROUND(ST_Distance(p.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_waterways WHERE name ILIKE '%Torrens%' LIMIT 1)::geography)) AS distance_meters, "
            "ST_AsGeoJSON(p.geometry) AS geojson "
            "FROM osm_parks p "
            "WHERE ST_DWithin("
            "p.geometry::geography, "
            "(SELECT geometry FROM osm_waterways WHERE name ILIKE '%Torrens%' LIMIT 1)::geography, "
            "500) "
            "AND p.name IS NOT NULL "
            "ORDER BY p.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_waterways WHERE name ILIKE '%Torrens%' LIMIT 1) "
            "LIMIT 50;"
        )
    },
    # Pattern 22: School count per suburb ranked
    {
        "role": "user",
        "content": "Which suburbs have the most schools?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT b.name AS suburb, "
            "COUNT(s.id) AS school_count, "
            "ST_AsGeoJSON(b.geometry) AS geojson "
            "FROM osm_boundaries b "
            "LEFT JOIN osm_schools s ON ST_Within(s.geometry, b.geometry) "
            "WHERE b.admin_level = '9' "
            "GROUP BY b.id, b.name, b.geometry "
            "HAVING COUNT(s.id) > 0 "
            "ORDER BY school_count DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 23: Hospital + nearby pharmacy chain — multi-hop style
    {
        "role": "user",
        "content": "For each hospital, show the nearest pharmacy"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT DISTINCT ON (h.id) "
            "h.name AS hospital, "
            "p.name AS nearest_pharmacy, "
            "ROUND(ST_Distance(h.geometry::geography, p.geometry::geography)) AS distance_m, "
            "ST_AsGeoJSON(h.geometry) AS geojson "
            "FROM osm_hospitals h "
            "CROSS JOIN LATERAL ("
            "SELECT id, name, geometry FROM osm_pharmacies "
            "ORDER BY osm_pharmacies.geometry <-> h.geometry "
            "LIMIT 1"
            ") p "
            "WHERE h.name IS NOT NULL "
            "ORDER BY h.id, distance_m "
            "LIMIT 50;"
        )
    },
    # Pattern 24: Directional query — north of a reference point using ST_Azimuth
    {
        "role": "user",
        "content": "Show restaurants north of Adelaide CBD"
    },
    {
        "role": "assistant",
        "content": (
            "WITH ref AS ("
            "SELECT ST_Centroid(geometry) AS geom "
            "FROM osm_all WHERE name ILIKE '%Adelaide CBD%' LIMIT 1"
            ") "
            "SELECT r.name, r.amenity, "
            "ROUND(ST_Distance(r.geometry::geography, (SELECT geom FROM ref)::geography)) AS distance_m, "
            "DEGREES(ST_Azimuth((SELECT geom FROM ref), ST_Centroid(r.geometry))) AS bearing_deg, "
            "ST_AsGeoJSON(r.geometry) AS geojson "
            "FROM osm_restaurants r, ref "
            "WHERE r.geometry IS NOT NULL "
            "AND ("
            "DEGREES(ST_Azimuth(ref.geom, ST_Centroid(r.geometry))) <= 22.5 "
            "OR DEGREES(ST_Azimuth(ref.geom, ST_Centroid(r.geometry))) >= 337.5"
            ") "
            "ORDER BY r.geometry <-> (SELECT geom FROM ref) "
            "LIMIT 50;"
        )
    },
    # Pattern 25: Large parks — qualitative size descriptor
    {
        "role": "user",
        "content": "Show me large parks in Adelaide"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT name, leisure, "
            "ROUND(ST_Area(geometry::geography)) AS area_m2, "
            "ST_AsGeoJSON(geometry) AS geojson "
            "FROM osm_parks "
            "WHERE name IS NOT NULL "
            "AND ST_Area(geometry::geography) > 50000 "
            "ORDER BY ST_Area(geometry::geography) DESC "
            "LIMIT 50;"
        )
    },
    # Pattern 26: Walkable distance — fuzzy proximity qualifier
    {
        "role": "user",
        "content": "Show schools within walking distance of Rundle Mall"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT s.name, "
            "ROUND(ST_Distance(s.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Rundle Mall%' LIMIT 1)::geography)) AS distance_m, "
            "ST_AsGeoJSON(s.geometry) AS geojson "
            "FROM osm_schools s "
            "WHERE ST_DWithin("
            "s.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Rundle Mall%' LIMIT 1)::geography, "
            "800) "
            "ORDER BY s.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Rundle Mall%' LIMIT 1) "
            "LIMIT 50;"
        )
    },
    # Pattern 27: Dense suburb — qualitative density with per-km² ranking
    {
        "role": "user",
        "content": "Which suburbs have the densest concentration of cafes?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT b.name AS suburb, "
            "COUNT(r.id) AS cafe_count, "
            "ROUND(ST_Area(b.geometry::geography) / 1000000.0, 4) AS area_km2, "
            "ROUND((COUNT(r.id) / (ST_Area(b.geometry::geography) / 1000000.0))::numeric, 2) AS cafes_per_km2, "
            "ST_AsGeoJSON(b.geometry) AS geojson "
            "FROM osm_boundaries b "
            "LEFT JOIN osm_restaurants r ON ST_Within(r.geometry, b.geometry) AND r.amenity = 'cafe' "
            "WHERE b.admin_level = '9' "
            "GROUP BY b.id, b.name, b.geometry "
            "HAVING COUNT(r.id) > 0 "
            "ORDER BY cafes_per_km2 DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 28: Directional — east of a suburb
    {
        "role": "user",
        "content": "Show parks east of Norwood"
    },
    {
        "role": "assistant",
        "content": (
            "WITH ref AS ("
            "SELECT ST_Centroid(geometry) AS geom "
            "FROM osm_boundaries WHERE name ILIKE '%Norwood%' ORDER BY admin_level DESC LIMIT 1"
            ") "
            "SELECT p.name, "
            "ROUND(ST_Distance(p.geometry::geography, (SELECT geom FROM ref)::geography)) AS distance_m, "
            "DEGREES(ST_Azimuth((SELECT geom FROM ref), ST_Centroid(p.geometry))) AS bearing_deg, "
            "ST_AsGeoJSON(p.geometry) AS geojson "
            "FROM osm_parks p, ref "
            "WHERE p.geometry IS NOT NULL "
            "AND DEGREES(ST_Azimuth(ref.geom, ST_Centroid(p.geometry))) BETWEEN 67.5 AND 112.5 "
            "ORDER BY p.geometry <-> (SELECT geom FROM ref) "
            "LIMIT 50;"
        )
    },
    # Pattern 29: Isolated features — qualitative 'isolated' descriptor
    {
        "role": "user",
        "content": "Which pharmacies are most isolated from hospitals?"
    },
    {
        "role": "assistant",
        "content": (
            "SELECT p.name AS pharmacy, "
            "ROUND(MIN(ST_Distance(p.geometry::geography, h.geometry::geography))) AS nearest_hospital_m, "
            "ST_AsGeoJSON(p.geometry) AS geojson "
            "FROM osm_pharmacies p "
            "CROSS JOIN osm_hospitals h "
            "WHERE p.name IS NOT NULL "
            "GROUP BY p.id, p.name, p.geometry "
            "ORDER BY nearest_hospital_m DESC "
            "LIMIT 20;"
        )
    },
    # Pattern 30: Spatial reasoning context injection — example with context block
    {
        "role": "user",
        "content": (
            "[SPATIAL REASONING CONTEXT]\n"
            "Intent: proximity search\n"
            "Distance: 1,500 m (short walk — short walk or cycling distance ≤1.5 km)\n"
            "Recommended PostGIS operators: ST_DWithin(a::geography, b::geography, 1500)\n"
            "[/SPATIAL REASONING CONTEXT]\n"
            "Show restaurants within a short walk of Glenelg Beach"
        )
    },
    {
        "role": "assistant",
        "content": (
            "SELECT r.name, r.amenity, r.cuisine, "
            "ROUND(ST_Distance(r.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg%' LIMIT 1)::geography)) AS distance_m, "
            "ST_AsGeoJSON(r.geometry) AS geojson "
            "FROM osm_restaurants r "
            "WHERE ST_DWithin("
            "r.geometry::geography, "
            "(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg%' LIMIT 1)::geography, "
            "1500) "
            "ORDER BY r.geometry <-> (SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg%' LIMIT 1) "
            "LIMIT 100;"
        )
    },
]


def query_requires_device_location(query: str) -> bool:
    """Return True when the query refers to the user's current location."""
    normalized = query.strip().lower()
    if normalized == "me":
        return True
    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in DEVICE_LOCATION_PATTERNS
    )


def _build_device_location_context(coords: tuple[float, float]) -> str:
    """
    Build the device-location context string injected into the system prompt.
    Only called when the user explicitly requested a 'near me' query and the
    browser returned GPS coordinates. This is the ONLY case where coordinates
    are injected into the prompt at runtime.
    """
    lat, lng = coords
    return (
        f"\nDEVICE GPS CONTEXT: The user's current location is "
        f"latitude={lat}, longitude={lng}. "
        f"Use ST_SetSRID(ST_MakePoint({lng}, {lat}), 4326) for this location. "
        f"Do NOT use a subquery — use the exact coordinates above."
    )


def generate_sql(
    user_query: str,
    schema_context: str,
    allowed_tables: set[str],
    device_coords: tuple[float, float] | None = None,
    model: str | None = None,
    max_retries: int = 3,
    conversation_history: list[dict] | None = None,
    spatial_intent_context: str | None = None,
) -> tuple[str, str | None]:
    """
    Generate a validated PostGIS SQL query from natural language.

    Place names are resolved entirely inside PostGIS via subqueries.
    Device GPS coordinates are injected only for 'near me' queries.
    conversation_history is a list of prior {"role": "user"|"assistant", "content": str}
    turns injected between few-shots and the current query so the LLM can resolve
    follow-up references ("those", "same area", "now filter by...").
    spatial_intent_context is an optional [SPATIAL REASONING CONTEXT] block produced
    by SpatialReasoner.decompose_intent() that provides operator hints to the LLM.

    Returns (sql, error). If successful, error is None.
    """
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

    validator = SQLValidator(allowed_tables)

    if query_requires_device_location(user_query) and device_coords is None:
        return "", (
            "This query refers to your current location. "
            "Please share your device location and try again."
        )

    # Build system prompt — only inject coords for device location queries
    system_content = SYSTEM_PROMPT.format(schema_context=schema_context)
    if device_coords is not None:
        system_content += _build_device_location_context(device_coords)

    # Build user query message — prepend spatial reasoning context if available
    user_message_content = user_query
    if spatial_intent_context:
        user_message_content = spatial_intent_context + "\n" + user_query

    messages = [
        {"role": "system", "content": system_content},
        *FEW_SHOT_EXAMPLES,
        *(conversation_history or []),
        {"role": "user", "content": user_message_content},
    ]

    last_error = None

    for attempt in range(max_retries):
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)
            response = client.chat(
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

            import sys
            print(f"\n[SQL GEN attempt {attempt+1}]\n{raw_sql}\n", file=sys.stderr, flush=True)

            try:
                validated_sql = validator.validate(raw_sql)
                return validated_sql, None
            except ValidationError as e:
                last_error = str(e)
                messages.append({"role": "assistant", "content": raw_sql})
                messages.append({
                    "role": "user",
                    "content": (
                        f"ERROR: {last_error}. "
                        f"Please fix the query. Output ONLY valid PostGIS SQL, no markdown or explanation. "
                        f"NEVER use ST_Extent — use osm_boundaries subquery instead. "
                        f"NEVER invent table names like 'boundary', 'suburb', 'area' — only osm_* tables exist. "
                        f"For suburb boundaries use: (SELECT geometry FROM osm_boundaries WHERE name ILIKE '%place%' ORDER BY admin_level DESC LIMIT 1). "
                        f"Available tables: {sorted(allowed_tables)}"
                    ),
                })

        except Exception as e:
            error_str = str(e)
            if "connection refused" in error_str.lower() or "connect" in error_str.lower():
                return "", (
                    "Cannot connect to Ollama. Make sure Ollama is running: "
                    f"`ollama serve` and the model is pulled: `ollama pull {model}`"
                )
            last_error = f"Ollama error: {error_str}"

    return "", f"Failed after {max_retries} attempts. Last error: {last_error}"


def should_skip_analysis(result) -> bool:
    """
    Return True when the analysis LLM call adds no value and should be skipped.

    Skipped when:
    - Zero rows returned
    - Pure count/aggregate result (no geometry, ≤ 3 columns, all numeric)
    - Non-spatial result with a single scalar value (e.g. SELECT COUNT(*))
    """
    if result.row_count == 0:
        return True

    if not result.has_geometry and result.raw_rows:
        # Single-row, single-column scalar (e.g. COUNT query)
        if len(result.raw_rows) == 1 and len(result.raw_rows[0]) <= 2:
            return True
        # Small tabular result where every value is numeric — just show the table
        if len(result.raw_rows) <= 5 and result.columns:
            import numbers
            all_numeric = all(
                isinstance(v, (int, float, numbers.Number))
                for row in result.raw_rows
                for v in row.values()
                if v is not None
            )
            if all_numeric:
                return True

    return False


def generate_analysis_stream(
    user_query: str,
    sql: str,
    stats: dict,
    model: str | None = None,
    spatial_prompt: str | None = None,
):
    """
    Streaming variant of generate_analysis().

    Yields text chunks as they arrive from the LLM so callers can use
    st.write_stream() for perceived latency improvement.  Falls back to
    yielding the full string at once if the model/path doesn't support streaming.

    spatial_prompt: optional pre-built prompt from SpatialReasoner.explain_results()
    that provides richer spatial context.  When provided it replaces the default prompt.
    """
    analysis_model = os.getenv("ANALYSIS_MODEL", "").strip()
    analysis_key   = os.getenv("ANALYSIS_API_KEY", "").strip()

    count        = stats.get("count", 0)
    geom_types   = stats.get("geometry_types", "unknown")
    sample_names = stats.get("sample_names", [])
    extra_stats  = {k: v for k, v in stats.items()
                    if k not in ("count", "geometry_types", "sample_names")}

    # Use the enriched spatial prompt if provided, else fall back to the default
    if spatial_prompt:
        prompt = spatial_prompt
    else:
        prompt = (
            f"User asked: \"{user_query}\"\n"
            f"SQL executed: {sql}\n"
            f"Results: {count} features returned.\n"
            f"Geometry types: {geom_types}\n"
            f"Sample names: {', '.join(str(n) for n in sample_names[:5]) or 'N/A'}\n"
            f"Statistics: {extra_stats}\n\n"
            f"Provide:\n"
            f"1. A 2-3 sentence spatial interpretation of what was found and why it matters.\n"
            f"2. Any notable patterns, clusters, or distributions visible in the data.\n"
            f"Be concise and factual."
        )

    if analysis_model and analysis_key:
        try:
            import openai
            client = openai.OpenAI(api_key=analysis_key)
            stream = client.chat.completions.create(
                model=analysis_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return
        except Exception:
            pass

    if analysis_model and not analysis_key:
        use_thinking = os.getenv("ANALYSIS_THINKING", "false").lower() == "true"
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)
            kwargs = dict(
                model=analysis_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 1.0 if use_thinking else 0.3, "num_predict": 800},
                stream=True,
            )
            if use_thinking:
                kwargs["think"] = True
            in_thinking = True if use_thinking else False
            for chunk in client.chat(**kwargs):
                msg = chunk.get("message", {})
                # While thinking tokens are flowing, suppress them from the stream
                if use_thinking and in_thinking:
                    if not msg.get("thinking"):
                        in_thinking = False
                    continue
                text = msg.get("content", "")
                if text:
                    yield text
            return
        except Exception:
            pass

    # Fallback: local Ollama OLLAMA_MODEL, streaming
    try:
        local_model = model or os.getenv("OLLAMA_MODEL", "gemma4:e4b")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)
        for chunk in client.chat(
            model=local_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 300},
            stream=True,
        ):
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text
    except Exception:
        return


def generate_analysis(
    user_query: str,
    sql: str,
    stats: dict,
    model: str | None = None,
    spatial_prompt: str | None = None,
) -> str:
    """
    Call the analysis LLM to produce a natural language spatial summary.

    Uses ANALYSIS_MODEL env var if set, otherwise falls back to OLLAMA_MODEL.
    Returns a plain text paragraph. Returns empty string on failure.
    spatial_prompt: optional pre-built prompt from SpatialReasoner.explain_results().
    """
    analysis_model = os.getenv("ANALYSIS_MODEL", "").strip()
    analysis_key   = os.getenv("ANALYSIS_API_KEY", "").strip()

    count        = stats.get("count", 0)
    geom_types   = stats.get("geometry_types", "unknown")
    sample_names = stats.get("sample_names", [])
    extra_stats  = {k: v for k, v in stats.items()
                    if k not in ("count", "geometry_types", "sample_names")}

    if spatial_prompt:
        prompt = spatial_prompt
    else:
        prompt = (
            f"User asked: \"{user_query}\"\n"
            f"SQL executed: {sql}\n"
            f"Results: {count} features returned.\n"
            f"Geometry types: {geom_types}\n"
            f"Sample names: {', '.join(str(n) for n in sample_names[:5]) or 'N/A'}\n"
            f"Statistics: {extra_stats}\n\n"
            f"Provide:\n"
            f"1. A 2-3 sentence spatial interpretation of what was found and why it matters.\n"
            f"2. Any notable patterns, clusters, or distributions visible in the data.\n"
            f"Be concise and factual."
        )

    if analysis_model and analysis_key:
        # Cloud LLM via OpenAI-compatible API (e.g. openai/gpt-4o)
        try:
            import openai
            client = openai.OpenAI(api_key=analysis_key)
            resp = client.chat.completions.create(
                model=analysis_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    if analysis_model and not analysis_key:
        # Ollama-hosted analysis model (local or cloud-backed via Ollama).
        # ANALYSIS_THINKING=true enables extended chain-of-thought reasoning
        # (supported by gemma4 and other thinking-capable Ollama models).
        use_thinking = os.getenv("ANALYSIS_THINKING", "false").lower() == "true"
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)
            kwargs = dict(
                model=analysis_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 1.0 if use_thinking else 0.3, "num_predict": 800},
                stream=False,
            )
            if use_thinking:
                kwargs["think"] = True
            response = client.chat(**kwargs)
            if use_thinking:
                thinking = response["message"].get("thinking", "")
                if thinking:
                    import sys
                    print(f"\n[ANALYSIS THINKING]\n{thinking}\n[/ANALYSIS THINKING]\n", file=sys.stderr, flush=True)
            return response["message"]["content"].strip()
        except Exception:
            pass

    try:
        local_model = model or os.getenv("OLLAMA_MODEL", "gemma4:e4b")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)
        response = client.chat(
            model=local_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 300},
            stream=False,
        )
        return response["message"]["content"].strip()
    except Exception:
        return ""
