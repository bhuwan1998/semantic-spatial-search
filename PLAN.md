# GeoAgentic-RAG: Migration & Enhancement Plan

## Overview

This document captures the complete plan to migrate the Natural Language Spatial Search
proof-of-concept from a local SpatiaLite/GeoPackage stack to a production-aligned
PostgreSQL + PostGIS + pgvector + Apache AGE architecture, while also adding:

- **Hybrid RAG** for smarter, token-efficient schema retrieval
- **Spatial analysis agent** for natural-language result interpretation
- **Graph topology** via Apache AGE (openCypher, AWS Neptune migration path)
- **Chat interface** with persistent map panel

---

## Reference Paper

**GeoAgentic-RAG: A Multi-Agent framework for autonomous geospatial reasoning and visual
insight generation with LLM** (Liang et al., 2026, *International Journal of Applied Earth
Observation and Geoinformation*, 147, 105195)

Key ideas adopted from the paper:

| Paper concept | This implementation |
|---|---|
| Semantic-Spatial Fusion (pgvector embeddings for schema retrieval) | `core/rag.py` — query → nomic-embed-text → pgvector cosine search |
| Multi-agent collaboration (Task Plan Agent, Spatial Retrieve Agent) | Two-LLM pipeline: SQL generation + analysis agent |
| Self-Reflection module (error recovery loop, +14.1% HS) | Enhanced retry loop in `core/llm.py` |
| ReAct Thought-Action-Observation loop | SQL generation → execution → LLM analysis → response |
| Closed-loop workflow (retrieval → analysis → language generation) | `core/analyst.py` spatial summary + follow-up suggestions |
| PostGIS + PGVector knowledge base | PostgreSQL 15 + PostGIS + pgvector + Apache AGE |
| Natural language QA interface | Streamlit chat UI + persistent map panel |

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Streamlit Chat App (app.py)                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │
│  │   Chat Panel (40%)   │  │        Map Panel (60%)               │ │
│  │                      │  │   Folium map — always persistent     │ │
│  │ [User message]       │  │   Updates on each query response     │ │
│  │ [AI: summary+stats]  │  │                                      │ │
│  │ [SQL expander]       │  │                                      │ │
│  │ [Follow-up chips]    │  │                                      │ │
│  │ [chat_input box]     │  │                                      │ │
│  └──────────────────────┘  └──────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       ↓                        ↓                        ↓
  core/rag.py           core/llm.py                core/analyst.py
  Embed query →         PostGIS SQL gen            Stats + LLM summary
  pgvector search →     (RAG-retrieved schema)     follow-up suggestions
  top-k table context   Ollama local LLM           ANALYSIS_MODEL (env)
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                ↓
                       core/executor.py
                       psycopg3 + PostGIS
                       Returns GeoDataFrame
                                │
                                ↓
            ┌───────────────────────────────────────────┐
            │          PostgreSQL 15 (Docker)            │
            │  PostGIS + pgvector + Apache AGE           │
            │                                            │
            │  12 tables: osm_schools, osm_parks, ...   │
            │  + osm_all unified view                    │
            │  + table_descriptions (vector(768))        │
            │  + AGE graph: osm_spatial                  │
            │    Nodes: School, Park, Hospital, ...      │
            │    Edges: NEAR, WITHIN, CONNECTED_TO       │
            └───────────────────────────────────────────┘
                                ↑
                       data/setup_db.py
                       osmnx → PostGIS loader
                       + pgvector embedding indexer
                       + AGE graph builder
```

---

## Technology Stack

| Component | Technology | Notes |
|---|---|---|
| Spatial database | PostgreSQL 15 + PostGIS 3 | Replaces SpatiaLite/GeoPackage |
| Vector embeddings | pgvector (v0.8+) | Schema RAG, 768-dim nomic-embed-text |
| Graph topology | Apache AGE | openCypher, AWS Neptune migration path |
| Container runtime | Docker + Docker Compose | Single container for all three extensions |
| Embedding model | `nomic-embed-text` via Ollama | 768 dimensions, fully local |
| SQL generation LLM | Ollama (local, `OLLAMA_MODEL`) | Same as current, PostGIS dialect |
| Analysis LLM | `ANALYSIS_MODEL` env var | Optional cloud model; falls back to Ollama |
| Place name resolution | PostGIS subqueries against osm_all / osm_boundaries | All place names resolved inside the DB at query time — no geocoder in query path |
| Python DB driver | psycopg3 (`psycopg[binary]`) + psycopg-pool | Replaces sqlite3 |
| Data ingestion | osmnx → `to_postgis()` | Same 12 OSM layers, same tag filters |

### Why PostgreSQL 15 specifically

Apache AGE officially supports PostgreSQL 11–16. The `apache/age` Docker image is built on
PG 15. pgvector and PostGIS both support PG 15. All three extensions coexist in one
container on PG 15 — requiring only a custom `Dockerfile` to compile pgvector into the
AGE base image.

---

## Database Schema

### 12 Spatial Tables (one per OSM layer)

Each table mirrors the current GeoPackage layer structure:

```sql
CREATE TABLE osm_schools (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    amenity     text,
    -- ... other OSM attribute columns per layer
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX ON osm_schools USING GIST (geometry);
```

Layers: `osm_schools`, `osm_hospitals`, `osm_restaurants`, `osm_pharmacies`,
`osm_roads`, `osm_waterways`, `osm_railways`, `osm_parks`, `osm_buildings`,
`osm_landuse`, `osm_natural`, `osm_boundaries`

### Unified View

```sql
CREATE VIEW osm_all AS
    SELECT id, osm_id, name, 'schools' AS layer, geometry FROM osm_schools
    UNION ALL
    SELECT id, osm_id, name, 'hospitals' AS layer, geometry FROM osm_hospitals
    -- ... all 12 tables
```

### pgvector Table (Schema RAG)

```sql
CREATE TABLE table_descriptions (
    id          serial PRIMARY KEY,
    table_name  text NOT NULL,
    description text NOT NULL,     -- natural language description for LLM
    embedding   vector(768)        -- nomic-embed-text output
);
-- No index needed at 12 rows; hnsw added if feature-level embeddings added later
```

### Apache AGE Graph (`osm_spatial`)

Nodes per feature:
```cypher
(:School   {osm_id, name, layer, suburb, postcode})
(:Park     {osm_id, name, layer, suburb})
(:Hospital {osm_id, name, layer, suburb})
-- etc. for each layer type
```

Relationship edges:
```cypher
(a)-[:NEAR        {distance_m: float}]->(b)   -- ST_DWithin 500m pairs
(a)-[:WITHIN      {boundary_name: text}]->(b)  -- spatial join with admin boundaries
(a)-[:CONNECTED_TO {road_name: text}]->(b)     -- road network adjacency
```

---

## PostGIS SQL Dialect (Migration from SpatiaLite)

This is the most critical change — all LLM prompts, few-shot examples, and SQL validation
rules must switch from SpatiaLite to PostGIS syntax.

| SpatiaLite (current) | PostGIS (new) | Notes |
|---|---|---|
| `MakePoint(lon, lat, 4326)` | `ST_SetSRID(ST_MakePoint(lon, lat), 4326)` | |
| `AsGeoJSON(CastAutomagic(geom))` | `ST_AsGeoJSON(geometry)` | Works on all geom types natively |
| `ST_Distance(a, b) < 500` | `ST_DWithin(a::geography, b::geography, 500)` | Index-accelerated in PostGIS |
| `ST_Distance(a, b) * 111320` | `ST_Distance(a::geography, b::geography)` | `::geography` returns meters directly |
| No KNN operator | `ORDER BY geom <-> ref LIMIT n` | Uses GiST index for fast KNN |
| `CastAutomagic()` | Not needed | PostGIS handles mixed types natively |
| `EnableGpkgAmphibiousMode()` | Not needed | No GeoPackage binary blobs in PostGIS |
| No `ST_DWithin` | `ST_DWithin(a::geography, b::geography, dist_m)` | Correct meter-based proximity |

### Distance Pattern (PostGIS best practice)

```sql
-- Proximity filter (uses spatial index) + KNN sort (uses GiST)
WHERE ST_DWithin(geometry::geography,
                 ST_SetSRID(ST_MakePoint(lng, lat), 4326)::geography,
                 2000)         -- 2000 metres
ORDER BY geometry <-> ST_SetSRID(ST_MakePoint(lng, lat), 4326)
LIMIT 10;
```

---

## Place Name Resolution Strategy

### Design principle

**All place name resolution happens inside PostGIS at query time.** There is no geocoder
in the query path. No hardcoded coordinate dictionaries, no Nominatim calls, no runtime
HTTP requests to resolve place names.

The LLM is instructed to generate SQL subqueries that look up place names directly from
the database:

```sql
-- Named place → centroid via osm_all (parks, landmarks, buildings, etc.)
(SELECT ST_Centroid(geometry) FROM osm_all WHERE name ILIKE '%Glenelg Beach%' LIMIT 1)

-- Suburb/boundary → polygon via osm_boundaries
(SELECT geometry FROM osm_boundaries WHERE name ILIKE '%Norwood%' LIMIT 1)
```

This means the coordinates used in every spatial query are sourced from the same PostGIS
data as the query results — consistent, accurate, and requiring no external service.

### The only exception: device GPS

The one case where coordinates are injected into the LLM prompt at runtime is **"near me"**
queries. These require the user's actual GPS coordinates from the browser, which genuinely
cannot be resolved from the database. The app requests these via `streamlit-js-eval` and
injects them via `_build_device_location_context()` in `core/llm.py`.

### `core/geocoder.py`

Reduced to a single function: `get_adelaide_center()`, which returns the default map
center for `app.py` on startup. This has nothing to do with query resolution.

### Data preparation (not runtime)

`geopy` (Nominatim) is available in `data/setup_db.py` for data preparation tasks such as
assigning suburb labels to AGE graph nodes when the PostGIS boundary join returns no result.
It is never called during a user query.

---

## Hybrid RAG Pipeline

### Problem with current approach

The full schema (~3,000 tokens) is injected into every LLM call regardless of which tables
are relevant to the query. For a query about schools, the model receives detailed schema for
roads, waterways, railways, etc. — adding noise and wasting context window.

### Solution: pgvector semantic schema retrieval

```
User query: "Find schools near Glenelg Beach"
     ↓
Embed with nomic-embed-text (768-dim):
     [0.12, -0.34, 0.89, ...]
     ↓
pgvector cosine search against table_descriptions:
     SELECT table_name, description
     FROM table_descriptions
     ORDER BY embedding <=> $query_embedding
     LIMIT 4
     → ["osm_schools", "osm_boundaries", "osm_parks", "osm_roads"]
     ↓
Inject only those 4 table schemas into LLM prompt (~800 tokens vs 3,000)
```

This mirrors the paper's **Semantic-Spatial Fusion** approach adapted for SpatiaLite → PostGIS.

### Graceful degradation

If the embedding model is unavailable or pgvector query fails, `core/rag.py` falls back to
injecting the full schema — same behaviour as the current system.

---

## Spatial Analysis Agent

After each query executes, a second LLM call generates analytical output:

### Stats computed directly from GeoDataFrame

- Feature count by geometry type
- For polygons: total area (m²), largest/smallest feature
- For points with `dist_m` column: min/max/avg distance
- Unique name count (if `name` column present)
- Top-3 most common attribute values (amenity type, road class, etc.)

### LLM analysis prompt

```
Given:
- User query: "{query}"
- SQL executed: "{sql}"
- Results: {count} features returned
  - Geometry types: {breakdown}
  - Stats: {computed_stats}
  - Sample names: {top_5_names}

Provide:
1. A 2-3 sentence spatial interpretation of what was found and why it's significant
2. Any notable patterns, clusters, or distributions
3. Three follow-up queries the user might want to ask next
```

### Output (`AnalysisResult` dataclass)

```python
@dataclass
class AnalysisResult:
    summary: str          # 2-3 sentence spatial interpretation
    stats: dict           # computed statistics dict
    followups: list[str]  # 3 contextual follow-up suggestions
```

---

## Apache AGE Graph

### Purpose

The PostGIS tables handle spatial geometry and indexing. The AGE graph captures
**relationships and topology** that are expensive to recompute from scratch on each query:

- "Which parks are accessible from this school (within 500m)?"
- "What facilities are clustered together in Norwood?"
- "What is connected to this waterway?"

### Graph construction (in `data/setup_db.py`)

1. For each feature in each layer: create a labelled node with `osm_id`, `name`, `layer`,
   `suburb` (assigned via PostGIS spatial join with boundaries).
2. For cross-layer pairs within 500m: create `NEAR` edges (computed via `ST_DWithin`).
3. For features within administrative boundaries: create `WITHIN` edges.
4. For road/waterway network adjacency: create `CONNECTED_TO` edges.

### AWS Neptune migration path

Apache AGE uses **openCypher** — the same query language supported by AWS Neptune's
openCypher endpoint. Cypher queries written against AGE are directly portable to Neptune
with only a connection layer change (psycopg3 → Neptune Bolt/HTTP endpoint). The graph
model (node labels, edge types, property names) transfers unchanged.

---

## Chat Interface Design

### Layout

```
st.set_page_config(layout="wide")
col_chat, col_map = st.columns([2, 3])
```

### Chat panel (left, 40%)

- `st.chat_message("user")` / `st.chat_message("assistant")` for conversation history
- Each assistant message contains:
  - Natural language spatial summary (from analysis agent)
  - `st.expander("View SQL")` — syntax-highlighted generated SQL
  - `st.expander("Statistics")` — count, area, distance stats table
  - Three follow-up suggestion `st.button` chips — click to auto-submit
- `st.chat_input("Ask a spatial question...")` pinned at bottom
- `st.expander("Database Schema")` — collapsible schema browser (moved from sidebar)
- `st.expander("Example queries")` — clickable example chips

### Map panel (right, 60%)

- Folium map always rendered, never disappears between messages
- Basemap selector (`st.selectbox`) above the map
- Map updates (`st.session_state.current_map_html`) on each successful query
- Same colour-coded geometry rendering: red=Point, blue=Line, green=Polygon
- Device location marker (amber dot + 250m circle) when GPS coordinates available

### Session state

```python
st.session_state.messages = [
    {
        "role": "user",
        "content": "Find 5 schools near Adelaide CBD"
    },
    {
        "role": "assistant",
        "content": "Found 5 schools within 2km of Adelaide CBD...",
        "sql": "SELECT ...",
        "stats": {"count": 5, "avg_distance_m": 1240, ...},
        "followups": ["Show hospitals near these schools", ...],
        "map_html": "<html>...</html>"
    }
]
```

---

## File Change Summary

### New files

| File | Purpose |
|---|---|
| `Dockerfile` | PG 15 + AGE (pre-installed) + pgvector (compiled from source) + PostGIS |
| `docker-compose.yml` | Single-container service, named volume, health check, env vars |
| `data/setup_db.py` | Replaces `setup_data.py`: osmnx → PostGIS + embed index + AGE graph |
| `core/db.py` | psycopg3 connection pool with AGE search_path configuration |
| `core/rag.py` | Hybrid RAG: query embedding → pgvector cosine search → schema context |
| `core/analyst.py` | Stats computation + second LLM call for summary + follow-ups |
| `core/graph.py` | Apache AGE Cypher interface: build graph, query relationships |

### Modified files

| File | Change summary |
|---|---|
| `core/schema.py` | Rewrite: `information_schema` + `geometry_columns` replaces GeoPackage PRAGMA |
| `core/executor.py` | Rewrite: psycopg3 + PostGIS geometry handling replaces sqlite3 + SpatiaLite |
| `core/validator.py` | Update function allowlist (PostGIS in, SpatiaLite-only out) |
| `core/llm.py` | PostGIS system prompt + few-shots; accept RAG schema; analysis LLM function |
| `core/geocoder.py` | Minor: geopy import; `reverse_geocode()`; `assign_suburb_via_postgis()` |
| `app.py` | Full rewrite: chat + persistent map layout; conversation state management |
| `requirements.txt` | Add psycopg3, psycopg-pool, numpy, geopy, age |
| `.env.example` | Add DB connection vars + EMBEDDING_MODEL + ANALYSIS_MODEL + ANALYSIS_API_KEY |
| `setup.sh` | Replace spatialite check with Docker check; update data setup step |

### Unchanged

- `core/geocoder.py` `ADELAIDE_LANDMARKS` dict (all ~50 entries kept)
- `LAYER_DEFS` (same 12 layers, same OSM tags, same column pruning logic)
- `build_map_html()` / `build_default_map_html()` / `GEOM_COLORS` (identical rendering)
- `QueryResult` dataclass interface (`gdf`, `columns`, `row_count`, `has_geometry`, `error`, `raw_rows`)
- `query_requires_device_location()` device location detection
- Self-correction retry loop (up to 3 attempts, error fed back to model)
- `EXAMPLE_QUERIES` list (moved from sidebar to chat panel expander)

---

## Implementation Steps

| Step | File(s) | Description |
|---|---|---|
| 1 | `Dockerfile` | PG 15 + AGE base + pgvector compiled + PostGIS apt install |
| 2 | `docker-compose.yml` | Service definition, volume, health check |
| 3 | `requirements.txt` | Add 5 new packages |
| 4 | `.env.example` | DB + embedding + analysis env vars |
| 5 | `core/db.py` | psycopg3 connection pool |
| 6 | `data/setup_db.py` | Data load + embedding index + AGE graph |
| 7 | `core/schema.py` | PostgreSQL schema introspection |
| 8 | `core/rag.py` | Hybrid RAG retrieval |
| 9 | `core/executor.py` | PostGIS query execution |
| 10 | `core/validator.py` | PostGIS function allowlist |
| 11 | `core/llm.py` | PostGIS prompt + RAG integration + analysis function |
| 12 | `core/analyst.py` | Analysis agent |
| 13 | `core/graph.py` | AGE Cypher interface |
| 14 | `core/geocoder.py` | geopy + PostGIS spatial join geocoding |
| 15 | `app.py` | Chat + persistent map UI |
| 16 | `setup.sh` | Docker-first bootstrap |

---

## Environment Variables

```bash
# Ollama (SQL generation LLM)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4:e4b

# Embedding model (for pgvector RAG)
EMBEDDING_MODEL=nomic-embed-text

# Analysis LLM (optional — leave blank to use OLLAMA_MODEL)
# Examples: openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3.1:70b
ANALYSIS_MODEL=
ANALYSIS_API_KEY=

# PostgreSQL connection
DB_HOST=localhost
DB_PORT=5432
DB_NAME=geospatial
DB_USER=geo
DB_PASSWORD=geo
```

---

## Production Migration Path

| This PoC | Production (AWS) |
|---|---|
| Docker single container (PG 15 + PostGIS + pgvector + AGE) | RDS PostgreSQL + PostGIS, Aurora pgvector |
| Apache AGE (openCypher) | AWS Neptune (openCypher endpoint) — same Cypher queries |
| geopy Nominatim (public API) | Self-hosted Nominatim or AWS Location Service |
| Ollama local LLM | `ANALYSIS_MODEL` → AWS Bedrock / OpenAI API |
| `setup_db.py` (osmnx) | osm2pgsql pipeline for full PBF import |
| Single-user Streamlit | Streamlit Cloud or containerised deployment |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PostgreSQL version | 15 | AGE Docker image constraint (supports PG 11–16) |
| Graph DB | Apache AGE | Same DB, openCypher, direct Neptune migration path |
| Geocoding (PoC) | geopy + PostGIS spatial join | Zero new servers; boundaries table handles unannotated features |
| Embedding model | `nomic-embed-text` 768-dim | Fully local, no new deps beyond existing ollama client |
| Vector index | No index (12 rows); HNSW if feature embeddings added later | Scale-appropriate |
| Distance in PostGIS | `::geography` cast | Meters natively, correct on curved earth, no × 111320 heuristic |
| Nearest neighbor | `ST_DWithin` filter + `<->` sort | DWithin uses spatial index; `<->` handles KNN via GiST |
| Table layout | 12 separate tables + `osm_all` unified view | Clear per-layer LLM prompting + cross-layer queries via view |
| Analysis LLM | Configurable via `ANALYSIS_MODEL` | Supports local Ollama and cloud APIs interchangeably |
| Session model | `st.session_state.messages` list | Full multi-turn conversation history with map state per turn |
