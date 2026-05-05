# Performance & Production Improvements Plan

Profiled on: 2026-05-05  
Environment: MacOS, Docker (apache/age PG15), Ollama local, gemma4:31b-cloud

---

## Profiling Summary

The pipeline has 4 sequential blocking steps:

| Step | Typical time | Bottleneck |
|---|---|---|
| RAG embed (Ollama) | 500ms–2s | Cold Ollama round-trip |
| LLM SQL generation | 5–30s | Main offender — 31b model on local hardware |
| PostGIS query execute | 14–80ms | Fast, well-indexed, not the problem |
| Analysis LLM | 5–30s | Second full model call, fully blocking |

The database is **not** the bottleneck. All query times are 14–80ms. Every second of latency is LLM round-trips.

Key findings from `EXPLAIN ANALYZE`:
- Every `WHERE name ILIKE '%place%'` on `osm_all` does a **full sequential scan across all 12 union members** — costs up to 3347 planner units, hits 51,869 road rows on every name lookup
- `osm_all` is a plain `VIEW` — every subquery re-executes the 12-table UNION
- `shared_buffers=128MB`, `work_mem=4MB` — well below what a dedicated container should use
- No trigram indexes on `name` columns — `ILIKE` cannot use any index

---

## Improvements

### 1. Add `pg_trgm` trigram indexes on `name` columns
**Impact: High | Effort: Low | Status: TODO**

Every `WHERE name ILIKE '%suburb%'` or `WHERE name ILIKE '%place%'` does a full sequential scan. Adding GIN trigram indexes makes these use an index instead.

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX osm_schools_name_trgm      ON osm_schools      USING gin (name gin_trgm_ops);
CREATE INDEX osm_hospitals_name_trgm    ON osm_hospitals     USING gin (name gin_trgm_ops);
CREATE INDEX osm_restaurants_name_trgm  ON osm_restaurants   USING gin (name gin_trgm_ops);
CREATE INDEX osm_pharmacies_name_trgm   ON osm_pharmacies    USING gin (name gin_trgm_ops);
CREATE INDEX osm_roads_name_trgm        ON osm_roads         USING gin (name gin_trgm_ops);
CREATE INDEX osm_waterways_name_trgm    ON osm_waterways     USING gin (name gin_trgm_ops);
CREATE INDEX osm_railways_name_trgm     ON osm_railways      USING gin (name gin_trgm_ops);
CREATE INDEX osm_parks_name_trgm        ON osm_parks         USING gin (name gin_trgm_ops);
CREATE INDEX osm_buildings_name_trgm    ON osm_buildings     USING gin (name gin_trgm_ops);
CREATE INDEX osm_landuse_name_trgm      ON osm_landuse       USING gin (name gin_trgm_ops);
CREATE INDEX osm_natural_name_trgm      ON osm_natural       USING gin (name gin_trgm_ops);
CREATE INDEX osm_boundaries_name_trgm   ON osm_boundaries    USING gin (name gin_trgm_ops);
```

Add to `docker/initdb.sql` and run once on existing DB.

---

### 2. Materialise `osm_all` as a MATERIALIZED VIEW
**Impact: High | Effort: Low | Status: TODO**

`osm_all` is currently a plain `VIEW`. Every subquery against it (place name resolution, radius searches) re-runs the 12-table UNION from scratch. Convert to a `MATERIALIZED VIEW` with its own GiST and trigram indexes. Refresh only after data reloads.

```sql
CREATE MATERIALIZED VIEW osm_all_mat AS
  -- same UNION ALL as current osm_all view
  SELECT id, osm_id, name, geometry, 'school' AS layer FROM osm_schools
  UNION ALL
  SELECT id, osm_id, name, geometry, 'hospital' AS layer FROM osm_hospitals
  -- ... etc
;

CREATE INDEX osm_all_mat_geom_idx ON osm_all_mat USING gist (geometry);
CREATE INDEX osm_all_mat_name_trgm ON osm_all_mat USING gin (name gin_trgm_ops);
```

Update `docker/initdb.sql` and replace `osm_all` references in queries or create a view alias.  
Refresh after setup: `REFRESH MATERIALIZED VIEW osm_all_mat;`

---

### 3. PostgreSQL memory tuning
**Impact: Medium | Effort: Low | Status: TODO**

Current defaults are too conservative for a dedicated container.

In `docker-compose.yml`, add to the `db` service:
```yaml
command: >
  postgres
  -c shared_buffers=512MB
  -c work_mem=32MB
  -c effective_cache_size=2GB
  -c maintenance_work_mem=128MB
  -c checkpoint_completion_target=0.9
  -c wal_buffers=16MB
  -c random_page_cost=1.1
```

The `work_mem=32MB` improvement particularly helps `ST_Area`, `ST_Intersection`, and sort operations on large geometry sets. `random_page_cost=1.1` tells the planner the data is likely cached (SSD/memory), preferring index scans.

---

### 4. Stream analysis LLM output
**Impact: High (perceived speed) | Effort: Medium | Status: TODO**

Both SQL gen and analysis calls block until the full response arrives. The analysis call (second LLM call) takes 5–30s with no feedback to the user.

Switch `generate_analysis()` in `core/llm.py` to use `stream=True` and pipe tokens to Streamlit via `st.write_stream()`. The user sees output start within ~1s instead of waiting 30s for the full paragraph.

```python
# core/llm.py — generate_analysis streaming version
response = client.chat(model=..., messages=..., stream=True)
for chunk in response:
    yield chunk["message"]["content"]

# app.py
with st.chat_message("assistant"):
    st.write_stream(generate_analysis(...))
```

---

### 5. Run RAG embed + graph context fetch concurrently
**Impact: Medium | Effort: Low | Status: TODO**

Currently sequential:
```
embed(query) → graph_context(query) → generate_sql(...)
```

These are fully independent. Run in parallel with `ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as pool:
    rag_future   = pool.submit(rag.retrieve, user_query)
    graph_future = pool.submit(get_graph_context_for_query, user_query)

relevant_tables = rag_future.result()
graph_ctx       = graph_future.result()
```

Saves 500ms–2s on every query by overlapping the Ollama embed call with the AGE graph query.

---

### 6. LRU cache on RAG embeddings
**Impact: Medium | Effort: Low | Status: TODO**

The same query phrasing re-embeds on every submission. Repeated or similar follow-up queries hit Ollama each time.

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple[float, ...]:
    vec = self._embed(text)
    return tuple(vec) if vec else None
```

Cache keyed on the exact query string. Saves the full embed round-trip for repeated/similar queries.

---

### 7. Skip analysis LLM call for aggregate/count queries
**Impact: Medium | Effort: Low | Status: TODO**

Queries like "how many restaurants by type?" return no geometry and don't need a narrative analysis. Detect this and skip the second LLM call entirely.

```python
# core/analyst.py
if not result.has_geometry and result.row_count < 5:
    # Simple count/aggregate — skip LLM, generate summary from stats directly
    return AnalysisResult(summary="", stats=stats, followups=[])
```

Saves 5–30s on every aggregate query.

---

### 8. CLUSTER heavy tables on geometry
**Impact: Low–Medium | Effort: Low | Status: TODO**

Physical row ordering matching the GiST index reduces I/O on spatial range scans. Most impactful for `osm_roads` (14MB, 51k rows) and `osm_parks`.

```sql
CLUSTER osm_roads     USING osm_roads_geom_idx;
CLUSTER osm_parks     USING osm_parks_geom_idx;
CLUSTER osm_landuse   USING osm_landuse_geom_idx;
CLUSTER osm_buildings USING osm_buildings_geom_idx;
ANALYZE;
```

Run after initial data load in `data/setup_db.py`.

---

### 9. Add `admin_level` index on `osm_boundaries`
**Impact: Medium | Effort: Low | Status: TODO**

Every suburb containment query filters `WHERE b.admin_level = '9'`. Currently a seq scan over 462 rows — small but runs many times per query as a subquery.

```sql
CREATE INDEX osm_boundaries_admin_level_idx ON osm_boundaries (admin_level);
CREATE INDEX osm_boundaries_geom_admin_idx  ON osm_boundaries USING gist (geometry) WHERE admin_level = '9';
```

The partial GiST index (`WHERE admin_level = '9'`) makes suburb boundary lookups significantly faster.

---

## Recommended Implementation Order

| Priority | Item | Expected gain |
|---|---|---|
| 1 | `pg_trgm` name indexes (#1) | Eliminates seq scans on ILIKE — 10–50× faster name resolution |
| 2 | Materialise `osm_all` (#2) | Eliminates repeated 12-table UNION recomputation |
| 3 | `admin_level` index on boundaries (#9) | Faster suburb subqueries |
| 4 | PostgreSQL memory tuning (#3) | Better geometry operation performance |
| 5 | Skip analysis for aggregates (#7) | Saves 5–30s on ~30% of queries |
| 6 | Stream analysis output (#4) | Perceived latency massively improved |
| 7 | Concurrent RAG + graph (#5) | Saves 500ms–2s per query |
| 8 | LRU embed cache (#6) | Free speed for repeated queries |
| 9 | CLUSTER tables (#8) | Minor spatial scan improvement |

Items 1–4 are pure DB/infrastructure changes with no application code required.  
Items 5–8 are application-level changes in `core/` and `app.py`.
