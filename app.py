"""
Natural Language Spatial Search — Tabbed Chat + Map Interface.

Tab 1 (Chat): full conversation with follow-up context, suggested chips, schema browser.
Tab 2 (Map):  persistent Folium map, updates after every query, basemap switcher.

Powered by:
  - PostGIS SQL generation (Ollama local LLM) with conversation history
  - Hybrid RAG schema retrieval (pgvector + nomic-embed-text)
  - Spatial analysis agent (second LLM call)
  - Apache AGE property graph context
"""

import os

from dotenv import load_dotenv
import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_js_eval import get_geolocation

from core.schema import introspect_db, format_schema_for_llm, get_table_names
from core.rag import HybridRAG, retrieve_cached, prefetch, prefetch_async
from core.llm import generate_sql, query_requires_device_location
from core.executor import execute_query
from core.analyst import analyse, analyse_stream
from core.graph import get_graph_context_for_query, get_graph_summary, graph_is_available
from core.spatial_reasoner import decompose_intent
from core.cypher_generator import generate_cypher
from core.cypher_executor import execute_cypher

load_dotenv()

# ─── configuration ────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

# Max prior conversation turns passed to the LLM for follow-up context.
# Each turn = 1 user message + 1 assistant SQL message.
# Keep small to avoid blowing the context window.
CONVERSATION_HISTORY_TURNS = 3

BASEMAP_OPTIONS = {
    "Dark (CartoDB Dark Matter)": "CartoDB dark_matter",
    "Light (CartoDB Positron)":   "CartoDB positron",
    "OpenStreetMap":              "OpenStreetMap",
}

GEOM_COLORS = {
    "Point":              "#e74c3c",
    "MultiPoint":         "#e74c3c",
    "LineString":         "#3498db",
    "MultiLineString":    "#3498db",
    "Polygon":            "#2ecc71",
    "MultiPolygon":       "#27ae60",
    "GeometryCollection": "#9b59b6",
}

# Override colours by feature_type value (used in green-vs-buildings and UNION ALL queries)
FEATURE_TYPE_COLORS = {
    "park":     "#27ae60",   # green
    "building": "#e67e22",   # orange
    "green":    "#27ae60",
    "concrete": "#e67e22",
    "cafe":     "#9b59b6",
    "school":   "#3498db",
    "hospital": "#e74c3c",
    "pharmacy": "#f39c12",
    "road":     "#95a5a6",
    "waterway": "#2980b9",
    "railway":  "#7f8c8d",
}

EXAMPLE_QUERIES = [
    "Find 5 schools near Adelaide CBD",
    "Show me restaurants within 2km of Glenelg Beach",
    "What are the largest parks by area?",
    "Show all primary roads",
    "How many restaurants are there by type?",
    "Find parks with 'creek' in the name",
    "Show me hospitals near the University of Adelaide",
    "Which roads intersect parks?",
    "Show all waterways",
    "What landuse types are near Adelaide Airport?",
    "Find pharmacies near North Adelaide",
    "Show railway lines",
    "Compare green areas vs buildings in the Adelaide CBD",
    "Which suburbs have the most restaurants per square kilometre?",
    "Which hospitals have no pharmacy within 1km?",
    "Rank suburbs by cafe density",
    "Which schools have no restaurant within 500 metres?",
    "Show parks near the River Torrens",
    "Which suburbs have the most schools?",
    "For each hospital, show the nearest pharmacy",
    # Graph-traversal queries (routed to Cypher / Apache AGE)
    "Find all amenities reachable in 2 hops from schools in the Adelaide CBD",
    "What is the shortest path between a school and a pharmacy via NEAR edges?",
    "Find clusters where a school, hospital and park are all mutually near each other",
    "Which suburbs are connected to Norwood via shared features?",
    "Show restaurants reachable from parks via NEAR graph edges",
]

# ─── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Geo-Agentic Spatial Search",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── schema (cached) ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_schema():
    """Load and cache the database schema from PostgreSQL."""
    try:
        tables = introspect_db()
        schema_text = format_schema_for_llm(tables)
        table_names = get_table_names(tables)
        return tables, schema_text, table_names
    except Exception:
        return None, None, None


# ─── map helpers ──────────────────────────────────────────────────────────────

# Adelaide city viewport — default view locked to Adelaide CBD at zoom 12
ADELAIDE_CENTER = (-34.9285, 138.6007)
ADELAIDE_DEFAULT_ZOOM = 12

# Hard bounding box for South Australia — prevents panning/zooming outside SA
SA_BOUNDS = [[-38.5, 128.0], [-26.0, 141.0]]  # [[south, west], [north, east]]
SA_MIN_ZOOM = 7
SA_MAX_ZOOM = 18

def _create_base_map(basemap_name: str) -> folium.Map:
    tiles_arg = BASEMAP_OPTIONS.get(basemap_name, "CartoDB dark_matter")
    m = folium.Map(
        location=list(ADELAIDE_CENTER),
        zoom_start=ADELAIDE_DEFAULT_ZOOM,
        tiles=None,
        min_zoom=SA_MIN_ZOOM,
        max_zoom=SA_MAX_ZOOM,
        max_bounds=True,
        min_lat=SA_BOUNDS[0][0],
        max_lat=SA_BOUNDS[1][0],
        min_lon=SA_BOUNDS[0][1],
        max_lon=SA_BOUNDS[1][1],
    )
    folium.TileLayer(tiles=tiles_arg, name=basemap_name).add_to(m)
    for name, tiles in BASEMAP_OPTIONS.items():
        if name != basemap_name:
            folium.TileLayer(tiles=tiles, name=name).add_to(m)
    # NOTE: do NOT call m.fit_bounds(SA_BOUNDS) here — that would zoom out to
    # all of South Australia on every render. The SA bounds are enforced via
    # max_bounds=True and the min/max lat/lon parameters above (pan restriction).
    return m


def _add_location_marker(m: folium.Map, device_coords: tuple[float, float] | None) -> None:
    if device_coords is None:
        return
    plugins.LocateControl(
        auto_start=False, flyTo=True, showCompass=True,
        strings={"title": "Show my location"},
    ).add_to(m)
    lat, lng = device_coords
    folium.CircleMarker(
        location=[lat, lng], radius=9, color="#111827", weight=2,
        fill=True, fill_color="#f59e0b", fill_opacity=0.95,
        tooltip="Your current location",
    ).add_to(m)
    folium.Circle(
        location=[lat, lng], radius=250, color="#f59e0b", weight=2,
        fill=True, fill_color="#fbbf24", fill_opacity=0.12,
    ).add_to(m)


def build_default_map_html(basemap: str, device_coords=None) -> str:
    m = _create_base_map(basemap)
    _add_location_marker(m, device_coords)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def build_map_html(gdf: gpd.GeoDataFrame, basemap: str, device_coords=None, spatial_intent=None) -> str:
    m = _create_base_map(basemap)
    _add_location_marker(m, device_coords)

    if gdf is not None and not gdf.empty:
        bounds = gdf.total_bounds
        fit = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
        if device_coords:
            lat, lng = device_coords
            fit[0][0] = min(fit[0][0], lat)
            fit[0][1] = min(fit[0][1], lng)
            fit[1][0] = max(fit[1][0], lat)
            fit[1][1] = max(fit[1][1], lng)
        # Clamp result fit_bounds to SA extents so we never zoom outside SA
        fit[0][0] = max(fit[0][0], SA_BOUNDS[0][0])
        fit[0][1] = max(fit[0][1], SA_BOUNDS[0][1])
        fit[1][0] = min(fit[1][0], SA_BOUNDS[1][0])
        fit[1][1] = min(fit[1][1], SA_BOUNDS[1][1])
        m.fit_bounds(fit)

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # Prefer feature_type-based colour (UNION ALL / green-vs-buildings queries)
            feature_type = str(row.get("feature_type", "")).lower() if "feature_type" in gdf.columns else ""
            color = (
                FEATURE_TYPE_COLORS.get(feature_type)
                or GEOM_COLORS.get(geom.geom_type, "#95a5a6")
            )
            popup_parts = [
                f"<b>{col}</b>: {row[col]}"
                for col in gdf.columns
                if col != "geometry" and row[col] is not None
                and str(row[col]) not in ("", "None")
            ]
            popup_html = "<br>".join(popup_parts) or "No attributes"

            gtype = geom.geom_type
            if gtype in ("Point", "MultiPoint"):
                pts = [geom] if gtype == "Point" else list(geom.geoms)
                for pt in pts:
                    folium.CircleMarker(
                        location=[pt.y, pt.x], radius=7, color=color,
                        fill=True, fill_opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=300),
                    ).add_to(m)
            elif gtype in ("LineString", "MultiLineString"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x, c=color: {"color": c, "weight": 3, "opacity": 0.8},
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(m)
            elif gtype in ("Polygon", "MultiPolygon"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x, c=color: {
                        "color": c, "weight": 2, "fillColor": c, "fillOpacity": 0.3,
                    },
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(m)

    # ── Spatial reasoning overlays ────────────────────────────────────────────
    if spatial_intent is not None:
        _add_spatial_overlays(m, gdf, spatial_intent)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def _add_spatial_overlays(m: folium.Map, gdf, intent) -> None:
    """Add proximity rings and directional annotations based on SpatialIntent."""
    import math

    # Compute centroid of results to use as reference for overlays
    if gdf is None or gdf.empty:
        return
    try:
        # Project to a metre-based CRS before computing centroid to avoid the
        # "Geometry is in a geographic CRS" warning and get accurate coordinates.
        # EPSG:7854 = GDA2020 / MGA zone 54 — covers Adelaide.
        projected = gdf.to_crs(epsg=7854)
        centroid_proj = projected.geometry.centroid
        # Convert centroid back to WGS84 for Folium (which expects lon/lat degrees)
        import geopandas as _gpd
        centroid_wgs = _gpd.GeoSeries(centroid_proj, crs=7854).to_crs(epsg=4326)
        cx = float(centroid_wgs.x.mean())
        cy = float(centroid_wgs.y.mean())
    except Exception:
        return

    # Proximity rings — show if proximity intent detected
    dist_m = getattr(intent, "explicit_distance_m", None)
    has_fuzzy = getattr(intent, "has_fuzzy_proximity", False)
    primary = getattr(intent, "primary_intent", "unknown")

    if dist_m is not None and primary in (
        "proximity_search", "nearest_neighbour", "gap_analysis"
    ):
        tier = getattr(intent, "proximity_tier", "")
        folium.Circle(
            location=[cy, cx],
            radius=dist_m,
            color="#f59e0b",
            weight=2,
            fill=True,
            fill_color="#fbbf24",
            fill_opacity=0.06,
            tooltip=f"Search radius: {dist_m:,.0f} m ({tier})",
            dash_array="6",
        ).add_to(m)
    elif has_fuzzy:
        from core.spatial_concepts import DEFAULT_PROXIMITY_M
        tier = getattr(intent, "proximity_tier", "nearby")
        folium.Circle(
            location=[cy, cx],
            radius=DEFAULT_PROXIMITY_M,
            color="#94a3b8",
            weight=1,
            fill=True,
            fill_color="#cbd5e1",
            fill_opacity=0.05,
            tooltip=f"Fuzzy proximity: {DEFAULT_PROXIMITY_M:,} m ({tier})",
            dash_array="4",
        ).add_to(m)

    # Directional arrow annotation — add a simple bearing label marker
    directions = getattr(intent, "directions", [])
    if directions:
        _BEARING_MAP = {
            "north": 0, "northeast": 45, "east": 90, "southeast": 135,
            "south": 180, "southwest": 225, "west": 270, "northwest": 315,
        }
        _DIR_ARROW = {
            "north": "↑", "northeast": "↗", "east": "→", "southeast": "↘",
            "south": "↓", "southwest": "↙", "west": "←", "northwest": "↖",
        }
        for d in directions[:2]:  # show at most 2 directional indicators
            bearing_deg = _BEARING_MAP.get(d)
            arrow = _DIR_ARROW.get(d, "→")
            if bearing_deg is None:
                continue
            # Place the label 1.5 km from the centroid in the given direction
            offset_m = 1500
            rad = math.radians(bearing_deg)
            # Approximate degree offsets (1° lat ≈ 111 km, 1° lon ≈ 111*cos(lat) km)
            dlat = (offset_m / 111_000) * math.cos(rad)
            dlon = (offset_m / (111_000 * abs(math.cos(math.radians(cy))))) * math.sin(rad)
            folium.Marker(
                location=[cy + dlat, cx + dlon],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:20px;color:#f59e0b;'
                        f'text-shadow:0 0 4px #000;font-weight:bold;">'
                        f'{arrow} {d.title()}</div>'
                    ),
                    icon_size=(90, 28),
                    icon_anchor=(45, 14),
                ),
                tooltip=f"Directional filter: {d} of reference",
            ).add_to(m)


# ─── device location helper ───────────────────────────────────────────────────

def get_device_location(component_key: str):
    location = get_geolocation(component_key=component_key)
    if not location:
        return None, None, True
    if "error" in location:
        msg  = location["error"].get("message", "Unknown geolocation error.")
        code = location["error"].get("code")
        if code == 1:
            return None, "Location permission denied. Allow browser location access.", False
        return None, f"Unable to get device location: {msg}", False
    coords = location.get("coords", {})
    lat = coords.get("latitude")
    lng = coords.get("longitude")
    if lat is None or lng is None:
        return None, "Could not read device coordinates.", False
    return (lat, lng), None, False


# ─── conversation history builder ─────────────────────────────────────────────

def _build_conversation_history(messages: list[dict], max_turns: int) -> list[dict]:
    """
    Extract the last `max_turns` user/assistant exchange pairs from the session
    message list and return them as plain {"role", "content"} dicts suitable for
    injection into the LLM prompt.

    Assistant messages are reduced to just their SQL so the context window stays
    small — the LLM only needs to know *what was queried*, not the full rendered output.
    """
    history = []
    # Walk backwards to find the last N complete turns (user + assistant SQL)
    pairs = []
    i = len(messages) - 1
    while i >= 0 and len(pairs) < max_turns:
        msg = messages[i]
        if msg["role"] == "assistant" and not msg.get("error"):
            # look for the preceding user message
            if i > 0 and messages[i - 1]["role"] == "user":
                pairs.append((messages[i - 1], msg))
                i -= 2
                continue
        i -= 1

    # Reverse so oldest turn comes first
    for user_msg, asst_msg in reversed(pairs):
        history.append({"role": "user", "content": user_msg["content"]})
        sql = asst_msg.get("sql", "")
        if sql:
            history.append({"role": "assistant", "content": sql})

    return history


# ─── session state init ───────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "messages":               [],    # list of chat message dicts
        "current_map_html":       None,  # latest map HTML
        "current_gdf":            None,  # latest GeoDataFrame for basemap rerender
        "current_intent":         None,  # latest SpatialIntent for map overlays
        "basemap_selection":      "Dark (CartoDB Dark Matter)",
        "device_location":        None,
        "pending_location_query": None,
        "location_request_key":   0,
        "active_tab":             0,     # 0 = Chat, 1 = Map, 2 = Graph Explorer
        "rag_prefetched":         False, # guard: only prefetch once per process
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Warm the RAG cache for all example queries in the background so that
    # clicking any chip is instant — the embed is already done.
    if not st.session_state["rag_prefetched"]:
        prefetch(EXAMPLE_QUERIES)
        st.session_state["rag_prefetched"] = True


# ─── cypher pipeline (graph-traversal queries) ────────────────────────────────

def _run_cypher_pipeline(
    user_query: str,
    intent,
    all_tables,
    device_coords=None,
) -> dict:
    """
    Execute a graph-traversal query via Cypher → AGE → PostGIS geometry join.

    Called by run_pipeline when intent.requires_graph is True and the AGE graph
    is available.  Falls back to the SQL pipeline on Cypher generation failure.
    """
    # Generate Cypher
    cypher, error = generate_cypher(user_query, intent, model=OLLAMA_MODEL)
    if error:
        return {"role": "assistant", "error": f"Cypher generation failed: {error}", "cypher": None, "sql": None}

    # Execute Cypher + back-join geometry
    result = execute_cypher(cypher)
    if result.error:
        return {"role": "assistant", "error": result.error, "cypher": cypher, "sql": None}

    # Build a QueryResult-compatible object for analyse_stream
    from types import SimpleNamespace
    qr = SimpleNamespace(
        gdf=result.gdf,
        columns=result.columns,
        row_count=result.row_count,
        has_geometry=result.has_geometry,
        error=result.error,
        raw_rows=result.raw_rows,
    )

    # Analyse results
    analysis = analyse_stream(user_query, cypher, qr, intent=intent)

    # Build map
    basemap = st.session_state.get("basemap_selection", "Dark (CartoDB Dark Matter)")
    if result.has_geometry and result.gdf is not None and not result.gdf.empty:
        map_html = build_map_html(result.gdf, basemap, device_coords, spatial_intent=intent)
        st.session_state["current_gdf"] = result.gdf
        st.session_state["current_intent"] = intent
    else:
        map_html = st.session_state.get("current_map_html") or build_default_map_html(
            basemap, device_coords
        )

    # Build table dataframe
    import pandas as pd
    if result.has_geometry and result.gdf is not None:
        table_df = result.gdf.drop(columns=["geometry"], errors="ignore").reset_index(drop=True)
    elif result.raw_rows:
        table_df = pd.DataFrame(result.raw_rows)
    else:
        table_df = None

    return {
        "role":           "assistant",
        "summary":        analysis.summary,
        "stats":          analysis.stats,
        "followups":      analysis.followups,
        "sql":            None,
        "cypher":         cypher,          # shown in "View Cypher" expander
        "map_html":       map_html,
        "table_df":       table_df,
        "row_count":      result.row_count,
        "error":          None,
        "_stream":        analysis.stream,
        "spatial_intent": intent,
    }


# ─── core pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    schema_text: str,
    table_names: set[str],
    all_tables,
    device_coords=None,
) -> dict:
    """
    Full RAG → Spatial Reasoning → SQL → Execute → Analyse pipeline.
    Passes the last CONVERSATION_HISTORY_TURNS exchanges as context to the LLM
    so follow-up queries resolve references to prior results.
    Returns a result dict stored in messages.
    """
    from concurrent.futures import ThreadPoolExecutor

    ALWAYS_INCLUDE = {"osm_all", "osm_boundaries"}

    # 1. RAG + graph context + spatial intent decomposition — all independent.
    with ThreadPoolExecutor(max_workers=3) as pool:
        rag_future    = pool.submit(retrieve_cached, user_query)
        graph_future  = pool.submit(get_graph_context_for_query, user_query)
        intent_future = pool.submit(decompose_intent, user_query)

    relevant_tables = rag_future.result()
    graph_ctx       = graph_future.result()
    intent          = intent_future.result()

    if relevant_tables:
        from core.schema import format_schema_for_llm as _fmt
        want = set(relevant_tables) | ALWAYS_INCLUDE
        filtered_tables = [t for t in all_tables if t.name in want]
        rag_schema = _fmt(filtered_tables)
        rag_table_names = {t.name for t in filtered_tables} | table_names
    else:
        rag_schema = schema_text
        rag_table_names = table_names | ALWAYS_INCLUDE

    if graph_ctx:
        rag_schema = rag_schema + f"\n\n{graph_ctx}"

    # 2. Build conversation history for follow-up context
    history = _build_conversation_history(
        st.session_state["messages"], CONVERSATION_HISTORY_TURNS
    )

    # ── Engine routing ─────────────────────────────────────────────────────
    # Graph-traversal intents (multi-hop, path, cluster) are executed as
    # Cypher against Apache AGE.  All other intents use PostGIS SQL.
    if intent and intent.requires_graph and graph_is_available():
        return _run_cypher_pipeline(
            user_query, intent, all_tables, device_coords
        )

    # 3. Generate SQL (with spatial intent context injected)
    sql, error = generate_sql(
        user_query=user_query,
        schema_context=rag_schema,
        allowed_tables=rag_table_names,
        device_coords=device_coords,
        model=OLLAMA_MODEL,
        conversation_history=history,
        spatial_intent_context=intent.llm_context if intent else None,
    )

    if error:
        return {"role": "assistant", "error": error, "sql": None}

    # 4. Execute query
    result = execute_query(sql)

    if result.error:
        return {"role": "assistant", "error": result.error, "sql": sql}

    # 5. Analyse results (streaming — summary rendered live in _process_query)
    analysis = analyse_stream(user_query, sql, result, intent=intent)

    # 6. Build map HTML (with spatial reasoning overlays)
    basemap = st.session_state.get("basemap_selection", "Dark (CartoDB Dark Matter)")
    if result.has_geometry and result.gdf is not None and not result.gdf.empty:
        map_html = build_map_html(result.gdf, basemap, device_coords, spatial_intent=intent)
        st.session_state["current_gdf"] = result.gdf
        st.session_state["current_intent"] = intent
    else:
        map_html = st.session_state.get("current_map_html") or build_default_map_html(
            basemap, device_coords
        )

    # 7. Build table dataframe
    if result.has_geometry and result.gdf is not None:
        table_df = result.gdf.drop(columns=["geometry"], errors="ignore").reset_index(drop=True)
        # UNION ALL comparison queries: split geometry rows from stat/count summary rows
        if "count" in table_df.columns and "feature_type" in table_df.columns:
            count_rows = (
                table_df[table_df["count"].notna()][["feature_type", "count"]]
                .reset_index(drop=True)
            )
            geom_rows = (
                table_df[table_df["count"].isna()]
                .drop(columns=["count"], errors="ignore")
                .reset_index(drop=True)
            )
            table_df = geom_rows if not geom_rows.empty else table_df
            if not count_rows.empty:
                comparison = {
                    row["feature_type"]: row["count"]
                    for _, row in count_rows.iterrows()
                }
                if "comparison" not in analysis.stats:
                    analysis.stats["comparison"] = comparison
    elif result.raw_rows:
        table_df = pd.DataFrame(result.raw_rows)
    else:
        table_df = None

    return {
        "role":           "assistant",
        "summary":        analysis.summary,   # "" when streaming; filled by _process_query
        "stats":          analysis.stats,
        "followups":      analysis.followups,
        "sql":            sql,
        "map_html":       map_html,
        "table_df":       table_df,
        "row_count":      result.row_count,
        "error":          None,
        "_stream":        analysis.stream,    # generator or None; consumed once, then dropped
        "spatial_intent": intent,             # SpatialIntent for reasoning trace UI
    }


# ─── render helpers ───────────────────────────────────────────────────────────

def _render_assistant_message(msg: dict) -> None:
    """Render a single assistant message inside st.chat_message context."""
    if msg.get("error"):
        st.error(msg["error"])
        if msg.get("sql"):
            with st.expander("Generated SQL"):
                st.code(msg["sql"], language="sql")
        return

    row_count = msg.get("row_count", 0)
    st.markdown(f"**{row_count} result{'s' if row_count != 1 else ''} found**")

    summary = msg.get("summary", "")
    if summary:
        st.write(summary)

    # Comparison metric cards (UNION ALL suburb-comparison / green-vs-buildings queries)
    stats      = msg.get("stats", {})
    comparison = stats.get("comparison", {})
    if comparison:
        st.markdown("**Summary:**")
        cols = st.columns(len(comparison))
        for i, (label, val) in enumerate(comparison.items()):
            with cols[i]:
                # Format: floats with 4dp for ratios, integers with comma for areas
                try:
                    fval = float(val)
                    display = f"{fval:,.4f}" if fval != int(fval) else f"{int(fval):,}"
                except (TypeError, ValueError):
                    display = str(val)
                # Prettify label
                pretty = label.replace("_", " ").title()
                st.metric(pretty, display)

    # Spatial reasoning trace panel
    spatial_intent = msg.get("spatial_intent")
    if spatial_intent is not None:
        trace = spatial_intent.to_display_dict()
        if trace:
            with st.expander("Spatial Reasoning Trace", expanded=False):
                source = trace.pop("Source", "rule-based")
                badge_color = "#6366f1" if "llm" in source else "#64748b"
                st.markdown(
                    f'<span style="background:{badge_color};color:#fff;'
                    f'padding:2px 8px;border-radius:4px;font-size:0.75rem;">'
                    f'{source}</span>',
                    unsafe_allow_html=True,
                )
                for key, val in trace.items():
                    st.markdown(f"**{key}:** {val}")

    if msg.get("cypher"):
        with st.expander("View Cypher"):
            st.code(msg["cypher"], language="cypher")
    elif msg.get("sql"):
        with st.expander("View SQL"):
            st.code(msg["sql"], language="sql")

    display_stats = {
        k: v for k, v in stats.items()
        if k not in ("sample_names", "comparison")
    }
    if display_stats and len(display_stats) > 1:
        with st.expander("Statistics"):
            st.json(display_stats)

    if msg.get("table_df") is not None and not msg["table_df"].empty:
        with st.expander("Data Table"):
            st.dataframe(msg["table_df"], width="stretch")

    # Follow-up suggestion chips
    followups = msg.get("followups", [])
    if followups:
        st.caption("Suggested follow-ups:")
        cols = st.columns(len(followups))
        for i, fq in enumerate(followups):
            with cols[i]:
                if st.button(fq, key=f"fup_{hash(fq)}_{id(msg)}", width="stretch"):
                    st.session_state["_pending_chat_input"] = fq
                    st.rerun()


# ─── query processor (shared by chat input + follow-up chips) ─────────────────

def _process_query(user_query: str, all_tables, schema_text, table_names) -> None:
    """
    Run the full pipeline for user_query, append messages, update map state.
    Called from both the chat input handler and the pending-location handler.
    """
    st.session_state["messages"].append({"role": "user", "content": user_query})

    device_coords = st.session_state.get("device_location")

    # Handle device-location queries
    if query_requires_device_location(user_query) and device_coords is None:
        request_key = f"geo_{st.session_state['location_request_key']}"
        coords, location_error, waiting = get_device_location(request_key)
        if waiting:
            st.session_state["pending_location_query"] = user_query
            st.info("Waiting for browser location...")
            st.stop()
        elif location_error:
            st.session_state["messages"].append(
                {"role": "assistant", "error": location_error, "sql": None}
            )
            st.session_state["location_request_key"] += 1
            st.rerun()
        else:
            device_coords = coords
            st.session_state["device_location"] = coords
            st.session_state["location_request_key"] += 1

    with st.spinner("Thinking..."):
        result_msg = run_pipeline(
            user_query, schema_text, table_names, all_tables,
            device_coords=device_coords,
        )

    # Stream the analysis summary live if a generator is present
    stream = result_msg.pop("_stream", None)
    if stream is not None:
        with st.chat_message("assistant"):
            st.markdown(f"**{result_msg.get('row_count', 0)} result{'s' if result_msg.get('row_count', 0) != 1 else ''} found**")
            accumulated = st.write_stream(stream)
            result_msg["summary"] = accumulated or ""
    elif not result_msg.get("summary"):
        count = result_msg.get("row_count", 0)
        result_msg["summary"] = (
            "No features were found matching your query."
            if count == 0
            else f"Found {count} feature{'s' if count != 1 else ''} matching your query."
        )

    st.session_state["messages"].append(result_msg)
    if result_msg.get("map_html"):
        st.session_state["current_map_html"] = result_msg["map_html"]

    st.rerun()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    _init_session()

    all_tables, schema_text, table_names = load_schema()

    if schema_text is None:
        st.error(
            "Cannot connect to the database. "
            "Make sure PostgreSQL is running: `docker-compose up -d` "
            "and data is loaded: `python data/setup_db.py`"
        )
        st.stop()

    st.title("Geo-Agentic Spatial Search")
    st.caption(
        "Ask questions about Adelaide's spatial data in plain English. "
        "Powered by PostGIS · pgvector · Apache AGE · Ollama."
    )

    tab_chat, tab_map, tab_graph = st.tabs(["Chat", "Map", "Graph Explorer"])

    # ── Tab 1: Chat ───────────────────────────────────────────────────────────
    with tab_chat:
        # Schema browser + example queries in the sidebar-style expanders at top
        col_tools, col_spacer = st.columns([1, 2])
        with col_tools:
            with st.expander("Database Schema", expanded=False):
                for table in all_tables:
                    if table.name == "osm_all":
                        continue
                    geom_desc = f" ({table.geometry_type})" if table.geometry_type else ""
                    st.markdown(f"**{table.name}** — {table.row_count} rows{geom_desc}")
                    col_names = [
                        c.name for c in table.columns
                        if not c.is_geometry and c.name not in ("id", "osm_id")
                    ]
                    if col_names:
                        st.caption(", ".join(col_names))

            with st.expander("Example queries", expanded=False):
                for q in EXAMPLE_QUERIES:
                    if st.button(q, key=f"ex_{hash(q)}", width="stretch"):
                        st.session_state["_pending_chat_input"] = q
                        st.rerun()

        st.divider()

        # Conversation history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    _render_assistant_message(msg)

        # Chat input
        # prefetch_async fires when the user submits — the embed is in-flight
        # while the pipeline starts, so it usually completes before RAG needs it.
        pending    = st.session_state.pop("_pending_chat_input", None)
        user_input = st.chat_input("Ask a spatial question...", key="chat_input_box")

        # Fire prefetch as soon as we have the query text
        if user_input:
            prefetch_async(user_input.strip())

        query_to_run = pending or user_input

        if query_to_run and query_to_run.strip():
            _process_query(
                query_to_run.strip(), all_tables, schema_text, table_names
            )

        # Resume a pending device-location query from the previous render cycle
        pending_loc_query = st.session_state.get("pending_location_query")
        if pending_loc_query:
            request_key = f"geo_{st.session_state['location_request_key']}"
            coords, location_error, waiting = get_device_location(request_key)
            if not waiting:
                st.session_state["pending_location_query"] = None
                st.session_state["location_request_key"] += 1
                if location_error:
                    st.session_state["messages"].append(
                        {"role": "assistant", "error": location_error, "sql": None}
                    )
                    st.rerun()
                else:
                    st.session_state["device_location"] = coords
                    with st.spinner("Thinking..."):
                        result_msg = run_pipeline(
                            pending_loc_query, schema_text, table_names, all_tables,
                            device_coords=coords,
                        )
                    st.session_state["messages"].append(result_msg)
                    if result_msg.get("map_html"):
                        st.session_state["current_map_html"] = result_msg["map_html"]
                    st.rerun()

    # ── Tab 2: Map ────────────────────────────────────────────────────────────
    with tab_map:
        basemap = st.selectbox(
            "Basemap",
            list(BASEMAP_OPTIONS.keys()),
            index=0,
            key="basemap_selection",
            label_visibility="collapsed",
        )

        # Re-render when basemap changes
        current_gdf = st.session_state.get("current_gdf")
        if current_gdf is not None:
            st.session_state["current_map_html"] = build_map_html(
                current_gdf, basemap,
                device_coords=st.session_state.get("device_location"),
                spatial_intent=st.session_state.get("current_intent"),
            )

        map_html = st.session_state.get("current_map_html") or build_default_map_html(
            basemap, device_coords=st.session_state.get("device_location")
        )
        components.html(map_html, height=700, scrolling=False)

        # Show which query produced the current map
        messages = st.session_state.get("messages", [])
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), None
        )
        if last_user:
            st.caption(f"Showing results for: *{last_user}*")

    # ── Tab 3: Graph Explorer ─────────────────────────────────────────────────
    with tab_graph:
        st.markdown("### Apache AGE Graph Explorer")
        st.caption(
            "Run openCypher queries directly against the `osm_spatial` property graph. "
            "Node labels: School, Hospital, Restaurant, Pharmacy, Park, Building, "
            "Road, Waterway, Railway, Landuse, NaturalFeature, Boundary. "
            "Edge types: NEAR (distance_m), WITHIN (boundary_name)."
        )

        graph_available = graph_is_available()
        if not graph_available:
            st.warning(
                "Graph is not available. Ensure the database is running and "
                "`python data/setup_db.py` has been run to build the AGE graph."
            )
        else:
            # Edge summary
            with st.expander("Graph Statistics", expanded=True):
                summary = get_graph_summary()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("NEAR edges", f"{summary.get('near_total', 0):,}")
                with col2:
                    st.metric("WITHIN edges", f"{summary.get('within_total', 0):,}")

                near_pairs = summary.get("near", {})
                if near_pairs:
                    st.markdown("**NEAR edge pairs:**")
                    pairs_data = [
                        {"Relationship": k, "Count": v}
                        for k, v in sorted(near_pairs.items(), key=lambda x: -x[1])
                    ]
                    st.dataframe(pd.DataFrame(pairs_data), width="stretch", hide_index=True)

            # Example Cypher queries
            cypher_examples = [
                "MATCH (s:School)-[r:NEAR]->(p:Park) RETURN s.name AS school, p.name AS park, r.distance_m AS distance_m ORDER BY r.distance_m LIMIT 10",
                "MATCH (h:Hospital)-[r:NEAR]->(p:Pharmacy) RETURN h.name AS hospital, p.name AS pharmacy, r.distance_m AS distance_m ORDER BY r.distance_m LIMIT 10",
                "MATCH (r:Restaurant)-[n:NEAR]->(p:Park) RETURN r.name AS restaurant, p.name AS park, n.distance_m AS distance_m ORDER BY n.distance_m LIMIT 10",
                "MATCH (s:School)-[:WITHIN]->(b:Boundary) RETURN b.name AS suburb, count(s) AS school_count ORDER BY school_count DESC LIMIT 15",
                "MATCH (h:Hospital)-[:NEAR]->(p:Pharmacy) WITH h, count(p) AS nearby_pharmacies RETURN h.name AS hospital, nearby_pharmacies ORDER BY nearby_pharmacies DESC LIMIT 10",
            ]
            with st.expander("Example Cypher queries", expanded=False):
                for ex in cypher_examples:
                    if st.button(ex[:80] + ("..." if len(ex) > 80 else ""), key=f"cypher_ex_{hash(ex)}", width="stretch"):
                        st.session_state["_pending_cypher"] = ex
                        st.rerun()

            # Cypher input
            pending_cypher = st.session_state.pop("_pending_cypher", None)
            cypher_input = st.text_area(
                "Cypher query",
                value=pending_cypher or "",
                height=100,
                placeholder="MATCH (s:School)-[r:NEAR]->(p:Park) RETURN s.name, p.name, r.distance_m LIMIT 10",
                key="cypher_input_box",
            )
            run_cypher = st.button("Run Cypher", type="primary")

            if run_cypher and cypher_input.strip():
                from core.graph import _cypher
                with st.spinner("Running Cypher query..."):
                    rows = _cypher(cypher_input.strip())

                if not rows:
                    st.info("No results returned.")
                else:
                    # Parse agtype rows into a DataFrame
                    import json
                    parsed = []
                    for row in rows:
                        parsed_row = {}
                        for i, val in enumerate(row):
                            val_str = str(val)
                            try:
                                parsed_row[f"col_{i}"] = json.loads(val_str)
                            except (json.JSONDecodeError, TypeError):
                                # Strip surrounding quotes from agtype strings
                                parsed_row[f"col_{i}"] = val_str.strip('"')
                        parsed.append(parsed_row)

                    df = pd.DataFrame(parsed)
                    st.success(f"{len(df)} rows returned")
                    st.dataframe(df, width="stretch")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Settings")
        st.caption(f"Model: `{OLLAMA_MODEL}`")
        st.caption(f"Follow-up context: last {CONVERSATION_HISTORY_TURNS} turns")
        if st.button("Clear conversation"):
            st.session_state["messages"]         = []
            st.session_state["current_map_html"] = None
            st.session_state["current_gdf"]      = None
            st.rerun()


if __name__ == "__main__":
    main()
