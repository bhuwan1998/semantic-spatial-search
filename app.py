"""
Natural Language Spatial Search - Streamlit Application

A proof-of-concept app that translates natural language queries into
SpatiaLite SQL and visualizes results on an interactive map.
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

from core.schema import introspect_gpkg, format_schema_for_llm, get_table_names
from core.llm import generate_sql, query_requires_device_location
from core.executor import execute_query
from core.geocoder import get_adelaide_center

# Load .env file (OLLAMA_HOST, OLLAMA_MODEL, GPKG_PATH, etc.)
load_dotenv()

# --- Configuration ---
GPKG_PATH = os.environ.get("GPKG_PATH", "data/adelaide_osm.gpkg")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Available basemap tile layers (name -> folium tiles arg)
BASEMAP_OPTIONS = {
    "Dark (CartoDB Dark Matter)": "CartoDB dark_matter",
    "Light (CartoDB Positron)": "CartoDB positron",
    "OpenStreetMap": "OpenStreetMap"
}

# Attribution strings for custom tile URLs
BASEMAP_ATTR = {
    "Satellite (Esri WorldImagery)": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
    "Topo (OpenTopoMap)": 'Map data &copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors, SRTM | Style &copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
    "Watercolor (Stamen)": 'Map tiles by <a href="https://stamen.com">Stamen Design</a>, under <a href="https://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="https://openstreetmap.org">OpenStreetMap</a>',
}

# Geometry type to color mapping
GEOM_COLORS = {
    "Point": "#e74c3c",
    "MultiPoint": "#e74c3c",
    "LineString": "#3498db",
    "MultiLineString": "#3498db",
    "Polygon": "#2ecc71",
    "MultiPolygon": "#27ae60",
    "GeometryCollection": "#9b59b6",
}

# Example queries for the sidebar
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
    "Find parks near me",
]


def get_device_location(
    component_key: str,
) -> tuple[tuple[float, float] | None, str | None, bool]:
    """Request the browser's geolocation and normalize the response."""
    location = get_geolocation(component_key=component_key)

    if not location:
        return None, None, True

    if "error" in location:
        error = location["error"]
        code = error.get("code")
        message = error.get("message", "Unknown geolocation error.")

        if code == 1:
            return None, (
                "Location permission was denied. "
                "Allow browser location access to search near you."
            ), False
        return None, f"Unable to retrieve device location: {message}", False

    coords = location.get("coords", {})
    latitude = coords.get("latitude")
    longitude = coords.get("longitude")

    if latitude is None or longitude is None:
        return None, "Unable to read device coordinates from the browser response.", False

    return (latitude, longitude), None, False


def execute_search(
    user_query: str,
    schema_text: str,
    table_names: set[str],
    device_coords: tuple[float, float] | None = None,
):
    """Generate SQL, execute it, and store results in session state."""
    with st.spinner("Generating spatial SQL..."):
        sql, error = generate_sql(
            user_query=user_query,
            schema_context=schema_text,
            allowed_tables=table_names,
            device_coords=device_coords,
            model=OLLAMA_MODEL,
        )

    if error:
        st.session_state["last_error"] = f"SQL Generation Error: {error}"
        return

    st.session_state["last_sql"] = sql

    with st.spinner("Executing query..."):
        result = execute_query(GPKG_PATH, sql)

    if result.error:
        st.session_state["last_error"] = f"Execution Error: {result.error}"
        st.session_state["last_sql"] = sql
        return

    st.session_state["last_row_count"] = result.row_count
    st.session_state["last_has_geometry"] = result.has_geometry

    if result.has_geometry and result.gdf is not None and not result.gdf.empty:
        selected_basemap = st.session_state.get("basemap_selection", "Dark (CartoDB Dark Matter)")
        st.session_state["last_map_html"] = build_map_html(
            result.gdf,
            selected_basemap,
            device_coords=st.session_state.get("device_location"),
        )
        st.session_state["last_gdf"] = result.gdf
        st.session_state["last_basemap"] = selected_basemap
        st.session_state["last_table_df"] = result.gdf.drop(
            columns=["geometry"], errors="ignore"
        ).reset_index(drop=True)
        st.session_state["last_geom_types"] = result.gdf.geom_type.unique().tolist()
    elif result.raw_rows:
        st.session_state["last_table_df"] = pd.DataFrame(result.raw_rows)

    st.session_state["query_history"].append({
        "query": user_query,
        "sql": sql,
        "count": result.row_count,
    })


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Spatial Search - Natural Language",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_schema(gpkg_path: str):
    """Load and cache the database schema."""
    tables = introspect_gpkg(gpkg_path)
    schema_text = format_schema_for_llm(tables)
    table_names = get_table_names(tables)
    return tables, schema_text, table_names


def _create_base_map(basemap_name: str) -> folium.Map:
    """Create a Folium map with the selected basemap and layer control."""
    center = get_adelaide_center()
    tiles_arg = BASEMAP_OPTIONS[basemap_name]
    attr = BASEMAP_ATTR.get(basemap_name)

    # Built-in folium tile names vs custom URLs
    if tiles_arg.startswith("http"):
        m = folium.Map(location=list(center), zoom_start=12, tiles=None)
        folium.TileLayer(
            tiles=tiles_arg, attr=attr or "", name=basemap_name, max_zoom=19,
        ).add_to(m)
    else:
        m = folium.Map(location=list(center), zoom_start=12, tiles=None)
        folium.TileLayer(tiles=tiles_arg, name=basemap_name).add_to(m)

    # Add all other basemaps as toggleable layers
    for name, tiles in BASEMAP_OPTIONS.items():
        if name == basemap_name:
            continue
        extra_attr = BASEMAP_ATTR.get(name)
        if tiles.startswith("http"):
            folium.TileLayer(
                tiles=tiles, attr=extra_attr or "", name=name, max_zoom=19,
            ).add_to(m)
        else:
            folium.TileLayer(tiles=tiles, name=name).add_to(m)

    return m


def _add_current_location_controls(
    m: folium.Map,
    device_coords: tuple[float, float] | None = None,
):
    """Add current-location controls and overlays to the map."""
    plugins.LocateControl(
        auto_start=False,
        flyTo=True,
        keepCurrentZoomLevel=False,
        showCompass=True,
        strings={"title": "Show my location"},
    ).add_to(m)

    if device_coords is None:
        return

    lat, lng = device_coords
    folium.CircleMarker(
        location=[lat, lng],
        radius=9,
        color="#111827",
        weight=2,
        fill=True,
        fill_color="#f59e0b",
        fill_opacity=0.95,
        tooltip="Your current location",
        popup=folium.Popup("Your current location", max_width=220),
    ).add_to(m)
    folium.Circle(
        location=[lat, lng],
        radius=250,
        color="#f59e0b",
        weight=2,
        fill=True,
        fill_color="#fbbf24",
        fill_opacity=0.12,
    ).add_to(m)


def build_default_map_html(
    basemap_name: str,
    device_coords: tuple[float, float] | None = None,
) -> str:
    """Build a default Folium map centered on Adelaide with no overlays."""
    m = _create_base_map(basemap_name)
    _add_current_location_controls(m, device_coords)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def build_map_html(
    gdf: gpd.GeoDataFrame,
    basemap_name: str,
    device_coords: tuple[float, float] | None = None,
) -> str:
    """Build a Folium map and return its HTML string."""
    m = _create_base_map(basemap_name)
    _add_current_location_controls(m, device_coords)

    if gdf is not None and not gdf.empty:
        # Auto-fit bounds to data
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        fit_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
        if device_coords is not None:
            lat, lng = device_coords
            fit_bounds[0][0] = min(fit_bounds[0][0], lat)
            fit_bounds[0][1] = min(fit_bounds[0][1], lng)
            fit_bounds[1][0] = max(fit_bounds[1][0], lat)
            fit_bounds[1][1] = max(fit_bounds[1][1], lng)
        m.fit_bounds(fit_bounds)

        # Add features to map
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            geom_type = geom.geom_type
            color = GEOM_COLORS.get(geom_type, "#95a5a6")

            # Build popup content from non-geometry columns
            popup_parts = []
            for col in gdf.columns:
                if col == "geometry":
                    continue
                val = row[col]
                if val is not None and str(val) != "None" and str(val) != "":
                    popup_parts.append(f"<b>{col}</b>: {val}")
            popup_html = "<br>".join(popup_parts) if popup_parts else "No attributes"

            if geom_type in ("Point", "MultiPoint"):
                if geom_type == "Point":
                    points = [geom]
                else:
                    points = list(geom.geoms)
                for pt in points:
                    folium.CircleMarker(
                        location=[pt.y, pt.x],
                        radius=7,
                        color=color,
                        fill=True,
                        fill_opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=300),
                    ).add_to(m)

            elif geom_type in ("LineString", "MultiLineString"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x, c=color: {
                        "color": c,
                        "weight": 3,
                        "opacity": 0.8,
                    },
                ).add_to(m)

            elif geom_type in ("Polygon", "MultiPolygon"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x, c=color: {
                        "color": c,
                        "weight": 2,
                        "fillColor": c,
                        "fillOpacity": 0.3,
                    },
                ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def render_sidebar(tables, table_names):
    """Render the sidebar with schema info and example queries."""
    with st.sidebar:
        st.header("Database Schema")

        # Layer overview
        for table in tables:
            with st.expander(
                f"{table.name} ({table.row_count} rows)",
                expanded=False,
            ):
                if table.geometry_type:
                    st.caption(f"Geometry: {table.geometry_type} (SRID {table.srid})")

                cols = [c for c in table.columns if not c.is_geometry and c.name != "fid"]
                if cols:
                    col_names = [c.name for c in cols]
                    st.text(f"Columns: {', '.join(col_names)}")

                if table.sample_values:
                    for col_name, vals in list(table.sample_values.items())[:3]:
                        if col_name == "fid":
                            continue
                        st.text(f"  {col_name}: {', '.join(vals[:3])}")

        st.divider()
        st.header("Example Queries")
        st.caption("Click to use:")

        for query in EXAMPLE_QUERIES:
            if st.button(query, key=f"example_{hash(query)}", width="stretch"):
                st.session_state["pending_query"] = query
                st.rerun()

        st.divider()
        st.header("Basemap")
        basemap_names = list(BASEMAP_OPTIONS.keys())
        st.selectbox(
            "Select basemap:",
            basemap_names,
            index=0,
            key="basemap_selection",
        )

        st.divider()
        st.caption(f"Model: {OLLAMA_MODEL}")
        st.caption(f"Database: {GPKG_PATH}")


def main():
    """Main application entry point."""
    setup_page()

    st.title("Natural Language Spatial Search")
    st.caption(
        "Ask questions about Adelaide's spatial data in plain English. "
        "Powered by Llama 3.1 via Ollama + SpatiaLite."
    )

    # Check if database exists
    if not os.path.exists(GPKG_PATH):
        st.error(
            f"Database not found at `{GPKG_PATH}`. "
            f"Run `python data/setup_data.py` first to download the Adelaide OSM data."
        )
        st.stop()

    # Load schema
    try:
        tables, schema_text, table_names = load_schema(GPKG_PATH)
    except Exception as e:
        st.error(f"Error loading database schema: {e}")
        st.stop()

    # Render sidebar
    render_sidebar(tables, table_names)

    # Initialize session state
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "last_sql" not in st.session_state:
        st.session_state["last_sql"] = None
    if "last_map_html" not in st.session_state:
        st.session_state["last_map_html"] = None
    if "last_table_df" not in st.session_state:
        st.session_state["last_table_df"] = None
    if "last_row_count" not in st.session_state:
        st.session_state["last_row_count"] = None
    if "last_has_geometry" not in st.session_state:
        st.session_state["last_has_geometry"] = False
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_geom_types" not in st.session_state:
        st.session_state["last_geom_types"] = []
    if "last_gdf" not in st.session_state:
        st.session_state["last_gdf"] = None
    if "last_basemap" not in st.session_state:
        st.session_state["last_basemap"] = None
    if "device_location" not in st.session_state:
        st.session_state["device_location"] = None
    if "pending_location_query" not in st.session_state:
        st.session_state["pending_location_query"] = None
    if "location_request_key" not in st.session_state:
        st.session_state["location_request_key"] = 0
    if "show_my_location_request" not in st.session_state:
        st.session_state["show_my_location_request"] = False

    # If an example query was clicked, pre-fill the input
    default_value = st.session_state.pop("pending_query", "")

    # Query input
    user_query = st.text_input(
        "Ask a spatial question:",
        value=default_value,
        placeholder="e.g., Find 5 schools near Adelaide CBD",
        key="query_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button("Search", type="primary", width="stretch")
    with col2:
        if st.button("Show My Location", width="content"):
            st.session_state["show_my_location_request"] = True
            st.rerun()

    # Process query -- only regenerate SQL + execute on button click
    if run_button and user_query and user_query.strip():
        # Clear previous results
        st.session_state["last_sql"] = None
        st.session_state["last_map_html"] = None
        st.session_state["last_table_df"] = None
        st.session_state["last_row_count"] = None
        st.session_state["last_has_geometry"] = False
        st.session_state["last_error"] = None
        st.session_state["last_geom_types"] = []

        device_coords = None
        if query_requires_device_location(user_query):
            request_key = f"geo_request_{st.session_state['location_request_key']}"
            with st.spinner("Requesting your device location..."):
                device_coords, location_error, waiting_for_location = get_device_location(
                    component_key=request_key
                )

            if waiting_for_location:
                st.session_state["pending_location_query"] = user_query
                st.info("Waiting for your browser to return device location...")
                st.rerun()
            elif location_error:
                st.session_state["last_error"] = location_error
                st.session_state["device_location"] = None
                st.session_state["pending_location_query"] = None
            else:
                st.session_state["device_location"] = device_coords
                st.session_state["pending_location_query"] = None
                st.session_state["location_request_key"] += 1

        if not st.session_state["last_error"] and st.session_state["pending_location_query"] is None:
            execute_search(user_query, schema_text, table_names, device_coords=device_coords)

    elif run_button:
        st.warning("Please enter a query.")

    pending_location_query = st.session_state.get("pending_location_query")
    if pending_location_query:
        request_key = f"geo_request_{st.session_state['location_request_key']}"
        device_coords, location_error, waiting_for_location = get_device_location(
            component_key=request_key
        )

        if not waiting_for_location:
            if location_error:
                st.session_state["last_error"] = location_error
                st.session_state["device_location"] = None
                st.session_state["pending_location_query"] = None
                st.session_state["location_request_key"] += 1
                st.rerun()

            st.session_state["device_location"] = device_coords
            st.session_state["pending_location_query"] = None
            st.session_state["location_request_key"] += 1
            execute_search(
                pending_location_query,
                schema_text,
                table_names,
                device_coords=device_coords,
            )

    if st.session_state.get("show_my_location_request"):
        request_key = f"geo_preview_{st.session_state['location_request_key']}"
        device_coords, location_error, waiting_for_location = get_device_location(
            component_key=request_key
        )

        if waiting_for_location:
            st.info("Waiting for your browser to return device location...")
        elif location_error:
            st.session_state["last_error"] = location_error
            st.session_state["show_my_location_request"] = False
            st.session_state["location_request_key"] += 1
            st.rerun()
        else:
            st.session_state["device_location"] = device_coords
            st.session_state["show_my_location_request"] = False
            st.session_state["location_request_key"] += 1
            if st.session_state.get("last_gdf") is not None:
                selected_basemap = st.session_state.get("basemap_selection", "Dark (CartoDB Dark Matter)")
                st.session_state["last_map_html"] = build_map_html(
                    st.session_state["last_gdf"],
                    selected_basemap,
                    device_coords=device_coords,
                )
                st.session_state["last_basemap"] = selected_basemap
            else:
                st.session_state["last_map_html"] = build_default_map_html(
                    st.session_state.get("basemap_selection", "Dark (CartoDB Dark Matter)"),
                    device_coords=device_coords,
                )

    # --- Persistent map: always visible ---
    selected_basemap = st.session_state.get("basemap_selection", "Dark (CartoDB Dark Matter)")

    # Re-render results map if basemap selection changed since last render
    if (
        st.session_state.get("last_gdf") is not None
        and st.session_state.get("last_basemap") != selected_basemap
    ):
        st.session_state["last_map_html"] = build_map_html(
            st.session_state["last_gdf"],
            selected_basemap,
            device_coords=st.session_state.get("device_location"),
        )
        st.session_state["last_basemap"] = selected_basemap

    # Show the map with results overlaid, or a default Adelaide view
    map_html = st.session_state.get("last_map_html")
    if map_html:
        components.html(map_html, height=550, scrolling=False)

        # Legend
        geom_types = st.session_state.get("last_geom_types", [])
        legend_items = []
        for gt in geom_types:
            color = GEOM_COLORS.get(gt, "#95a5a6")
            legend_items.append(f":{color[1:]}[●] {gt}")
        if legend_items:
            st.caption(" | ".join(legend_items))
    else:
        # Default Adelaide map -- always shown on load
        components.html(
            build_default_map_html(
                selected_basemap,
                device_coords=st.session_state.get("device_location"),
            ),
            height=550,
            scrolling=False,
        )

    # --- Results info and SQL below the map ---
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
        if st.session_state.get("last_sql"):
            with st.expander("Generated SQL", expanded=True):
                st.code(st.session_state["last_sql"], language="sql")
        st.info("Try rephrasing your question.")

    elif st.session_state.get("last_sql"):
        row_count = st.session_state.get("last_row_count")
        if row_count is not None:
            st.success(f"Found {row_count} results")

        with st.expander("Generated SQL", expanded=False):
            st.code(st.session_state["last_sql"], language="sql")

        table_df = st.session_state.get("last_table_df")
        if table_df is not None and not table_df.empty:
            with st.expander("Data Table", expanded=False):
                st.dataframe(table_df, width="stretch")

        elif row_count == 0:
            st.info("No results found. Try a different query.")

    # Show query history
    if st.session_state.get("query_history"):
        with st.expander("Query History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state["query_history"])):
                st.text(f"{len(st.session_state['query_history']) - i}. {entry['query']}")
                st.code(entry["sql"], language="sql")
                st.caption(f"Results: {entry['count']}")
                st.divider()


if __name__ == "__main__":
    main()
