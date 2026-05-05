"""
Geocoder Module - Minimal helpers for map centering only.

DESIGN PRINCIPLE: All place name resolution happens inside PostGIS at query time
via LLM-generated subqueries against osm_boundaries and osm_all. There are no
hardcoded coordinate dictionaries and no Nominatim calls in the query path.

The only function kept here is get_adelaide_center(), used by app.py to set the
default map view on startup — nothing to do with query resolution.

Nominatim (via geopy) is available in data/setup_db.py for data preparation only,
e.g. assigning suburb labels to graph nodes where the OSM boundary data is sparse.
"""


def get_adelaide_center() -> tuple[float, float]:
    """Return Adelaide CBD coordinates for the default map center."""
    return (-34.9285, 138.6007)
