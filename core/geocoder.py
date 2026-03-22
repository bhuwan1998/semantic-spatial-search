"""
Geocoder Module - Resolves Adelaide place names to coordinates.

Provides a hardcoded dictionary of well-known Adelaide landmarks and suburbs,
with a fallback to Nominatim (OpenStreetMap geocoder) for unknown places.
"""

# Adelaide CBD center: -34.9285, 138.6007

ADELAIDE_LANDMARKS = {
    # CBD and inner city
    "adelaide cbd": (-34.9285, 138.6007),
    "adelaide": (-34.9285, 138.6007),
    "city center": (-34.9285, 138.6007),
    "city centre": (-34.9285, 138.6007),
    "rundle mall": (-34.9228, 138.6040),
    "victoria square": (-34.9290, 138.5999),
    "adelaide oval": (-34.9156, 138.5961),
    "adelaide central market": (-34.9302, 138.5978),
    "central market": (-34.9302, 138.5978),
    "adelaide railway station": (-34.9210, 138.5963),
    "parliament house": (-34.9212, 138.5995),
    "adelaide town hall": (-34.9262, 138.5999),
    "adelaide convention centre": (-34.9217, 138.5912),

    # Parks and gardens
    "adelaide botanic garden": (-34.9176, 138.6088),
    "botanic garden": (-34.9176, 138.6088),
    "elder park": (-34.9210, 138.5906),
    "rymill park": (-34.9240, 138.6090),
    "bonython park": (-34.9173, 138.5803),
    "park lands": (-34.9260, 138.6000),

    # Beaches
    "glenelg": (-34.9818, 138.5149),
    "glenelg beach": (-34.9818, 138.5149),
    "henley beach": (-34.9216, 138.4963),
    "semaphore": (-34.8360, 138.4848),
    "brighton": (-35.0185, 138.5225),
    "west beach": (-34.9490, 138.5090),

    # Suburbs - North
    "north adelaide": (-34.9070, 138.5950),
    "prospect": (-34.8832, 138.5927),
    "walkerville": (-34.8940, 138.6150),
    "medindie": (-34.8980, 138.6040),
    "nailsworth": (-34.8870, 138.5990),

    # Suburbs - East
    "norwood": (-34.9210, 138.6300),
    "burnside": (-34.9420, 138.6540),
    "glen osmond": (-34.9540, 138.6480),
    "magill": (-34.9170, 138.6680),
    "unley": (-34.9490, 138.6050),

    # Suburbs - South
    "marion": (-35.0120, 138.5560),
    "morphett vale": (-35.1290, 138.5200),
    "noarlunga": (-35.1460, 138.4960),
    "reynella": (-35.0970, 138.5340),
    "hallett cove": (-35.0790, 138.5170),

    # Suburbs - West
    "port adelaide": (-34.8470, 138.5060),
    "hindmarsh": (-34.9070, 138.5720),
    "thebarton": (-34.9180, 138.5680),
    "torrensville": (-34.9200, 138.5590),

    # Education
    "university of adelaide": (-34.9200, 138.6040),
    "flinders university": (-35.0190, 138.5680),
    "university of south australia": (-34.9090, 138.5700),
    "unisa": (-34.9090, 138.5700),

    # Hills
    "mount lofty": (-34.9770, 138.7080),
    "stirling": (-35.0010, 138.7210),
    "crafers": (-35.0220, 138.7160),
    "mount barker": (-35.0690, 138.8580),

    # Major infrastructure
    "adelaide airport": (-34.9461, 138.5306),
    "adelaide zoo": (-34.9117, 138.6010),
    "royal adelaide hospital": (-34.9210, 138.5870),

    # River
    "river torrens": (-34.9180, 138.5960),
    "torrens river": (-34.9180, 138.5960),
}


def geocode(place_name: str) -> tuple[float, float] | None:
    """
    Resolve a place name to (latitude, longitude) coordinates.

    Returns (lat, lng) tuple or None if not found.
    First checks the hardcoded Adelaide landmarks dictionary,
    then falls back to Nominatim.
    """
    # Normalize the input
    normalized = place_name.strip().lower()

    # Check hardcoded landmarks
    if normalized in ADELAIDE_LANDMARKS:
        return ADELAIDE_LANDMARKS[normalized]

    # Partial match: check if the query is a substring of any landmark
    for key, coords in ADELAIDE_LANDMARKS.items():
        if normalized in key or key in normalized:
            return coords

    # Fallback to Nominatim (OpenStreetMap geocoder)
    try:
        from urllib.request import urlopen, Request
        import json

        query = f"{place_name}, Adelaide, South Australia"
        url = (
            f"https://nominatim.openstreetmap.org/search?"
            f"q={query}&format=json&limit=1"
        )
        req = Request(url, headers={"User-Agent": "GeoSpatialSearch/1.0"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data:
                lat = float(data[0]["lat"])
                lng = float(data[0]["lon"])
                return (lat, lng)
    except Exception:
        pass

    return None


def get_adelaide_center() -> tuple[float, float]:
    """Return Adelaide CBD coordinates."""
    return (-34.9285, 138.6007)
