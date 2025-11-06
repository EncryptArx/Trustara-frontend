from geopy.geocoders import Nominatim


def get_geo_tag() -> dict:
    """
    MVP placeholder: returns a static example or attempts coarse lookup.
    Real implementation should use request IP or EXIF metadata.
    """
    try:
        # Placeholder: return example city/country
        return {"country": "India", "city": "Bengaluru"}
    except Exception:
        return {"country": None, "city": None}


