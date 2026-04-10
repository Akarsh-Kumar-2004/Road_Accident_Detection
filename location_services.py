from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import requests


DEFAULT_LOCATION = {
    "lat": 26.839901,
    "lon": 75.561668,
    "place": "Bagru, Jaipur, Rajasthan, India",
    "city": "Bagru",
    "region": "Rajasthan",
    "country": "India",
    "source": "configured-location",
}


def get_current_pc_location() -> Dict[str, object]:
    configured_location = _get_env_location()
    if configured_location is not None:
        return configured_location

    windows_location = _get_windows_pc_location()
    if windows_location is not None:
        return windows_location

    ip_location = _get_ip_geolocation()
    if ip_location is not None:
        return ip_location

    return DEFAULT_LOCATION.copy()


def get_nearest_hospitals(lat: float, lon: float, radius_meters: int = 7000, limit: int = 3) -> List[Dict[str, object]]:
    return _fetch_nearby_places("hospital", lat, lon, radius_meters, limit)


def get_nearest_police_stations(
    lat: float, lon: float, radius_meters: int = 7000, limit: int = 3
) -> List[Dict[str, object]]:
    return _fetch_nearby_places("police", lat, lon, radius_meters, limit)


def _fetch_nearby_places(
    amenity: str, lat: float, lon: float, radius_meters: int, limit: int
) -> List[Dict[str, object]]:
    query = """
    [out:json];
    (
      node["amenity"="{amenity}"](around:{radius},{lat},{lon});
      way["amenity"="{amenity}"](around:{radius},{lat},{lon});
      relation["amenity"="{amenity}"](around:{radius},{lat},{lon});
    );
    out center;
    """.format(amenity=amenity, radius=radius_meters, lat=lat, lon=lon)

    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=12,
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except Exception:
        return []

    places: List[Dict[str, object]] = []
    for element in elements:
        tags = element.get("tags", {})
        place_lat = element.get("lat", element.get("center", {}).get("lat"))
        place_lon = element.get("lon", element.get("center", {}).get("lon"))
        if place_lat is None or place_lon is None:
            continue

        places.append(
            {
                "name": tags.get("name", amenity.title()),
                "lat": place_lat,
                "lon": place_lon,
            }
        )

    places.sort(key=lambda item: _distance_sq(lat, lon, item["lat"], item["lon"]))
    return places[:limit]


def get_route_geometry(start: Tuple[float, float], end: Tuple[float, float]) -> Optional[List[List[float]]]:
    start_lon, start_lat = start[1], start[0]
    end_lon, end_lat = end[1], end[0]
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
    )

    try:
        response = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=10)
        response.raise_for_status()
        routes = response.json().get("routes", [])
        if not routes:
            return None
        coordinates = routes[0]["geometry"]["coordinates"]
        return [[lat, lon] for lon, lat in coordinates]
    except Exception:
        return None


def build_google_maps_route_url(start: Tuple[float, float], end: Tuple[float, float]) -> str:
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={start[0]},{start[1]}&destination={end[0]},{end[1]}&travelmode=driving"
    )


def build_google_maps_place_url(location: Tuple[float, float]) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={location[0]},{location[1]}"


def build_openstreetmap_route_url(start: Tuple[float, float], end: Tuple[float, float]) -> str:
    return (
        "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car"
        f"&route={start[0]}%2C{start[1]}%3B{end[0]}%2C{end[1]}"
    )


def _get_env_location() -> Optional[Dict[str, object]]:
    lat = os.getenv("ACCIDENT_PC_LAT")
    lon = os.getenv("ACCIDENT_PC_LON")
    if not lat or not lon:
        return None

    try:
        latitude = float(lat)
        longitude = float(lon)
    except ValueError:
        return None

    place = os.getenv("ACCIDENT_PC_PLACE", f"{latitude:.5f}, {longitude:.5f}")
    city = os.getenv("ACCIDENT_PC_CITY")
    region = os.getenv("ACCIDENT_PC_REGION")
    country = os.getenv("ACCIDENT_PC_COUNTRY")

    return {
        "lat": latitude,
        "lon": longitude,
        "place": place,
        "city": city or place,
        "region": region or "Unknown",
        "country": country or "Unknown",
        "source": "environment-override",
    }


def _get_windows_pc_location() -> Optional[Dict[str, object]]:
    if os.name != "nt":
        return None

    powershell_command = """
    Add-Type -AssemblyName System.Device
    $watcher = New-Object System.Device.Location.GeoCoordinateWatcher
    $started = $watcher.TryStart($false, [TimeSpan]::FromSeconds(10))
    if (-not $started -or $watcher.Position.Location.IsUnknown) {
        exit 1
    }

    $location = $watcher.Position.Location
    [pscustomobject]@{
        lat = $location.Latitude
        lon = $location.Longitude
        accuracy = $location.HorizontalAccuracy
    } | ConvertTo-Json -Compress
    """

    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", powershell_command],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    try:
        raw_location = json.loads(completed.stdout)
        latitude = float(raw_location["lat"])
        longitude = float(raw_location["lon"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    reverse_geocoded = _reverse_geocode(latitude, longitude) or {}
    return _build_location_payload(
        latitude,
        longitude,
        reverse_geocoded,
        source="windows-location-service",
    )


def _get_ip_geolocation() -> Optional[Dict[str, object]]:
    providers = [
        ("https://ipapi.co/json/", _parse_ipapi_response),
        ("https://ipwho.is/", _parse_ipwhois_response),
    ]

    for url, parser in providers:
        try:
            response = requests.get(url, timeout=8, headers=_request_headers())
            response.raise_for_status()
            parsed = parser(response.json())
        except Exception:
            continue

        if parsed is not None:
            return parsed

    return None


def _parse_ipapi_response(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if latitude is None or longitude is None:
        return None

    return _build_location_payload(
        float(latitude),
        float(longitude),
        {
            "city": payload.get("city"),
            "state": payload.get("region"),
            "country": payload.get("country_name"),
            "display_name": ", ".join(
                part for part in [payload.get("city"), payload.get("region"), payload.get("country_name")] if part
            ),
        },
        source="ip-geolocation",
    )


def _parse_ipwhois_response(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not payload.get("success", True):
        return None

    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if latitude is None or longitude is None:
        return None

    return _build_location_payload(
        float(latitude),
        float(longitude),
        {
            "city": payload.get("city"),
            "state": payload.get("region"),
            "country": payload.get("country"),
            "display_name": ", ".join(
                part for part in [payload.get("city"), payload.get("region"), payload.get("country")] if part
            ),
        },
        source="ip-geolocation",
    )


def _reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, object]]:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon},
            headers=_request_headers(),
            timeout=8,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _build_location_payload(
    lat: float,
    lon: float,
    details: Dict[str, object],
    *,
    source: str,
) -> Dict[str, object]:
    address = details.get("address", {}) if isinstance(details.get("address"), dict) else {}
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("hamlet")
        or details.get("city")
        or "Unknown"
    )
    region = address.get("state") or details.get("state") or "Unknown"
    country = address.get("country") or details.get("country") or "Unknown"
    place = details.get("display_name") or ", ".join(part for part in [city, region, country] if part and part != "Unknown")
    if not place:
        place = f"{lat:.5f}, {lon:.5f}"

    return {
        "lat": lat,
        "lon": lon,
        "place": place,
        "city": city,
        "region": region,
        "country": country,
        "source": source,
    }


def _request_headers() -> Dict[str, str]:
    return {"User-Agent": "AccidentDetectionSystem/1.0"}


def _distance_sq(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return (lat1 - lat2) ** 2 + (lon1 - lon2) ** 2
