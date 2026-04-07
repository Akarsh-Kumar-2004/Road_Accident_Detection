from __future__ import annotations

import os
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

def _distance_sq(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return (lat1 - lat2) ** 2 + (lon1 - lon2) ** 2
