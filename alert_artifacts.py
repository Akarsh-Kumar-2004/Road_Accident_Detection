from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from location_services import get_route_geometry


def create_incident_map_image(
    output_path: str,
    location: Dict[str, object],
    hospitals: List[Dict[str, object]],
    police_stations: List[Dict[str, object]],
) -> str:
    accident_point = (float(location["lat"]), float(location["lon"]))
    figure, axis = plt.subplots(figsize=(10, 7))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#fcfcfc")

    axis.scatter(accident_point[1], accident_point[0], c="#d62828", s=220, label="Accident", zorder=5)
    axis.annotate(
        f"Accident\n{location['place']}",
        (accident_point[1], accident_point[0]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="#d62828",
        weight="bold",
    )

    _plot_places(axis, hospitals, "#2a9d8f", "Hospital")
    _plot_places(axis, police_stations, "#1d4ed8", "Police")
    _add_route(axis, accident_point, hospitals[0] if hospitals else None, "#2a9d8f")
    _add_route(axis, accident_point, police_stations[0] if police_stations else None, "#1d4ed8")
    _set_plot_bounds(axis, accident_point, hospitals, police_stations)

    axis.set_title("Accident Response Snapshot", fontsize=16, weight="bold")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_places(axis, places: List[Dict[str, object]], color: str, label: str) -> None:
    if not places:
        return

    longitudes = [place["lon"] for place in places]
    latitudes = [place["lat"] for place in places]
    axis.scatter(longitudes, latitudes, c=color, s=110, label=label, zorder=4)

    for index, place in enumerate(places, start=1):
        axis.annotate(
            f"{index}. {place['name']}",
            (place["lon"], place["lat"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )


def _add_route(
    axis,
    start: Tuple[float, float],
    destination: Optional[Dict[str, object]],
    color: str,
) -> None:
    if not destination:
        return

    route = get_route_geometry(start, (destination["lat"], destination["lon"]))
    if route:
        latitudes = [point[0] for point in route]
        longitudes = [point[1] for point in route]
        axis.plot(longitudes, latitudes, color=color, linewidth=2.5, alpha=0.8)


def _set_plot_bounds(
    axis,
    accident_point: Tuple[float, float],
    hospitals: List[Dict[str, object]],
    police_stations: List[Dict[str, object]],
) -> None:
    latitudes = [accident_point[0]]
    longitudes = [accident_point[1]]

    for place in hospitals + police_stations:
        latitudes.append(place["lat"])
        longitudes.append(place["lon"])

    lat_padding = max((max(latitudes) - min(latitudes)) * 0.25, 0.01)
    lon_padding = max((max(longitudes) - min(longitudes)) * 0.25, 0.01)

    axis.set_xlim(min(longitudes) - lon_padding, max(longitudes) + lon_padding)
    axis.set_ylim(min(latitudes) - lat_padding, max(latitudes) + lat_padding)
