
"""
DATS 6401 Midterm
Buffalo Range Airport — Geographic Compass Overlay

Creates a compass centered on the airport coordinates.
Wind direction is calculated using circular statistics.

Source of wind observations: NASA POWER.
"""

import numpy as np
import pydeck as pdk


AIRPORT_LAT = -21.0167
AIRPORT_LON = 31.5833

COMPASS_DIRECTIONS = [
    ("N", 0),
    ("NE", 45),
    ("E", 90),
    ("SE", 135),
    ("S", 180),
    ("SW", 225),
    ("W", 270),
    ("NW", 315),
]


def destination_point(latitude, longitude, bearing, distance_km):
    """Calculate a point at a given bearing and distance."""
    earth_radius_km = 6371.0088

    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    bearing_rad = np.radians(bearing)
    angular_distance = distance_km / earth_radius_km

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1) * np.sin(angular_distance)
        * np.cos(bearing_rad)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance)
        * np.cos(lat1),
        np.cos(angular_distance)
        - np.sin(lat1) * np.sin(lat2)
    )

    return [float(np.degrees(lon2)), float(np.degrees(lat2))]


def circular_mean_direction(degrees):
    """Compute circular mean of wind directions in degrees."""
    angles = np.asarray(degrees, dtype=float)
    angles = angles[np.isfinite(angles)]

    if len(angles) == 0:
        return None

    radians = np.radians(angles % 360)

    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))

    # A nearly zero resultant has no meaningful mean direction.
    if np.hypot(mean_sin, mean_cos) < 1e-6:
        return None

    return float(
        np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
    )


def create_compass_layers(
    wind_directions=None,
    show_compass=True,
    show_wind=True,
    radius_km=8,
):
    """
    Return PyDeck layers for an airport-centered compass.

    Wind direction follows meteorological convention:
    the arrow points toward the direction FROM which
    the wind originates.
    """
    layers = []

    if show_compass:
        compass_paths = []
        compass_labels = []

        for label, bearing in COMPASS_DIRECTIONS:
            endpoint = destination_point(
                AIRPORT_LAT, AIRPORT_LON, bearing, radius_km
            )

            compass_paths.append({
                "name": label,
                "path": [
                    [AIRPORT_LON, AIRPORT_LAT],
                    endpoint
                ]
            })

            label_point = destination_point(
                AIRPORT_LAT,
                AIRPORT_LON,
                bearing,
                radius_km * 1.15
            )

            compass_labels.append({
                "label": label,
                "position": label_point
            })

        layers.append(
            pdk.Layer(
                "PathLayer",
                data=compass_paths,
                get_path="path",
                get_color=[45, 90, 160, 180],
                get_width=3,
                width_min_pixels=2,
                pickable=True
            )
        )

        layers.append(
            pdk.Layer(
                "TextLayer",
                data=compass_labels,
                get_position="position",
                get_text="label",
                get_size=20,
                get_color=[15, 45, 100, 255],
                get_text_anchor="'middle'",
                get_alignment_baseline="'center'"
            )
        )

    if show_wind and wind_directions is not None:
        mean_direction = circular_mean_direction(wind_directions)

        if mean_direction is not None:
            endpoint = destination_point(
                AIRPORT_LAT,
                AIRPORT_LON,
                mean_direction,
                radius_km * 0.85
            )

            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{
                        "path": [
                            [AIRPORT_LON, AIRPORT_LAT],
                            endpoint
                        ],
                        "name": (
                            f"Mean wind FROM "
                            f"{mean_direction:.1f}°"
                        )
                    }],
                    get_path="path",
                    get_color=[225, 55, 45, 255],
                    get_width=8,
                    width_min_pixels=5,
                    pickable=True
                )
            )

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": endpoint,
                        "radius": 180
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color=[225, 55, 45, 255],
                    pickable=True
                )
            )

    # Keep the airport clearly marked at the center.
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=[{
                "name": "Buffalo Range Airport (FVCZ)",
                "position": [AIRPORT_LON, AIRPORT_LAT],
                "radius": 200
            }],
            get_position="position",
            get_radius="radius",
            get_fill_color=[255, 185, 0, 255],
            get_line_color=[30, 30, 30, 255],
            stroked=True,
            line_width_min_pixels=2,
            pickable=True
        )
    )

    return layers
