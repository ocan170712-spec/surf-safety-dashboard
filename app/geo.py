"""Approximate lat/lon centroids for US states that appear in the shark
attack data. The source data has no coordinates, only state names, so
points are jittered around these centroids purely for visualization -
they do not represent exact incident locations."""

STATE_CENTROIDS = {
    "Florida": (27.7, -81.5), "California": (36.8, -119.6), "Hawaii": (20.8, -156.3),
    "Oregon": (44.0, -120.5), "North Carolina": (35.6, -79.0), "South Carolina": (33.9, -80.9),
    "Texas": (31.0, -99.9), "New Jersey": (40.1, -74.7), "Georgia": (32.9, -83.4),
    "Virginia": (37.4, -78.7), "New York": (43.0, -75.5), "Alabama": (32.8, -86.8),
    "Massachusetts": (42.4, -71.4), "Washington": (47.4, -121.5), "Puerto Rico": (18.2, -66.5),
    "Louisiana": (31.0, -92.0), "Mississippi": (32.7, -89.7), "Delaware": (39.0, -75.5),
    "Maine": (45.4, -69.2), "Connecticut": (41.6, -72.7), "Rhode Island": (41.7, -71.5),
    "Maryland": (39.0, -76.7),
}
