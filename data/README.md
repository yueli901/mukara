# Data Directory

This directory contains data preparation code and helper assets used to build Mukara inputs.

## Subdirectories
- `highway_network/`: trunk road topology and edge-feature extraction notebooks.
- `landuse_poi/`: OSM-based land use and POI preprocessing scripts.
- `population_and_employment/`: population/employment rasterization workflows.
- `traffic_volume/`: traffic volume download, cleaning, and aggregation notebooks.

## Notes
- Heavy intermediate/raw datasets are intentionally excluded from Git.
- Follow `config.py` for expected processed file paths at training time.
