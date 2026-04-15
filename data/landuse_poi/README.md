# Land Use and POI Data

Workflows for deriving rasterized land use, road-level, and POI features from OSM data.

## Files
- `download_pbfs_polys.ipynb`: fetch region `.osm.pbf` files and `.poly` boundaries.
- `data_cleaning_land_use_poi-local.py`: local PBF processing (recommended workflow).
- `data_cleaning_land_use_poi-ohsome.py`: deprecated remote API workflow.
- `aggregate_by_index.ipynb`: aggregate per-grid intermediate outputs into model-ready tensors.
- `polys/`: region boundary files for clipping and batching.

## Output Expectations
Processed file consumed by training:
- `landuse_and_poi-230101.h5`
