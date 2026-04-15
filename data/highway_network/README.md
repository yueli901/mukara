# Highway Network Data

Preprocessing assets for the interurban highway graph.

## Files
- `download_edge_features_google.ipynb`: current notebook for edge feature extraction via Google Routes API.
- `download_edge_features_ors.ipynb`: legacy OpenRouteService extraction workflow.
- `visualize.ipynb`: exploratory plots and QA checks for network features.

## Output Expectations
Processed files consumed by training:
- `edge_features.csv`
- `node_coordinates.csv`
