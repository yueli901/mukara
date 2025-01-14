import os
import logging
from pyrosm import OSM
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from shapely.geometry import Polygon

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pyrosm")

# Function to parse .poly file and extract bounds
def parse_poly_file(poly_file):
    with open(poly_file, 'r') as file:
        lines = file.readlines()
    coords = []
    for line in lines:
        if line.strip().upper() in ["NONE", "END"] or line.strip().isdigit():
            continue
        x, y = map(float, line.split())
        coords.append((x, y))
    return Polygon(coords)

# Function to process a single grid cell
def process_grid_cell(args):
    idx, cell, landuse_data, road_data, poi_data, tags, output_folder, logger = args

    cell_geom = cell.geometry
    cell_area = cell_geom.area
    temp_file = os.path.join(output_folder, f"{idx}.csv")
    
    # Skip processing if the temp file already exists
    if os.path.exists(temp_file):
        return

    # Initialize temporary data for the cell
    channels = len(tags["landuse"]) + len(tags["road"]) + len(tags["poi"])  # Dynamic based on tags
    temp_data = np.zeros(channels)

    ### Process landuse
    for i, landuse_type in enumerate(tags["landuse"]):
        try:
            features = landuse_data[(landuse_data["landuse"] == landuse_type) & landuse_data.intersects(cell_geom)]
            if not features.empty:
                intersecting_areas = features.intersection(cell_geom).area.sum()
                relative_area = intersecting_areas / cell_area
                temp_data[i] = relative_area
        except Exception as e:
            # print(f"Error processing landuse for type {landuse_type}: {e}")
            # Attempt to fix invalid geometries and retry
            landuse_data["geometry"] = landuse_data["geometry"].apply(
                lambda geom: geom.buffer(0) if not geom.is_valid else geom
            )
            features = landuse_data[(landuse_data["landuse"] == landuse_type) & landuse_data.intersects(cell_geom)]
            if not features.empty:
                intersecting_areas = features.intersection(cell_geom).area.sum()
                relative_area = intersecting_areas / cell_area
                temp_data[i] = relative_area

    ### Process roads
    road_start_idx = len(tags["landuse"])  # Start after landuse channels
    for i, (road_level, road_types) in enumerate(tags["road"].items(), start=road_start_idx):
        try:
            # Filter road data for the current road level
            filtered_roads = road_data[road_data["highway"].isin(road_types)]
            # Check for intersection with the grid cell
            intersecting_roads = filtered_roads[filtered_roads.intersects(cell_geom)]
            # Calculate the total length of intersecting road segments
            if not intersecting_roads.empty:
                total_length = intersecting_roads.intersection(cell_geom).length.sum()
                temp_data[i] = total_length  # Total length of intersecting roads
        except Exception as e:
            # print(f"Error processing roads for level {road_level}: {e}")
            # Attempt to fix invalid geometries and retry
            road_data["geometry"] = road_data["geometry"].apply(
                lambda geom: geom.buffer(0) if not geom.is_valid else geom
            )
            filtered_roads = road_data[road_data["highway"].isin(road_types)]
            intersecting_roads = filtered_roads[filtered_roads.intersects(cell_geom)]
            if not intersecting_roads.empty:
                total_length = intersecting_roads.intersection(cell_geom).length.sum()
                temp_data[i] = total_length

    ### Process POIs
    poi_start_idx = road_start_idx + len(tags["road"])  # Start after road channels
    for i, (poi_category, subcategories) in enumerate(tags["poi"].items(), start=poi_start_idx):
        for column_name, valid_tags in subcategories.items():
            try:
                # Filter POIs that belong to the current subcategory and intersect the grid cell
                features = poi_data[
                    poi_data[column_name].isin(valid_tags) & poi_data.intersects(cell_geom)
                ]
                if not features.empty:
                    poi_count = len(features)
                    temp_data[i] += poi_count  # Add POI count to the corresponding channel
            except Exception as e:
                # print(f"Error processing POIs for category {poi_category}, column {column_name}: {e}")
                # Attempt to fix invalid geometries and retry
                poi_data["geometry"] = poi_data["geometry"].apply(
                    lambda geom: geom.buffer(0) if not geom.is_valid else geom
                )
                features = poi_data[
                    poi_data[column_name].isin(valid_tags) & poi_data.intersects(cell_geom)
                ]
                if not features.empty:
                    poi_count = len(features)
                    temp_data[i] += poi_count
                
    # if np.any(temp_data):  # Check if temp_data has any non-zero values
    #     print(f"Grid {idx} has non-zero values: {temp_data}")

    # Save temporary results to CSV
    temp_df = pd.DataFrame([temp_data], columns=[
        *tags["landuse"], *tags["road"].keys(), *tags["poi"].keys()
    ])
    temp_df.to_csv(temp_file, index=False)


def process_pbf_file(pbf_file, grid_file, poly_file, parent_folder, log_folder):
    base_name = os.path.splitext(os.path.basename(pbf_file))[0]
    log_file = os.path.join(log_folder, f"{base_name}.log")
    output_folder = os.path.join(parent_folder, base_name)
    os.makedirs(output_folder, exist_ok=True)

    # Set up a separate logger for each PBF file
    logger = logging.getLogger(base_name)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler for the specific log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Skip if log indicates completion
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            if any("Processing completed successfully" in line for line in f):
                print(f"Skipping {base_name} as it is already completed.")
                return

    logger.info(f"Starting processing for {base_name}.")

    # Load grid shapefile
    logger.info("Loading grid shapefile...")
    grid = gpd.read_file(grid_file)

    # Parse .poly file
    logger.info("Parsing .poly file...")
    poly_bounds = parse_poly_file(poly_file)
    poly_geom = gpd.GeoSeries([poly_bounds], crs="EPSG:4326").to_crs(epsg=27700)

    # Filter useful grid cells
    logger.info("Filtering useful grid cells...")
    useful_cells = grid[grid.intersects(poly_geom.unary_union)]

    # Collect existing results
    processed_indices = {int(f.split(".")[0]) for f in os.listdir(output_folder) if f.endswith(".csv")}

    # Process only unprocessed grids
    useful_cells = useful_cells[~useful_cells.index.isin(processed_indices)]

    # Load OSM data
    logger.info("Loading OSM data...")
    osm = OSM(pbf_file)

    tags = {
        "landuse": ["residential", "commercial", "industrial", "retail"],
        "road": {
            "high_level": ["motorway", "trunk", "primary"],
            "medium_level": ["secondary", "tertiary"],
            "low_level": ["residential", "unclassified"]
        },
        "poi": {
            "transport": {
                "amenity": ["bus_station", "parking"],
                "railway": ["station", "stop", "tram_stop"]
            },
            "food": {
                "amenity": ["bar", "cafe", "restaurant"]
            },
            "health": {
                "amenity": ["clinic", "hospital", "pharmacy"]
            },
            "education": {
                "amenity": ["school", "college", "kindergarten", "university"]
            },
            "retail": {
                "shop": ["supermarket", "department_store", "mall"]
            }
        }
    }

    # Extract data
    logger.info("Extracting OSM data...")
    ### Land use
    landuse_data = osm.get_data_by_custom_criteria(custom_filter={"landuse": tags["landuse"]}, filter_type="keep")
    if landuse_data is not None and not landuse_data.empty:
        landuse_data = landuse_data[["landuse", "geometry"]]  # Keep only relevant columns
        landuse_data = landuse_data[landuse_data.geometry.type.isin(["Polygon", "MultiPolygon"])]  # Keep only polygons
    else:
        landuse_data = gpd.GeoDataFrame(columns=["landuse", "geometry"])  # Create an empty GeoDataFrame if no data
        
    ### Road
    road_data = osm.get_network(network_type="all")  # To include all road levels
    if road_data is not None and not road_data.empty:
        # Filter for specific road levels
        valid_road_types = tags["road"]["high_level"] + tags["road"]["medium_level"] + tags["road"]["low_level"]
        road_data = road_data[road_data["highway"].isin(valid_road_types)]

        # Retain only necessary columns
        road_data = road_data[["highway", "geometry"]]

        # Filter for valid geometries (MULTILINESTRING only)
        road_data = road_data[road_data.geometry.type == "MultiLineString"]
    else:
        road_data = gpd.GeoDataFrame(columns=["highway", "geometry"])  # Create an empty GeoDataFrame if no data

    ### POIs
    # Dynamically generate POI filter
    poi_filter = {}
    for poi_category, subcategories in tags["poi"].items():
        for key, values in subcategories.items():
            if key not in poi_filter:
                poi_filter[key] = []
            poi_filter[key].extend(values)

    poi_data = osm.get_data_by_custom_criteria(custom_filter=poi_filter, filter_type="keep")

    if poi_data is not None and not poi_data.empty:
        # Retain only relevant columns
        relevant_columns = ["geometry", "amenity", "shop", "railway"]
        poi_data = poi_data[relevant_columns]

        # Filter for relevant geometries (Point, Polygon, MultiPolygon)
        poi_data = poi_data[poi_data.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])]

        # Transform Polygons and MultiPolygons to Points by calculating their centroids
        poi_data["geometry"] = poi_data.geometry.apply(
            lambda geom: geom.centroid if geom.geom_type in ["Polygon", "MultiPolygon"] else geom
        )
    else:
        # Create an empty GeoDataFrame with expected columns if no data
        poi_data = gpd.GeoDataFrame(columns=["geometry", "amenity", "shop", "railway"])

    # Transform OSM data to match the grid CRS
    landuse_data = landuse_data.to_crs(epsg=27700) if not landuse_data.empty else landuse_data
    road_data = road_data.to_crs(epsg=27700) if not road_data.empty else road_data
    poi_data = poi_data.to_crs(epsg=27700) if not poi_data.empty else poi_data

    # Parallel processing
    logger.info("Processing grid cells in parallel...")
    args = [(idx, cell, landuse_data, road_data, poi_data, tags, output_folder, logger) for idx, cell in useful_cells.iterrows()]
    with Pool(processes=min(12, cpu_count())) as pool:
        list(tqdm(pool.imap_unordered(process_grid_cell, args, chunksize=200), total=len(args)))

    logger.info(f"Processing completed successfully for {base_name}.")

def main():
    # Paths and parameters
    pbf_folder = "pbfs/230101"
    poly_folder = "polys"
    parent_folder = "temp"
    log_folder = "logs"
    grid_file = "../population_and_employment/grid_1km_653_573/grid_cells.shp"
    os.makedirs(parent_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Extract region names from filenames (ignoring the date and extensions)
    def extract_region_name(filename):
        parts = filename.split("-")
        return "-".join(parts[:-1])

    # Create mappings
    pbf_files = {extract_region_name(file): os.path.join(pbf_folder, file) for file in os.listdir(pbf_folder) if file.endswith(".osm.pbf")}
    poly_files = {os.path.splitext(file)[0]: os.path.join(poly_folder, file) for file in os.listdir(poly_folder) if file.endswith(".poly")}

    # Process each PBF file
    for region_name, pbf_file in tqdm(pbf_files.items(), desc="Overall Progress"):
        poly_file = poly_files[region_name]
        process_pbf_file(pbf_file, grid_file, poly_file, parent_folder, log_folder)


if __name__ == "__main__":
    main()