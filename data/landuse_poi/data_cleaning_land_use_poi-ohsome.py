import os
import geopandas as gpd
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import requests
import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pyrosm")

# Set up logging
log_file = "grid_processing.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global variables for workers
global_vars = {}


def initialize_worker(grid_data, output_directory, landuse, road, poi, year_value):
    """
    Initialize global variables for each worker.
    """
    global global_vars
    global_vars["grid"] = grid_data
    global_vars["output_dir"] = output_directory
    global_vars["landuse_filters"] = landuse
    global_vars["road_filters"] = road
    global_vars["poi_filters"] = poi
    global_vars["year"] = year_value


def get_ohsome_aggregation(bbox, date, filter_, agg_type):
    """
    Fetches data from OhSome API's aggregation endpoint.
    """
    base_url = f"https://api.ohsome.org/v1/elements/{agg_type}"
    data = {
        "bboxes": bbox,
        "time": date,
        "filter": filter_,
        "format": "json"
    }
    try:
        response = requests.post(base_url, data=data)
        response.raise_for_status()
        result = response.json()
        return result["result"][0]["value"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {filter_}: {e}")
        return None


def process_grid(idx):
    """
    Processes a single grid cell and saves results to a CSV file named after the grid index.
    """
    logging.info(f"Processing {idx}")

    # Access global variables
    grid = global_vars["grid"]
    output_dir = global_vars["output_dir"]
    landuse_filters = global_vars["landuse_filters"]
    road_filters = global_vars["road_filters"]
    poi_filters = global_vars["poi_filters"]
    year = global_vars["year"]

    output_file = os.path.join(output_dir, f"{idx}.csv")
    if os.path.exists(output_file):
        logging.info(f"Skipping {idx}, already processed.")
        return

    row = grid.loc[idx]
    bounds = row.geometry.bounds
    bbox = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
    record = {"idx": idx}

    # Process land use
    for i, filter_ in enumerate(landuse_filters):
        area = get_ohsome_aggregation(bbox, f"{year}-01-01", filter_, "area")
        if area is not None:
            record[f"landuse_{i}"] = area
        else:
            logging.warning(f"Data collection not completed for {idx} during landuse processing.")
            return

    # Process roads
    for i, filter_ in enumerate(road_filters):
        length = get_ohsome_aggregation(bbox, f"{year}-01-01", filter_, "length")
        if length is not None:
            record[f"road_{i}"] = length
        else:
            logging.warning(f"Data collection not completed for {idx} during road processing.")
            return

    # Process POIs
    for key in poi_filters.keys():
        count = get_ohsome_aggregation(bbox, f"{year}-01-01", poi_filters[key], "count")
        if count is not None:
            record[key] = count
        else:
            logging.warning(f"Data collection not completed for {idx} during POI processing.")
            return

    # Save results
    pd.DataFrame([record]).to_csv(output_file, index=False)
    logging.info(f"Processed and saved grid {idx}")


def main():
    # Load grid shapefile
    grid_file = "grid_1km_653_573/grid_cells.shp"
    grid = gpd.read_file(grid_file)
    grid = grid.to_crs(epsg=4326)

    # Filters
    landuse_filters = [
        "landuse=residential",
        "landuse=commercial",
        "landuse=industrial",
        "landuse=retail",
        "landuse=education",
        "landuse=institutional"
    ]

    road_filters = [
        "highway in (motorway,trunk,primary)",  # High-level
        "highway in (secondary,tertiary)",      # Medium-level
        "highway in (residential,unclassified)" # Low-level
    ]

    poi_filters = {
        "transport": "amenity in (bus_station,parking) or railway in (station,stop,tram_stop)",
        "food": "amenity in (bar,cafe,restaurant)",
        "health": "amenity in (clinic,hospital,pharmacy)",
        "education": "amenity in (school,college,kindergarten,university)",
        "retail": "shop in (supermarket,department_store,mall)"
    }

    # Output directory
    output_dir = "grid_1km_653_573/individual_grids"
    os.makedirs(output_dir, exist_ok=True)

    # Default year
    year = "2023"

    # Intersecting indices
    indices_file = "intersecting_indices.txt"
    if not os.path.exists(indices_file):
        logging.info("Identifying intersecting grid cells...")
        lsoa_union = grid.unary_union  # Replace with actual LSOA union geometry
        intersecting_indices = [
            idx for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="Processing Grid Cells")
            if row.geometry.intersects(lsoa_union)
        ]
        with open(indices_file, "w") as f:
            for idx in intersecting_indices:
                f.write(f"{idx}\n")
        logging.info(f"Intersecting indices saved to {indices_file}")
    else:
        with open(indices_file, "r") as f:
            intersecting_indices = [int(line.strip()) for line in tqdm(f, desc="Loading Indices")]

    logging.info(f"Number of intersecting grid cells: {len(intersecting_indices)}")

    # Multiprocessing
    logging.info("Starting multiprocessing...")
    with Pool(processes=min(16, cpu_count()), initializer=initialize_worker,
              initargs=(grid, output_dir, landuse_filters, road_filters, poi_filters, year)) as pool:
        list(tqdm(pool.imap_unordered(process_grid, intersecting_indices), total=len(intersecting_indices)))
    logging.info("Multiprocessing completed.")


if __name__ == "__main__":
    main()