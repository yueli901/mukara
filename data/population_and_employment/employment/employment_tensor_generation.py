import os
import numpy as np
import h5py
import shutil
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

# Paths to input files and folders
GRID_FILE = '../grid_1km_653_573/grid_cells.shp'
LSOA_FILE = '../boundaries/Lower layer Super Output Areas (December 2011) Boundaries EW BFC (V3)/Lower_layer_Super_Output_Areas_Dec_2011_Boundaries_Full_Clipped_BFC_EW_V3_2022_1117503576712596763/LSOA_2011_EW_BFC_V3.shp'
EMPLOYMENT_FOLDER = 'processed_employment_data'
TEMP_FOLDER = 'temp_employment'
OUTPUT_FILE = 'employment_tensor.h5'

# Employment types and sectors
EMPLOYMENT_TYPES = ["full_time_employees", "part_time_employees", "employment"]
NUM_TYPES = len(EMPLOYMENT_TYPES)
SECTORS = [f"sector_{i+1}" for i in range(18)]  # Sectors are generic for now
NUM_SECTORS = len(SECTORS)

# Years to process
YEARS = range(2015, 2023)
NUM_YEARS = len(YEARS)
NUM_GRIDS_Y, NUM_GRIDS_X = 653, 573
NUM_CHANNELS = NUM_TYPES * NUM_SECTORS

BATCH_SIZE = 1000  # Adjust batch size based on memory and performance

def process_batch(batch_indices, grid_geoms, lsoa_gdf, employment_dicts, lsoa_areas):
    """
    Process a batch of grid cells instead of individual ones.
    """
    for idx in batch_indices:
        temp_file = os.path.join(TEMP_FOLDER, f"pixel_{idx}.npy")
        if os.path.exists(temp_file):
            continue

        grid = grid_geoms[idx]
        pixel_employment = np.zeros((NUM_YEARS, NUM_CHANNELS), dtype=np.float32)

        # Find intersecting LSOAs
        intersecting_lsoas = lsoa_gdf[lsoa_gdf.intersects(grid)]
        if not intersecting_lsoas.empty:
            for _, lsoa in intersecting_lsoas.iterrows():
                intersection = grid.intersection(lsoa.geometry)
                overlap_ratio = intersection.area / lsoa_areas[lsoa['LSOA11CD']]

                # Add weighted employment data
                for year_idx, year in enumerate(YEARS):
                    for type_idx, emp_type in enumerate(EMPLOYMENT_TYPES):
                        employment_data = employment_dicts[(year, emp_type)]
                        lsoa_code = lsoa['LSOA11CD']
                        for sector_idx, sector in enumerate(SECTORS):
                            employment_value = employment_data.get(lsoa_code, {}).get(sector, 0)
                            channel_idx = type_idx * NUM_SECTORS + sector_idx
                            pixel_employment[year_idx, channel_idx] += employment_value * overlap_ratio

        # Save result to temp file
        np.save(temp_file, pixel_employment)

def main():
    # Setup and data loading
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    grid_gdf = gpd.read_file(GRID_FILE)
    lsoa_gdf = gpd.read_file(LSOA_FILE).to_crs(grid_gdf.crs)
    lsoa_areas = lsoa_gdf.set_index('LSOA11CD')['geometry'].area.to_dict()

    # Load employment data into dictionaries
    print("Loading employment data...")
    employment_dicts = {}
    for year in YEARS:
        for emp_type in EMPLOYMENT_TYPES:
            file_path = os.path.join(EMPLOYMENT_FOLDER, f"employment_{year}_{emp_type}.csv")
            df = pd.read_csv(file_path)
            df = df.rename(columns={df.columns[1]: "LSOA11CD"})  # Ensure consistent LSOA column name
            sector_columns = [col for col in df.columns[2:20]]  # Select the 18 sector columns
            sector_renames = {old_col: f"sector_{i+1}" for i, old_col in enumerate(sector_columns)}
            df = df.rename(columns=sector_renames)
            employment_dict = df.set_index("LSOA11CD")[SECTORS].to_dict('index')
            employment_dicts[(year, emp_type)] = employment_dict

    grid_geoms = grid_gdf.geometry
    total_indices = list(range(len(grid_geoms)))

    # Divide grid indices into batches
    batches = [total_indices[i:i+BATCH_SIZE] for i in range(0, len(total_indices), BATCH_SIZE)]

    # Parallel processing with batches
    print("Processing grid cells in batches...")
    Parallel(n_jobs=23)(
        delayed(process_batch)(
            batch, grid_geoms, lsoa_gdf, employment_dicts, lsoa_areas
        )
        for batch in tqdm(batches, desc="Batch Processing")
    )

    # Assemble results
    print("Assembling the employment tensor...")
    employment_tensor = np.zeros((NUM_YEARS, NUM_GRIDS_Y, NUM_GRIDS_X, NUM_CHANNELS), dtype=np.float32)
    for idx in tqdm(total_indices, desc="Tensor Assembly"):
        temp_file = os.path.join(TEMP_FOLDER, f"pixel_{idx}.npy")
        if os.path.exists(temp_file):
            row, col = divmod(idx, NUM_GRIDS_X)
            pixel_employment = np.load(temp_file)
            employment_tensor[:, row, col, :] = pixel_employment

    # Save the tensor to HDF5 file
    print(f"Saving employment tensor to {OUTPUT_FILE}...")
    with h5py.File(OUTPUT_FILE, 'w') as h5f:
        h5f.create_dataset(
            'employment_tensor', 
            data=employment_tensor.astype(np.float32),  # Ensure float32
            compression='gzip', 
            compression_opts=4  # Compression level: 1-9 (4 is balanced)
        )
    print(f"Employment tensor saved successfully with shape {employment_tensor.shape} and dtype float32.")

    # Delete temporary files
    if os.path.exists(TEMP_FOLDER):
        print(f"Deleting temporary files in {TEMP_FOLDER}...")
        shutil.rmtree(TEMP_FOLDER)  # Deletes the entire temp folder and its contents
        print("Temporary files deleted successfully.")

    print("All tasks completed!")


if __name__ == '__main__':
    main()