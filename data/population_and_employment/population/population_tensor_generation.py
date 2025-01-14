import os
import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
import h5py
import shutil

# Paths to input files and folders
GRID_FILE = 'grid_1km_653_573/grid_cells.shp'
LSOA_FILE = 'boundaries/Lower layer Super Output Areas (December 2021) Boundaries EW BFC (V10)/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10_8562115581115271145/LSOA_2021_EW_BFC_V10.shp'
POPULATION_FOLDER = 'population'
TEMP_FOLDER = 'temp'
OUTPUT_FILE = 'population_tensor.npy'

# List of population files
POPULATION_FILES = [
    'population_male_aged_0_to_15.csv',
    'population_male_aged_16_to_24.csv',
    'population_male_aged_25_to_49.csv',
    'population_male_aged_50_to_64.csv',
    'population_male_aged_65+.csv',
    'population_female_aged_0_to_15.csv',
    'population_female_aged_16_to_24.csv',
    'population_female_aged_25_to_49.csv',
    'population_female_aged_50_to_64.csv',
    'population_female_aged_65+.csv'
]

# Years to process
YEARS = range(2015, 2023)
NUM_YEARS = len(YEARS)
NUM_CHANNELS = len(POPULATION_FILES)
NUM_GRIDS_Y, NUM_GRIDS_X = 653, 573

BATCH_SIZE = 1000  # Adjust batch size based on memory and performance

def process_batch(batch_indices, grid_geoms, lsoa_gdf, population_dicts, lsoa_areas):
    """
    Process a batch of grid cells instead of individual ones.
    """
    for idx in batch_indices:
        temp_file = os.path.join(TEMP_FOLDER, f"pixel_{idx}.npy")
        if os.path.exists(temp_file):
            continue

        grid = grid_geoms[idx]
        row, col = divmod(idx, NUM_GRIDS_X)
        pixel_population = np.zeros((NUM_YEARS, NUM_CHANNELS), dtype=np.float32)

        # Find intersecting LSOAs
        intersecting_lsoas = lsoa_gdf[lsoa_gdf.intersects(grid)]
        if not intersecting_lsoas.empty:
            for _, lsoa in intersecting_lsoas.iterrows():
                intersection = grid.intersection(lsoa.geometry)
                overlap_ratio = intersection.area / lsoa_areas[lsoa['LSOA21CD']]

                for channel_idx, pop_dict in enumerate(population_dicts):
                    for year_idx, year in enumerate(YEARS):
                        lsoa_code = lsoa['LSOA21CD']
                        lsoa_pop = pop_dict.get(lsoa_code, {}).get(str(year), 0)
                        pixel_population[year_idx, channel_idx] += lsoa_pop * overlap_ratio

        # Save result to temp file
        np.save(temp_file, pixel_population)


def main():
    # Setup and data loading (as before)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    grid_gdf = gpd.read_file(GRID_FILE)
    lsoa_gdf = gpd.read_file(LSOA_FILE).to_crs(grid_gdf.crs)
    lsoa_areas = lsoa_gdf.set_index('LSOA21CD')['geometry'].area.to_dict()

    print("Loading population data...")
    population_data = []
    for file in POPULATION_FILES:
        file_path = os.path.join(POPULATION_FOLDER, file)
        df = pd.read_csv(file_path)
        pop_dict = df.set_index('mnemonic').loc[:, [str(year) for year in YEARS]].to_dict('index')
        population_data.append(pop_dict)

    grid_geoms = grid_gdf.geometry
    total_indices = list(range(len(grid_geoms)))

    # Divide grid indices into batches
    batches = [total_indices[i:i+BATCH_SIZE] for i in range(0, len(total_indices), BATCH_SIZE)]

    # Parallel processing with batches
    print("Processing grid cells in batches...")
    Parallel(n_jobs=23)(
        delayed(process_batch)(
            batch, grid_geoms, lsoa_gdf, population_data, lsoa_areas
        )
        for batch in tqdm(batches, desc="Batch Processing")
    )

    # Assemble results (unchanged)
    print("Assembling the population tensor...")
    population_tensor = np.zeros((NUM_YEARS, NUM_GRIDS_Y, NUM_GRIDS_X, NUM_CHANNELS), dtype=np.float32)
    for idx in tqdm(total_indices, desc="Tensor Assembly"):
        temp_file = os.path.join(TEMP_FOLDER, f"pixel_{idx}.npy")
        if os.path.exists(temp_file):
            row, col = divmod(idx, NUM_GRIDS_X)
            pixel_population = np.load(temp_file)
            population_tensor[:, row, col, :] = pixel_population

    # Save the tensor to HDF5 file
    print(f"Saving population tensor to {OUTPUT_FILE}...")
    with h5py.File(OUTPUT_FILE, 'w') as h5f:
        h5f.create_dataset(
            'population_tensor', 
            data=population_tensor.astype(np.float32),  # Ensure float32
            compression='gzip', 
            compression_opts=4  # Compression level: 1-9 (4 is balanced)
        )
    print(f"Population tensor saved successfully with shape {population_tensor.shape} and dtype float32.")

    # Delete temporary files
    if os.path.exists(TEMP_FOLDER):
        print(f"Deleting temporary files in {TEMP_FOLDER}...")
        shutil.rmtree(TEMP_FOLDER)  # Deletes the entire temp folder and its contents
        print("Temporary files deleted successfully.")

    print("All tasks completed!")


if __name__ == '__main__':
    main()