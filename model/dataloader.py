import tensorflow as tf
import h5py
import os
import numpy as np
import random
import pandas as pd
import geopandas as gpd

from model.utils import Scaler
from config import PATH, DATA, TRAINING

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

def adjacency_matrix():
	"""
	Output two lists, both are node index, paired to each other.
	"""
	sensor_data = pd.read_csv(os.path.join(PATH["data"], PATH["edge_features"]))
	src = sensor_data["Origin"].to_list()
	# src = [x - 1 for x in src]
	dst = sensor_data["Destination"].to_list()
	# dst = [x - 1 for x in dst]
	return src, dst


# def sensor_index():
#     df = pd.read_csv(os.path.join(PATH["data"], PATH["edge_features"]))
#     sensor_idx = {sensor_id: i for i, sensor_id in enumerate(df['Id'])}
#     return sensor_idx


def get_node_positions():
    """
    Map nodes to grid cells.
    Output: array of shape (181, 2) where each entry contains the row and col (position on the grid) for each node.
    """
    nodes_df = pd.read_csv(os.path.join(PATH["data"], PATH["node_features"]))
    # Convert latitude and longitude to shapely Point objects
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df.Longitude, nodes_df.Latitude),
        crs='EPSG:4326'
    )

    grid_gdf = gpd.read_file(os.path.join(PATH["data"], PATH["grid"], 'grid_cells.shp'))
    
    # Convert grid to the same CRS
    grid_gdf = grid_gdf.to_crs(nodes_gdf.crs)
    
    # Perform a spatial join to find which grid cell each node belongs to
    nodes_with_grid = gpd.sjoin(nodes_gdf, grid_gdf, how="left", predicate="within")

    grid_shape = (int(PATH['grid'].split('_')[-2]), int(PATH['grid'].split('_')[-1])) # height and width, named in grid path
    nodes_with_grid['row'] = nodes_with_grid['index_right'] // grid_shape[1]
    nodes_with_grid['col'] = nodes_with_grid['index_right'] % grid_shape[1]

    node_positions = nodes_with_grid[['row', 'col']].to_numpy()

    return tf.convert_to_tensor(node_positions)


def get_static_features():
    """
    population_and_employment shape (year=8, height, width, c=2)
    landuse_and_poi shape (height, width, 12)
    z-score normalize
    return grid static feature (year=8, height, width, c=14)
    """
    with h5py.File(os.path.join(PATH["data"], PATH["population_and_employment"]), 'r') as f:  
        pe = f['features'][..., [item for sublist in [DATA['population'], DATA['employment']] for item in sublist]]
    year, row, col, c = pe.shape
    pe = np.reshape(pe, (year * row * col, -1))
    pe = (pe - np.mean(pe, axis=0)) / (np.std(pe, axis=0) + 1e-8)	
    pe = np.reshape(pe, (year, row, col, -1))
    pe = tf.convert_to_tensor(pe, dtype=tf.float32)

    with h5py.File(os.path.join(PATH["data"], PATH["landuse_and_poi"]), "r") as h5f:
        lp = h5f["features"][..., [item for item in DATA['landuse_poi']]]
    lp = np.reshape(lp, (row * col, -1))
    lp = (lp - np.mean(lp, axis=0)) / (np.std(lp, axis=0) + 1e-8)	
    lp = np.reshape(lp, (row, col, -1))
    lp = tf.convert_to_tensor(lp, dtype=tf.float32)
    lp = tf.broadcast_to(lp, (year, row, col, lp.shape[-1]))
    static_features = tf.concat([pe, lp], axis=-1)
    print(f"Grid features shape {static_features.shape}")

    return static_features


def get_edge_features():
    """edge_features shape (498, 5) and z-score normalized"""
    edge_features_df = pd.read_csv(os.path.join(PATH["data"], PATH["edge_features"]))
    numerical_columns = ["Distance (km)", "Duration (s)", "Straight_line_distance", "Average_speed (km/h)", "Detour_factor"]
    edge_features = edge_features_df[numerical_columns].to_numpy()
    edge_features = (edge_features - np.mean(edge_features, axis=0)) / (np.std(edge_features, axis=0) + 1e-8)
    edge_features_tensor = tf.convert_to_tensor(edge_features, dtype=tf.float32)
    return edge_features_tensor


def get_gt():
	"""average daily traffic volume, shape (8, 498), z-score normalize"""
	with h5py.File(os.path.join(PATH["data"], PATH["ground_truth"]), 'r') as f:  
		data = f['data'][:]
	scaler = Scaler(data)
	data_normalized = scaler.transform(data)
	return data_normalized, scaler