PATH = {
    'param': 'param',
    'data': 'data',
    'grid': 'population_and_employment/grid_1km_653_573',
    'population_and_employment': 'population_and_employment/population_and_employment.h5',
    'landuse_and_poi': 'landuse_poi/landuse_and_poi-230101.h5',
    'edge_features': 'highway_network/edge_features.csv',
    'node_features': "highway_network/node_coordinates.csv",
    'ground_truth': 'traffic_volume/average_daily_volumes_2015-2022.h5',
    'evaluate': 'eval/logs',
}

DATA = {
'population': [1, 2, 3, 4, 5, 6, 7],
'employment': [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
'landuse_poi': [],
}

MODEL = {
    # Model name
    'model': 'model_full',
    # MLP
    'activation': 'relu',
    'hiddens': [16],
    'output': [16],
    # CNN
    'roi_size': 25,
    'depth_cnn': 3,
    'channels': [16, 32, 64],
    'kernel_size': 3,
    'strides': 1, 
    'pool_size': 2,
    'pool_strides': 2,
    'output_dense': 16, # node embedding size
    # GAT
    'depth_gat': 5,
    'input_gat': 16, # node embedding size
    'output_gat': 16, # node embedding size
    'num_heads': 4,
    }

TRAINING = {
    'seed': 4,
    'lr': 1e-3,
    'epoch': 50,
    'step': 7,
    'clip_gradient': 5,
    'train_prop': 0.8,
    'batch_size': 500, # spatial batch, 500 ensures all sensors of a year in one batch
    'loss_function': 'GEH',
    'eval_metrics': ['GEH', 'MAE']
    }
