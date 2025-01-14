import tensorflow as tf
from tensorflow.keras import layers
import dgl
from config import MODEL, TRAINING
import numpy as np

from model.dataloader import get_edge_features

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])

# import datetime
# current_time = datetime.datetime.now()
# formatted_time = current_time.strftime('%Y%m%d-%H%M%S')

class MLP(tf.keras.Sequential):
    """ Multilayer perceptron. """
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        """
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        out_act: bool
            Whether to apply activation function after the last dense layer.
        """
        super(MLP, self).__init__(**kwargs)
        for i, h in enumerate(hiddens):
            activation = None if i == len(hiddens) - 1 and not out_act else act_type
            self.add(tf.keras.layers.Dense(
                h, activation=activation, kernel_initializer=weight_initializer, 
                name=f'dense_{i}'  # Ensure unique name for each layer
            ))


class Model(tf.keras.Model):
    """
    The model.
    Input: a year of grid feature (row=653, col=574, d=2)
    Output: a year of average daily traffic volume (e=498,)
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        # edge features
        self.edge_features = get_edge_features()
        self.edge_features_embeddings = MLP(MODEL['hiddens'] + MODEL['output'], act_type=MODEL['activation'], out_act=False)

        # MLP for edge embedding and final output
        self.edge_update_mlp = MLP(MODEL['hiddens'] + [1], act_type=MODEL['activation'], out_act=False)

    def call(self, grid_static_feature):

        self.edge_embeddings = self.edge_features_embeddings(self.edge_features)
        traffic_volume = self.edge_update_mlp(self.edge_embeddings)  # [e, 1]
        traffic_volume = tf.squeeze(traffic_volume)
    
        return traffic_volume