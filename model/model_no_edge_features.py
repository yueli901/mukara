import tensorflow as tf
from tensorflow.keras import layers
import dgl
from config import MODEL, TRAINING
import numpy as np

from model.dataloader import adjacency_matrix, get_node_positions

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


class CNN(tf.keras.Model):
    def __init__(self, 
                 depth: int, 
                 channels: list, 
                 kernel_size: int, 
                 strides: int, 
                 pool_size: int,
                 pool_strides: int,
                 output_dense: int,
                 weight_initializer=None, 
                 **kwargs):
        """
        For generating node embeddings from ROI.
        """
        super(CNN, self).__init__(**kwargs)
        
        # Create layers dynamically based on the depth and channels
        self.conv_layers = []
        for i in range(depth):
            self.conv_layers.append(
                layers.Conv2D(
                    filters=channels[i],
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    activation="relu",
                    kernel_initializer=weight_initializer,
                    name=f"conv_{i}"
                )
            )
            self.conv_layers.append(
                layers.MaxPooling2D(
                    pool_size=pool_size,
                    strides=pool_strides,
                    padding="same",
                    name=f"maxpool_{i}"
                )
            )
        
        self.flatten = layers.Flatten(name="flatten")
        
        # Add a dense layer to convert to 1D node embeddings
        self.dense_output = layers.Dense(
            units=output_dense,
            activation="relu",
            kernel_initializer=weight_initializer,
            name="dense_output"
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass of the CNN head.
        """
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense_output(x)
        return x


class Model(tf.keras.Model):
    """
    The model.
    Input: a year of grid feature (row=653, col=574, d=2)
    Output: a year of average daily traffic volume (e=498,)
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        # Graph initialization
        src, dst = adjacency_matrix()
        self.g = dgl.graph((src, dst), num_nodes=len(set(src)))

        # Node positions
        self.node_positions = get_node_positions()

        # CNN for grid feature extraction
        self.roi_size = MODEL['roi_size']
        self.cnn = CNN(
            depth=MODEL['depth_cnn'],
            channels=MODEL['channels'],
            kernel_size=MODEL['kernel_size'],
            strides=MODEL['strides'],
            pool_size=MODEL['pool_size'],
            pool_strides=MODEL['pool_strides'],
            output_dense=MODEL['output_dense'],
            weight_initializer='he_normal'
        )

        # GAT layers for node embedding updates
        self.gat_layers = [GraphAttentionLayer(MODEL['input_gat'], MODEL['output_gat'], MODEL['num_heads'], depth) for depth in range(MODEL['depth_gat'])]

        # MLP for edge embedding and final output
        self.edge_update_mlp = MLP(MODEL['hiddens'] + [1], act_type=MODEL['activation'], out_act=False)

    def call(self, grid_static_feature):
        """
        grid_static_feature [1, row, col, c] for one year
        """
        # Extract node embeddings using CNN
        rois = self.get_rois(grid_static_feature)
        node_embeddings = self.cnn(rois)  # [n, roi_size, roi_size, c] to [n, d]
        
        # Assign node embeddings to the graph
        self.g.ndata['embedding'] = node_embeddings

        # Apply GAT layers
        for gat_layer in self.gat_layers:
            self.g.ndata['embedding'] = gat_layer(self.g, self.g.ndata['embedding'])

        # Extract the updated node embeddings
        updated_node_embeddings = self.g.ndata['embedding'] # [n, d]

        # Compute edge embeddings and scalar outputs
        traffic_volume = self.compute_edge_embeddings(updated_node_embeddings)
        return traffic_volume

    def get_rois(self, grid_static_feature):
        """
        Extracts small region of interests (rois, or windows) of data around each sensor position from a TensorFlow tensor.
        """
        if self.roi_size % 2 == 0:
            raise ValueError("roi_size must be an odd number to ensure the roi is centered.")
        
        pad_size = self.roi_size // 2
        
        # Pad the grid data with zeros
        padded_grid = tf.pad(
            tf.squeeze(grid_static_feature, axis=0), 
            paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
            mode='CONSTANT',
            constant_values=0
        )
        
        def extract_roi(pos):
            row, col = tf.unstack(pos)
            roi = padded_grid[row:row + self.roi_size, col:col + self.roi_size, :]
            return roi
        
        rois = tf.map_fn(extract_roi, self.node_positions, fn_output_signature=tf.TensorSpec((self.roi_size, self.roi_size, grid_static_feature.shape[-1])))
        
        return rois

    def compute_edge_embeddings(self, node_embeddings):
        """
        Compute edge embeddings and final scalar outputs based on updated node embeddings.
        """
        src_emb = tf.gather(node_embeddings, self.g.edges()[0])  # Source node embeddings [e, d]
        dst_emb = tf.gather(node_embeddings, self.g.edges()[1])  # Destination node embeddings [e, d]

        # Concatenate source and destination embeddings to form edge embeddings
        edge_embeddings = tf.concat([src_emb, dst_emb], axis=-1)  # [e, 2d]

        # Update edge embeddings using MLP
        traffic_volume = self.edge_update_mlp(edge_embeddings)  # [e, 1]
        traffic_volume = tf.squeeze(traffic_volume)

        return traffic_volume


class GraphAttentionLayer(tf.keras.layers.Layer):
    """
    Single Graph Attention Layer with explicit naming for layers and heads to avoid scope overlap.
    """

    def __init__(self, input_dim, output_dim, num_heads, layer_idx, **kwargs):
        """
        layer_idx: Index of the GAT layer to ensure unique names for each layer.
        """
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_heads = []

        # Loop through each head and create an AttentionHead with a unique name
        for i in range(num_heads):
            self.attention_heads.append(
                AttentionHead(input_dim, output_dim, head_idx=i, layer_idx=layer_idx, name=f'attention_head_{layer_idx}_{i}')
            )
        self.update_mlp = MLP(
            MODEL['hiddens'] + [MODEL['output_gat']], 
            act_type=MODEL['activation'], 
            out_act=False, 
            name=f"gat_update_mlp_{layer_idx}"
        )

    def call(self, g, node_embeddings):
        """
        Forward pass for the GAT layer.
        """
        # Compute outputs from each attention head
        head_outputs = [head(g, node_embeddings) for head in self.attention_heads]

        # Concatenate the outputs from all attention heads
        node_embeddings_concat = tf.concat(head_outputs, axis=-1)

        # Update node embeddings using MLP
        node_embeddings_update = self.update_mlp(node_embeddings_concat)
        return node_embeddings_update


class AttentionHead(tf.keras.layers.Layer):
    """
    Single attention head for GAT with unique naming.
    """

    def __init__(self, input_dim, output_dim, head_idx, layer_idx, **kwargs):
        """
        head_idx: Index of the attention head to ensure unique names.
        layer_idx: Index of the GAT layer.
        """
        super(AttentionHead, self).__init__(**kwargs)
        self.output_dim = output_dim

        # Explicitly name each variable to avoid overlap
        self.W_n = self.add_weight(
            shape=(input_dim, output_dim), 
            initializer='glorot_uniform', 
            trainable=True, 
            name=f'attention_W_n_{layer_idx}_{head_idx}'
        )
        self.a_src = self.add_weight(
            shape=(output_dim, 1), 
            initializer='glorot_uniform', 
            trainable=True, 
            name=f'attention_a_src_{layer_idx}_{head_idx}'
        )
        self.a_dst = self.add_weight(
            shape=(output_dim, 1), 
            initializer='glorot_uniform', 
            trainable=True, 
            name=f'attention_a_dst_{layer_idx}_{head_idx}'
        )

    def call(self, g, node_embeddings):
        """
        Forward pass for a single attention head.
        Input: g, node_embeddings [n, c=input_dim]
        Output: updated node_embeddings [n, d=output_dim] 
        """
        # Linear transformation of node embeddings
        node_embeddings_transformed = tf.matmul(node_embeddings, self.W_n)  # [n, d] * [d, d]

        # Assign transformed node embeddings to graph
        g.ndata['embedding'] = node_embeddings_transformed

        # Perform message passing with attention score computation
        g.update_all(self.message_func, self.reduce_func)

        # Return updated node embeddings
        return g.ndata.pop('updated_embedding')

    def message_func(self, edges):
        """
        Message function for GAT that calculates raw attention scores and stores them in edata.
        """
        # Compute raw attention scores
        h_src = tf.matmul(edges.src['embedding'], self.a_src)
        h_dst = tf.matmul(edges.dst['embedding'], self.a_dst)
        e = tf.nn.leaky_relu(h_src + h_dst)

        # Return source node embeddings as messages
        return {'e': e, 'm': edges.src['embedding']}

    def reduce_func(self, nodes):
        """
        Reduce function for GAT that calculates normalized attention coefficients and updates node embeddings.
        """
        # Retrieve raw attention scores from mailbox
        e = nodes.mailbox['e']  # Shape: [n, num_neighbors, 1]

        # Compute attention coefficients by applying softmax to raw scores
        alpha = tf.nn.softmax(e, axis=1)  # Normalize across neighbors

        # Retrieve messages (source node embeddings) from mailbox
        m = nodes.mailbox['m']  # Shape: [n, num_neighbors, d]

        # Compute attention-weighted sum of neighbor embeddings
        updated_embedding = tf.nn.relu(tf.reduce_sum(alpha * m, axis=1))  # Shape: [N, d]

        # Return the updated node embeddings
        return {'updated_embedding': updated_embedding}
