import tensorflow as tf
# To disable all GPUs
tf.config.set_visible_devices([], 'GPU')

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import os
import logging
import datetime
import random

from config import PATH, TRAINING, MODEL
import importlib
imported_module = importlib.import_module(f"model.{MODEL['model']}")
Model = getattr(imported_module, "Model")
from model.dataloader import get_gt, get_static_features
import model.utils as utils

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

# Set logging
# current_time = datetime.datetime.now()
# formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename=os.path.join(PATH["evaluate"],'training_log.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', filemode='w')

# Define a function for the training loop
def train_model(g, grid_static_features, label, scaler, train_test_index):
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING['lr'])

    for epoch in range(TRAINING['epoch']):
        for step in range(grid_static_features.shape[0]):  # each year as a step

            # Extract indices of training sensors
            train_indices = tf.where(train_test_index)
            train_indices = tf.random.shuffle(train_indices)  # Shuffle the training sensor indices for stochastic updates

            batch_size = min(TRAINING['batch_size'], len(train_indices))

            # Loop until all training sensors are processed
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i + batch_size]

                with tf.GradientTape() as tape:
                    pred = g(grid_static_features[step:step+1, ...])  # Prediction can only be made for all sensors in the year simultaneously
                    batch_pred = tf.gather(pred, batch_indices)
                    batch_label = tf.gather(label[step], batch_indices)
                    loss, _ = compute_loss(
                        batch_label, batch_pred, scaler, tf.gather(train_test_index, batch_indices), TRAINING['loss_function']
                    )
                
                grads = tape.gradient(loss, g.trainable_variables)
                grads = [tf.clip_by_value(grad, -TRAINING['clip_gradient'], TRAINING['clip_gradient']) for grad in grads]
                optimizer.apply_gradients(zip(grads, g.trainable_variables))

            # Evaluate model and get the loss as a dictionary
            train_loss = evaluate_model(g, grid_static_features, scaler, train_test_index)
            valid_loss = evaluate_model(g, grid_static_features, scaler, ~train_test_index)

            train_log_message = f"Epoch {epoch}, Step {step}, Train Loss: " + ", ".join(
                [f"{key}: {value:.2f}" for key, value in train_loss.items()]
            )
            logging.info(train_log_message)

            valid_log_message = f"Epoch {epoch}, Step {step}, Valid Loss: " + ", ".join(
                [f"{key}: {value:.2f}" for key, value in valid_loss.items()]
            )
            logging.info(valid_log_message)

            if epoch == TRAINING['epoch']-1 and step == TRAINING['step']:
                return


def compute_loss(label, pred, scaler, train_test_index, loss_function):
    """
    label shape (batch_size,) tf, one batch only, z scale
    pred shape (batch_size,) tf, one batch only, z scale
    train_test_index shape (batch_size,) numpy boolean
    """

    train_test_index = tf.convert_to_tensor(train_test_index, dtype=tf.bool)
    label_selected = tf.boolean_mask(label, train_test_index)
    pred_selected = tf.boolean_mask(pred, train_test_index)

    # Handle potential NaNs in labels: create a mask for valid (non-NaN) labels
    valid_mask = ~tf.math.is_nan(label_selected)
    label_valid = tf.boolean_mask(label_selected, valid_mask)
    pred_valid = tf.boolean_mask(pred_selected, valid_mask)

    # Compute loss
    loss = getattr(utils, loss_function)(label_valid, pred_valid, scaler)
    valid_count = tf.cast(tf.size(pred_valid), tf.float32)

    return loss, valid_count


# Evaluation function remains unchanged
def evaluate_model(g, grid_static_features, scaler, train_test_index):
    loss = {metric: 0.0 for metric in TRAINING['eval_metrics']}
    count = 0.0
    for y in range(grid_static_features.shape[0]):  # each year as a step
        pred = g(grid_static_features[y:y+1, ...])
        for metric in TRAINING['eval_metrics']:
            loss_temp, valid_count = compute_loss(label[y], pred, scaler, train_test_index, metric)
            loss[metric] += loss_temp * valid_count
        count += valid_count
    for metric in TRAINING['eval_metrics']:
        loss[metric] /= count

    return loss

# Main
if __name__ == '__main__':
    np.random.seed(TRAINING['seed'])
    tf.random.set_seed(TRAINING['seed'])
    random.seed(TRAINING['seed'])

    # add timestamp for model name
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    filename = f"{formatted_time}.h5"
    name = os.path.join(PATH["param"], filename)

    # initialize model
    g = Model() 

    # Data setup
    label, scaler = get_gt() # (8, 498)
    grid_static_features = get_static_features() # (8, 653, 574, 2)
    train_test_index = utils.train_test_sampler((label.shape[1],), TRAINING['train_prop'])

    # Training
    train_model(g, grid_static_features, label, scaler, train_test_index)

    # Save model parameters
    g.save_weights(name)
    print(g.summary())
    print("Model saved successfully.")

    # log file
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # copy the current model config to the end of log file
    with open('config.py', 'r') as config_file:
        config_content = config_file.read()

    with open(os.path.join(PATH["evaluate"], f"training_log.log"), 'a') as log_file:
        log_file.write('\n\n# Contents of config.py\n')
        log_file.write(config_content)