import tensorflow as tf
from config import TRAINING

tf.random.set_seed(TRAINING['seed'])

class Scaler:
	def __init__(self, data):
		# Ignore NaNs in mean and std calculations
		valid_data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
		count = tf.reduce_sum(tf.cast(~tf.math.is_nan(data), tf.float32))

		self.mean = tf.reduce_sum(valid_data) / count
		self.std = tf.sqrt(tf.reduce_sum(tf.square(valid_data - self.mean)) / count)

		tf.print("Mean:", self.mean, "Standard Deviation:", self.std)

	def transform(self, data):
		# Standardize data while ignoring NaNs
		return tf.where(tf.math.is_nan(data), data, (data - self.mean) / (self.std + 1e-8))
    
	def inverse_transform(self, data):
		# Reverse standardization while ignoring NaNs
		return tf.where(tf.math.is_nan(data), data, data * self.std + self.mean)


def MSE_z(label_z, pred_z, scaler):
	return tf.reduce_mean(tf.square(label_z - pred_z))

def MAE(label_z, pred_z, scaler):
	pred = scaler.inverse_transform(pred_z)
	label = scaler.inverse_transform(label_z)
	return tf.reduce_mean(tf.abs(label - pred))

def MSE(label_z, pred_z, scaler):
	pred = scaler.inverse_transform(pred_z)
	label = scaler.inverse_transform(label_z)
	return tf.reduce_mean(tf.square(label - pred))

def GEH(label_z, pred_z, scaler):
	pred = scaler.inverse_transform(pred_z)
	label = scaler.inverse_transform(label_z)
	gehs = tf.sqrt(2 * tf.square(pred - label) / (tf.abs(pred) + label + 1e-8)) # pred for small volume sensors can be negative
	return tf.reduce_mean(gehs)


def train_test_sampler(shape, true_probability):
    """
    Generate a boolean tensor of a given shape with True or False values based on a specified probability for True.
    In this context, randomly choose the number of train and test sensors.
    """
    # Generate random values between 0 and 1 with the given shape using TensorFlow
    random_values = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    
    # Create a boolean tensor where values are True if random_values < true_probability
    boolean_tensor = random_values < true_probability
    
    return boolean_tensor
