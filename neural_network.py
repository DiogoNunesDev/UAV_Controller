from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow import concat
import numpy as np

import tensorflow as tf

tf.get_logger().setLevel('ERROR')


class NeuralNetwork:
  
  def __init__(self, input_dim, output_dim):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.model = self.create_model()
    
  def create_model(self):
    """Creates and returns a neural network model."""
    model = Sequential()
    model.add(Dense(128, input_shape=(self.input_dim,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(self.output_dim, activation='tanh'))  # Limita as saídas entre -1 e 1
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
  
  def model_to_genome(self, model):
    """Converts a Keras model to a genome (1D numpy array)."""
    genome = []
    for layer in model.layers:
      weights = layer.get_weights()
      for w in weights:
        genome.extend(w.flatten())
    return np.array(genome)

  def genome_to_model(self, genome):
    """Converts a genome back to a Keras model by setting the weights."""
    offset = 0
    for layer in self.model.layers:
      layer_weights = []
      for weight in layer.get_weights():
        shape = weight.shape
        size = np.prod(shape)
        layer_weights.append(genome[offset:offset + size].reshape(shape))
        offset += size
      if len(layer_weights) == len(layer.get_weights()):
        layer.set_weights(layer_weights)
      else:
        raise ValueError(f"Expected {len(layer.get_weights())} weights but got {len(layer_weights)}.")
    return self.model
  
  def generate_random_genome(self, nn):
    """Generates a random genome corresponding to the neural network's parameters."""
    genome = []
    for layer in nn.model.layers:
      weights = layer.get_weights()
      for w in weights:
        # Generating random weights and biases with the same shape as the model's weights and biases
        random_weights = np.random.randn(*w.shape)
        genome.extend(random_weights.flatten())
    return np.array(genome)
  