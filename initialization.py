import tensorflow as tf
import numpy as np


def fan_in_init(input_layer_dim):
    return tf.random_uniform_initializer(minval=-1.0 / np.sqrt(input_layer_dim), maxval=1.0 / np.sqrt(input_layer_dim))


def uniform_init(minval, maxval):
    tf.random_uniform_initializer(minval, maxval)