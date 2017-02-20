from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import h5py
import numpy as np
import tensorflow as tf

def create_net():
    X = tf.placeholder(shape=(None, 28, 28, 3), dtype=tf.float32)
    Y = tf.placeholder(shape=(None, 4), dtype=tf.float32)

    # network = tf.reshape(X, [-1, 28, 28, 3])

    network = conv_2d(X, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4, activation='softmax')

    return X, Y, network
