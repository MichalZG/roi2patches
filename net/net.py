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
# Data loading and preprocessing
"""
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Create a hdf5 dataset from CIFAR-10 numpy array
import h5py
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('cifar10_X', data=X)
h5f.create_dataset('cifar10_Y', data=Y)
h5f.create_dataset('cifar10_X_test', data=X_test)
h5f.create_dataset('cifar10_Y_test', data=Y_test)
h5f.close()

# Load hdf5 dataset
h5f = h5py.File('data.h5', 'r')
X = h5f['cifar10_X']
Y = h5f['cifar10_Y']
X_test = h5f['cifar10_X_test']
Y_test = h5f['cifar10_Y_test']
"""

h5f = h5py.File('/home/pi/Programs/python-programs/roi2patches/data/datasets/28x28_rgb_eq_test.h5', 'r')

dset_X = h5f['X'][:1000]
dset_Y = h5f['Y'][:1000]
dset_X_test = h5f['X_val'][:1000]
dset_Y_test = h5f['Y_val'][:1000]
dset_Y = to_categorical(dset_Y[...]-1, 4)
dset_Y_test = to_categorical(dset_Y_test[...]-1, 4)


# Real-time data preprocessing
img_prep = ImagePreprocessing()
# img_prep.add_featurewise_zero_center()
# img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
with tf.Graph().as_default():
    
    from net_model import create_net

    X, Y, network = create_net()
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(network, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        batch_size = 100

        for epoch in range(5):
            avg_cost = 0.
            total_batch = dset_X.shape[0]
            for i in range(0, total_batch, batch_size):
                batch_xs = dset_X[i:i+batch_size]
                batch_ys = dset_Y[i:i+batch_size]
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
                cost = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += cost/total_batch
                if i % 20 == 0:
                    print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,
                          "Loss:", str(cost))
    
        saver.save(sess, 'my-model')
with tf.Graph().as_default():
    
    from net_model import create_net

    X, Y, network = create_net()
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(network, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('my-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        batch_x = batch_xs
        predictions = sess.run(network, feed_dict={X: batch_x})
        for i, p in enumerate(predictions):
            print(p, batch_ys[i])

