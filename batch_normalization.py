import tensorflow as tf
import numpy as np

def batch_norm(batch) :

    epsilon = 1e-5
    beta = tf.Variable(tf.constant(0.0, shape=[1]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[1]), trainable=True)
    mean, variance = tf.nn.moments(batch, axes=[0])

    norm_batch = tf.nn.batch_normalization(batch, mean, variance, beta, gamma, epsilon)

    return norm_batch