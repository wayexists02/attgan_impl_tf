import tensorflow as tf
import numpy as np


class Genc():

    def __init__(self):
        self.layers = []

    def build(self, X, params):
        with tf.name_scope("genc"):
            layer1 = self._conv_layer(X, params["W1_enc"], params["b1_enc"], 2)
            layer2 = self._conv_layer(layer1, params["W2_enc"], params["b2_enc"], 2)
            layer3 = self._conv_layer(layer2, params["W3_enc"], params["b3_enc"], 2)
            layer4 = self._conv_layer(layer3, params["W4_enc"], params["b4_enc"], 2)

            self.layers.extend([layer1, layer2, layer3, layer4])

        return layer4

    def _conv_layer(self, X, W, b, s):
        layer = tf.nn.conv2d(X, W, strides=(1, s, s, 1), padding="SAME") + b
        
        mean, var = tf.nn.moments(layer, axes=[1, 2, 3], keep_dims=True)
        layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

        layer = tf.nn.tanh(layer)

        return layer

