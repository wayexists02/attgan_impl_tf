import tensorflow as tf
import numpy as np


class D():

    def __init__(self):
        pass

    def build(self, X, params):

        with tf.name_scope("d"):
            layer1 = self._conv_layer(X, params["W1_d"], params["b1_d"], 2)
            layer2 = self._conv_layer(layer1, params["W2_d"], params["b2_d"], 2)
            layer3 = self._conv_layer(layer2, params["W3_d"], params["b3_d"], 2)
            layer4 = self._conv_layer(layer3, params["W4_d"], params["b4_d"], 2)

            layer4 = tf.reshape(layer4, (tf.shape(X)[0], -1))

            layer5_d = self._fc_layer(layer4, params["W5_fc_d"], params["b5_fc_d"])
            layer6_d = self._fc_layer(layer5_d, params["W6_fc_d"], params["b6_fc_d"], bn=False)

            layer5_att = self._fc_layer(layer4, params["W5_fc_att"], params["b5_fc_att"])
            layer6_att = self._fc_layer(layer5_att, params["W6_fc_att"], params["b6_fc_att"], bn=False)

        return layer6_d, layer6_att

    def _conv_layer(self, X, W, b, s):
        layer = tf.nn.conv2d(X, W, strides=(1, s, s, 1), padding="SAME") + b

        mean, var = tf.nn.moments(layer, axes=[1, 2, 3], keep_dims=True)
        layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

        layer = tf.nn.tanh(layer)

        return layer

    def _fc_layer(self, X, W, b, bn=True):
        layer = tf.matmul(X, W) + b
        
        if bn:
            mean, var = tf.nn.moments(layer, axes=[1], keep_dims=True)
            layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

        layer = tf.nn.tanh(layer)

        return layer
