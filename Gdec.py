import tensorflow as tf
import numpy as np


class Gdec():

    def __init__(self):
        self.layers = []

    def build(self, X, params, enc_layers):
        with tf.name_scope("gdec"):
            layer1 = self._dconv_layer(X, params["W1_dec"], params["b1_dec"], 2)
            layer2 = self._dconv_layer(layer1, params["W2_dec"], params["b2_dec"], 2, enc_layers[-2])
            layer3 = self._dconv_layer(layer2, params["W3_dec"], params["b3_dec"], 2, enc_layers[-3])
            layer4 = self._dconv_layer(layer3, params["W4_dec"], params["b4_dec"], 2)

            self.layers.extend([layer1, layer2, layer3, layer4])

        return layer4

    def _dconv_layer(self, X, W, b, s, skip_conn=None):
        n = tf.shape(X)[0]
        h, w = X.get_shape().as_list()[1:3]
        if s == 2:
            h *= 2
            w *= 2

        c = W.get_shape().as_list()[2]

        if skip_conn is not None:
            X = tf.nn.tanh(X + skip_conn)
            
        layer = tf.nn.conv2d_transpose(X, W, (n, h, w, c), strides=(1, s, s, 1), padding="SAME") + b

        mean, var = tf.nn.moments(layer, axes=[1, 2, 3], keep_dims=True)
        layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)
        
        layer = tf.nn.tanh(layer)

        return layer
