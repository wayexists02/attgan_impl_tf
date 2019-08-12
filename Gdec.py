import tensorflow as tf
import numpy as np


class Gdec():
    """
    Decoder part of generator (auto encoder)
    """

    def __init__(self):
        self.layers = []

    def build(self, X, params, enc_layers):
        """
        Build G-decoder
        
        Arguments:
        ----------
        :X input tensor
        :params parameters (weights) dictionary
        :enc_layers layers of encoder. it is used to do skip connection
        """
        
        with tf.name_scope("gdec"):
            layer1 = self._dconv_layer(X, params["W1_dec"], params["b1_dec"], 2)
            layer2 = self._dconv_layer(layer1, params["W2_dec"], params["b2_dec"], 2, enc_layers[-2])
            layer3 = self._dconv_layer(layer2, params["W3_dec"], params["b3_dec"], 2, enc_layers[-3])
            layer4 = self._dconv_layer(layer3, params["W4_dec"], params["b4_dec"], 2, enc_layers[-4])
            layer5 = self._dconv_layer(layer4, params["W5_dec"], params["b5_dec"], 1, bn=False)

            self.layers.extend([layer1, layer2, layer3, layer4, layer5])

        return layer5

    def _dconv_layer(self, X, W, b, s, skip_conn=None, bn=True, actv=tf.nn.tanh, skip_conn_actv=tf.nn.tanh):
        """
        build deconv layer.
        
        Arguments:
        ----------
        :X input tensor
        :W weight variable
        :b bias variable
        :s stride
        :skip_conn layer for skip connection
        :bn whether skip connection will be added
        """
        
        # retrieve useful information
        n = tf.shape(X)[0]
        h, w = X.get_shape().as_list()[1:3]
        if s == 2:
            h *= 2
            w *= 2

        c = W.get_shape().as_list()[2]

        # if skip connection is added, add it.
        if skip_conn is not None:
            X = X + skip_conn
            if skip_conn_actv is not None:
                X = skip_conn_actv(X)
            
        # apply transposed convolution
        layer = tf.nn.conv2d_transpose(X, W, (n, h, w, c), strides=(1, s, s, 1), padding="SAME") + b

        # apply batch norm
        if bn is True:
            mean, var = tf.nn.moments(layer, axes=[1, 2, 3], keep_dims=True)
            layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)
        
        # activation
        if actv is not None:
            layer = actv(layer)

        return layer
