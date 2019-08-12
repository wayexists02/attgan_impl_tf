import tensorflow as tf
import numpy as np


class Genc():
    """
    G-encoder.
    Encoder part of autoencoder
    """

    def __init__(self):
        self.layers = []

    def build(self, X, params):
        """
        Build encoder part of generator.
        
        Arguments:
        ----------
        :X input tensor
        :params parameters dictionary
        """
        
        with tf.name_scope("genc"):
            layer1 = self._conv_layer(X, params["W1_enc"], params["b1_enc"], 1)
            layer2 = self._conv_layer(layer1, params["W2_enc"], params["b2_enc"], 2)
            layer3 = self._conv_layer(layer2, params["W3_enc"], params["b3_enc"], 2)
            layer4 = self._conv_layer(layer3, params["W4_enc"], params["b4_enc"], 2)
            layer5 = self._conv_layer(layer4, params["W5_enc"], params["b5_enc"], 2)
            
            self.mean = self._conv_layer(layer5, params["W6_enc_mean"], params["b6_enc_mean"], 1, bn=False, actv=tf.nn.tanh)
            self.logvar = self._conv_layer(layer5, params["W6_enc_var"], params["b6_enc_var"], 1, bn=False, actv=tf.nn.sigmoid)
            
            std = tf.math.exp(self.logvar*0.5)
            normal = tf.random_normal(tf.shape(std), dtype=tf.float32)
            
            latent = normal * std + self.mean

            self.layers.extend([layer1, layer2, layer3, layer4, layer5])

        return latent
    
    def loss_function(self):
        loss = 0.5 * (tf.math.exp(self.logvar) + self.mean**2 - self.logvar - 1)
        return tf.reduce_mean(loss)

    def _conv_layer(self, X, W, b, s, bn=True, actv=tf.nn.tanh):
        """
        build convolution layer.
        
        Arguments:
        ----------
        :X input tensor
        :W weight variable for this layer
        :b bias variable for this layer
        :s stride value
        :bn whether batch norm will be applied
        """
        
        layer = tf.nn.conv2d(X, W, strides=(1, s, s, 1), padding="SAME") + b
        
        if bn is True:
            mean, var = tf.nn.moments(layer, axes=[1, 2, 3], keep_dims=True)
            layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

        if actv is not None:
            layer = actv(layer)

        return layer

