import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from functools import partial

from settings import *


class Discriminator(models.Model):

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)

        self.features = models.Sequential([
            self._conv_module(32, 3, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_module(64, 3, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_module(96, 3, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_module(128, 3, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_module(128, 3, actv=tf.nn.leaky_relu, batch_norm=True),
        ], name="disc_features")

        self.classifier = models.Sequential([
            self._fc_module(256, actv=tf.nn.leaky_relu, dropout=True),
            self._fc_module(64, actv=tf.nn.leaky_relu, dropout=True),
            self._fc_module(NUM_ATT, actv=tf.nn.sigmoid),
        ], name="disc_classifier")

        self.discriminator = models.Sequential([
            self._fc_module(256, actv=tf.nn.leaky_relu, dropout=True),
            self._fc_module(64, actv=tf.nn.leaky_relu, dropout=True),
            self._fc_module(1, actv=tf.nn.sigmoid),
        ], name="disc_discriminator")

    @tf.function
    def call(self, inputs, training=False):
        n = tf.shape(inputs)[0]
        
        x = self.features(inputs, training=training)
        x = tf.reshape(x, (n, -1))

        c = self.classifier(x, training=training)
        d = self.discriminator(x, training=training)
        return c, d

    def _conv_module(self, n, f, actv=None, input_shape=None, batch_norm=False, dropout=False):
        if input_shape is None:
            conv = partial(layers.Conv2D)
        else:
            conv = partial(layers.Conv2D, input_shape=input_shape)

        modules = [
            conv(n, (f, f), strides=2, padding="SAME"),
        ]

        if batch_norm is True:
            modules.append(layers.BatchNormalization())

        if actv is not None:
            modules.append(layers.Activation(actv))

        if dropout is True:
            modules.append(layers.Dropout(0.5))

        return models.Sequential(modules)

    def _fc_module(self, n, actv=None, input_shape=None, batch_norm=False, dropout=False):
        if input_shape is None:
            dense = partial(layers.Dense)
        else:
            dense = partial(layers.Dense, input_shape=input_shape)

        modules = [
            dense(n)
        ]

        if batch_norm is True:
            modules.append(layers.BatchNormalization())

        if actv is not None:
            modules.append(layers.Activation(actv))

        if dropout is True:
            modules.append(layers.Dropout(0.5))

        return models.Sequential(modules)
