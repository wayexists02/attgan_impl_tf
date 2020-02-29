import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers
from functools import partial
from settings import *


class Gdec(models.Model):

    def __init__(self, *args, **kwargs):
        super(Gdec, self).__init__(*args, **kwargs)

        self.decoders = [
            self._conv_transpose_module(128, 5, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_transpose_module(96, 5, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_transpose_module(64, 5, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_transpose_module(32, 5, actv=tf.nn.leaky_relu, batch_norm=True),
            self._conv_transpose_module(3, 5, actv=tf.nn.tanh),
        ]

    @tf.function
    def call(self, inputs, skip_conn, training=False):

        x = inputs

        for layer, skip in zip(self.decoders, skip_conn[::-1]):
            if skip is not None:
                x = layer(x + skip, training=training)
            else:
                x = layer(x, training=training)

        return x

    def _conv_transpose_module(self, n, f, actv=None, input_shape=None, batch_norm=False, dropout=False):
        if input_shape is None:
            conv = partial(layers.Conv2DTranspose)
        else:
            conv = partial(layers.Conv2DTranspose, input_shape=input_shape)

        modules = [
            conv(n, (f, f), strides=2, padding="SAME", output_padding=1),
        ]

        if batch_norm is True:
            modules.append(layers.BatchNormalization())

        if actv is not None:
            modules.append(layers.Activation(actv))

        if dropout is True:
            modules.append(layers.Dropout(0.5))

        return models.Sequential(modules)
