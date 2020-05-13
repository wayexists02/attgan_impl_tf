import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers
from functools import partial
from settings import *


class Genc(models.Model):

    def __init__(self, *args, **kwargs):
        super(Genc, self).__init__(*args, **kwargs)

        self.encoders = [
            self._conv_module(32, 5, actv=tf.nn.leaky_relu, batch_norm=True), # 112
            self._conv_module(64, 5, actv=tf.nn.leaky_relu, batch_norm=True), # 56
            self._conv_module(128, 5, actv=tf.nn.leaky_relu, batch_norm=True), # 28
            self._conv_module(256, 5, actv=tf.nn.leaky_relu, batch_norm=True), # 14
            self._conv_module(512*2, 5, actv=None, batch_norm=False), # 7
        ]

    def call(self, inputs, training=False):

        x = inputs
        skip_conn = []

        for layer in self.encoders:
            x = layer(x, training=training)
            skip_conn.append(x)

        skip_conn[0] = None
        # skip_conn[2] = None
        skip_conn[3] = None
        skip_conn[4] = None

        return x, skip_conn

    def _conv_module(self, n, f, actv=None, input_shape=None, batch_norm=False, dropout=False):
        if input_shape is None:
            conv = partial(layers.Conv2D)
        else:
            conv = partial(layers.Conv2D, input_shape=input_shape)

        modules = [
            conv(n, (f, f), strides=2, padding="SAME")
        ]

        if batch_norm is True:
            modules.append(layers.BatchNormalization())

        if actv is not None:
            modules.append(layers.Activation(actv))

        if dropout is True:
            modules.append(layers.Dropout(0.5))

        return models.Sequential(modules)
