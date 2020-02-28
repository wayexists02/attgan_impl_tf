import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers


class Genc(models.Model):

    def __init__(self, *args, **kwargs):
        super(Genc, self).__init__(*args, **kwargs)

        self.encoders = [
            self._conv_module(32, 5),
            self._conv_module(64, 5),
            self._conv_module(96, 5),
            self._conv_module(128, 5),
            self._conv_module(128, 5, None),
        ]

    def call(self, inputs, training=False):

        x = inputs
        skip_conn = []

        for layer in self.encoders:
            x = layer(x, training=training)
            skip_conn.append(x)

        skip_conn[0] = None
        skip_conn[-1] = None

        return x, skip_conn

    def _conv_module(self, n, f, actv=tf.nn.leaky_relu):
        if actv is not None:
            return models.Sequential([
                layers.Conv2D(n, (f, f), strides=2, padding="SAME"),
                layers.BatchNormalization(),
                layers.Activation(actv),
            ])
        else:
            return models.Sequential([
                layers.Conv2D(n, (f, f), strides=2, padding="SAME"),
                layers.BatchNormalization(),
            ])
