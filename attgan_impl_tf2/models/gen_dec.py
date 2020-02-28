import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers


class Gdec(models.Model):

    def __init__(self, *args, **kwargs):
        super(Gdec, self).__init__(*args, **kwargs)

        self.decoders = [
            self._conv_transpose_module(128, 5),
            self._conv_transpose_module(96, 5),
            self._conv_transpose_module(64, 5),
            self._conv_transpose_module(32, 5),
            self._conv_transpose_module(3, 5, tf.nn.tanh),
        ]

    def call(self, inputs, skip_conn, training=False):

        x = inputs

        for layer, skip in zip(self.decoders, skip_conn[::-1]):
            if skip is not None:
                x = layer(x + skip, training=training)
            else:
                x = layer(x, training=training)

        return x

    def _conv_transpose_module(self, n, f, actv=tf.nn.leaky_relu):
        if actv is not None:
            return models.Sequential([
                layers.Conv2DTranspose(n, (f, f), strides=2, padding="SAME", output_padding=1),
                layers.BatchNormalization(),
                layers.Activation(actv),
            ])
        else:
            return models.Sequential([
                layers.Conv2DTranspose(n, (f, f), strides=2, padding="SAME", output_padding=1),
                layers.BatchNormalization(),
            ])
