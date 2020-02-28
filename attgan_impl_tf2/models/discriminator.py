import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models

from settings import *


class Discriminator(models.Model):

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)

        self.features = models.Sequential([
            self._conv_module(32, 3),
            self._conv_module(64, 3),
            self._conv_module(96, 3),
            self._conv_module(128, 3),
            self._conv_module(128, 3),
        ])

        self.classifier = models.Sequential([
            self._fc_module(256),
            self._fc_module(64),
            self._fc_module(NUM_ATT, actv=tf.nn.sigmoid),
        ])

        self.discriminator = models.Sequential([
            self._fc_module(256),
            self._fc_module(64),
            self._fc_module(1, actv=tf.nn.sigmoid),
        ])

    def call(self, inputs, training=False):
        n = tf.shape(inputs)[0]
        
        x = self.features(inputs, training=training)
        x = tf.reshape(x, (n, -1))

        c = self.classifier(x, training=training)
        d = self.discriminator(x, training=training)
        return c, d

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

    def _fc_module(self, n, actv=tf.nn.leaky_relu):
        if type(actv) is not tf.nn.sigmoid:
            return models.Sequential([
                layers.Dense(n),
                layers.Activation(actv),
                layers.Dropout(0.5)
            ])
        else:
            return models.Sequential([
                layers.Dense(n),
                layers.Activation(actv)
            ])
