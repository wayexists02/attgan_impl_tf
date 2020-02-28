import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from . import gen_dec, gen_enc, discriminator


class AttGAN(models.Model):

    def __init__(self, *args, **kwargs):
        super(AttGAN, self).__init__(*args, **kwargs)

        self.encoder = gen_enc.Genc()
        self.decoder = gen_dec.Gdec()
        self.disc = discriminator.Discriminator()

    def call(self, inputs, att, training=False):
        z, skip_conn = self.encoder(inputs, training=training)

        mean = z[:, :, :, :64]
        logvar = z[:, :, :, 64:]

        sample_z = self._reparameterize(mean, logvar)
        att_sample_z = self._add_attribute(sample_z, att)
        reconstructed = self.decoder(att_sample_z, skip_conn, training=training)

        c, d = self.disc(reconstructed, training=training)

        return reconstructed, c, d

    def _reparameterize(self, mean, logvar):
        r = tf.random.normal(tf.shape(mean))
        sd = tf.math.exp(logvar/2)

        sample = r * sd + mean
        return sample

    def _add_attribute(self, sample_z, att):
        n = tf.shape(sample_z)[0]
        h = tf.shape(sample_z)[1]
        w = tf.shape(sample_z)[2]

        att_tiled = tf.tile(tf.reshape(att, (n, 1, 1, -1)), [1, h, w, 1])
        att_sample_z = tf.concat([sample_z, att_tiled], axis=-1)

        return att_sample_z
