import tensorflow as tf
import numpy as np
import os
import time
import shutil
import pathlib


def wgan_gp(x_origin, x_reconstructed, D):
    n = x_origin.shape[0]
    r = tf.random.uniform(minval=0.0, maxval=1.0, shape=(n, 1, 1, 1))

    samples = (x_reconstructed - x_origin) * r + x_origin

    with tf.GradientTape() as tape:
        tape.watch(samples)
        _, d = D(samples, training=True)
        d = tf.reshape(d, (n, 1, 1, 1))

    grad = tape.gradient(d, samples)
    grad = tf.reshape(grad, (n, -1))
    norm = tf.norm(grad + 1e-6, axis=-1)
    gp = tf.reduce_mean((norm - 1)**2)

    # print(gp)

    return gp

def generate_attribute(n, num_att):
    att = np.random.uniform(0, 1, size=(n, num_att))
    att[att >= 0.5] = 1.
    att[att < 0.5] = 0.

    return att.astype(np.float32)

def save_model_with_source(model, path, source_dir):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)

    path1 = os.path.join(path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))).replace("\\", "/")
    os.mkdir(path1)

    path2 = os.path.join(path1, pathlib.Path(source_dir).name).replace("\\", "/")

    shutil.copytree(source_dir, path2)
    
    tf.saved_model.save(model, path1)
    print("Model was saved.")

def load_model(path):
    model = tf.saved_model.load(path)
    return model
