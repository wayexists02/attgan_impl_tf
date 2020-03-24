import tensorflow as tf
import numpy as np
import os
import time
import shutil
import pathlib


def wgan_gp(x_origin, x_reconstructed, D):
    n = x_origin.shape[0]
    r = tf.random.uniform(minval=0.0, maxval=1.0, shape=(n, 1, 1, 1))

    samples = x_origin*r + x_reconstructed*(1 - r)

    with tf.GradientTape() as tape:
        tape.watch(samples)
        _, d = D(samples, training=False)

    grad = tape.gradient(d, [samples])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0)**2)

    # print(gp)

    return gp

def generate_attribute(att):
    new_att = np.copy(att)
    np.random.shuffle(new_att)

    return new_att

def save_model_with_source(model, ckpt, path, source_dir, step):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)

    path_ckpt_dir = os.path.join(path, time.strftime(f"%Y-%m-%d-%H-%M-%S-Epoch{step}", time.localtime(time.time()))).replace("\\", "/")
    os.mkdir(path_ckpt_dir)

    path_to_source = os.path.join(path_ckpt_dir, pathlib.Path(source_dir).name).replace("\\", "/")
    path_to_ckpt = os.path.join(path_ckpt_dir, "ckpt").replace("\\", "/")

    shutil.copytree(source_dir, path_to_source)
    
    ckpt.save(path_to_ckpt)
    print("Model was saved.")

def load_model(ckpt, path):
    model = ckpt.restore(path)
    return model
