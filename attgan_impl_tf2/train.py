import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

from models import attgan
from tensorflow.keras import losses, optimizers, metrics
import numpy as np
import tensorflow as tf
import dataloader
import utils

from settings import *


train_dloader = dataloader.DataLoader("train", BATCH_SIZE)
valid_dloader = dataloader.DataLoader("valid", BATCH_SIZE)
test_dloader = dataloader.DataLoader("test", BATCH_SIZE)

model = attgan.AttGAN()

# criterion_reconstruction = losses.MeanAbsoluteError()
# criterion_adversarial = losses.BinaryCrossentropy()
# criterion_attribute = losses.BinaryCrossentropy()

def criterion_MAE(y_true, y_pred):
    n = y_true.shape[0]

    loss = tf.math.abs(y_true - y_pred)
    loss = tf.reshape(loss, shape=(n, -1))
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss

def criterion_BCE(y_true, y_pred):
    n = y_true.shape[0]

    loss = - y_true * tf.math.log(y_pred + 1e-8) - (1 - y_true) * tf.math.log(1 - y_pred + 1e-8)
    loss = tf.reshape(loss, (n, -1))
    loss = tf.reduce_sum(loss, axis=1)
    loss = tf.reduce_mean(loss)

    return loss

optimizer = optimizers.Adam(learning_rate=ETA)

loss_mean_gen = metrics.Mean()
loss_mean_dis = metrics.Mean()

log_file = sys.argv[1]
print("Logging into " + log_file)

def generator_loss(x, a, b, x_rec_a, x_rec_b, d_x, d_rec_a, d_rec_b, c_x, c_rec_a, c_rec_b, training=False):
    n = x.shape[0]

    loss_rec = criterion_MAE(x, x_rec_a)
    loss_adv_gen = tf.reduce_mean(-d_rec_b, axis=0)
    loss_att_gen = criterion_BCE(a, c_rec_a) + criterion_BCE(b, c_rec_b)
    
    loss_gen = loss_rec*100 + loss_adv_gen + loss_att_gen

    print("GENERATOR")
    print("LOSS_GEN:", loss_gen)
    # print("REC:", loss_rec)
    # print("ADV:", loss_adv_gen)
    # print("ATT:", loss_att_gen)
    # print()
    
    return loss_gen

def discriminator_loss(x, a, b, x_rec_a, x_rec_b, d_x, d_rec_a, d_rec_b, c_x, c_rec_a, c_rec_b, training=False):
    loss_adv_dis = tf.reduce_mean(-d_x + d_rec_b, axis=0)
    loss_att_dis = criterion_BCE(a, c_x)

    if training is True:
        gp = utils.wgan_gp(x, x_rec_b, model.disc)
        loss_dis = loss_adv_dis + loss_att_dis + 10*gp
    else:
        loss_dis = loss_adv_dis + loss_att_dis

    print("DISCRIMINATOR")
    print("LOSS_DIS:", loss_dis)
    # print("ADV:", loss_adv_dis)
    # print("ATT:", loss_att_dis)
    # print("GP:", gp)
    # print()
    
    return loss_dis

def inference(x, a, training=False):
    n = x.shape[0]
    num_att = a.shape[1]
    b = utils.generate_attribute(n, num_att)
        
    x_rec_a, c_rec_a, d_rec_a = model(x, a, training=training)
    x_rec_b, c_rec_b, d_rec_b = model(x, b, training=training)
    c_x, d_x = model.disc(x, training=training)

    # print(c_x)
    
    loss_gen = generator_loss(x, a, b, x_rec_a, x_rec_b, d_x, d_rec_a, d_rec_b, c_x, c_rec_a, c_rec_b, training=training)
    loss_dis = discriminator_loss(x, a, b, x_rec_a, x_rec_b, d_x, d_rec_a, d_rec_b, c_x, c_rec_a, c_rec_b, training=training)

    return loss_gen, loss_dis

def step(x, a):
    with tf.GradientTape(persistent=True) as tape:
        loss_gen, loss_dis = inference(x, a, training=True)

    grad_gen = tape.gradient(loss_gen, [*model.encoder.trainable_variables, *model.decoder.trainable_variables])
    optimizer.apply_gradients(zip(grad_gen, [*model.encoder.trainable_variables, *model.decoder.trainable_variables]))

    grad_dis = tape.gradient(loss_dis, model.disc.trainable_variables)
    optimizer.apply_gradients(zip(grad_dis, model.disc.trainable_variables))

    return loss_gen, loss_dis

def get_mean_loss(loss_mean_obj):
    loss = loss_mean_obj.result()
    loss_mean_obj.reset_states()
    return loss

less_valid_loss = 10000000.0

for e in range(EPOCHS):
    
    for x_batch, att_batch in train_dloader.next_batch():
        loss_gen, loss_dis = step(x_batch, att_batch)

        loss_mean_gen(loss_gen)
        loss_mean_dis(loss_dis)
        
    train_loss_gen = get_mean_loss(loss_mean_gen)
    train_loss_dis = get_mean_loss(loss_mean_dis)
    
    for x_batch, att_batch in valid_dloader.next_batch():
        loss_gen, loss_dis = inference(x_batch, att_batch)

        loss_mean_gen(loss_gen)
        loss_mean_dis(loss_dis)
        
    valid_loss_gen = get_mean_loss(loss_mean_gen)
    valid_loss_dis = get_mean_loss(loss_mean_dis)
    
    print(f"Epochs {e+1}/{EPOCHS}")
    print(f"Train generator loss: {train_loss_gen:.8f}")
    print(f"Train discriminator loss: {train_loss_dis:.8f}")
    print(f"Valid generator loss: {valid_loss_gen:.8f}")
    print(f"Valid discriminator loss: {valid_loss_dis:.8f}")

    with open(log_file, "a") as logfile:
        logfile.write(f"Epochs {e+1}/{EPOCHS}\n")
        logfile.write(f"Train generator loss: {train_loss_gen:.8f}\n")
        logfile.write(f"Train discriminator loss: {train_loss_dis:.8f}\n")
        logfile.write(f"Valid generator loss: {valid_loss_gen:.8f}\n")
        logfile.write(f"Valid discriminator loss: {valid_loss_dis:.8f}\n")

    if less_valid_loss > valid_loss_gen:
        less_valid_loss = valid_loss_gen
        utils.save_model_with_source(model, "ckpts/attgan", "models")
