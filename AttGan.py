import tensorflow as tf
import numpy as np

from Genc import Genc
from Gdec import Gdec
from D import D
from env import *


class AttGan():
    
    def __init__(self, eta, num_att):
        self.eta = eta
        self.num_att = num_att

        self.loss_g = None
        self.loss_d = None
        self.Gparams = []
        self.Dparams = []

        self.graph = None
        self.sess = None
        self.saver = None

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def build(self):

        print("Building Attgan...")

        self.graph = tf.Graph()

        with self.graph.as_default():
            
            with tf.name_scope("attgan"):
                self.X_src, self.X_att_a, self.X_att_b = self._init_placeholder()
                self.params = self._init_parameters()

                optimizer_g = tf.train.AdamOptimizer(learning_rate=self.eta)
                optimizer_d = tf.train.AdamOptimizer(learning_rate=self.eta)

                print("Building generator...")

                enc = Genc()
                Z = enc.build(self.X_src, self.params)

                Z_a = tf.concat([Z, tf.tile(tf.reshape(self.X_att_a, (tf.shape(self.X_att_a)[0], 1, 1, tf.shape(self.X_att_a)[1])), [1, 6, 6, 1])], axis=3)
                Z_b = tf.concat([Z, tf.tile(tf.reshape(self.X_att_b, (tf.shape(self.X_att_b)[0], 1, 1, tf.shape(self.X_att_b)[1])), [1, 6, 6, 1])], axis=3)
                
                dec_a = Gdec()
                rec_a = dec_a.build(Z_a, self.params)

                dec_b = Gdec()
                rec_b = dec_b.build(Z_b, self.params)
                
                self.reconstructed = rec_b

                print("Generator was built.")

                print("Building discriminator...")

                d_a = D()
                _d_a, att_a = d_a.build(rec_a, self.params)
                
                d_b = D()
                _d_b, att_b = d_b.build(rec_b, self.params)

                print("Discriminator was built.")

                rec_loss = tf.reduce_mean((rec_a - self.X_src)**2) + tf.reduce_mean((rec_b - self.X_src)**2)
                gen_d_loss = tf.reduce_mean((_d_a - 1)**2) + tf.reduce_mean((_d_b - 1)**2)
                att_loss_a = tf.reduce_mean((att_a - self.X_att_a)**2)
                att_loss_b = tf.reduce_mean((att_b - self.X_att_b)**2)
                #gp = self._gradient_penalty(self.X_src, rec_a)
                
                self.loss_g = 20.0*rec_loss + 10.0*att_loss_b + 5.0*gen_d_loss
                self.loss_d = 0.5*tf.reduce_mean((_d_a + 1)**2) + 0.5*tf.reduce_mean((_d_b + 1)**2) + 1.0*att_loss_a# + 15.0*gp

                self.op_train_g = optimizer_g.minimize(self.loss_g, var_list=self.Gparams)
                grad_d = optimizer_d.compute_gradients(self.loss_d, var_list=self.Dparams)
                grad_d_ = []
                for grad, var in grad_d:
                    if len(grad.get_shape().as_list()) == 4:
                        grad_ = tf.clip_by_norm(grad, 1e-14, axes=[1, 2, 3])
                    else:
                        grad_ = tf.clip_by_norm(grad, 1e-14, axes=[1])
                    grad_d_.append((grad_, var))
                    
                self.op_train_d = optimizer_d.apply_gradients(grad_d_)

                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())

                self.saver = tf.train.Saver()

        print("Attgan was built.")

    def step(self, X_src, X_att_a, X_att_b):
        with self.graph.as_default():
            feed_dict = {
                    self.X_src: X_src,
                    self.X_att_a: X_att_a,
                    self.X_att_b: X_att_b
            }
            _, loss_g = self.sess.run([self.op_train_g, self.loss_g], feed_dict=feed_dict)
            _, loss_d = self.sess.run([self.op_train_d, self.loss_d], feed_dict=feed_dict)
#             print(self.sess.run(self.reconstructed, feed_dict=feed_dict)[0,0])

        return loss_g, loss_d

    def convert(self, X_src, X_att):
        with self.graph.as_default():
            feed_dict = {
                    self.X_src: X_src,
                    self.X_att_b: X_att
            }
            reconstructed = self.sess.run(self.reconstructed, feed_dict=feed_dict)

        return reconstructed
    
    def save(self, path):
        with self.graph.as_default():
            self.saver.save(self.sess, path)
            print("Attgan was saved.")
            
    def load(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)
            print("Attgan was restored.")
            
    def _gradient_penalty(self, x_a, x_b):
        alpha = tf.random_uniform(shape=tf.shape(x_a), minval=0., maxval=1.)
        inter = x_a + alpha * (x_b - x_a)
        d = D()
        pred, _ = d.build(inter, self.params)
        grad = tf.gradients([pred], inter)[0]
        norm = tf.norm(tf.reshape(grad, shape=(tf.shape(x_a)[0], 96*96*3)), axis=1)
        gp = tf.reduce_mean((norm - 1)**2)
        return gp

    def _kl_divergence(self, _from, _to):
        with tf.name_scope("kl"):
            kl = tf.reduce_mean(_from * tf.log((_from + 1e-8)/(_to + 1e-8)))

        return kl

    def _init_parameters(self):
        params = dict()

        with tf.name_scope("params"):
            params["W1_enc"] = tf.Variable(tf.random_normal((5, 5, 3, 16)), dtype=tf.float32)
            params["b1_enc"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W2_enc"] = tf.Variable(tf.random_normal((5, 5, 16, 16)), dtype=tf.float32)
            params["b2_enc"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)
            
            params["W3_enc"] = tf.Variable(tf.random_normal((5, 5, 16, 32)), dtype=tf.float32)
            params["b3_enc"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)
            
            params["W4_enc"] = tf.Variable(tf.random_normal((5, 5, 32, 32)), dtype=tf.float32)
            params["b4_enc"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W1_dec"] = tf.Variable(tf.random_normal((5, 5, 32, 32+self.num_att)), dtype=tf.float32)
            params["b1_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W2_dec"] = tf.Variable(tf.random_normal((5, 5, 16, 32)), dtype=tf.float32)
            params["b2_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W3_dec"] = tf.Variable(tf.random_normal((5, 5, 16, 16)), dtype=tf.float32)
            params["b3_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W4_dec"] = tf.Variable(tf.random_normal((5, 5, 3, 16)), dtype=tf.float32)
            params["b4_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 3)), dtype=tf.float32)
            
            params["W1_d"] = tf.Variable(tf.random_normal((5, 5, 3, 16)), dtype=tf.float32)
            params["b1_d"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W2_d"] = tf.Variable(tf.random_normal((5, 5, 16, 16)), dtype=tf.float32)
            params["b2_d"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W3_d"] = tf.Variable(tf.random_normal((5, 5, 16, 32)), dtype=tf.float32)
            params["b3_d"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W4_d"] = tf.Variable(tf.random_normal((5, 5, 32, 32)), dtype=tf.float32)
            params["b4_d"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W5_fc_d"] = tf.Variable(tf.random_normal((6*6*32, 128)), dtype=tf.float32)
            params["b5_fc_d"] = tf.Variable(tf.random_normal((1, 128)), dtype=tf.float32)

            params["W6_fc_d"] = tf.Variable(tf.random_normal((128, 1)), dtype=tf.float32)
            params["b6_fc_d"] = tf.Variable(tf.random_normal((1, 1)), dtype=tf.float32)

            params["W5_fc_att"] = tf.Variable(tf.random_normal((6*6*32, 128)), dtype=tf.float32)
            params["b5_fc_att"] = tf.Variable(tf.random_normal((1, 128)), dtype=tf.float32)

            params["W6_fc_att"] = tf.Variable(tf.random_normal((128, self.num_att)), dtype=tf.float32)
            params["b6_fc_att"] = tf.Variable(tf.random_normal((1, self.num_att)), dtype=tf.float32)
            
        for k in params.keys():
            if "enc" in k or "dec" in k:
                self.Gparams.append(params[k])
            else:
                self.Dparams.append(params[k])

        return params
    
    def _init_placeholder(self):
        with tf.name_scope("in"):
            X_src = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 3), name="X_src")
            X_att_a = tf.placeholder(tf.float32, shape=(None, self.num_att), name="X_att_a")
            X_att_b = tf.placeholder(tf.float32, shape=(None, self.num_att), name="X_att_b")

        return X_src, X_att_a, X_att_b

