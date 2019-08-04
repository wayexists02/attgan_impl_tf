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
        
        self.X_src = []
        self.X_att_a = []
        self.X_att_b = []
        
        self.loss_g = None
        self.loss_d = None
        
        self.reconstructed = None

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def build(self):

        print("Building Attgan...")

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.params = self._init_parameters()
            
            gp = tf.Variable(GP, dtype=tf.float32)
            self.Dparams.append(gp)

            optimizer_g = tf.train.AdamOptimizer(learning_rate=self.eta)
            optimizer_d = tf.train.AdamOptimizer(learning_rate=self.eta)
            
            reconstructed_list = []
            g_grad_var_pairs = []
            d_grad_var_pairs = []
            
            loss_g_list = []
            loss_d_list = []
            
            for i in range(len(GPU_INDEX)):
                with tf.device(f"/gpu:{i}"):
                    with tf.name_scope(f"attgan_{i}"):
                        X_src, X_att_a, X_att_b = self._init_placeholder()

                        self.X_src.append(X_src)
                        self.X_att_a.append(X_att_a)
                        self.X_att_b.append(X_att_b)

                        print(f"Building generator {i}...")

                        enc = Genc()
                        Z = enc.build(X_src, self.params)

                        Z_a = tf.concat([Z, tf.tile(tf.reshape(X_att_a, (tf.shape(X_att_a)[0], 1, 1, tf.shape(X_att_a)[1])), [1, 6, 6, 1])], axis=3)
                        Z_b = tf.concat([Z, tf.tile(tf.reshape(X_att_b, (tf.shape(X_att_b)[0], 1, 1, tf.shape(X_att_b)[1])), [1, 6, 6, 1])], axis=3)

                        dec_a = Gdec()
                        rec_a = dec_a.build(Z_a, self.params, enc.layers)

                        dec_b = Gdec()
                        rec_b = dec_b.build(Z_b, self.params, enc.layers)

                        reconstructed_list.append(rec_b)

                        print(f"Generator {i} was built.")

                        print(f"Building discriminator {i}...")

                        d_a = D()
                        _d_a, att_a = d_a.build(rec_a, self.params)

                        d_b = D()
                        _d_b, att_b = d_b.build(rec_b, self.params)

                        print(f"Discriminator {i} was built.")

                        rec_loss = tf.reduce_mean((rec_a - X_src)**2) + tf.reduce_mean((rec_b - X_src)**2)
                        gen_d_loss = tf.reduce_mean((_d_a - 1)**2) + tf.reduce_mean((_d_b - 1)**2)
                        dis_d_loss = tf.reduce_mean((_d_a + 1)**2) + tf.reduce_mean((_d_b + 1)**2)
                        att_loss_a = tf.reduce_mean((att_a - X_att_a)**2)
                        att_loss_b = tf.reduce_mean((att_b - X_att_b)**2)

                        loss_g = 25.0*rec_loss + 10.0*att_loss_b + 1.0*gen_d_loss
                        loss_d = 0.5*dis_d_loss + 1.0*att_loss_a + 15.0*(gp**2)

                        loss_g_list.append(loss_g)
                        loss_d_list.append(loss_d)

                        g_grad_var_pair = optimizer_g.compute_gradients(loss_g, var_list=self.Gparams)
                        d_grad_var_pair = self._compute_discriminator_gradients(optimizer_d, loss_d, gp)

                        g_grad_var_pairs.append(g_grad_var_pair)
                        d_grad_var_pairs.append(d_grad_var_pair)
                    
            self.reconstructed = tf.concat(reconstructed_list, axis=0)
                    
            g_avg_grad_var_pair = self._average_gradients(g_grad_var_pairs)
            d_avg_grad_var_pair = self._average_gradients(d_grad_var_pairs)
            
            self.op_train_g = optimizer_g.apply_gradients(g_avg_grad_var_pair)
            self.op_train_d = optimizer_d.apply_gradients(d_avg_grad_var_pair)
            
            self.loss_g = tf.reduce_mean(loss_g_list)
            self.loss_d = tf.reduce_mean(loss_d_list)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

        print("Attgan was built.")

    def step(self, X_src, X_att_a, X_att_b):
        
        X_src_dict = self._divide_placeholder(self.X_src, X_src)
        X_att_a_dict = self._divide_placeholder(self.X_att_a, X_att_a)
        X_att_b_dict = self._divide_placeholder(self.X_att_b, X_att_b)
        
        with self.graph.as_default():
            feed_dict = dict()
            
            for k in X_src_dict:
                feed_dict[k] = X_src_dict[k]
            for k in X_att_a_dict:
                feed_dict[k] = X_att_a_dict[k]
            for k in X_att_b_dict:
                feed_dict[k] = X_att_b_dict[k]
                
            _, loss_g = self.sess.run([self.op_train_g, self.loss_g], feed_dict=feed_dict)
            _, loss_d = self.sess.run([self.op_train_d, self.loss_d], feed_dict=feed_dict)

        return loss_g, loss_d

    def convert(self, X_src, X_att):
        
        X_src_dict = self._divide_placeholder(self.X_src, X_src)
        X_att_dict = self._divide_placeholder(self.X_att_b, X_att)
        
        with self.graph.as_default():
            feed_dict = dict()
            
            for k in X_src_dict:
                feed_dict[k] = X_src_dict[k]
            for k in X_att_dict:
                feed_dict[k] = X_att_dict[k]
                
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
            
    def _divide_placeholder(self, tensor, placeholder):
        n = placeholder.shape[0]
        size = n // len(GPU_INDEX)
        
        placeholder_dict = dict()
        
        for i in range(len(GPU_INDEX)):
            start = i*size
            end = (i+1)*size
            
            if i == len(GPU_INDEX) - 1:
                end = n
                
            placeholder_dict[tensor[i]] = placeholder[start:end]
            
        return placeholder_dict
            
    def _compute_discriminator_gradients(self, optimizer, loss, gp):
        grad_var_pair = optimizer.compute_gradients(loss, var_list=self.Dparams)
        new_grad_var_pair = []
        
        for grad, var in grad_var_pair:
            if var is not gp:
                _grad = tf.where(tf.norm(grad) < 1e-22, tf.zeros_like(grad), tf.clip_by_norm(grad, gp))
            else:
                _grad = grad
                
            new_grad_var_pair.append((_grad, var))
            
        return new_grad_var_pair
    
    def _average_gradients(self, grad_var_pairs):
        avg_grad_var_pair = []
        
        for grad_var_list in zip(*grad_var_pairs):
            avg_grad = 0.0
            var = None
            
            cnt = 0
            
            for _grad, _var in grad_var_list:
                var = _var
                if _grad is None:
                    continue
                avg_grad += _grad
                cnt += 1
                
            if cnt > 0:
                avg_grad = avg_grad / cnt
                
            avg_grad_var_pair.append((avg_grad, var))
            
        return avg_grad_var_pair

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

