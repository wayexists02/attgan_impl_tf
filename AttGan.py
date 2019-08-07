import tensorflow as tf
import numpy as np

from Genc import Genc
from Gdec import Gdec
from D import D
from env import *


class AttGan():
    """
    Model of Attribute-GAN
    
    I implemented generator using AutoEncoder instead of VAE.
    """
    
    def __init__(self, eta, num_att):
        """
        Constructor of AttGan
        
        Arguments:
        ----------
        :eta learning rate
        :num_att number of attributes.
        """
        
        self.eta = eta            # learning rate
        self.num_att = num_att    # the number of attributes

        # instance variables regarding model.
        self.loss_g = None        # loss of generator
        self.loss_d = None        # loss of discriminator
        self.Gparams = []         # trainable parameters of generator
        self.Dparams = []         # trainable parameters of discriminator

        self.graph = None         # graph for Att-GAN
        self.sess = None          # tensorflow session to run model
        self.saver = None         # tensorflow saver object to save model
        
        self.X_src = []           # list of placeholder of images. length of this array should be same as number of available gpus
        self.X_att_a = []         # list of placeholder of original attributes. length of this array should be same as number of available gpus
        self.X_att_b = []         # list of placeholder of modified attributes. length of this array should be same as number of available gpus
        
        self.reconstructed = None # tensor for generating image with modified attributes

    def __del__(self):
        """
        Destructor
        If tensorflow session is alive, close it.
        """
        
        if self.sess is not None:
            self.sess.close()

    def build(self):
        """
        Build Attribute-GAN architecture.
        """

        print("Building Attgan...")

        # create tensorflow graph to store architecture of attgan
        self.graph = tf.Graph()

        # open graph
        with self.graph.as_default():
            
            # create parameters.
            self.params = self._init_parameters()
            
            # tensorflow variable for gradients penalty of discriminator.
            gp = tf.Variable(GP, dtype=tf.float32)
            self.Dparams.append(gp)

            # optimizer for generator and discriminator.
            optimizer_g = tf.train.AdamOptimizer(learning_rate=self.eta)
            optimizer_d = tf.train.AdamOptimizer(learning_rate=self.eta)
            
            # to use multi-gpu, we should store results of each gpus into an array.
            # Later, we should concatenate this results into a single tensor.
            reconstructed_list = []
            g_grad_var_pairs = []
            d_grad_var_pairs = []
            
            loss_g_list = []
            loss_d_list = []
            
            # construct graphs for each gpu
            for i in range(len(GPU_INDEX)):
                
                # using i-th gpu
                with tf.device(f"/gpu:{i}"):
                    
                    # name scope for plot graph prettier in tensorboard
                    with tf.name_scope(f"attgan_{i}"):
                        
                        # initialize placeholder
                        X_src, X_att_a, X_att_b = self._init_placeholder()

                        # put placeholders into list for map the placeholders into real data
                        self.X_src.append(X_src)
                        self.X_att_a.append(X_att_a)
                        self.X_att_b.append(X_att_b)

                        print(f"Building generator {i}...")

                        # construct encoder of AE (generator)
                        enc = Genc()
                        Z = enc.build(X_src, self.params)

                        # add attributes to latent vector
                        Z_a = tf.concat([Z, tf.tile(tf.reshape(X_att_a, (tf.shape(X_att_a)[0], 1, 1, tf.shape(X_att_a)[1])), [1, 12, 12, 1])], axis=3)
                        Z_b = tf.concat([Z, tf.tile(tf.reshape(X_att_b, (tf.shape(X_att_b)[0], 1, 1, tf.shape(X_att_b)[1])), [1, 12, 12, 1])], axis=3)

                        # construct decoder of AE for reconstructing image with original attributes
                        dec_a = Gdec()
                        rec_a = dec_a.build(Z_a, self.params, enc.layers)

                        # construct decoder of AE for reconstructing image with modified attributes
                        dec_b = Gdec()
                        rec_b = dec_b.build(Z_b, self.params, enc.layers)

                        # append reconstructed image into list. Later, I concatenate them into single tensor
                        reconstructed_list.append(rec_b)

                        print(f"Generator {i} was built.")

                        print(f"Building discriminator {i}...")

                        # construct discriminator for classifying whether the reconstructed image is real or fake, and classifying attributes.
                        d_a = D()
                        _d_a, att_a = d_a.build(rec_a, self.params)

                        d_b = D()
                        _d_b, att_b = d_b.build(rec_b, self.params)

                        print(f"Discriminator {i} was built.")

                        # compute loss tensor
                        rec_loss = tf.reduce_mean((rec_a - X_src)**2) + tf.reduce_mean((rec_b - X_src)**2)   # reconstruction loss for auto encoder
                        gen_d_loss = tf.reduce_mean((_d_a - 1)**2) + tf.reduce_mean((_d_b - 1)**2)           # adversarial loss for generator
                        dis_d_loss = tf.reduce_mean((_d_a + 1)**2) + tf.reduce_mean((_d_b + 1)**2)           # adversarial loss for discriminator
                        att_loss_a = tf.reduce_mean((att_a - X_att_a)**2)                                    # attributes cross entropy for predicting original attributes
                        att_loss_b = tf.reduce_mean((att_b - X_att_b)**2)                                    # attributes cross entropy for predicting modified attributes

                        loss_g = 15.0*rec_loss + 10.0*att_loss_b + 1.0*gen_d_loss                            # construct generator loss
                        loss_d = 1.0*dis_d_loss + 10.0*att_loss_a + 15.0*(gp**2)                             # construct discriminator loss

                        loss_g_list.append(loss_g)                                                           # add losses to list to collect them into single gpu
                        loss_d_list.append(loss_d)                                                           # add losses to list to collect them into single gpu

                        g_grad_var_pair = optimizer_g.compute_gradients(loss_g, var_list=self.Gparams)       # compute gradients to average gradients computed in every gpus
                        d_grad_var_pair = self._compute_discriminator_gradients(optimizer_d, loss_d, gp)     # compute gradients to average gradients computed in every gpus

                        g_grad_var_pairs.append(g_grad_var_pair)                                             # list for collecting gradients
                        d_grad_var_pairs.append(d_grad_var_pair)                                             # list for collecting gradients
                    
            # I will use this tensor to compute image with modified attributes.
            self.reconstructed = tf.concat(reconstructed_list, axis=0)
            self.rec_tb = tf.summary.image("test image", self.reconstructed)
                    
            # average gradients computed in each gpus to apply gradients.
            g_avg_grad_var_pair = self._average_gradients(g_grad_var_pairs)
            d_avg_grad_var_pair = self._average_gradients(d_grad_var_pairs)
            
            # apply gradients operation
            self.op_train_g = optimizer_g.apply_gradients(g_avg_grad_var_pair)
            self.op_train_d = optimizer_d.apply_gradients(d_avg_grad_var_pair)
            
            # final loss for generator and discriminator
            self.loss_g = tf.reduce_mean(loss_g_list)
            self.loss_d = tf.reduce_mean(loss_d_list)

            # open session and, initialize variables.
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            
            tf.summary.FileWriter("./logdir/", self.graph)
            self.rec_rb = tf.summary.merge_all()

            # create saver object.
            self.saver = tf.train.Saver()

        print("Attgan was built.")

    def step(self, X_src, X_att_a, X_att_b):
        """
        Gradient descent using one batch.
        
        Arguments:
        ----------
        :X_src images batch, shaped of (batch_size, HEIGHT, WIDTH, 3)
        :X_att_a original attributes, shaped of (batch_size, num_attributes)
        :X_att_b modified attributes, shaped of (batch_size, num_attributes)
        
        Returns:
        --------
        :loss_g loss of generator
        :loss_d loss of discriminator
        """
        
        # divide data into n-slice where n is the number of gpu
        X_src_dict = self._divide_placeholder(self.X_src, X_src)
        X_att_a_dict = self._divide_placeholder(self.X_att_a, X_att_a)
        X_att_b_dict = self._divide_placeholder(self.X_att_b, X_att_b)
        
        # open AttGAN graph
        with self.graph.as_default():
            feed_dict = dict()
            
            # map data slice into placeholder
            for k in X_src_dict:
                feed_dict[k] = X_src_dict[k]
            for k in X_att_a_dict:
                feed_dict[k] = X_att_a_dict[k]
            for k in X_att_b_dict:
                feed_dict[k] = X_att_b_dict[k]
                
            # gradient descent operation. compute losses
            _, loss_g = self.sess.run([self.op_train_g, self.loss_g], feed_dict=feed_dict)
            _, loss_d = self.sess.run([self.op_train_d, self.loss_d], feed_dict=feed_dict)

        return loss_g, loss_d

    def convert(self, X_src, X_att):
        """
        convert image into similar image with given attributes
        
        Arguments:
        ----------
        :X_src images batch.
        :X_att attributes we want to apply to image.
        """
        
        # divide data into n-slice where n is the number of gpus
        X_src_dict = self._divide_placeholder(self.X_src, X_src)
        X_att_dict = self._divide_placeholder(self.X_att_b, X_att)
        
        # open AttGAN graph.
        with self.graph.as_default():
            feed_dict = dict()
            
            # map data slices into placeholder
            for k in X_src_dict:
                feed_dict[k] = X_src_dict[k]
            for k in X_att_dict:
                feed_dict[k] = X_att_dict[k]
                
            # compute reconstructed images with given attributes
            reconstructed, _ = self.sess.run([self.reconstructed, self.rec_tb], feed_dict=feed_dict)

        return reconstructed
    
    def save(self, path):
        """
        Save model into path
        
        Arguments:
        ----------
        :path path to save model
        """
        
        with self.graph.as_default():
            self.saver.save(self.sess, path)
            print("Attgan was saved.")
            
    def load(self, path):
        """
        Load model stored in path.
        
        Arguments:
        ----------
        :path path model was saved in.
        """
        
        with self.graph.as_default():
            self.saver.restore(self.sess, path)
            print("Attgan was restored.")
            
    def _divide_placeholder(self, tensor, data):
        """
        divide data into n-slice where n is the number of gpus.
        
        Arguments:
        ----------
        :tensor tensor we want to data map into.
        :data data we want to divide.
        
        Returns:
        --------
        :data_dict divided data
        """
        
        # retrieve useful information about data.
        n = data.shape[0]
        size = n // len(GPU_INDEX)
        
        data_dict = dict()
        
        # divide data
        for i in range(len(GPU_INDEX)):
            start = i*size
            end = (i+1)*size
            
            if i == len(GPU_INDEX) - 1:
                end = n
                
            data_dict[tensor[i]] = data[start:end]
            
        return data_dict
            
    def _compute_discriminator_gradients(self, optimizer, loss, gp):
        """
        Compute discriminator gradients with gradients penalty.
        To suppress learning speed of discriminator, we penalize gradients of discriminator.
        
        Arguments:
        ----------
        :loss discriminator loss
        :gp tensorflow variable for clipping gradients of discriminator
        """
        
        # compute gradients with respect to loss
        grad_var_pair = optimizer.compute_gradients(loss, var_list=self.Dparams)
        new_grad_var_pair = []
        
        # clipping gradients of discriminator by norm
        for grad, var in grad_var_pair:
            if var is not gp:
                _grad = tf.where(tf.norm(grad) < 1e-22, tf.zeros_like(grad), tf.clip_by_norm(grad, gp))
            else:
                _grad = grad
                
            new_grad_var_pair.append((_grad, var))
            
        return new_grad_var_pair
    
    def _average_gradients(self, grad_var_pairs):
        """
        Compute average gradients computed by each gpus.
        
        Arguments:
        ----------
        :grad_var_pairs list of pairs of gradient and variable [(grad1, var1), (grad2, var2), ...]
        """
        
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
            
            params["W5_enc"] = tf.Variable(tf.random_normal((5, 5, 32, 64)), dtype=tf.float32)
            params["b5_enc"] = tf.Variable(tf.random_normal((1, 1, 1, 64)), dtype=tf.float32)

            params["W1_dec"] = tf.Variable(tf.random_normal((5, 5, 32, 64+self.num_att)), dtype=tf.float32)
            params["b1_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W2_dec"] = tf.Variable(tf.random_normal((5, 5, 32, 32)), dtype=tf.float32)
            params["b2_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W3_dec"] = tf.Variable(tf.random_normal((5, 5, 16, 32)), dtype=tf.float32)
            params["b3_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W4_dec"] = tf.Variable(tf.random_normal((5, 5, 16, 16)), dtype=tf.float32)
            params["b4_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W5_dec"] = tf.Variable(tf.random_normal((5, 5, 3, 16)), dtype=tf.float32)
            params["b5_dec"] = tf.Variable(tf.random_normal((1, 1, 1, 3)), dtype=tf.float32)
            
            params["W1_d"] = tf.Variable(tf.random_normal((5, 5, 3, 16)), dtype=tf.float32)
            params["b1_d"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W2_d"] = tf.Variable(tf.random_normal((5, 5, 16, 16)), dtype=tf.float32)
            params["b2_d"] = tf.Variable(tf.random_normal((1, 1, 1, 16)), dtype=tf.float32)

            params["W3_d"] = tf.Variable(tf.random_normal((5, 5, 16, 32)), dtype=tf.float32)
            params["b3_d"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W4_d"] = tf.Variable(tf.random_normal((5, 5, 32, 32)), dtype=tf.float32)
            params["b4_d"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32)

            params["W5_d"] = tf.Variable(tf.random_normal((5, 5, 32, 64)), dtype=tf.float32)
            params["b5_d"] = tf.Variable(tf.random_normal((1, 1, 1, 64)), dtype=tf.float32)

            params["W6_fc_d"] = tf.Variable(tf.random_normal((12*12*64, 256)), dtype=tf.float32)
            params["b6_fc_d"] = tf.Variable(tf.random_normal((1, 256)), dtype=tf.float32)

            params["W7_fc_d"] = tf.Variable(tf.random_normal((256, 64)), dtype=tf.float32)
            params["b7_fc_d"] = tf.Variable(tf.random_normal((1, 64)), dtype=tf.float32)

            params["W8_fc_d"] = tf.Variable(tf.random_normal((64, 1)), dtype=tf.float32)
            params["b8_fc_d"] = tf.Variable(tf.random_normal((1, 1)), dtype=tf.float32)

            params["W6_fc_att"] = tf.Variable(tf.random_normal((12*12*64, 256)), dtype=tf.float32)
            params["b6_fc_att"] = tf.Variable(tf.random_normal((1, 256)), dtype=tf.float32)

            params["W7_fc_att"] = tf.Variable(tf.random_normal((256, 64)), dtype=tf.float32)
            params["b7_fc_att"] = tf.Variable(tf.random_normal((1, 64)), dtype=tf.float32)

            params["W8_fc_att"] = tf.Variable(tf.random_normal((64, self.num_att)), dtype=tf.float32)
            params["b8_fc_att"] = tf.Variable(tf.random_normal((1, self.num_att)), dtype=tf.float32)
            
        # assign variables into each parameters list
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

