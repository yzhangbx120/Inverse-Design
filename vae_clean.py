from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf

from layers import Dense

import matplotlib.pyplot as plt


def print_loss(loss):
    print(loss)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class VAE():
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }

    def __init__(self, architecture=[], d_hyperparams={}):
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()
        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
        assert len(self.architecture) > 2, \
            "Architecture must have more layers! (input, 1+ hidden, latent)"

        self._buildTrainGraph()
        self.sesh.run(tf.global_variables_initializer())
 
    
    def _buildTrainGraph(self):
    
        self.variables_to_restore = []
        
        weight_idx = 0
    
        self.x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.architecture[0]], name="x")
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        # encoding / "recognition": q(z|x)
        current_input = self.x_in
        for layer_i, n_output in enumerate(self.architecture[1:-1]):
            n_input = current_input.get_shape()[1].value
            stddev = tf.cast((2 / n_input)**0.5, tf.float32)
            W = tf.Variable(tf.random_normal([n_input, n_output], stddev = stddev), name='weights'+str(weight_idx))
            weight_idx = weight_idx+1
            b = tf.Variable(tf.zeros([n_output]))
            W = tf.nn.dropout(W,dropout)
            output = self.nonlinearity(tf.matmul(current_input,W)+b)
            current_input = output
        
        h_encoded = current_input 
        
        n_input = current_input.get_shape()[1].value
        stddev = tf.cast((2 / n_input)**0.5, tf.float32)
        W = tf.Variable(tf.random_normal([n_input, self.architecture[-1]], stddev = stddev), name='weights'+str(weight_idx))
        weight_idx = weight_idx+1
        b = tf.Variable(tf.zeros([self.architecture[-1]]))
        W = tf.nn.dropout(W,dropout)
        z_mean = tf.identity(tf.matmul(h_encoded,W)+b)
        W = tf.Variable(tf.random_normal([n_input, self.architecture[-1]], stddev = stddev), name='weights'+str(weight_idx))
        weight_idx = weight_idx+1
        b = tf.Variable(tf.zeros([self.architecture[-1]]))
        W = tf.nn.dropout(W,dropout)
        z_log_sigma = tf.identity(tf.matmul(h_encoded,W)+b)

        # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
        self.z = self.sampleGaussian(z_mean, z_log_sigma)
        
        current_input = self.z
        decoding_weights = []
        decoding_bias = []
        for layer_i, n_output in enumerate(reversed(self.architecture[1:-1])):
            n_input = current_input.get_shape()[1].value
            stddev = tf.cast((2 / n_input)**0.5, tf.float32)
            W = tf.Variable(tf.random_normal([n_input, n_output], stddev = stddev), name='weights'+str(weight_idx))
            weight_idx = weight_idx+1
            b = tf.Variable(tf.zeros([n_output]))
            self.variables_to_restore.append(W)
            self.variables_to_restore.append(b)
            W = tf.nn.dropout(W,dropout)
            output = self.nonlinearity(tf.matmul(current_input,W)+b)
            decoding_weights.append(W)
            decoding_bias.append(b)
            current_input = output
        
        n_input = current_input.get_shape()[1].value
        stddev = tf.cast((2 / n_input)**0.5, tf.float32)
        W = tf.Variable(tf.random_normal([n_input, self.architecture[0]], stddev = stddev), name='weights'+str(weight_idx))
        weight_idx = weight_idx+1
        b = tf.Variable(tf.zeros([self.architecture[0]]))
        self.variables_to_restore.append(W)
        self.variables_to_restore.append(b)
        W = tf.nn.dropout(W,dropout)
        self.x_reconstructed = tf.identity(self.squashing(tf.matmul(current_input,W)+b))
        decoding_weights.append(W)
        decoding_bias.append(b)
 
        self.opt_z = tf.Variable(tf.random_normal([1,self.architecture[-1]], stddev = 1.0))
        
        self.outer_z = tf.placeholder(tf.float32,shape=[1,self.architecture[-1]])
        self.assigen_op = self.opt_z.assign(self.outer_z)
        current_input = self.opt_z 
        for i in range(len(decoding_weights)):
            if i < len(decoding_weights)-1:
                output = self.nonlinearity(tf.matmul(current_input,decoding_weights[i])+decoding_bias[i])
            else:
                output = tf.identity(self.squashing(tf.matmul(current_input,decoding_weights[i])+decoding_bias[i]))
            current_input = output 
        self.opt_reconstructed = current_input
        self.opt_mask = tf.placeholder(tf.float32, [256,64,1])
        self.opt_inv_mask = 1.0-self.opt_mask
        self.opt_image = tf.placeholder(tf.float32, [256,64,1])
        
        opt_contextual_loss = tf.reduce_sum(tf.abs(tf.multiply(self.opt_mask,tf.reshape(self.opt_reconstructed,[256,64,1]))-tf.multiply(self.opt_mask,self.opt_image)))
        opt_regularization_loss = tf.square(tf.reduce_sum(tf.multiply(self.opt_inv_mask,tf.reshape(self.opt_reconstructed,[256,64,1])))-tf.reduce_sum(tf.multiply(self.opt_mask,self.opt_image)))
        opt_smoothing_loss = tf.reduce_sum(tf.image.total_variation(tf.reshape(self.opt_reconstructed,[1,256,64,1])))
        self.opt_loss = opt_contextual_loss+0.1*opt_regularization_loss+0.2*opt_smoothing_loss
          
        self.opt_train = tf.contrib.opt.ScipyOptimizerInterface(self.opt_loss,var_list=[self.opt_z],options={'maxiter': 1000})


    def sampleGaussian(self, mu, log_sigma):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        # reparameterization trick
        epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
        return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

    @staticmethod
    def crossEntropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        # bound by clipping to avoid nan
        obs_ = tf.clip_by_value(obs, offset, 1 - offset)
        return -tf.reduce_sum(actual * tf.log(obs_) +
                              (1 - actual) * tf.log(1 - obs_), 1)

    @staticmethod
    def l1_loss(obs, actual):
        """L1 loss (a.k.a. LAD), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        return tf.reduce_sum(tf.abs(obs - actual) , 1)

    @staticmethod
    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
        return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 - tf.exp(2 * log_sigma), 1)

    def accuracy(self,test_init_data, pred_test_data):
        diff = np.absolute(test_init_data-pred_test_data)
        acc = 1.0-np.mean(diff)
        return acc 
    
    def predict_second(self, n_examples):  
        if True:
            test_data_ = np.loadtxt('./acc_test_data_original_form.dat')
            test_data = 255-test_data_
            test_data = (test_data>128)+0
        print('load finished...')
        
        if True:
            saver = tf.train.Saver(self.variables_to_restore)
            
        if True:
            saver.restore(self.sesh,'./model')
            
        if True:
            # test_data = test_data[0:10,:]
            num_sam = 3
            all_pred_test_data = np.zeros_like(test_data)
            for i in range(len(test_data)):
            
                ori_image = test_data[i,:].reshape([256,64,1]).astype(np.float32)
                mask = np.ones([256,64,1])
                mask[:,0:32,:] = 0.0
                
                isContinue = True
                iter = 0
                current_best_z = None 
                current_best_loss = 1.e20
                while isContinue and iter < 10:
                    iter += 1
                    rand_z = np.random.normal(0.,1.,[1,self.architecture[-1]])
                    self.sesh.run(self.assigen_op, feed_dict = {self.outer_z:rand_z})
                    
                    self.opt_train.minimize(self.sesh,feed_dict={self.opt_mask: mask, self.opt_image: ori_image})
                    
                    pred_images = self.sesh.run(self.opt_reconstructed)
                    
                    loss = self.sesh.run(self.opt_loss,feed_dict={self.opt_mask: mask, self.opt_image: ori_image})
                    if (loss < 200.):
                        isContinue = False
                    if (loss < current_best_loss):
                        current_best_loss = loss
                        current_best_z = rand_z
                
                if isContinue:
                    rand_z = current_best_z
                    self.sesh.run(self.assigen_op, feed_dict = {self.outer_z:rand_z})
                    pred_images = self.sesh.run(self.opt_reconstructed)
                    
                loss = self.sesh.run(self.opt_loss,feed_dict={self.opt_mask: mask, self.opt_image: ori_image})
                print(i, loss)
                all_pred_test_data[i,:] = pred_images.reshape(all_pred_test_data[i,:].shape)
                
            all_pred_test_data = 255-all_pred_test_data*255
            all_pred_test_data = all_pred_test_data.astype(np.int32)
            np.savetxt('./data/pred_by_vae_100_original_form.dat',all_pred_test_data,fmt='%i',delimiter=' ')
            
            
         