# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:52:32 2020

@author: Danish
"""


import tensorflow as tf

class BatchGenerator:
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size
        self.counter = 0
        
    def get_nextBatch(self, batch_shape):
        batch = self.X[self.counter:self.counter + self.batch_size]
        #Getting the batch of t+1 time
        counter2 = self.counter + self.batch_size
        batch_t_plus1 = self.X[counter2:counter2 + self.batch_size]
        batch_reshaped = batch.reshape(batch_shape)
        
        #incrementing the counter
        self.counter += self.batch_size
        return batch, batch_reshaped, batch_t_plus1
    
    def get_disc_data(generator, X, X_reshaped, x_t1):
        #prediction of generator x_t1_hat given x_t
        x_t1_hat = generator.predict(X_reshaped)
        #we concatenate X={x1....xt} & x_t1_hat to get {x1....xt, x_t1_hat} as fake data X_fake
        X_fake = tf.concat([X, x_t1_hat], axis=0, name='fake_data_Concat')
        #we concatenate X={x1....xt} & x_t1 to get {x1....xt, x_t1} as real data X_real.
        X_real =  tf.concat([X, x_t1], axis=0, name='real_data_concat')
        #concatenating real and fake data to prepare the input for discrminator.
        X_disc = tf.concat([X_real,X_fake], axis=0, name='Disc_X_Concat')
        #Generating targets for discriminator, 1s for real and 0s for fake.
        Y_disc = tf.concat([tf.ones_like(X_real), tf.zeros_like(X_fake)], axis=0, name='Disc_Y_Concat')
        return X_disc, Y_disc
    