# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:20:44 2020

@author: Danish
"""


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model

input_shape = (1,7)

class GAN:
    """

    Parameters
    ----------
    n_features : TYPE
        DESCRIPTION.
    input_shape : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    dropout : TYPE, optional
        DESCRIPTION. The default is 0.2.

    Returns
    -------
    None.

    """
    def __init__(self, n_features, input_shape, optimizer, dropout=0.2, alpha=0.2, gen_loss=None, **kwargs):

        self.n_features = n_features
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.dropout = dropout
        self.aplha = alpha
        self.gen_loss = gen_loss
        self.gen_summary = kwargs.get('gen_summary', False)
        self.disc_summary = kwargs.get('disc_summary', False)
        self.gan_summary = kwargs.get('gan_summary', False)
        self.compile_gen = kwargs.get('compile_gen', True)
        
    def get_generator(self):
        gen_input = Input(input_shape)
        #LSTM layer
        lstm = LSTM(units=50, activation='relu', return_sequences=False, input_shape=input_shape)(gen_input)
        lstm = Dropout(self.dropout)(lstm)
        #Fully connected layer
        fc = Dense(units=self.n_features)(lstm)
        fc = LeakyReLU(alpha=self.aplha)(fc)
        
        generator = Model(inputs=gen_input, outputs=fc)
        if self.gen_summary:
            generator.summary()
        if self.compile_gen:
            if self.gen_loss is None:
                raise TypeError('gen_loss can not be None, when kwargs["compile_gen"] is True. Provide valid keras loss function string')
            generator.compile(loss=self.gen_loss, optimizer=self.optimizer)
        return generator 
    
    def get_discriminator(self):
        #input layer
        disc_input = Input((self.n_features))
        #1st hidden layer
        h1 = Dense(units=72)(disc_input)
        h1 = LeakyReLU(alpha=self.aplha)(h1)
        #2nd hidden layer
        h2 = Dense(units=100)(h1)
        h2 = LeakyReLU(alpha=self.aplha)(h2)
        #3rd hidden layer
        h3 = Dense(units=10)(h2)
        h3 = LeakyReLU(alpha=self.aplha)(h3)
        #output layer
        fc = Dense(units=1, activation='sigmoid')
        
        discriminator = Model(disc_input, fc)
        
        
        
        
        

