# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:20:44 2020

@author: Danish
"""


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    def __init__(self, n_features, input_shape, batch_size, optimizer, dropout=0.2, alpha=0.2, gen_loss=None, **kwargs):
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
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.2.
        gen_loss : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : 
            disc_label_smoothing: 
                Label smoothing for discriminator min_max_loss. The amount of smoothing for positive labels. 
                This technique is taken from `Improved Techniques for Training GANs` 
                (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
            loss_collection: 
                Collection to which this loss will be added. Loss collection for generator and discriminator.
                tf.compat.v1.GraphKeys.LOSSES
            reduction:
                A `tf.losses.Reduction` to apply to loss. e.g: Reduction.SUM_BY_NONZERO_WEIGHTS
            
        Returns
        -------
        None.

        """
        self.n_features = n_features
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.dropout = dropout
        self.aplha = alpha
        self.gen_loss = gen_loss
        self.gen_summary = kwargs.get('gen_summary', False)
        self.disc_summary = kwargs.get('disc_summary', False)
        self.gan_summary = kwargs.get('gan_summary', False)
        self.compile_gen = kwargs.get('compile_gen', False)
        self.disc_label_smoothing = kwargs.get('disc_label_smoothing', 0.25)
        self.loss_collection = kwargs.get('loss_collection', tf.compat.v1.GraphKeys.LOSSES)
        self.reduction = kwargs.get('reduction', tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    
    def minimax_discriminator_loss(self, Y_true, Y_pred, real_weights=1.0, gen_weights=1.0, summaries=False):
        """
        Original minimax discriminator loss for GANs, with label smoothing. Note that this loos is not 
        recommended to use. A more practically seful loss is `modified_discriminator_loss`.
        
        L = - real_weights * log(sigmoid(D(x))) - generated_weights * log(1 - sigmoid(D(G(z))))
        
        See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more details.
        
        Parameters
        ----------
        Y_true : list
            A list containing 2 elements. First element Discriminator output on real data, second element
            Discriminator output on generated data. e.g [Y_hat_real, Y_hat_fake]
        Y_hat_fake : list
            A list containing 2 elements. First element Targets for real data, second element targets for
            generated/fake data. e.g [Y_real, Y_fake]
            
        real_weights : TYPE, optional
            Optional `Tensor` whose rank is either 0, or the same rank as `real_data`, and must be 
            broadcastable to `real_data` (i.e., all dimensions must be either `1`, or the same as 
            the corresponding dimension). The default is 1.0.
        gen_weights : TYPE, optional
            Same as `real_weights`, but for `generated_data`. The default is 1.0.
        summaries : bool, optional
            Whether or not to add summaries for the loss.. The default is False.

        Returns
        -------
        loss : tensor
            A loss Tensor. The shape depends on `reduction`..

        """
        Y_hat_real = Y_pred[0]
        Y_hat_fake = Y_pred[1]
        label_smoothing = self.disc_label_smoothing
        loss_collection = self.loss_collection
        reduction = self.reduction
        with tf.name_scope('Discriminator_MiniMax_Loss') as scope:
      
          # -log((1 - label_smoothing) - sigmoid(D(x)))
          loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_hat_real), Y_hat_real, real_weights, 
                                                                   label_smoothing, scope, reduction=reduction)
          # -log(- sigmoid(D(G(x))))
          loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
              tf.zeros_like(Y_hat_fake),
              Y_hat_fake,
              gen_weights,
              scope=scope,
              loss_collection=None,
              reduction=reduction)
      
          loss = loss_on_real + loss_on_generated
          tf.compat.v1.losses.add_loss(loss, loss_collection)
      
          if summaries:
            tf.compat.v1.summary.scalar('discriminator_gen_minimax_loss',
                                        loss_on_generated)
            tf.compat.v1.summary.scalar('discriminator_real_minimax_loss',
                                        loss_on_real)
            tf.compat.v1.summary.scalar('discriminator_minimax_loss', loss)
      
        return loss
    
    def discriminator_loss(self, y_true, y_pred):
        """
        

        Parameters
        ----------
        y_true : TYPE
            Targets or labels, half of length conatining ones and half of length containing zeros.
        y_pred : TYPE
            Predictions by Discriminator.

        Returns
        -------
        None.

        """
        with tf.compat.v1.variable_scope('Discriminator_Loss'):
            """ 
                tf.split() split the given tensors into two equal parts, pythonic slicing can't be used
                here because the shape of the loss at compile time is dynamic e.g (?, probablity), so the
                first axis is not determined and by using pythonic slicing it will generate.
                ValueError: slice index 4 of dimension 1 out of bounds. for 'loss_2/dense_34_loss/
                Discriminator_Loss/strided_slice_1' (op: 'StridedSlice') with input shapes: [?,1]
                
                Under the hood the thing that is happening can be visualized as:
                    
                #Discriminator output on real data
                Y_hat_real = y_pred[0:2]
                #Discriminator output on fake data
                Y_hat_fake = y_pred[2, 4]
            """
            #tf.split split the tensors and returns a list of two elements as 
            #[Targets for real data, Targets for fake data]
            Y_true =  tf.split(y_true, num_or_size_splits=2, axis=0)
            #tf.split split the tensors and returns a list of two elements as 
            #[Discriminator output on real data, Discriminator output on fake data]
            Y_pred = tf.split(y_pred, num_or_size_splits=2, axis=0)
            loss = self.minimax_discriminator_loss(Y_true, Y_pred, real_weights=1.0, 
                                                   gen_weights=1.0, summaries=False)
            return  loss
    
    def get_generator(self):
        with tf.name_scope('Generator'):
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
    
    def get_discriminator(self, disc_loss):
        """
        Creates the discriminator Model as mentioned in https://doi.org/10.1016/j.procs.2019.01.256

        Parameters
        ----------
        disc_loss : TYPE
            DESCRIPTION.

        Returns
        -------
        discriminator : TYPE
            DESCRIPTION.

        """
        with tf.name_scope('Discriminator'):
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
            fc = Dense(units=1, activation='sigmoid')(h3)
        
        discriminator = Model(disc_input, fc)
        if self.disc_summary:
            discriminator.summary()
        discriminator.compile(loss=disc_loss, optimizer=self.optimizer)
        return discriminator
    
    def test(self):
        self.get_discriminator(self.discriminator_loss)
    
    

