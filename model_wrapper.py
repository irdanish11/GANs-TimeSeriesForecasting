# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:20:44 2020

@author: Danish
"""


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from DataGenerator import BatchGenerator

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
            label_smoothing: 
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
        self.optimizer = optimizer
        self.dropout = dropout
        self.aplha = alpha
        self.gen_loss = gen_loss
        self.history_epoch = {'Disc_Loss':[], 'Disc_Accuracy':[], 'Gen_Loss':[], 'Gen_Acc':[], 'Batch_Data':[]}
        self.history_batch = {'Disc_Loss':[], 'Disc_Accuracy':[], 'Gen_Loss':[], 'Gen_Acc':[], 'Batch_Data':[]}
        self.gen_summary = kwargs.get('gen_summary', False)
        self.disc_summary = kwargs.get('disc_summary', False)
        self.gan_summary = kwargs.get('gan_summary', True)
        self.gan_summary = kwargs.get('gan_summary', False)
        self.compile_gen = kwargs.get('compile_gen', False)
        self.label_smoothing = kwargs.get('label_smoothing', 0.25)
        self.loss_collection = kwargs.get('loss_collection', tf.compat.v1.GraphKeys.LOSSES)
        self.reduction = kwargs.get('reduction', tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    
    def minimax_discriminator_loss(self, Y_true, Y_pred, real_weights=1.0, gen_weights=1.0, summaries=False):
        """
        Original minimax discriminator loss for GANs, with label smoothing. Note that this loos is not 
        recommended to use. A more practically seful loss is `modified_discriminator_loss`.
        
        L = - real_weights * log(sigmoid(D(x))) - generated_weights * log(1 - sigmoid(D(G(z))))
        
        See `Generative Adversarial Nets <https://arxiv.org/abs/1406.2661>`_ for more details.
        
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
        Y_real = Y_true[0]
        Y_fake = Y_true[1]
        label_smoothing = 0.25#self.label_smoothing
        loss_collection = self.loss_collection
        reduction = self.reduction
        with tf.name_scope('Discriminator_MinMax_Loss') as scope:
            # -log((1 - label_smoothing) - sigmoid(D(x)))
            loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(Y_real, Y_hat_real, real_weights, label_smoothing, 
                                                                     scope, loss_collection=None, reduction=reduction)
            # -log(- sigmoid(D(G(x))))
            loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(Y_fake, Y_hat_fake, gen_weights, scope=scope,
                                                                          loss_collection=None, reduction=reduction)
            D_loss = loss_on_real + loss_on_generated
            tf.compat.v1.losses.add_loss(D_loss, loss_collection)

            if summaries:
              tf.compat.v1.summary.scalar('discriminator_gen_minimax_loss', loss_on_generated)
              tf.compat.v1.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
              tf.compat.v1.summary.scalar('discriminator_minimax_loss', D_loss)

        return D_loss
    
    def discriminator_loss(self, y_true, y_pred):
        """
        Creates a Keras type custom loss function for Discriminator MinMax Loss, to work with model.compile.
        
        Parameters
        ----------
        y_true : tensor
            Targets or labels, half of length conatining ones and half of length containing zeros.
        y_pred : tensor
            Predictions by Discriminator.

        Returns
        -------
        A tensor having the same dimension as the ouptut of model.

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

            D_loss = self.minimax_discriminator_loss(Y_true, Y_pred, real_weights=1.0, gen_weights=1.0, 
                                                   summaries=True)
            return  D_loss
    
    def minmax_generator_loss(self, x_t1, x_t1_hat, Y_hat_gan, lambda1=1.0, lambda2=1.0, summaries=False):
        """
        The generator loss G_loss which along with D_loss used to optimize the value function. 
        Particularly, we combine the Mean Square Error (MSE) with the generator loss of a classical 
        GAN to constitute the G_loss of our architecture.
            G_loss = λ1*g_mse + λ2*g_loss.
        The loss function G_loss is composed by g_mse and g_loss with λ1 and λ2, respectively. λ1 
        and λ2 are hyper-parameters. For more see: 
        `Stock Market Prediction Based on Generative Adversarial Network <https://doi.org/10.1016/j.procs.2019.01.256>`_

        Parameters
        ----------
        x_t1 : TYPE
            Real Data at time step t+1.
        x_t1_hat : TYPE
            Fake data generated by generator at time t+1.
        Y_hat_gan : TYPE
            Ouput of discrminator which is extracted using gan(combined model), not the discriminator only.
        lambda1 : TYPE, optional
            Values for hyper paramets lambda 1. For more check equation 10 in the paper. The default is 1.0.
        lambda2 : TYPE, optional
            Same as lambda1. The default is 1.0.

        Returns
        -------
        None.

        """
        loss_collection = self.loss_collection
        reduction = self.reduction
        with tf.name_scope('Generator_MinMax_Loss') as scope:
            #usually in MSE (labels-prediction)^2 but in eq-8 of paper (predictions-labels), 
            #that is why we change the parameter passing order below
            g_mse = tf.compat.v1.losses.mean_squared_error(labels=x_t1_hat, predictions=x_t1, scope=scope,
                                                           loss_collection=None, reduction=reduction)
            g_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_hat_gan), Y_hat_gan, scope,
                                                               loss_collection=None, reduction=reduction)
            #GAN loss
            G_loss = (lambda1*g_mse) + (lambda2*g_loss)
            tf.compat.v1.losses.add_loss(G_loss, loss_collection)
      
            if summaries:
              tf.compat.v1.summary.scalar('g_mse_loss', g_mse)
              tf.compat.v1.summary.scalar('gan_g_loss', g_loss)
              tf.compat.v1.summary.scalar('gan_G_loss', G_loss)
    
        return G_loss
            
            
    
    def get_generator(self, name):
        """
        

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        generator : TYPE
            DESCRIPTION.

        """
        with tf.name_scope('Generator'):
            gen_input = Input(input_shape)
            #LSTM layer
            lstm = LSTM(units=50, activation='relu', return_sequences=False, input_shape=input_shape)(gen_input)
            lstm = Dropout(self.dropout)(lstm)
            #Fully connected layer
            fc = Dense(units=self.n_features)(lstm)
            fc = LeakyReLU(alpha=self.aplha)(fc)
            
        generator = Model(inputs=gen_input, outputs=fc, name=name)
        if self.gen_summary:
            generator.summary()
        if self.compile_gen:
            if self.gen_loss is None:
                raise TypeError('gen_loss can not be None, when kwargs["compile_gen"] is True. Provide valid keras loss function string')
            generator.compile(loss=self.gen_loss, optimizer=self.optimizer)
        return generator 
    
    def get_discriminator(self, disc_loss, name):
        """
        Creates the discriminator Model as mentioned in `Stock Market Prediction Based on Generative Adversarial Network <https://doi.org/10.1016/j.procs.2019.01.256>`_

        Returns
        -------
        discriminator : object to the Model class.
            Keras engine training Model object.

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
        
        discriminator = Model(disc_input, fc, name=name)
        if self.disc_summary:
            discriminator.summary()
        discriminator.compile(loss=disc_loss, optimizer=self.optimizer, metrics=['accuracy'])
        return discriminator
    
    def get_gan_model(self, name):
        generator = self.get_generator(name='Generator')
        discriminator = self.get_discriminator(self.discriminator_loss, name='Discriminator')
        #Make the discriminator untrainable when we are training the generator.  
        #This doesn't effect the discriminator by itself
        discriminator.trainable = False
        
        #Combine the two models to create the GAN
        gan_input = Input(shape=input_shape)
        fake_data = generator(gan_input)
        gan_output = discriminator(fake_data)
        
        gan = Model(gan_input, gan_output, name=name)
        if self.gan_summary:
            gan.summary()  
        #gan.compile(loss=gan_loss, optimizer=self.optimizer)
        return generator, discriminator, gan
    
    def train_generator(self, generator, gan, X_reshaped, x_t1, X_fake):
        with tf.GradientTape() as tape:
            #prediction of generator x_t1_hat given x_t
            x_t1_hat = generator(X_reshaped)
            #For info regarding model.predict(X) and model(X) see: https://stackoverflow.com/questions/60159714/when-to-use-model-predictx-vs-modelx-in-tensorflow 
            #Prediction of GAN as whole model, where discrminator is non trainable.
            Y_hat_gan = gan(X_fake) 
            #Generating targets for GAN, here we are giving ones because we want our generator to 
            #generate data that looks like real.
            Y_gan = tf.ones_like(Y_hat_gan)
            #Computing G_loss
            G_loss = self.minmax_generator_loss(x_t1, x_t1_hat, Y_hat_fake, lambda1=1.0, lambda2=1.0, summaries=True)
            
            
        
    
    def train_GAN(self, X_train, epochs, batch_size, batch_shape, name, gan_summary=False):
        generator, discriminator, gan_model = self.get_gan_model(name)
        gan_model.summary()
        steps_per_epoch = len(X_train)//batch_size
        for epoch in range(1, epochs+1):
            bg = BatchGenerator(X_train, batch_size=32)
            for batch in range(1, steps_per_epoch):
                # X_reshaped is used data to get the output from generator model. X is the data in
                # orignal dimensions e.g [batches,features], while X_reshaped is the data for LSTM 
                # layer as LSTM layer 3D data so X_reshaped has dimensions [batches, timesteps, features]
                #whereas x_t1 is the data at time t+1 or next batch
                X, X_reshaped, x_t1 = bg.get_nextBatch(batch_shape)
                #Getting the data for discrimnator training.
                X_disc, Y_disc, X_fake = bg.get_disc_gan_data(generator, X, X_reshaped, x_t1)
                #train discriminator
                metrics = discriminator.train_on_batch(X_disc, Y_disc)
                self.history_batch['Disc_Loss'].append(metrics[0])
                self.history_batch['Disc_Accuracy'].append(metrics[1])
                    

                
    
    def test(self):
        self.get_discriminator(self.discriminator_loss)
    
    

