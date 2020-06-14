# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:31:12 2020

@author: Danish
"""


import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
from ModelWrapper import GAN

# generate random data just to check the working of model
X, y = make_blobs(n_samples=50000, centers=3, n_features=7)
X = np.array(X, dtype=np.float32)

gan = GAN(n_features=7, optimizer=Adam(), ckpt_path='checkpoints', tb_path='./Tensorboard', 
          gen_summary=True, disc_summary=True)

history = gan.train_GAN(X, epochs=10, batch_size=64, batch_shape=(64, 1, 7), name='GAN',
              gan_summary=True)