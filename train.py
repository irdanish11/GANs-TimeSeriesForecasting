# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:31:12 2020

@author: Danish
"""


import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
from ModelWrapper import GAN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import load_model

#Reading the data set
path = r'C:\Users\danis\Documents\USFoods\Data'
df = pd.read_csv(path+'/Sales_Traffic_Covid_Mobility_Mapping.csv')
columns = list(df.columns)
columns.pop(0)
dataset = df.iloc[:, 1:15].values
sc = MinMaxScaler(feature_range = (0, 1))
scaled_data = sc.fit_transform(dataset)

X_train = scaled_data[0:int(len(scaled_data)*.95)]
X_test = scaled_data[int(len(scaled_data)*.95):-1]

# # generate random data just to check the working of model
# X, y = make_blobs(n_samples=50000, centers=3, n_features=7)
# X = np.array(X, dtype=np.float32)

gan = GAN(n_features=len(X_train[0]), optimizer=Adam(), ckpt_path='checkpoints', tb_path='./Tensorboard', 
          gen_summary=True, disc_summary=True)

history = gan.train_GAN(X_train, epochs=50, batch_size=32, batch_shape=(32, 1, 14), name='GAN',
              gan_summary=True)


################ Get Predictions ##################
def get_prediction(model, X_test, sc, columns):
    if len(np.shape(X_test)) == 2:
        X_test = np.expand_dims(X_test, axis=0)
    elif len(np.shape(X_test)) == 1:
        X_test = np.expand_dims(np.expand_dims(X_test, axis=0), axis=0)
    pred = model.predict(X_test)
    return pd.DataFrame(data=sc.inverse_transform(pred), columns=columns)

generator = load_model('./checkpoints/Generator.h5')

tmp = X_test[0]

pred = get_prediction(generator, X_test[0], sc, columns)

gan_model = load_model('./checkpoints/GAN_Model.h5')
gan_model.summary()
