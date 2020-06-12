# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:52:32 2020

@author: Danish
"""


class BatchGenerator:
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size
        self.counter = 0
        
    def get_nextBatch(self, batch_shape):
        batch = self.X[self.counter:self.counter + self.batch_size]
        batch_reshaped = batch.reshape(batch_shape)
        #incrementing the counter
        self.counter += self.batch_size
        return batch, batch_reshaped
    