# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:44:03 2020

@author: Danish
"""


import time
import sys
import  tensorflow as tf

class Timer:
    def __init__(self):
        self.begin = 0
    def restart(self):
        self.begin = time.time()
    def start(self):
        self.begin = time.time()
    def get_time_hhmmss(self, rem_batches):
        end = time.time()
        time_taken = end - self.begin
        reamin_time = time_taken*rem_batches
        #print('reamin time: '+str(reamin_time)+' Reamin Batches: '+str(rem_batches)+' Time Taken: '+str(time_taken))
        m, s = divmod(reamin_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str, time_taken
    
def PrintInline(string):
    sys.stdout.write('\r'+string)
    sys.stdout.flush() 
    
def TF_GPUsetup(GB=4):
    """
    Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU. Often Needed
    When GPU run out of memory. It would be one of the solution for the issue: Failed to 
    get convolution algorithm. This is probably because cuDNN failed to initialize,

    Parameters
    ----------
    GB : int, optional
        The amount of GPU memory you want to use. It is recommended to use  1 GB
        less than your total GPU memory. The default is 4.

    Returns
    -------
    None.

    """
    if type(GB)!=int:
        raise TypeError('Type of Parameter `GB` must be `int` and it should be 1 GB less than your GPU memory')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*GB))]
    if gpus:
      # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    print('\nTensorflow GPU installed: '+str(tf.test.is_built_with_cuda()))
    print('Is Tensorflow using GPU: '+str(tf.test.is_gpu_available()))