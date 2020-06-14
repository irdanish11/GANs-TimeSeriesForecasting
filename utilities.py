# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:44:03 2020

@author: Danish
"""


import time

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