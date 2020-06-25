# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:42:21 2020

@author: Danish
"""


import pandas as pd
from utilities import to_weeks, extract_sub_df
import os
import numpy as np
import csv

path = r'C:\Users\danis\Documents\USFoods'

mobility_df = pd.read_csv(path+'/Mobility_eth.csv', quoting=csv.QUOTE_NONE, error_bad_lines=False)

head_mo