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

mobility_df = pd.read_csv(path+'/Mobility/WeeklyData/weekly_20Apr/weekly_20Apr.csv')
mobility_df = mobility_df[['postal_code', 'date_range_start', 'date_range_end', 
                                   'raw_visit_counts', 'raw_visitor_counts']]
df_mb = mobility_df.head()
df_sales_tr = pd.read_csv(path+'/Data/Sales_Traffic_Mapping.csv')

zip_cd_mb = np.sort(mobility_df.postal_code.unique())
zip_cd_sl = np.sort(df_sales_tr.zip_cd.unique())

zip_cmn = np.intersect1d(zip_cd_mb, zip_cd_sl)

#, quoting=csv.QUOTE_NONE, error_bad_lines=False

sum(df_mb['visits_by_day'][0])

df_mb['date_range_start'][0][0:10]

