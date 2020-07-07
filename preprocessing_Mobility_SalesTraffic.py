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

#Getting Sales & Traffic Data
df_sales_tr = pd.read_csv(path+'/Data/Sales_Traffic_Mapping.csv')
zip_cd_sl = np.sort(df_sales_tr.zip_cd.unique())


path2 = path+'/Mobility/WeeklyData/weekly_mobility_data'
csv_files = os.listdir(path2)

final_df = []
for f in csv_files:
    print('Reading the CSV file!')
    mobility_df = pd.read_csv(path2+'/'+f)
    mobility_df = mobility_df[['postal_code', 'date_range_start', 'date_range_end', 
                                   'raw_visit_counts', 'raw_visitor_counts']]
    print('Processing the CSV file!')
    df_mb = mobility_df.head()
    zip_cd_mb = np.sort(mobility_df.postal_code.unique())
    zip_cmn = np.intersect1d(zip_cd_mb, zip_cd_sl)
    
    obj = mobility_df.date_range_start.copy()
    print('Performing Date to Week Conversion')
    fisc_yr_wk = to_weeks(obj, format_='%Y%m%d', splitter='-', convert=True, 
                          year_index=0, keep=10)
    mobility_df['fisc_wk'] = fisc_yr_wk
    mobility_df['fisc_wk2'] = fisc_yr_wk
    mobility_df['zip_cd'] = mobility_df['postal_code']
    
    mobility_df = mobility_df.set_index('postal_code').sort_index()
    
    dfs = []
    for z in zip_cmn:
        dfs.append(mobility_df.loc[z])
    df_tmp = pd.concat(dfs)
    final_df.append(df_tmp)

final_df = pd.concat(final_df)
final_df = final_df.reset_index(drop=True)

############# Sales Data ##############
df_sales_tr['fisc_wk2'] = df_sales_tr['fisc_wk']
df_sales_tr = df_sales_tr.set_index('fisc_wk2').sort_index()
dfs = []
for w in [17, 18]:
    dfs.append(df_sales_tr.loc[w])
    
    
    
    
    
    
    