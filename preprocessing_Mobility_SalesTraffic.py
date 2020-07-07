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
from tqdm import tqdm

path = r'C:\Users\danis\Documents\USFoods'

#Getting Sales & Traffic Data
df_sales_tr = pd.read_csv(path+'/Data/Sales_Traffic_Covid_Mapping.csv')
zip_cd_sl = np.sort(df_sales_tr.zip_cd.unique())


path2 = path+'/Mobility/WeeklyData/weekly_mobility_data'
csv_files = os.listdir(path2)

final_df = []
for f in csv_files:
    print('Reading the CSV file!')
    mobility_df = pd.read_csv(path2+'/'+f)
    mobility_df = mobility_df[['postal_code', 'date_range_start', 'raw_visit_counts',
                               'raw_visitor_counts', 'median_dwell']]
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
cmn_weeks = np. intersect1d(final_df.fisc_wk.unique(), df_sales_tr.fisc_wk.unique())
final_df = final_df.set_index('fisc_wk2').sort_index()

################# Mapping the Data #################
dfs_1 = []
for w in tqdm(cmn_weeks):
    df_st = extract_sub_df(df_sales_tr, w)
    df_st['zip_cd2'] = df_st['zip_cd']
    df_mob = extract_sub_df(final_df, w)
    df_mob['zip_cd2'] = df_mob['zip_cd']
    #setting index
    df_st = df_st.set_index('zip_cd2').sort_index()
    df_mob = df_mob.set_index('zip_cd2').sort_index() 
    cmn_zips = np.intersect1d(df_st.zip_cd.unique(), df_mob.zip_cd.unique())
    dfs_2 = []
    for z in cmn_zips:
        df1_s = extract_sub_df(df_st, z)
        df2_m = extract_sub_df(df_mob, z)
        if len(df1_s) < len(df2_m):
                size = len(df1_s)
        else:
            size = len(df2_m)
        df1_s = df1_s.reset_index(drop=True)
        df2_m = df2_m.reset_index(drop=True)
        df1_s = df1_s.loc[0:size-1]
        df2_m = df2_m.loc[0:size-1]
        #Adding matched entries
        df1_s['raw_visit_counts'] = df2_m['raw_visit_counts']
        df1_s['raw_visitor_counts'] = df2_m['raw_visitor_counts']
        df1_s['mob_median_dwell'] = df2_m['median_dwell']
        dfs_2.append(df1_s)
    dfs_1.append(pd.concat(dfs_2))

mapped_df = pd.concat(dfs_1)    
mapped_df = mapped_df.reset_index(drop=True).sort_index()    

mapped_df.to_csv(path+'/Data/Sales_Traffic_Covid_Mobility_Mapping.csv', index=False)    

