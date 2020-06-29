# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:59:59 2020

@author: Danish
"""

import pandas as pd
import numpy as np
from utilities import extract_sub_df
import pickle
from tqdm import tqdm

path = r'C:\Users\danis\Documents\USFoods'
df_covid = pd.read_csv(path+'/Data/Processed_COVID.csv')
df_sales_tr = pd.read_csv(path+'/Data/Sales_Traffic_Mapping.csv')
with open(path+'/Data/fips2zips', 'rb') as f:
    fips_2_zips = pickle.load(f)

#checking for how many weeks we have the data
weeks_covid = np.sort(df_covid.fisc_wk.unique())
weeks_sales_tr = df_sales_tr.fisc_wk.unique()

weeks_cmn = np.intersect1d(weeks_covid, weeks_sales_tr)
#we will find the non common values beteen the coinciding weeks and the rest of the
#weeks of sales and traffic mapped data. And for these we don't have covid data
weeks_not_cmn = np.setxor1d(weeks_cmn, weeks_sales_tr)

#setting fisc_wk as index
df_sales_tr['fisc_wk2'] = df_sales_tr['fisc_wk']
df_sales_tr = df_sales_tr.set_index('fisc_wk2').sort_index()

#extracting the weeks that does not have covid data, which are non-common weeks
#constructing df for non covid weeks
df_sales_Ncovid = []
for w in weeks_not_cmn:
    df_sales_Ncovid.append(extract_sub_df(df_sales_tr, w))
df_sales_Ncovid = pd.concat(df_sales_Ncovid) 
#creating the covid data for non-covid weeks that is why just adding 0's   
df_sales_Ncovid['cases'] = np.arange(0, len(df_sales_Ncovid))*0
df_sales_Ncovid['mortality'] = np.arange(0, len(df_sales_Ncovid))*0
df_sales_Ncovid = df_sales_Ncovid.reset_index()
df_sales_Ncovid = df_sales_Ncovid.drop(['fisc_wk2'], axis=1)


################ Mapping covid features to SalesTraffic Data ################
df_sales_tr['zip_cd2'] = df_sales_tr['zip_cd']
df_covid = df_covid.set_index('fisc_wk2').sort_index()
lst = []
lst_chng = []
df_weeks = []
dfs = []
for w in tqdm(weeks_cmn):
    df_st = extract_sub_df(df_sales_tr, w)
    df_cov = extract_sub_df(df_covid, w)
    #getting zip codes from both dataframes
    zips_st = np.sort(df_st.zip_cd.unique())
    
    #Mapping the zip codes accordingly
    changed = []
    df_cov = df_cov.reset_index()
    for row in range(len(df_cov)):
        fip = df_cov['fips'][row]
        for zip_ in fips_2_zips[fip]:
            if zip_ in zips_st:
                df_cov['zip_cd'][row] = zip_
                df_cov['zip_cd2'][row] = zip_
                changed.append(row)
    lst_chng.append(changed) 
    df_cov.set_index('fisc_wk2')   
    
    #getting zip codes from both dataframes
    zips_cov = np.sort(df_cov.zip_cd.unique())
    #getting common zip code
    """ Here is breakpoint we have very few number of common zip codes among 
        covid and sales traffic data"""
    zips_cmn = np.intersect1d(zips_cov, zips_st)
    #list of common zip codes
    lst.append(zips_cmn)
    df_st = df_st.set_index('zip_cd2').sort_index()
    df_cov = df_cov.set_index('zip_cd2').sort_index()
    df_zip = []
    #extrcating data for common zip codes
    for z in zips_cmn:
        df1_s = extract_sub_df(df_st, z)
        df2_c = extract_sub_df(df_cov, z)
        if len(df1_s) < len(df2_c):
                size = len(df1_s)
        else:
            size = len(df2_c)
        df1_s = df1_s.reset_index(drop=True)
        df2_c = df2_c.reset_index(drop=True)
        df1_s = df1_s.loc[0:size]
        df2_c = df2_c.loc[0:size]
        df1_s['cases'] = df2_c['cases']
        df1_s['mortality'] = df2_c['mortality']
        df1_s = df1_s.dropna()
        df_zip.append(df1_s)
    try:
        df_weeks.append(pd.concat(df_zip))
    except Exception as e:
        print('No Coinciding data found at zip level')
        
df_mapped = pd.concat(df_weeks)
df_mapped = df_mapped.reset_index(drop=True)


################# Concatenating final df #################
df_final = pd.concat([df_sales_Ncovid, df_mapped])
df_final = df_final.reset_index(drop=True)



