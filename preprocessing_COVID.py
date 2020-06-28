# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:41:53 2020

@author: Danish
"""


import pandas as pd
import glob
import os
from utilities import to_weeks, extract_sub_df
import numpy as np

path = r'C:\Users\danis\Documents\USFoods'
csv_files = os.listdir(path+'/COVID')
#removes the first file which is non csv
#csv_files.pop(0)

#reading all the csv files
covid_df = []
for f in csv_files:
    df = pd.read_csv(path+'/COVID/'+f)
    covid_df.append(df)
covid_df1 = covid_df.pop(18)
covid_df = pd.concat(covid_df)

############################ Converting Dates to Weeks ############################

#converting date to weeks
date = covid_df.date.copy()
date = date.reset_index(drop=True)
weeks0 = to_weeks(date, format_='%Y%m%d', splitter='-', year_index=0)
covid_df['fisc_wk'] = weeks0

#converting date to weeks
date = covid_df1.date.copy()
date = date.reset_index(drop=True)
weeks1 = to_weeks(date, format_='%d%m%Y', splitter='/', year_index=2)
covid_df1['fisc_wk'] = weeks1

############################ Adding zip codes ############################
#concatenated DF
covid_df = pd.concat([covid_df, covid_df1])

zip_df = pd.read_csv(path+'/zip_to_county.csv')

stcountyfp = zip_df.stcountyfp.unique()
fips = covid_df.fips.unique()
#verify that we have all fips id
fips_lst = []
for f in fips:
    if f in stcountyfp:
        fips_lst.append(f)
        
#creating a mapping of zip codes and fips
zip_df_tmp = zip_df.copy()
zip_df_tmp = zip_df_tmp.set_index('stcountyfp').sort_index()
fips_2_zips = {}
for f in fips_lst:
    df = extract_sub_df(zip_df_tmp, f)
    fips_2_zips[f] = df.zip.unique()

covid_df = covid_df.reset_index()
zip_cd = []                
for row in range(len(covid_df)):
    fip = covid_df['fips'][row]       
    zip_codes = fips_2_zips[fip]
    #getting the index to choose the zip code randomly
    idx = np.random.randint(0, len(zip_codes), size=1)[0]
    zip_cd.append(zip_codes[idx])
covid_df['zip_cd'] = zip_cd
#deleting unecessary columns
try:
    covid_df = covid_df.drop(['level_0', 'index'], axis=1)
except Exception as e:
    print('Requested Columns not found in the df')
covid_df['zip_cd2'] = covid_df['zip_cd']
covid_df['fisc_wk2'] = covid_df['fisc_wk']
covid_df.to_csv(path+'/Data/Processed_COVID.csv')













