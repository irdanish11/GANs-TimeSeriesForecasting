# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:41:53 2020

@author: Danish
"""


import pandas as pd
import glob
import os
from utilities import to_weeks

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
fips_zips = {}
for idx in range(len((zip_df))):
    fip = zip_df['stcountyfp'][idx]
    if fip in fips_lst:
        if fip not in fips_zips:
            fips_zips[fip] = zip_df['zip'][idx]
covid_df.to_csv(path+'/Data/Processed_COVID.csv')













