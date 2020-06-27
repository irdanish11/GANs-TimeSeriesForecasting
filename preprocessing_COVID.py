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
covid_df = pd.concat(covid_df)

#converting date to weeks
date = covid_df.date.copy()
date = date.reset_index(drop=True)
weeks = to_weeks(date, format_='%Y%m%d', splitter='-')

