# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:59:59 2020

@author: Danish
"""


import pandas as pd
import numpy as np

path = r'C:\Users\danis\Documents\USFoods'
df_covid = pd.read_csv(path+'/Data/Processed_COVID.csv')
df_sales_tr = pd.read_csv(path+'/Data/Sales_Traffic_Mapping.csv')

#checking for how many weeks we have the data
weeks_covid = np.sort(df_covid.fisc_wk.unique())
weeks_sales_tr = df_sales_tr.fisc_wk.unique()

weeks_cmn = np.intersect1d(weeks_covid, weeks_sales_tr)
#we will find the non common values beteen the coinciding weeks and the rest of the
#weeks of sales and traffic mapped data. And for these we don't have covid data
weeks_not_cmn = np.setxor1d(weeks_cmn, weeks_sales_tr)

#extracting the weeks that does not have covid data, which are non-common weeks
df_sales_tr['fisc_wk2'] = df_sales_tr['fisc_wk']
df_sales_tr = df_sales_tr.set_index('fisc_wk2').sort_index()
