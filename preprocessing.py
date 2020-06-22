# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:17 2020

@author: Danish
"""


import pandas as pd
from pathlib import Path
from utilities import to_weeks
import os

path = r'C:\Users\danis\Documents\USFoods'

#covid_df = pd.read_csv(path+'/COVIDData.csv')
mobility_df = pd.read_csv(path+'/Mobility_eth.csv')
sales_df1 = pd.read_csv(path+'/SalesData1.csv')
sales_df2 = pd.read_csv(path+'/SalesData2.csv')
sales_df3 = pd.read_csv(path+'/SalesData3.csv')
foot_traffic = pd.read_csv(path+'/Customer_Foot_Traffic.csv')




############ converting dates into weeks of Foot Traffic ############
foot_traffic_head = foot_traffic.head(20).copy()
obj = foot_traffic.clndr_dt.copy()
fisc_yr_wk = to_weeks(obj, format_='%Y%m%d', splitter='-', 
                      convert=True, year_index=0)
foot_traffic['fisc_wk'] = fisc_yr_wk



############### Sales Data ##############
def to_numeric(df, columns):
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    return df
#Converting the dtypes of the column from object to numeric. The fields containing 
#str will be replaced with NaN
sales_df1.dtypes 
sales_df1 = to_numeric(sales_df1, ['fisc_wk', 'cases_shipped'])
sales_df2.dtypes 
sales_df2 = to_numeric(sales_df2, ['fisc_wk', 'cases_shipped'])
sales_df3.dtypes 
sales_df3 = to_numeric(sales_df3, ['fisc_wk', 'cases_shipped'])

sales_df = pd.concat([sales_df1, sales_df2, sales_df3])



################# Sales Data to Foot Traffic ################

#droping infinities in the column
sales_df.dropna(subset=['fisc_wk'], inplace=True)
sales_df = sales_df.astype({'fisc_wk':'int'})
foot_traffic = foot_traffic.astype({'fisc_wk':'int'})

sales_df['fisc_wk2'] = sales_df['fisc_wk']
foot_traffic['div_nbr'] = foot_traffic['custo']


sales_head = sales_df.head(50)
traffic_head = foot_traffic.head(50)

div_num_sales = sales_df.div_nbr.unique()
div_num_trafic = foot_traffic.custo.unique()

#checking the for same div_num in sales_df & foot_traffic
same_div_num = []
for i in div_num_sales:
    if i in div_num_trafic:
        same_div_num.append(i)

sales_weeks = sales_df.fisc_wk.unique()
traffic_weeks = foot_traffic.fisc_wk.unique()
#removing the floating and NaN values.
sales_weeks.sort()
traffic_weeks.sort()

# Setting fisc_wk as index of sales data
new_sales_df = sales_df.set_index('fisc_wk2').sort_index()
# Setting custo/div_nbr as index of foot traffic
new_foot_traffic = foot_traffic.set_index('custo').sort_index()

#Extracting data for week 5 to 18 from sales_data, because don't need dta of other weeks
#because in foot_traffic we have data of only week 5 to 18
lst = []
for i in traffic_weeks:
    lst.append(new_sales_df.loc[i])
sales_df_5_18 = pd.concat(lst)
sales_df_5_18_nan = sales_df_5_18.copy()
sales_df_5_18 = sales_df_5_18.dropna()
sales_head = sales_df_5_18.head(50)
#Getting the div_nbr that are present in the data for week 5-18
div_nb_unq = sales_df_5_18.div_nbr.unique()

#Extracting data of 19 div_nbr from foot_traffic because sales_data contain data for 
#only those div_nbr
lst = []
for j in div_nb_unq:
    lst.append(new_foot_traffic.loc[j])
foot_traffic_16_div = pd.concat(lst) 
foot_traffic_16_div = foot_traffic_16_div.sort_index()   
foot_traffic_16_div = foot_traffic_16_div.dropna()
traffic_head = foot_traffic_16_div.head(50)



        
           

weeks = to_weeks(obj=foot_traffic.clndr_dt)

#checking the median dwell
data = {}
for i in range(len(foot_traffic.div_nbr)):
    if foot_traffic.div_nbr[i] == 2020:
        data[weeks[i]] = foot_traffic.median_dwell[i]

#Adds the column number of weeks by converting date into week number     
foot_traffic['fisc_yr_wk'] = weeks

################# Sales Data & Foot Traffic mapping #################
new_foot_traffic = foot_traffic.set_index('fisc_yr_wk').sort_index()
new_sales_df = sales_df.set_index('fisc_yr_wk').sort_index()
new_sales_df = new_sales_df.fillna(value=0)

t_sales = new_sales_df.loc[1]
t_traffic = new_foot_traffic.loc[1]



