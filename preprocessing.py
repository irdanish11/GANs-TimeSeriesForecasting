# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:17 2020

@author: Danish
"""


import pandas as pd

covid_df = pd.read_csv('./Data/COVIDData.csv')
mobility_df = pd.read_csv('./Data/Mobility_eth.csv')
sales_df = pd.read_csv('./Data/SalesData.csv')
foot_traffic = pd.read_csv('./Data/CustomerFootTraffic.csv')

div_num_sales = sales_df.div_nbr.unique()
div_num_trafic = foot_traffic.div_nbr.unique()

#checking the for same div_num in sales_df & foot_traffic
same_div_num = []
for i in div_num_sales:
    if i in div_num_trafic:
        same_div_num.append(i)
        
           
def to_weeks(obj, splitter='/', convert=True):       
    if convert:
        for j in range(len(obj)):
            split = obj[j].split(splitter)
            new = ['0'+i if len(i)<2  else i for i in split]
            if len(new[2])<4:
                new[2] = new[2]+'20'
            obj[j] = ''.join(new)
    obj = pd.to_datetime(obj, format='%m%d%Y')
    return obj.dt.week

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