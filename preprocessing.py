# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:17 2020

@author: Danish
"""


import pandas as pd
from pathlib import Path
from utilities import to_weeks, series_to_df
import os
import numpy as np

path = r'C:\Users\danis\Documents\USFoods'

#covid_df = pd.read_csv(path+'/COVIDData.csv')
#mobility_df = pd.read_csv(path+'/Mobility_eth.csv')
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
sales_df = sales_df.dropna()



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
foot_traffic['fisc_wk2'] = foot_traffic['fisc_wk']
foot_traffic_wk = foot_traffic.set_index('fisc_wk2').sort_index()
#

#Extracting data for week 6 to 18 from sales_data, because don't need dta of other weeks
#because in foot_traffic we have data of only week 6 to 18
lst = []
lst2 = []
for i in traffic_weeks:
    if i in sales_weeks:
        lst.append(new_sales_df.loc[i])
        lst2.append(foot_traffic_wk.loc[i])
sales_df_6_18 = pd.concat(lst)
new_foot_traffic = pd.concat(lst2)
sales_head = sales_df_6_18.head(50)
#Getting the div_nbr that are present in the data for week 6-18
div_nb_unq = sales_df_6_18.div_nbr.unique()
div_nb_unq2 = new_foot_traffic.div_nbr.unique()

#Extracting data of 16 div_nbr from foot_traffic because sales_data contain data for 
#only those div_nbr
new_foot_traffic = new_foot_traffic.set_index('custo').sort_index()
lst = []
for j in div_nb_unq:
    lst.append(new_foot_traffic.loc[j])
foot_traffic_16_div = pd.concat(lst) 
foot_traffic_16_div = foot_traffic_16_div.sort_index()   
foot_traffic_16_div = foot_traffic_16_div.dropna()
traffic_head = foot_traffic_16_div.head(50)

####################### Start Mapping ##################
sales_df_6_18['div_nbr2'] = sales_df_6_18['div_nbr']
foot_traffic_16_div['div_nbr2'] = foot_traffic_16_div['div_nbr']
foot_traffic_16_div['fisc_wk2'] = foot_traffic_16_div['fisc_wk']
#setting week as index
foot_traffic_16_div = foot_traffic_16_div.set_index('fisc_wk2').sort_index()
weeks_t = foot_traffic_16_div.fisc_wk.unique()
div_nbrs_t = foot_traffic_16_div.div_nbr.unique()
weeks_s = sales_df_6_18.fisc_wk.unique()
div_nbrs_s = sales_df_6_18.div_nbr.unique()
weeks_t == weeks_s
div_nbrs_t.sort() == div_nbrs_s.sort() 


data = ['fisc_wk', 'div_nbr', 'zip_cd', 'prod_nbr', 'cases_shipped', 'sales',
        'foot_traffic_by_day', 'cust_nbr', 'visits_that_day', 'median_dwell']

df_lst = []
for week in weeks_t:
    df_traffic = foot_traffic_16_div.loc[week]
    df_sales = sales_df_6_18.loc[week] 
    #Setting the div_nbr as index
    df_traffic = df_traffic.set_index('div_nbr2').sort_index()
    df_sales = df_sales.set_index('div_nbr2').sort_index()  
    #Getting the divison numbers present in respective  2 df
    div_tr = np.sort(df_traffic.div_nbr.unique())
    div_sa = np.sort(df_sales.div_nbr.unique()) 
    #extracting the data of each div_nbr one by one.
    lst = []
    for d in div_tr:
        if d in div_sa:
            df_temp_tr = pd.DataFrame(df_traffic.loc[d])
            df_temp_sa = pd.DataFrame([df_sales.loc[d]])
            #determine the size which will be used to extract the rows
            if len(df_temp_tr) < len(df_temp_sa):
                size = len(df_temp_tr)
            else:
                size = len(df_temp_sa)
            df_temp_tr = df_temp_tr.reset_index()
            df_temp_sa = df_temp_sa.reset_index() 
            df1_t = df_temp_tr.loc[0:size].copy()
            df2_s = df_temp_sa[0:size].copy()
            #Removing extra columns
            #fisc_wk & div_nbr are already present in df2_s, that is why drop here
            df1_t = df1_t.drop(['div_nbr2', 'clndr_dt', 'fisc_wk', 'div_nbr'], axis=1)
            df2_s = df2_s.drop(['div_nbr2', 'prod_desc', 'smplfd_menu_desc',
                                'cases_ordered'], axis=1)
            new_df = pd.concat([ df1_t,  df2_s], axis=1)
            new_df = new_df[data]
            new_df = new_df.dropna()
            lst.append(new_df)
    #concatenating all the dataframes based on the div_nbr for a single week
    week_df = pd.concat(lst)
    #appending the dataframe which contains the data for 1 week
    df_lst.append(week_df)
 #concatenating all the dataframes based on the weeks    
sales_traffic = pd.concat(df_lst)      

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



