#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:10:33 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import time

import read_datafiles as read
import csv_data as csv

start_time = time.time()


location = 'Vavihill' #    <----- THIS CAN BE CHANGED

pres_data = csv.main['pressure']
rain_data = csv.main['rain'] 
temp_data = csv.main['temperature'] 
wind_data = csv.main['wind'] 
PM_data = csv.main['PM25'][location] 


blocking_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit = 1015, 
                                     duration_limit = 5, 
                                     rain_limit = 0.5,
                                     info=False)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                         blocking_list, info=True)

titles = read.array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                   blocking_list, only_titles=True)

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


#%%

count = 0
for i, array in enumerate(totdata_list):
    array_title = titles[i]
    max_PM = np.max(array[2])
    if max_PM > 50:
        count += 1
   
        mean_dir = round(np.nanmean(array[3]))
        mean_speed = np.round(np.nanmean(array[4]),1)
        mean_temp = np.round(np.nanmean(array[5]),1)
        print(f'fig {count}: Mean wind dir is {mean_dir}°, mean wind speed is {mean_speed}m/s, and mean temp is {mean_temp}°C.')
        print(f'             {array_title}')
        read.plot_extra_blocking_array(array, array_title, extrainfo=True)
    

        
#%%

for i, array in enumerate(totdata_list):
    array_title = titles[i]
    read.plot_extra_blocking_array(array, array_title, extrainfo=True)
    
    if i > 30:
        raise ValueError(f"Showing {len(totdata_list)} graphs is too many!")
    

