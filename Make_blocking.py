#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:10:33 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import Read_datafiles as read
import time


#Stockholm_pressure     = 'smhi-Stockholm_pressure.csv'
#Vavihill_blackC        = 'Vavihill_blackC.csv'
#Sturup_temperature     = 'Sturup_temperature.csv'
#Lund_temperature       = 'Lund_temperature.csv' 
#Sturup_wind            = 'Sturup_wind.csv' 
#Sturup_temperature     = 'Sturup_temperature.csv'

Vavihill_PM25           = 'Vavihill_PM25.csv'
Hallahus_PM25           = 'Hallahus_PM25.csv'
Ängelholm_pressure      = 'Ängelholm_pressure.csv'
Sturup_pressure         = 'Sturup_pressure.csv'
Helsingborg_pressure    = 'Helsingborg_pressure.csv'
Helsingborg_wind        = 'Helsingborg_wind.csv' 
Helsingborg_rain        = 'Helsingborg_rain.csv' 
Hörby_wind              = 'Hörby_wind.csv'
Hörby_temperature       = 'Hörby_temperature.csv'
Hörby_rain              = 'Hörby_rain.csv'
Örja_rain               = 'Örja_rain.csv'
Vavihill_O3             = 'Vavihill_O3.csv'
Malmö_PM25              = 'Malmö_PM25.csv'

start_time = time.time()


PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_data = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)


#PM_data = read.get_pm_data(Malmö_PM25)


wind_data = read.get_wind_data(Hörby_wind)
temp_data = read.get_temp_data(Hörby_temperature)
rain_data = read.get_rain_data(Hörby_rain)
pres_data = read.get_pressure_data(Helsingborg_pressure)


SMHI_block_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit = 1015, 
                                     duration_limit = 5, 
                                     rain_limit = 0.5,
                                     info=False)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                         SMHI_block_list, info=True)

titles = read.array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                   SMHI_block_list, only_titles=True)

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
    

