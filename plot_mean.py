#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:38:36 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import read_datafiles as read
import csv_data as csv
import matplotlib.colors as mcolors
import time
  

start_time = time.time()

location = 'Vavihill' #    <----- THIS CAN BE CHANGED

pres_data = csv.main['pressure']
rain_data = csv.main['rain'] 
temp_data = csv.main['temperature'] 
wind_data = csv.main['wind'] 
PM_data   = csv.main['PM25'][location] 

blocking_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit=1015, 
                                     duration_limit=5, 
                                     rain_limit=0.5,
                                     info=False)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=1, info=True)

totdata_list_dates = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=1, only_titles=True)


block_datafile = pd.concat(blocking_list, ignore_index=True)
PM_without_blocking = PM_data[~PM_data['datetime_start'].isin(block_datafile['datetime'])]

pm_mean = np.nanmean(np.array(PM_without_blocking['pm2.5']))
pm_sigma = np.nanstd(np.array(PM_without_blocking['pm2.5']))

print(f'mean particle concentration is {np.round(pm_mean,1)} ± {np.round(pm_sigma,1)} µg/m³')

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


#%%


read.plot_mean(totdata_list, daystoplot=14, minpoints=8, 
          place=location, save=False, info=False,
          pm_mean=pm_mean, pm_sigma=pm_sigma)



#%%
""" Wind direction"""

dir_totdata_list = read.sort_wind_dir(totdata_list, pie=True, save=False, sort=0.5)



read.plot_dir_mean(dir_totdata_list, daystoplot=14, minpoints=8, 
              place=location, save=False,
              pm_mean=pm_mean, pm_sigma=pm_sigma)

#%%
""" Season"""

seasonal_totdata_list = read.sort_season(totdata_list, totdata_list_dates, pie=True, pieinfo=True, save=False)


read.plot_seasonal_mean(seasonal_totdata_list, daystoplot=14, minpoints=8, 
              place=location, save=False,
              pm_mean=pm_mean, pm_sigma=pm_sigma)


#%%
""" Blcoking strength """


pressure_totdata_list = read.sort_pressure(totdata_list, pie=True, pieinfo=True, save=False)


read.plot_pressure_mean(pressure_totdata_list, daystoplot=14, minpoints=8, 
              place=location, save=False,
              pm_mean=pm_mean, pm_sigma=pm_sigma)



#%%

dir_lower_lim = 80
dir_upper_lim = 190
daystoplot = 12
minum_allowed_datatsets = 6


dir_totdata_list = read.sort_wind_dir(totdata_list,
                                      lowerlim=dir_lower_lim,
                                      upperlim=dir_upper_lim) 

wind = f'{dir_lower_lim}° to {dir_upper_lim}°'

read.plot_mean(dir_totdata_list, daystoplot=daystoplot, minpoints=minum_allowed_datatsets, 
          wind=wind, place=location, pm_mean=pm_mean, save=False)


print(f'The number of blocking filtered between {dir_lower_lim}° to {dir_upper_lim} are now {len(dir_totdata_list)} datasets!')

for i, array in enumerate(dir_totdata_list):
    #read.plot_extra_blocking_array(array, extrainfo=True)
    if i > 30:
        raise ValueError(f"Showing {len(totdata_list)} graphs is too many!")


#%%%
"""
Here we filter on even higher pressure limits
"""

location = 'Vavihill' #    <----- THIS CAN BE CHANGED


PM_data   = csv.main['PM25'][location] 


blocking_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit=1025, 
                                     duration_limit=5, 
                                     rain_limit=0.5,
                                     info=False)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=1, info=True)


read.plot_mean(totdata_list, daystoplot=14, minpoints=8, 
          place=location, save=False, info=True,
          pm_mean=pm_mean, pm_sigma=pm_sigma)





