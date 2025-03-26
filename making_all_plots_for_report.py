#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:51:51 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import read_datafiles as read
import csv_data as csv
import time
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


info        = False  #<-------- CHANGE IF YOU WANT
press_lim   = 1015 
dur_lim     = 5 
rain_lim    = 0.5
mindatasets = 8
daystoplot  = 14


start_time = time.time()



###############################################################################

"""
Here we make the period plot
"""


pressure_data = csv.main['pressure']
rain_data = csv.main['rain']["Hörby"]
temp_data = csv.main['temperature'] 
wind_data = csv.main['wind']["Hörby"]
PM_data = csv.main['PM25']['Vavihill']



blocking_list = read.find_blocking(pressure_data, 
                                   rain_data, 
                                     pressure_limit=press_lim, 
                                     duration_limit=dur_lim, 
                                     rain_limit=rain_lim)

read.plot_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                       blocking_list,
                       start_time='2001-01-01', 
                       end_time='2001-12-31',
                       temp_plot=False,
                       save=True)

if not info: print('1. The period plot is done')

###############################################################################

"""
Here we make the histograms
"""

blocking_list = read.find_blocking(csv.histogram_main['pressure'], 
                                   csv.histogram_main['rain'], 
                                   pressure_limit=press_lim, 
                                   duration_limit=dur_lim, 
                                   rain_limit=rain_lim)


read.plot_blockings_by_year(blocking_list, lim=7, save=True)

read.plot_blockingsdays_by_year(blocking_list, 'all', save=True)
read.plot_blockingsdays_by_year(blocking_list, 'autumn', save=True)
read.plot_blockingsdays_by_year(blocking_list, 'summer', save=True)
read.plot_blockingsdays_by_year(blocking_list, 'winter', save=True)
read.plot_blockingsdays_by_year(blocking_list, 'spring', save=True)




if not info: print('2. The histograms are done')

###############################################################################

"""
Here we make the mean plots
"""

locationlist = ['Vavihill', 'Malmö']

for location in locationlist:
    PM_data   = csv.main['PM25'][location] 
    
    if location == "Malmö":
        rain_data = csv.main['rain']["Malmö"]
        wind_data = csv.main['wind']["Malmö"]
        
    if location == "Vavihill":
         rain_data = csv.main['rain']["Hörby"]
         wind_data = csv.main['wind']["Hörby"]

    blocking_list = read.find_blocking(pressure_data, rain_data, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=False)

    totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                                  temp_data, rain_data, 
                                                  blocking_list, 
                                                  cover=1, info=False)
    totdata_list_dates = read.array_extra_blocking_list(PM_data, wind_data, 
                                                  temp_data, rain_data, 
                                                  blocking_list, 
                                                  cover=1, only_titles=True)

    block_datafile = pd.concat(blocking_list, ignore_index=True)
    PM_without_blocking = PM_data[~PM_data['datetime_start'].isin(block_datafile['datetime'])]

    pm_mean = np.nanmean(np.array(PM_without_blocking['pm2.5']))
    pm_sigma = np.nanstd(np.array(PM_without_blocking['pm2.5']))

    read.plot_mean(totdata_list, daystoplot=daystoplot, minpoints=mindatasets, 
                   place=location, save=True, info=True, infosave=True,
                   pm_mean=pm_mean, pm_sigma=pm_sigma)
    
    if info: print(f" \n *** {location} ***")
    
    dir_totdata_list = read.sort_wind_dir(totdata_list, pieinfo=info)
    
    if info: print(" \n ")
    
    seasonal_totdata_list = read.sort_season(totdata_list, totdata_list_dates, 
                                             pie=False, pieinfo=info)
    if info: print(" \n ")

    pressure_totdata_list = read.sort_pressure(totdata_list, pieinfo=info)

    read.plot_dir_mean(dir_totdata_list, daystoplot=daystoplot, minpoints=mindatasets, 
                       place=location, save=True,
                       pm_mean=pm_mean, pm_sigma=pm_sigma)
    

    read.plot_seasonal_mean(seasonal_totdata_list, daystoplot=daystoplot, 
                            minpoints=mindatasets, place=location, save=True,
                            pm_mean=pm_mean, pm_sigma=pm_sigma)
    


    read.plot_pressure_mean(pressure_totdata_list, daystoplot=daystoplot, 
                            minpoints=mindatasets, place=location, save=True,
                            pm_mean=pm_mean, pm_sigma=pm_sigma)
    
    
if not info: print('3. The mean plots are now done')

plt.close('all')
    
if not info: print(f"Elapsed time: {time.time() - start_time:.0f} seconds")
