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
warnings.simplefilter("ignore", category=SyntaxWarning)



info = False          #<-------- CHANGE IF YOU WANT

press_lim   = 1014   # This is the pressure limit for classifying high pressure
dur_lim     = 5      # Minimum number of days for blocking
rain_lim    = 0.5    # Horly max rain rate
mindatasets = 8      # Minimum allowed of dattsets allowed when taking std and mean
daystoplot  = 14     # How long periods should the plots display
pm_coverege = 0.95    # How much PM2.5 coverge must the periods have


start_time = time.time()


###############################################################################

"""
Here we make the period plot
"""

locationlist = ['Vavihill', 'Malmö']

pressure_data = csv.main['pressure']
temp_data = csv.main['temperature'] 

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
                                     rain_limit=rain_lim)

    read.plot_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                       blocking_list,
                       start_time='2001-01-01', 
                       end_time='2001-12-31',
                       tempwind_plot=False,
                       locationsave=location)

if not info: print('1. The period plots are done')

###############################################################################

"""
Here we make the histograms
"""

blocking_list = read.find_blocking(csv.histogram_main['pressure'], 
                                   csv.histogram_main['rain'], 
                                   pressure_limit=press_lim, 
                                   duration_limit=dur_lim, 
                                   rain_limit=4*24) # This is avrege four 24 hours 


read.plot_blockings_by_year(blocking_list, lim1=7, lim2=10, save=True)


read.plot_blockingsdays_by_year(blocking_list, typ="all", save=True)



if not info: print('2. The histograms are done')

info

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
         
    if info: print(f" \n *** {location} ***")


    blocking_list = read.find_blocking(pressure_data, rain_data, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)

    totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                                  temp_data, rain_data, 
                                                  blocking_list, 
                                                  cover=pm_coverege, info=info)
    
    totdata_list_dates = read.array_extra_blocking_list(PM_data, wind_data, 
                                                  temp_data, rain_data, 
                                                  blocking_list, 
                                                  cover=pm_coverege, only_titles=True)

    block_datafile = pd.concat(blocking_list, ignore_index=True)
    PM_without_blocking = PM_data[~PM_data['datetime_start'].isin(block_datafile['datetime'])]

    pm_mean = np.nanmean(np.array(PM_without_blocking['pm2.5']))
    pm_sigma = np.nanstd(np.array(PM_without_blocking['pm2.5']))

    read.plot_mean(totdata_list, daystoplot=daystoplot, minpoints=mindatasets, 
                   place=location, save=True, info=True, infosave=True,
                   pm_mean=pm_mean, pm_sigma=pm_sigma)
    
    
    if info: print(" \n ")
    
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
