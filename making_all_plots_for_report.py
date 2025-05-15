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
warnings.simplefilter("ignore", category=UserWarning)

plt.close('all')

info   = False          #<-------- CHANGE IF YOU WANT
imsave = 'png'         # Can be pdf, png, False

press_lim   = 1014   # This is the pressure limit for classifying high pressure
dur_lim     = 5      # Minimum number of days for blocking
rain_lim    = 0.5    # Horly max rain rate
mindatasets = 8      # Minimum allowed of dattsets allowed when taking std and mean
daystoplot  = 14     # How long periods should the plots display
pm_coverege = 0.85   # How much PM2.5 coverge must the periods have
timediff    = '5 hours' # This determines how far away two differnet blocking can be 


start_time = time.time()


###############################################################################

"""
Here we make the frequency plots
"""

blocking_list = read.find_blocking(csv.histogram_main['pressure'], 
                                   csv.histogram_main['rain'], 
                                   pressure_limit=press_lim, 
                                   duration_limit=dur_lim, 
                                   rain_limit=2/24) # This is avrege four 24 hours 


read.plot_blockings_by_year(blocking_list, lim1=7, lim2=10, save=imsave)


read.plot_blockingsdays_by_year(blocking_list, typ="all", save=imsave)



if not info: print('1. The frequency plots are done')


###########################################################

"""
Here we make the period plot
"""


pressure_data = csv.main['pressure']
temp_data = csv.main['temperature'] 

PM_data_Vavihill   = csv.main['PM25']['Vavihill'] 
rain_data_Vavihill = csv.main['rain']["Hörby"]
wind_data_Vavihill = csv.main['wind']["Hörby"]
         
if info: print("*** Vavihill *** ")


blocking_list_Vavihill = read.find_blocking(pressure_data, rain_data_Vavihill, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)


PM_data_Malmö   = csv.main['PM25']['Malmö'] 
rain_data_Malmö = csv.main['rain']["Malmö"]
wind_data_Malmö = csv.main['wind']["Malmö"]

blocking_list_Malmö = read.find_blocking(pressure_data, rain_data_Malmö, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)


blocking_list_Malmö, blocking_list_Vavihill = read.date_calibrate_blockinglists(
    blocking_list_Malmö, 
    blocking_list_Vavihill, 
    timediff, info=info)


read.plot_period(PM_data_Vavihill, wind_data_Vavihill, rain_data_Vavihill, 
                 pressure_data, blocking_list_Vavihill,
                 start_time='2001-01-01', 
                 end_time='2001-12-31',
                 wind_plot=False,
                 locationsave="Vavihill",
                 save=imsave)

read.plot_period(PM_data_Malmö, wind_data_Malmö, rain_data_Malmö, 
                 pressure_data, blocking_list_Malmö,
                 start_time='2001-01-01', 
                 end_time='2001-12-31',
                 wind_plot=False,
                 locationsave="Malmö",
                 save=imsave)

if not info: print('2. The period plots are done')



##############################################################################


"""
Here we make the mean plots
"""
    
totdata_list_Vavihill, totdata_list_dates_Vavihill = read.array_blocking_list(
                                                  PM_data_Vavihill, 
                                                  wind_data_Vavihill, 
                                                  rain_data_Vavihill, 
                                                  blocking_list_Vavihill, 
                                                  cover=pm_coverege, 
                                                  info=info)

block_datafile_Vavihill = pd.concat(blocking_list_Vavihill, ignore_index=True)
PM_without_blocking_Vavihill = PM_data_Vavihill[~PM_data_Vavihill['datetime_start'].isin(block_datafile_Vavihill['datetime'])]



pm_mean_Vavihill= np.nanmean(np.array(PM_without_blocking_Vavihill['pm2.5']))
pm_sigma_Vavihill = np.nanstd(np.array(PM_without_blocking_Vavihill['pm2.5']))

if info: print("*** Malmö *** ")


totdata_list_Malmö, totdata_list_dates_Malmö = read.array_blocking_list(
                                                  PM_data_Malmö, 
                                                  wind_data_Malmö, 
                                                  rain_data_Malmö, 
                                                  blocking_list_Malmö, 
                                                  cover=pm_coverege, 
                                                  info=info)
    

block_datafile_Malmö = pd.concat(blocking_list_Malmö, ignore_index=True)
PM_without_blocking_Malmlö = PM_data_Malmö[~PM_data_Malmö['datetime_start'].isin(block_datafile_Malmö['datetime'])]

pm_mean_Malmö = np.nanmean(np.array(PM_without_blocking_Malmlö['pm2.5']))
pm_sigma_Malmö = np.nanstd(np.array(PM_without_blocking_Malmlö['pm2.5']))


read.plot_mean(totdata_list1=totdata_list_Vavihill, totdata_list2=totdata_list_Malmö,
               daystoplot=daystoplot, minpoints=mindatasets, 
               place1='Vavihill', place2='Malmö', 
               pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
               pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö, 
               save=imsave)
    
if info: print(" \n ")
    
dir_totdata_list_Vavihill = read.sort_wind_dir(totdata_list_Vavihill, pieinfo=info)
dir_totdata_list_Malmö = read.sort_wind_dir(totdata_list_Malmö, pieinfo=info)

    
if info: print(" \n ")
    
seasonal_totdata_list_Vavihill = read.sort_season(totdata_list_Vavihill, 
                                                  totdata_list_dates_Vavihill, 
                                                  pieinfo=info)
seasonal_totdata_list_Malmö = read.sort_season(totdata_list_Malmö, 
                                               totdata_list_dates_Malmö, 
                                               pieinfo=info)
if info: print(" \n ")

pressure_totdata_list_Vavihill = read.sort_pressure(totdata_list_Vavihill, pieinfo=info)
pressure_totdata_list_Malmö = read.sort_pressure(totdata_list_Malmö, pieinfo=info)

if info: print(" \n ")


read.plot_dir_mean(dir_totdata_list1=dir_totdata_list_Vavihill, 
                   dir_totdata_list2=dir_totdata_list_Malmö, 
                   daystoplot=daystoplot,  
                   minpoints=8,
                   place1='Vavihill', place2='Malmö', save=imsave,
                   pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                   pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)



read.plot_pressure_mean(pressure_totdata_list1=pressure_totdata_list_Vavihill, 
                        pressure_totdata_list2=pressure_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=imsave,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)


read.plot_seasonal_mean(seasonal_totdata_list1=seasonal_totdata_list_Vavihill, 
                        seasonal_totdata_list2=seasonal_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=imsave,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)
    

if not info: print('3. The mean plots are now done')

if imsave: plt.close('all')
    
if not info: print(f"Elapsed time: {time.time() - start_time:.0f} seconds")