#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:38:36 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Read_datafiles as read
import time
from scipy.signal import savgol_filter


Vavihill_PM25 = 'Vavihill_PM25.csv'
Hallahus_PM25 = 'Hallahus_PM25.csv'
Helsingborg_pressure = 'Helsingborg_pressure.csv'
Hörby_rain = 'Hörby_rain.csv'
Hörby_wind = 'Hörby_wind.csv'
Hörby_temperature = 'Hörby_temperature.csv'
Malmö_PM25 = 'Malmö_PM25.csv'

    

def plot_mean(totdata_list, daystoplot, wind=False, minpoints=5, 
              info=False, save=False, place='',
              pm_mean=False, pm_sigma=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour.
    """
    timelen = int(24*daystoplot)  # Initial length in hours

    # Create an array to store all the PM2.5 values
    PM_array = np.full((len(totdata_list), timelen), np.nan)
    

    # Populate the PM_array with data
    for i, array in enumerate(totdata_list):
        valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
        PM_array[i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs
    mean = np.nanmean(PM_array, axis=0)
    sigma = np.nanstd(PM_array, axis=0)  # Standard deviation for shading
    t = np.arange(timelen)/24  # Time axis in days   
      
    # Below we check the number of data points
    valid_counts_per_hour = np.sum(~np.isnan(PM_array), axis=0)
    if info: 
        plt.figure(figsize=(7, 4))
        plt.plot(t,valid_counts_per_hour, label='Number of datasets')
        plt.title(f'Number of datasets for {daystoplot} days at {place}')
        plt.xlabel('Time from start of blocking (days)')
        plt.ylabel('Number of datasets')
        plt.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5, label='Minimum number of datasets allowed')
        plt.yticks(np.arange(0, max(valid_counts_per_hour) + 1, 1))  
        plt.grid(axis='y', linestyle='--', alpha=0.6)  
        plt.tight_layout()
        plt.legend()
    
    # Give error if we loose to many data points
    if valid_counts_per_hour[timelen-1] < minpoints:
        raise ValueError(f'At day {daystoplot} there was {valid_counts_per_hour[timelen-1]} data points. This is to little to calculate a mean and standard deviation!')
        

    # Plot everything
    plt.figure(figsize=(8, 5))
    if wind:
        plt.title(f'Mean concentration of PM2.5 with wind between {wind} during first {daystoplot} days, {place}')
    else:
        plt.title(f'Mean concentration of PM2.5 during first {daystoplot} days, {place}')
    
    if pm_mean:
        plt.plot(t, pm_mean+t*0, label='Standard Mean', c='gray')
        plt.fill_between(t,  pm_mean+t*0 +  pm_sigma+t*0,  pm_mean+t*0 - pm_sigma, alpha=0.4, color='gray')

    plt.plot(t, mean, label='Mean Blocking Data', c='C0')
    plt.fill_between(t, mean + sigma, mean - sigma, alpha=0.4, color='C0')
    
    plt.xlabel('Time from start of blocking (days)')
    plt.ylabel('Mean Concentration [PM2.5 (µg/m³)]')
    plt.grid(True)
    plt.ylim(0,40)
    plt.tight_layout()
    plt.legend()
    
    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_{place}.pdf")
    plt.show()



def plot_dir_mean(dir_totdata_list, daystoplot, minpoints=5, place='',
                  labels=["North", "East", "South", "West"], 
                  pm_mean=False, pm_sigma=False, save=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each wind direction category in subplots.
    Only non-empty wind directions are plotted dynamically.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold"]  # Colors mapped to N, E, S, W

    # Filter out empty wind directions
    valid_data = [(totdata_list, label, color) for totdata_list, label, color in 
                  zip(dir_totdata_list, labels, colors) if len(totdata_list) > 0]
    
    if len(valid_data) == 0:
        print("No valid data to plot.")
        return
    
    # Create dynamic subplots based on available data
    fig, axes = plt.subplots(len(valid_data), 1, figsize=(8, 3 * len(valid_data)), sharex=True, sharey=True)
    fig.tight_layout(pad=5.0)
    fig.suptitle(f'Mean Concentration of PM2.5 During First {daystoplot} Days, {place}')

    if len(valid_data) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, (totdata_list, label, color) in zip(axes, valid_data):
        
        # Create an array to store all PM2.5 values
        PM_array = np.full((len(totdata_list), timelen), np.nan)

        # Populate PM_array with available data
        for i, array in enumerate(totdata_list):
            valid_len = min(len(array[2]), timelen)  # Avoid indexing errors
            PM_array[i, :valid_len] = array[2][:valid_len]

        # Compute mean and standard deviation, ignoring NaNs
        mean = np.nanmean(PM_array, axis=0)
        sigma = np.nanstd(PM_array, axis=0)
        t = np.arange(timelen)/24  # Convert time to days

        # Check if we have enough valid data points at the end
        valid_counts_per_hour = np.sum(~np.isnan(PM_array), axis=0)
        valid_indices = np.where(valid_counts_per_hour >= minpoints)[0]

        if len(valid_indices) == 0:
            non_nan_indices = np.where(~np.isnan(PM_array))[0]
            if len(non_nan_indices) == 0:
                continue  # Skip if no valid data
            hmax = np.max(non_nan_indices)
        else:
            hmax = valid_indices[-1]
            
        if pm_mean:
          ax.plot(t, pm_mean+t*0, label='Standard Mean', c='gray')
          ax.fill_between(t,  pm_mean+t*0 +  pm_sigma+t*0,  pm_mean+t*0 - pm_sigma, alpha=0.4, color='gray') 

        # Plot the mean and confidence interval
        ax.plot(t[:hmax + 1], mean[:hmax + 1], label=f'{label} Blocking Mean', color=color)
        ax.fill_between(t[:hmax + 1], mean[:hmax + 1] + sigma[:hmax + 1], 
                        mean[:hmax + 1] - sigma[:hmax + 1], alpha=0.3, color=color)
        

        # Set y-axis integer ticks and grid lines
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        ax.set_title(f'Direction: {label}')
        ax.set_ylabel('PM2.5 [µg/m³]')
        ax.set_ylim(0,)
        ax.legend()

    axes[-1].set_xlabel('Time from start of blocking (days)')

    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_dir_{place}.pdf")
    plt.show()



start_time = time.time()

PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_Vavihill = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)
PM_Malmö = read.get_pm_data(Malmö_PM25)
PM_datas = {'Malmö':PM_Malmö, 'Vavihill':PM_Vavihill}

location = 'Vavihill'

PM_data = PM_datas[location]

wind_data = read.get_wind_data(Hörby_wind)
temp_data = read.get_temp_data(Hörby_temperature)
rain_data = read.get_rain_data(Hörby_rain)
pres_data = read.get_pressure_data(Helsingborg_pressure)

SMHI_block_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit=1015, 
                                     duration_limit=5, 
                                     rain_limit=0.5,
                                     info=False)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              SMHI_block_list, 
                                              cover=1, info=True)


pm_mean = np.nanmean(np.array(PM_data['pm2.5']))
pm_sigma = np.nanstd(np.array(PM_data['pm2.5']))

print(f'mean particle concentration is {np.round(pm_mean,1)} ± {np.round(pm_sigma,1)} µg/m³')

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


#%%


plot_mean(totdata_list, daystoplot=7, minpoints=12, 
          place=location, save=False,
          pm_mean=pm_mean, pm_sigma=pm_sigma)



#%%

dir_totdata_list = read.sort_wind_dir(totdata_list, pie=False, save=False)


plot_dir_mean(dir_totdata_list, daystoplot=10, minpoints=8, 
              place=location, save=False,
              pm_mean=pm_mean, pm_sigma=pm_sigma)




#%%

dir_lower_lim = 125
dir_upper_lim = 140
daystoplot = 7
minum_allowed_datatsets = 6


dir_totdata_list = read.sort_wind_dir(totdata_list,
                                      lowerlim=dir_lower_lim,
                                      upperlim=dir_upper_lim) 

wind = f'{dir_lower_lim}° to {dir_upper_lim}°'

plot_mean(dir_totdata_list, daystoplot=daystoplot, minpoints=minum_allowed_datatsets, 
          wind=wind, place=location, pm_mean=pm_mean, save=False)


print(f'The number of blocking filtered between {dir_lower_lim}° to {dir_upper_lim} are now {len(dir_totdata_list)} datasets!')

for i, array in enumerate(dir_totdata_list):
    #read.plot_extra_blocking_array(array, extrainfo=True)
    if i > 30:
        raise ValueError(f"Showing {len(totdata_list)} graphs is too many!")



