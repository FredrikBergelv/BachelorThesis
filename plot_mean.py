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
  
def plot_mean(totdata_list, daystoplot, wind=False, minpoints=8, 
              info=False, infosave=False, save=False, place='',
              pm_mean=False, pm_sigma=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour.
    You must specify how many days you wish to plot, you can add a wind title, 
    the number of datasets needed, plot info, etc.
    """
    timelen = int(24 * daystoplot)  # Initial length in hours

    # Create an array to store all the PM2.5 values
    PM_array = np.full((len(totdata_list), timelen), np.nan)
    
    # Populate the PM_array with data
    for i, array in enumerate(totdata_list):
        valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
        PM_array[i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs
    mean = np.nanmean(PM_array, axis=0)
    sigma = np.nanstd(PM_array, axis=0)  # Standard deviation for shading
    t = np.arange(timelen) / 24  # Time axis in days   
    
    # Below we check the number of data points
    valid_counts_per_hour = np.sum(~np.isnan(PM_array), axis=0)

    if info: 
        plt.figure(figsize=(5, 3))
        plt.plot(t, valid_counts_per_hour, label='Number of datasets')
        plt.title(f'Number of datasets for {daystoplot} days at {place}')
        plt.xlabel('Time from start of blocking (days)')
        plt.ylabel('Number of datasets')
        plt.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5, label='Minimum number of datasets allowed')
        plt.yticks(np.arange(0, max(valid_counts_per_hour) + 1, 10))   
        plt.grid()
        plt.tight_layout()
        plt.legend()
        if infosave:
            plt.savefig(f"BachelorThesis/Figures/Meanplotinfo_{place}.pdf")
        plt.show()

            
    # Modify this section to check for available data at the end
    if valid_counts_per_hour[timelen - 1] < minpoints:
        print(f'Warning: At day {daystoplot}, there were only {valid_counts_per_hour[timelen - 1]} data points. Plotting may be incomplete.')

    # Plot everything
    plt.figure(figsize=(5, 5))
    if wind:
        plt.title(f'Mean concentration of PM2.5 with wind between {wind} during first {daystoplot} days, {place}')
    else:
        plt.title(f'Mean concentration of PM2.5, {place}')
    
    # You can also plot the Mean Standard, if wanted to
    if pm_mean:
        plt.plot(t, pm_mean + t * 0, label='Mean during no blocking', c='gray')
        plt.fill_between(t, pm_mean + t * 0 + pm_sigma + t * 0, pm_mean + t * 0 - pm_sigma, alpha=0.4, color='gray')

    # Plot mean values, filling with NaNs if data is not enough
    for i, points in enumerate(valid_counts_per_hour):
        if points < minpoints:
            mean[i] = np.nan  # Set the rest of the mean to NaN
            sigma[i] = np.nan  # Set the rest of the sigma to NaN

    plt.plot(t, mean, label='Mean during blocking', c='C0')
    plt.fill_between(t, mean + sigma, mean - sigma, alpha=0.4, color='C0')
    plt.plot(t, t*0+25, label='EU annual mean limit', c='r', linestyle='--')
    
    plt.xlabel('Time from start of blocking (days)')
    plt.ylabel('Mean Concentration [PM2.5 (µg/m³)]')
    plt.ylim(0, 40)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend()
    
    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_{place}.pdf")
    plt.show() 


def sort_wind_dir(totdata_list, upperlim=False, lowerlim=False, pie=False, save=False,
                  sort=0.5):
    """
    This function filters a list of blocking arrays by wind direction based on a 60% threshold.
    It returns five lists:
        sort_wind_dir[0] -> North (315° to 45°)
        sort_wind_dir[1] -> East (45° to 135°)
        sort_wind_dir[2] -> South (135° to 225°)
        sort_wind_dir[3] -> West (225° to 315°)
        sort_wind_dir[4] -> Non-directional (if no category reaches 50%)
    """
    N_totdata_list = []
    E_totdata_list = []
    S_totdata_list = []
    W_totdata_list = []
    Non_totdata_list = []
    personalized_totdata_list = []

    # Loop through the arrays to sort by wind direction percentage
    for array in totdata_list:
        wind_dir_values = array[3]  # Extract wind direction values
        total_values = len(wind_dir_values)

        # Count how many values fall into each category
        N_count = np.sum((wind_dir_values > 315) | (wind_dir_values < 45))
        E_count = np.sum((wind_dir_values > 45) & (wind_dir_values < 135))
        S_count = np.sum((wind_dir_values > 135) & (wind_dir_values < 225))
        W_count = np.sum((wind_dir_values > 225) & (wind_dir_values < 315))

        # Compute percentage of values in each category
        N_ratio = N_count / total_values
        E_ratio = E_count / total_values
        S_ratio = S_count / total_values
        W_ratio = W_count / total_values

        # Check if any category reaches the 60% threshold
        if N_ratio >= sort:
            N_totdata_list.append(array)
        elif E_ratio >= sort:
            E_totdata_list.append(array)
        elif S_ratio >= sort:
            S_totdata_list.append(array)
        elif W_ratio >= sort:
            W_totdata_list.append(array)
        else:
            Non_totdata_list.append(array)  # If none reach 50%, add to non-directional

        # If upper and lower limits are provided, filter based on them
        if upperlim is not False and lowerlim is not False:
            valid_count = np.sum((wind_dir_values > lowerlim) & (wind_dir_values < upperlim))
            valid_ratio = valid_count / total_values
            if valid_ratio >= sort:
                personalized_totdata_list.append(array)
            
    # Pie Chart Visualization
    if pie:
        lenN = len(N_totdata_list) 
        lenE = len(E_totdata_list)
        lenS = len(S_totdata_list)
        lenW = len(W_totdata_list)
        lenNon = len(Non_totdata_list)
        
        totlen = lenN + lenE + lenS + lenW + lenNon
        
        partN = len(N_totdata_list) / totlen
        partE = len(E_totdata_list) / totlen
        partS = len(S_totdata_list) / totlen
        partW = len(W_totdata_list) / totlen
        partNon = len(Non_totdata_list) / totlen

        # Prepare data for the pie chart
        sizes = [partN, partE, partS, partW, partNon]
        labels = ["North", "East", "South", "West", "No direction"]
        colors = ["royalblue", "tomato", "seagreen", "gold", "gray"]
        colors = [mcolors.to_rgba(c, alpha=0.7) for c in colors]

        # Plot the pie chart
        plt.figure(figsize=(5, 5))
        wedges, _, _ = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                               startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        
        # Equal aspect ratio ensures that pie chart is drawn as a circle
        plt.axis('equal')
        
        # Title
        plt.title('Distribution of Wind Directions', fontsize=14)

        if save:
            plt.savefig("BachelorThesis/Figures/PieChart.pdf", bbox_inches="tight")
        plt.show()
        
        # Print Summary
        print(f'It is important to note that {round(100 * partN,1)}% of the winds came from the north, {round(100 * partE,1)}% from the east, {round(100 * partS,1)}% from the south, {round(100 * partW,1)}% from the west, and {round(100 * partNon,1)}% from no specific direction.')        
    
    if upperlim:
        return personalized_totdata_list

    return N_totdata_list, E_totdata_list, S_totdata_list, W_totdata_list, Non_totdata_list  


def plot_dir_mean(dir_totdata_list, daystoplot, minpoints=8, place='',
                  labels=["North", "East", "South", "West", "No direction"], 
                  pm_mean=False, pm_sigma=False, save=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each wind direction category in subplots.
    Only non-empty wind directions are plotted dynamically.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold", "green"]  # Colors mapped to N, E, S, W, non

    # Filter out empty wind directions
    valid_data = [(totdata_list, label, color) for totdata_list, label, color in 
                  zip(dir_totdata_list, labels, colors) if len(totdata_list) > 0]
    
    # Create dynamic subplots based on available data
    fig, axes = plt.subplots(len(valid_data), 1, figsize=(8,  4* len(valid_data)), sharex=True, sharey=True)
    fig.tight_layout(pad=5.0)
    fig.suptitle(f'Mean Concentration of PM2.5, {place}')

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
        if PM_array.shape[0] == 0 or np.isnan(PM_array).all():
            mean = np.full(timelen, np.nan)
            sigma = np.full(timelen, np.nan)
        else:
            with np.errstate(all='ignore'):
                mean = np.nanmean(PM_array, axis=0)
                sigma = np.nanstd(PM_array, axis=0, ddof=0)
        t = np.arange(timelen)/24  # Convert time to days

        # Check if we have enough valid data points at the end
        valid_counts_per_hour = np.sum(~np.isnan(PM_array), axis=0)
        valid_indices = np.where(valid_counts_per_hour >= minpoints)[0]

        # We are looking for the maximal hour for which we have the minimum allowed datasets
        if len(valid_indices) == 0:
            non_nan_indices = np.where(~np.isnan(PM_array))[0]
            if len(non_nan_indices) == 0:
                continue  # Skip if no valid data
            hmax = np.max(non_nan_indices)
        else:
            hmax = valid_indices[-1]
            
        # You can also plot the Mean Standard, if wanted to
        if pm_mean:
          ax.plot(t, pm_mean+t*0, label='Mean during no blocking', c='gray')
          ax.fill_between(t,  pm_mean+t*0 +  pm_sigma+t*0,  pm_mean+t*0 - pm_sigma, alpha=0.4, color='gray') 

        # Plot the mean and confidence interval
        ax.plot(t[:hmax + 1], mean[:hmax + 1], label=f'Mean during {label} blocking', color=color)
        ax.fill_between(t[:hmax + 1], mean[:hmax + 1] + sigma[:hmax + 1], 
                        mean[:hmax + 1] - sigma[:hmax + 1], alpha=0.3, color=color)
        ax.plot(t, t*0+25, label='EU annual mean limit', c='r', linestyle='--')

        

        # Set y-axis integer ticks and grid lines
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        ax.set_title(f'Direction: {label}')
        ax.set_ylabel('PM2.5 [µg/m³]')
        ax.set_yticks(np.arange(0, 41, 5))  
        ax.legend()

    axes[-1].set_xlabel('Time from start of blocking (days)')
    fig.tight_layout()

    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_dir_{place}.pdf")
    plt.show()



start_time = time.time()

location = 'Malmö' #    <----- THIS CAN BE CHANGED

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


block_datafile = pd.concat(blocking_list, ignore_index=True)
PM_without_blocking = PM_data[~PM_data['datetime_start'].isin(block_datafile['datetime'])]

pm_mean = np.nanmean(np.array(PM_without_blocking['pm2.5']))
pm_sigma = np.nanstd(np.array(PM_without_blocking['pm2.5']))

print(f'mean particle concentration is {np.round(pm_mean,1)} ± {np.round(pm_sigma,1)} µg/m³')

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


#%%


plot_mean(totdata_list, daystoplot=14, minpoints=8, 
          place=location, save=False, info=False,
          pm_mean=pm_mean, pm_sigma=pm_sigma)



#%%

dir_totdata_list = sort_wind_dir(totdata_list, pie=True, save=False, sort=0.5)



plot_dir_mean(dir_totdata_list, daystoplot=14, minpoints=8, 
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



