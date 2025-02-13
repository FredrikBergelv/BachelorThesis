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


def plot_mean(totdata_list, daystoplot, save=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour.
    """
    timelen = int(24 * daystoplot) # Extarct the length we want to plot in hours
    
    # Create an array to store the sum of PM2.5 for each hour
    PM_sum_array = np.zeros(timelen)
    
    # Create another array to store all the PM2.5 values
    PM_array = [[0] * timelen for _ in range(len(totdata_list))]
    
    # Iterate through each blocking event and add the PM2.5 arrays
    for blocking in range(len(totdata_list)):
        array = totdata_list[blocking]
        if (len(array[2]) - timelen) < 0:
            raise ValueError("There is a hole in the data, check the cover parameter")
        PM_array[blocking] = array[2]
    
    # Calculate the sum of PM2.5 for each hour
    for hour in range(timelen):
        total_pm = 0
        for blocking in range(len(PM_array)):
            total_pm += PM_array[blocking][hour]
        PM_sum_array[hour] = total_pm / len(PM_array)
    
    # Apply Savitzky-Golay filter
    PM_sum_array_smooth = savgol_filter(PM_sum_array, window_length=24, polyorder=2)  # Adjust values as needed
    
    plt.figure(figsize=(8, 5))
    plt.title(f'Mean concentration of PM2.5 during first {daystoplot} days')
    plt.plot(range(len(PM_sum_array)), PM_sum_array, label='Mean Data')
    plt.plot(range(len(PM_sum_array)), PM_sum_array_smooth, label='Savitzky-Golay Filtered')
    plt.xlabel('Time from start of blocking (h)')
    plt.ylabel('Mean Concentration [PM2.5 (µg/m³)]')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig("BachelorThesis/Figures/Meanplot.pdf")
    plt.show()


start_time = time.time()

PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_data = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)


rain_data = read.get_rain_data(Hörby_rain)
pres_data = read.get_pressure_data(Helsingborg_pressure)

days = 5

SMHI_block_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit=1015, 
                                     duration_limit=days, 
                                     rain_limit=0.5,
                                     info=False)

totdata_list = read.array_blocking_list(PM_data, SMHI_block_list, 
                                        cover=1, info=True)

plot_mean(totdata_list, daystoplot=days, save=True)

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
