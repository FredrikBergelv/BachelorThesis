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


def array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, SMHI_block_list, 
                              only_titles=False, info=False):
    """
    This function takes in the particle data, wind data and the pressure blocking data
    It returns a list of arrays for each blocking period with wind, pressure, PM2.5
    To get the array do list[i]=array"
    array[0]=hours, 
    array[1]=pressure, 
    array[2]=pm2.5, 
    array[3]=wind dir,  
    array[4]=wind speed, 
    array[5]=temperature,
    array[6]=rain
    """
    totdata_list = []
    title_list = []
    counter = 0  #Counts number of plots we remove

    # Loop through each blocking period
    for i in range(len(SMHI_block_list)):
        block_data = SMHI_block_list[i]
        
        # Plotting the pressure data against the datetime for different locations
        start_time = block_data['datetime'].min()
        end_time = block_data['datetime'].max()
        
        # Filter all datasets to the overlapping time range
        PM_data_trimmed = PM_data[(PM_data['datetime_start'] >= start_time) & (PM_data['datetime_end'] <= end_time)]
        wind_data_trimmed = wind_data[(wind_data['datetime'] >= start_time) & (wind_data['datetime'] <= end_time)]
        temp_data_trimmed = temp_data[(temp_data['datetime'] >= start_time) & (temp_data['datetime'] <= end_time)]
        rain_data_trimmed = rain_data[(rain_data['datetime'] >= start_time) & (rain_data['datetime'] <= end_time)]

        # Drop rows with NaN in the PM2.5 column
        PM_data_trimmed = PM_data_trimmed.dropna(subset=['pm2.5'])
        
        # If PM_data is empty skip
        if PM_data_trimmed.empty:
            counter = counter + 1 
            continue
        
        # Since data is taken every hour, ensure that we have enough data
        expected_data = (end_time - start_time).total_seconds() / 3600  # Blocking duration in hours
        actual_data = len(PM_data_trimmed) # Data coverge of PM_data in hours
        coverage = actual_data/expected_data
        if coverage < 0.9:
            counter = counter + 1 
            continue
         
         # Store the dates if we want later on
        if only_titles == True:
                 title_list.append(f'Data from {start_time} to {end_time}') 
                 
                 
        # Merge PM_data_trimmed with block_data
        combined_data = pd.merge_asof(
            PM_data_trimmed,
            block_data,
            left_on='datetime_start',
            right_on='datetime',
            direction='nearest'
        )

        # Merge the result with wind_data_trimmed
        combined_data = pd.merge_asof(
            combined_data,
            wind_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )
        
        combined_data = pd.merge_asof(
            combined_data,
            temp_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )
        
        combined_data = pd.merge_asof(
            combined_data,
            rain_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )[['datetime', 'pressure', 'pm2.5', 'dir', 'speed', 'temp', 'rain' ]]
          
        totdata_list.append(combined_data)
                
    # Store all blocking values in arrays
    array_list = []
     
    # loop trhrough entire list. 
    for i in range(len(totdata_list)):
             datafile = totdata_list[i]
             
             # convert everything to arrays
             array = np.zeros((7,len(datafile)))
            
            # Since the data is for every hour the index is the hour since start
             for hour in range(len(datafile)):
                    array[0,hour], array[1,hour], array[2,hour] = hour, datafile['pressure'][hour], datafile['pm2.5'][hour]
                    array[3,hour], array[4,hour] = datafile['dir'][hour], datafile['speed'][hour]
                    array[5,hour], array[6,hour] = datafile['temp'][hour], datafile['rain'][hour]
             array_list.append(array)
        
    if only_titles == True:
        return title_list  
    if info == True:
        print(f'From a total of {len(SMHI_block_list)} high pressure bocking periods, {counter} plots were removed due to lack of PM2.5 data')
        print(f'resuting in {len(SMHI_block_list)-counter} relevant blocking periods')
    
    return array_list # Return list of all the datafiles

def plot_extra_blocking_array(array, array_title, extrainfo=True):
    """
    Plots the blocking data with four subplots: Pressure, PM2.5, Wind Direction, and Wind Speed.
    """
    
    time = array[0]/24
    pressure = array[1]   
    pm25 = array[2]
    wind_dir = array[3]
    wind_speed = array[4]
    temp = array[5]
    rain = array[6]
    
    # Create the figure and subplots    
    if extrainfo == False: 
        # Create the figure and subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        fig.suptitle(array_title, fontsize=16)
        
        # Plot Pressure
        axs[0].plot(time, pressure, label='Air Pressure', color='blue')
        axs[0].set_ylabel('Air Pressure (hPa)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM2.5 (µg/m³)')
        axs[1].legend()
        axs[1].set_ylim(0,60)
        axs[1].grid(True)
        axs[1].set_xlabel('Time from start of blocking period (days)')
    else:
        # Create the figure and subplots
        fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(array_title, fontsize=16)
        
        # Plot Pressure
        axs[0].plot(time, pressure, label='Air Pressure', color='blue')
        axs[0].set_ylabel('Air Pressure (hPa)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM2.5 (µg/m³)')
        axs[1].legend()
        axs[1].set_ylim(0,60)
        axs[1].grid(True)
        
        # Plot Wind Direction
        axs[2].scatter(time, wind_dir, label='Wind Direction', color='orange', s=7)
        axs[2].set_ylabel('Wind Direction (°)')
        axs[2].legend()
        axs[2].set_yticks([0, 90, 180, 270, 365])
        axs[2].set_ylim(0,365)
        axs[2].grid(True)
        
        # Plot Wind Speed
        axs[3].plot(time, wind_speed, label='Wind Speed', color='teal')
        axs[3].set_ylabel('Wind Speed (m/s)')
        axs[3].set_ylim(0,10)
        axs[3].legend()
        axs[3].grid(True)
        
        # Plot temp
        axs[4].plot(time, temp, label='Temperatre', color='red')
        axs[4].set_ylabel('Temperature (°C)')
        axs[4].legend()
        axs[4].grid(True)
        
        # Plot rain
        axs[5].plot(time, rain, label='Rain', color='darkblue')
        axs[5].set_ylabel('Rainfall (mm)')
        axs[5].set_xlabel('Time from start of blocking period (days)')
        axs[5].legend()
        axs[5].set_ylim(0,0.5)
        axs[5].grid(True)
        
        plt.show()

def extra_blocking_list(PM_data, wind_data, temp_data, rain_data, SMHI_block_list, 
                        info=False):
    """
    This function takes in the particle data, wind data and the pressure blocking data
    It returns a list of arrays for each blocking period with wind, pressure, PM2.5
    To get the array do list[i]=array"
    array[0]=hours, 
    array[1]=pressure, 
    array[2]=pm2.5, 
    array[3]=wind dir,  
    array[4]=wind speed, 
    array[5]=temperature,
    array[6]=rain
    """
    totdata_list = []
    counter = 0  #Counts number of plots we remove

    # Loop through each blocking period
    for i in range(len(SMHI_block_list)):
        block_data = SMHI_block_list[i]
        
        # Plotting the pressure data against the datetime for different locations
        start_time = block_data['datetime'].min()
        end_time = block_data['datetime'].max()
        
        # Filter all datasets to the overlapping time range
        PM_data_trimmed = PM_data[(PM_data['datetime_start'] >= start_time) & (PM_data['datetime_end'] <= end_time)]
        wind_data_trimmed = wind_data[(wind_data['datetime'] >= start_time) & (wind_data['datetime'] <= end_time)]
        temp_data_trimmed = temp_data[(temp_data['datetime'] >= start_time) & (temp_data['datetime'] <= end_time)]
        rain_data_trimmed = rain_data[(rain_data['datetime'] >= start_time) & (rain_data['datetime'] <= end_time)]

        # Drop rows with NaN in the PM2.5 column
        PM_data_trimmed = PM_data_trimmed.dropna(subset=['pm2.5'])
        
        # If PM_data is empty skip
        if PM_data_trimmed.empty:
            counter = counter + 1 
            continue
        
        # Since data is taken every hour, ensure that we have enough data
        expected_data = (end_time - start_time).total_seconds() / 3600  # Blocking duration in hours
        actual_data = len(PM_data_trimmed) # Data coverge of PM_data in hours
        coverage = actual_data/expected_data
        if coverage < 0.9:
            counter = counter + 1 
            continue
                 
        # Merge PM_data_trimmed with block_data
        combined_data = pd.merge_asof(
            PM_data_trimmed,
            block_data,
            left_on='datetime_start',
            right_on='datetime',
            direction='nearest'
        )

        # Merge the result with wind_data_trimmed
        combined_data = pd.merge_asof(
            combined_data,
            wind_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )
        
        combined_data = pd.merge_asof(
            combined_data,
            temp_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )
        
        combined_data = pd.merge_asof(
            combined_data,
            rain_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )[['datetime', 'pressure', 'pm2.5', 'dir', 'speed', 'temp', 'rain' ]]
          
        totdata_list.append(combined_data)
        
    if info == True:
        print(f'From a total of {len(SMHI_block_list)} high pressure bocking periods, {counter} plots were removed due to lack of PM2.5 data')
        print(f'resuting in {len(SMHI_block_list)-counter} relevant blocking periods')
    
    return totdata_list # Return list of all the datafiles

def plot_extra_blocking(data, extrainfo=True):
    """
    Plots the blocking data with four subplots: Pressure, PM2.5, Wind Direction, and Wind Speed.
    """
    
    time = data['datetime']
    pressure = data['pressure'] 
    pm25 = data['pm2.5']
    wind_dir = data['dir']
    wind_speed = data['speed']
    temp = data['temp']
    rain = data['rain']
    
    title = f'Plot from {min(time)} to {max(time)}'
    
    # Create the figure and subplots    
    if extrainfo == False: 
        # Create the figure and subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        fig.suptitle(title, fontsize=16)
        
        # Plot Pressure
        axs[0].plot(time, pressure, label='Air Pressure', color='blue')
        axs[0].set_ylabel('Air Pressure (hPa)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM2.5 (µg/m³)')
        axs[1].legend()
        axs[1].set_ylim(0,60)
        axs[1].grid(True)
        axs[1].set_xlabel('Time from start of blocking period (days)')
    else:
        # Create the figure and subplots
        fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(title, fontsize=16)
        
        # Plot Pressure
        axs[0].plot(time, pressure, label='Air Pressure', color='blue')
        axs[0].set_ylabel('Air Pressure (hPa)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM2.5 (µg/m³)')
        axs[1].legend()
        #axs[1].set_ylim(0,60)
        axs[1].grid(True)
        
        # Plot Wind Direction
        axs[2].scatter(time, wind_dir, label='Wind Direction', color='orange', s=7)
        axs[2].set_ylabel('Wind Direction (°)')
        axs[2].legend()
        axs[2].set_yticks([0, 90, 180, 270, 365])
        axs[2].set_ylim(0,365)
        axs[2].grid(True)
        
        # Plot Wind Speed
        axs[3].plot(time, wind_speed, label='Wind Speed', color='teal')
        axs[3].set_ylabel('Wind Speed (m/s)')
        axs[3].set_ylim(0,10)
        axs[3].legend()
        axs[3].grid(True)
        
        # Plot temp
        axs[4].plot(time, temp, label='Temperatre', color='red')
        axs[4].set_ylabel('Temperature (°C)')
        axs[4].legend()
        axs[4].grid(True)
        
        # Plot rain
        axs[5].plot(time, rain, label='Rain', color='darkblue')
        axs[5].set_ylabel('Rainfall (mm)')
        axs[5].set_xlabel('Time from start of blocking period (days)')
        axs[5].legend()
        axs[5].set_ylim(0,2)
        axs[5].grid(True)
        
        plt.show()
     




PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_data = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)

#PM_data = read.get_pm_data(Vavihill_O3, True)

wind_data = read.get_wind_data(Hörby_wind)
temp_data = read.get_temp_data(Hörby_temperature)
rain_data = read.get_rain_data(Hörby_rain)
pres_data = read.get_pressure_data(Helsingborg_pressure)


SMHI_block_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit = 1015, 
                                     duration_limit = 5, 
                                     rain_limit = 0.5,
                                     info=False)


#%%
totdata_list = extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                         SMHI_block_list, info=True)

for i in range(len(totdata_list)):
    data = totdata_list[i]
    plot_extra_blocking(data)
    if i > 30:
        raise ValueError(f"Showing {len(totdata_list)} graphs is too many!")

#%%

totdata_list = array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                         SMHI_block_list, info=True)

titles = array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, 
                                   SMHI_block_list, only_titles=True)

for i in range(len(totdata_list)):
    array = totdata_list[i]
    array_tiltle = titles[i]
    #plot_extra_blocking_array(array, array_tiltle, extrainfo=True)
    

