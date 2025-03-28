#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:04:20 2025

@author: Fredrik Bergelv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.colors as mcolors
from collections import defaultdict
import matplotlib.gridspec as gridspec
import re


"""
The functions down below are for reading datafiles with pandas, which gives 
dtatfiles with 'data' and 'datetime'. The functions can also plot if wanted to.
"""
def get_pressure_data(filename, 
                      plot=False):
    "This function takes a file path as argument and give you the pressuredata and datetime"
    "If one wants to, the pressure data can be plotted"
    
    # Load the necessary columns only using a regex for delimiters
    datafile = pd.read_csv(
        filename,
        sep = r'[:;/-]' , # Specify multiple delimiters
        skiprows = 11,    # Skip metadata rows
        engine='python',  # Specify engine to not get error 
        usecols = [0, 1, 2, 3, 6],  # Only load the Date, Hour, and Pressure columns
        names = ["year", "month", "day", "hour", "pressure"],  
        on_bad_lines = 'skip'             
    )
    
    # Combine the year, month, day, and hour columns into a single datetime column
    datafile['datetime'] = pd.to_datetime(datafile[['year', 'month', 'day', 'hour']])
    datafile = datafile.drop(columns=['year', 'month', 'day', 'hour'])
    
    if plot==True : 
        with open(filename, 'r') as file: # Read the file to extract the name
            
            lines = file.readlines()
            location = lines[1].strip().split(';')[0] # Here is the name of the location
            datatype = lines[4].strip().split(';')[0] # Here is the name of the data
        
        # Plotting the pressure data against the datetime
        plt.figure(figsize=(10, 6))
        plt.plot(datafile['datetime'], datafile['pressure'], label='Pressure')
        plt.xlabel('Date and Time')
        plt.ylabel('Pressure [hPa]')
        plt.title(location + ' - ' + datatype)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    return datafile # return a datafile of all the data 

def get_wind_data(filename, 
                  plot=False):
    "This function takes a file path as argument and give you the wind data and datetime"
    "If one wants to, the wind data can be plotted"
    
    # Load the necessary columns only using a regex for delimiters
    datafile = pd.read_csv(
        filename,
        sep = r'[:;/-]' , # Specify multiple delimiters
        skiprows = 15,    # Skip metadata rows
        engine='python',  # Specify engine to not get error 
        usecols = [0, 1, 2, 3, 6, 8],  # Only load the Date, Hour, and Pressure columns
        names = ["year", "month", "day", "hour", "dir", "speed"],  
        on_bad_lines = 'skip'             
        )
    
    # Combine the year, month, day, and hour columns into a single datetime column
    datafile['datetime'] = pd.to_datetime(datafile[['year', 'month', 'day', 'hour']])
    datafile = datafile.drop(columns=['year', 'month', 'day', 'hour'])
    
    if plot==True : 
        with open(filename, 'r') as file: # Read the file to extract the name
            
            lines = file.readlines()
            location = lines[1].strip().split(';')[0] # Here is the name of the location
            datatype = lines[4].strip().split(';')[0] # Here is the name of the data
        
        # Plotting the pressure data against the datetime
        plt.figure(figsize=(10, 6))
        plt.scatter(datafile['datetime'], datafile['dir'], label='wind direction', 
                    c='orange', s=7)
        plt.scatter(datafile['datetime'], datafile['speed'], label='wind speed',
                    c='teal')
        plt.xlabel('Date and Time')
        plt.ylabel('direction [degrees]')
        plt.title(location + ' - ' + datatype)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
    return datafile # return a datafile of all the data 

def get_rain_data(filename, 
                  plot=False):
    "This function takes a file path as argument and give you the wind data and datetime"
    "If one wants to, the wind data can be plotted"
    
    # Load the necessary columns only using a regex for delimiters
    datafile = pd.read_csv(
        filename,
        sep = r'[:;/-]' , # Specify multiple delimiters
        skiprows = 14,    # Skip metadata rows
        engine='python',  # Specify engine to not get error 
        usecols = [0, 1, 2, 3, 6],  # Only load the Date, Hour, and Pressure columns
        names = ["year", "month", "day", "hour", "rain"],  
        on_bad_lines = 'skip'             
        )
    
    # Combine the year, month, day, and hour columns into a single datetime column
    datafile['datetime'] = pd.to_datetime(datafile[['year', 'month', 'day', 'hour']])
    datafile = datafile.drop(columns=['year', 'month', 'day', 'hour'])
    
    if plot==True : 
        with open(filename, 'r') as file: # Read the file to extract the name
            
            lines = file.readlines()
            location = lines[1].strip().split(';')[0] # Here is the name of the location
            datatype = lines[4].strip().split(';')[0] # Here is the name of the data
        
        # Plotting the pressure data against the datetime
        plt.figure(figsize=(10, 6))
        plt.plot(datafile['datetime'], datafile['rain'], label='rain')
        plt.xlabel('Date and Time')
        plt.ylabel('Rainfall [mm]')
        plt.title(location + ' - ' + datatype)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
    return datafile # return a datafile of all the data 

def get_daily_rain_data(filename, plot=False):
    """
    Extracts daily rainfall data from a CSV file.
    
    Parameters:
        filename (str): Path to the CSV file.
        plot (bool): If True, plots the rainfall data.

    Returns:
        pd.DataFrame: A DataFrame containing datetime and rain data.
    """

    # Load the necessary columns only, handling multiple delimiters
    datafile = pd.read_csv(
        filename,
        delimiter=r'[:;/-]',  # Handle multiple delimiters
        skiprows=14,  # Skip metadata rows
        engine='python',  # Prevent parsing errors
        usecols=[12, 11, 10, 13],  # Load specific columns (year, month, day, rain)
        names=["year", "month", "day", "rain"],  
        on_bad_lines='skip'
    )
    
    # Combine year, month, and day columns into a single datetime column
    datafile['datetime'] = pd.to_datetime(datafile[['year', 'month', 'day']], errors='coerce')
    
    datafile['rain']=datafile['rain']/24
    
    # Drop unnecessary columns
    datafile.drop(columns=['year', 'month', 'day'], inplace=True)
    
    # Plot the data if requested
    if plot:
        with open(filename, 'r') as file:  
            lines = file.readlines()
            location = lines[1].strip().split(';')[0]  # Extract location name
            datatype = lines[4].strip().split(';')[0]  # Extract data type
        
        # Plot rainfall data
        plt.figure(figsize=(10, 6))
        plt.plot(datafile['datetime'], datafile['rain'], label='Rainfall (mm)', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.title(f"{location} - {datatype}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return datafile

def get_temp_data(filename, 
                  plot=False):
    "This function takes a file path as argument and give you the temp data and datetime"
    "If one wants to, the temp data can be plotted"
    
     # Load the necessary columns only using a regex for delimiters
    datafile = pd.read_csv(
         filename,
         sep = r'[:;/]' , # Specify multiple delimiters
         skiprows = 11,    # Skip metadata rows
         engine='python',  # Specify engine to not get error 
         usecols = [0, 1, 4],  # Only load the Date, Hour, and Temp columns
         names = ["year-month-day", "hour", "temp"],  
         on_bad_lines = 'skip'             
         )
    # Fix the dtaetime column
    datafile['datetime'] = pd.to_datetime(
        datafile['year-month-day'] + ' ' + datafile['hour'].astype(str), 
        format='%Y-%m-%d %H',  # Explicitly specify the format
        errors='coerce')

   
   # Drop the now redundant columns
    datafile = datafile.drop(columns=['year-month-day', 'hour'])
    
    if plot==True : 
        with open(filename, 'r') as file: # Read the file to extract the name
            
            lines = file.readlines()
            location = lines[1].strip().split(';')[0] # Here is the name of the location
            datatype = lines[4].strip().split(';')[0] # Here is the name of the data
        
        # Plotting the pressure data against the datetime
        plt.figure(figsize=(10, 6))
        plt.plot(datafile['datetime'], datafile['temp'], label='temperature')
        plt.xlabel('Date and Time')
        plt.ylabel('Temperature [degrees]')
        plt.title(location + ' - ' + datatype)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
    return datafile # return a datafile of all the data 
    
def get_pm_data(filename, 
                plot=False):
    """
    This function takes a file path as an argument and provides the PM2.5 data and datetime.
    Optionally, it can also plot the data.
    """
    # Load the necessary columns only using a regex for delimiters
    datafile = pd.read_csv(
        filename,
        sep=';',  
        skiprows = 28,  # Skip metadata rows
        engine='python',  # Specify engine to avoid errors
        usecols=[0, 1, 2],  # Only load the start date, end date and PM2.5 columns
        names=["datetime_s", "datetime_e", "pm2.5"],  
        on_bad_lines='skip'
        )
    # Convert datetime_s and datetime_e to datetime format
    datafile['datetime_start'] = pd.to_datetime(datafile['datetime_s'])
    datafile['datetime_end'] = pd.to_datetime(datafile['datetime_s'])
    datafile = datafile.drop(columns=['datetime_s', 'datetime_e'])
    if plot == True:
        with open(filename, 'r') as file: # Read the file to extract the name
            lines = file.readlines()
            location = lines[4].strip().split(';')[0]# Here is the name of the data
            
        # Plotting the PM2.5 data against datetime
        plt.figure(figsize=(10, 6))
        plt.plot(datafile['datetime_start'], datafile['pm2.5'], label='PM2.5', color='red')
        plt.xlabel('Date and Time')
        plt.ylabel('Concentration [PM2.5 (µg/m³)]')
        plt.title(f'{location[1:]} - Concentration of PM 2.5')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
    return datafile  # Return the cleaned data

"""
This function takes the filepath to a datafile and plots the yearly mean.
"""
def yearly_histogram(data, datatype, location=False, save=False):
    """
    This function takes a PM filepath 
    """
    if not location:
        location = data.replace('.csv', '')
    
    # Extract datat depending on datattpe
    if datatype == 'pm':
        data = get_pm_data(data)
    elif datatype == 'pres':
        data = get_pressure_data(data)
    elif datatype == 'temp':
        data = get_temp_data(data)
    elif datatype == 'rain':
        data = get_rain_data(data)
    else:
        raise ValueError(f'{datatype} is not valid datatype. Must be pm, pres, temp, rain. ')
        
    
    # Extract year and filter data from 2021 onwards
    if datatype == 'pm':
        data['year'] = data['datetime_start'].dt.year
    else :
        data['year'] = data['datetime'].dt.year

    
    # Calculate mean and standard deviation per year
    if datatype == 'pm':
        yearly_stats = data.groupby('year')['pm2.5'].agg(['mean', 'std'])
    elif datatype == 'pres':
        yearly_stats = data.groupby('year')['pressure'].agg(['mean', 'std'])
    elif datatype == 'temp':
        yearly_stats = data.groupby('year')['temp'].agg(['mean', 'std'])
    elif datatype == 'rain':
        yearly_stats = data.groupby('year')['rain'].agg(['mean', 'std'])
    
    # Extract values for plotting
    years = yearly_stats.index
    means = yearly_stats['mean']
    sigmas = yearly_stats['std']
    
    # Plot yearly mean PM2.5 concentrations with standard deviation as error bars
    plt.figure(figsize=(10, 6))
    plt.bar(years, means, yerr=sigmas, capsize=5, color='red', edgecolor='black', alpha=0.7)
    plt.xlabel('Year')
    
    if datatype == 'pm':
        plt.ylabel('Mean PM2.5 Concentration (µg/m³)')
        plt.title(f'Yearly Mean PM2.5 Concentration at {location}')
    elif datatype == 'pres':
        plt.ylabel('Mean Air Pressure (hPa)')
        plt.title(f'Yearly Mean Air Pressure at {location}')
        plt.ylim(975,1050)
    elif datatype == 'temp':
        plt.ylabel('Mean Temperature (°C)')
        plt.title(f'Yearly Mean temperature at {location}')
    elif datatype == 'rain':
        plt.ylabel('Mean Hourl Rainfall (mm)')
        plt.title(f'Yearly Mean Hourl Rainfall at {location}')
    
    plt.xticks(years, rotation=45)
    plt.xticks(years[::20]) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save:
        plt.savefig(f"BachelorThesis/Figures/yearly_{datatype}_{location}.pdf")
    plt.show()

"""
The functions are for extracting the blocking period from pressure data
and rain data. 
"""
def find_blocking(pres_data, rain_data, pressure_limit, duration_limit, 
                            rain_limit, plot=False, info=False):
    """
    This function takes a pandas datafile as argument and gives you the periods of blocking, 
    taking rainfall data into account (only hours with rainfall under the limit are included).
    If one wants to, the blocking data for each period can be plotted.
    """
    # WE MERGE THE DATASETS
    start_date = max(min(pres_data['datetime']), min(rain_data['datetime']))
    end_date = min(max(pres_data['datetime']), max(rain_data['datetime']))
    
    pres_data = pres_data[(pres_data['datetime'] >= start_date) & (pres_data['datetime'] <= end_date)]
    rain_data = rain_data[(rain_data['datetime'] >= start_date) & (rain_data['datetime'] <= end_date)]
    
    # Sort both datasets by datetime
    pres_data = pres_data.sort_values('datetime')
    rain_data = rain_data.sort_values('datetime')
    
    # Merge dataframes using merge_asof
    data = pd.merge_asof(pres_data, rain_data, on='datetime', direction='nearest')
    
    # Drop empty rows
    data.dropna(subset=['rain'])
    
    # Identify where we have high pressure and no rain, add new column
    data['highp'] = (data['pressure'] > pressure_limit) & (data['rain'] < rain_limit) 
    
    # Identify if next value is not high pressure or not (shift)(data['rain'] < rain_limit) 
    # If the next value is not the same, add the value of True(1) cumulative 
    # This gives a unique streak_id for each streak depending on the limit
    data['streak_id'] = (data['highp'] != data['highp'].shift()).cumsum()
    
    # Group by streak_id and calculate the duration of each streak
    streaks = data.groupby('streak_id').agg(
        start_date = ('datetime', 'first'),  # start is the first datetime in each id
        end_date = ('datetime', 'last'),     # end is the last datetime in each id
        duration_hours = ('datetime', lambda date: (date.max() - date.min()).total_seconds()/3600 + 1), # Calculate the duration, in hours
        highp = ('highp', 'max')             # Since all highp first/max, all give the same
    )
    # Filter for streaks with high pressure lasting at least the right number of days
    # We aslo filter the streaks which are over 100 days, due to problems whith combining data
    blocking = streaks[(streaks['highp'] == True) & (streaks['duration_hours']/24 >= duration_limit) & (streaks['duration_hours']/24 < 100)]
    blocking = blocking.drop(columns=['highp'])
    
    datalist = []  # We want to return a list of all the high pressure periods
    for index, row in blocking.iterrows():
        start_date, end_date = row['start_date'], row['end_date']  # Extract the datetime
        # Filter the data for this specific streak 
        streak_data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
        streak_data = streak_data.drop(columns=['highp', 'streak_id', "rain"])

        # Plot the streak if plot is True
        if plot:
            if len(blocking) > 50:
                raise ValueError(f"Showing {len(blocking)} graphs is too many!")
            plt.figure(figsize=(10, 6))
            plt.plot(streak_data['datetime'], streak_data['pressure'], label='Pressure')
            plt.xlabel('Date and Time')
            plt.ylabel('Pressure [hPa]')
            plt.title(f'High Pressure blocking ({start_date} to {end_date})')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()
        datalist.append(streak_data)
    
    if info:
        print(f'A total of {len(datalist)} high pressure blocking events were found between {min(data["datetime"].dt.date)} and {max(data["datetime"].dt.date)}')
        
    return datalist  # Return a list of all the blocking data


"""
The function is for extracting a certian period from <start> to <end>
and making that period into an array with all the data: pres, wind, temp, pm, rain. 
"""
def array_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                       start_time, end_time, info=False, plot=False, save=False):
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
    # Convert to dattime format from string
    start_time  = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
        
    # Filter all datasets to the overlapping time range
    PM_data_trimmed = PM_data[(PM_data['datetime_start'] >= start_time) & (PM_data['datetime_end'] <= end_time)]
    wind_data_trimmed = wind_data[(wind_data['datetime'] >= start_time) & (wind_data['datetime'] <= end_time)]
    temp_data_trimmed = temp_data[(temp_data['datetime'] >= start_time) & (temp_data['datetime'] <= end_time)]
    rain_data_trimmed = rain_data[(rain_data['datetime'] >= start_time) & (rain_data['datetime'] <= end_time)]
    pressure_data_trimmed = pressure_data[(pressure_data['datetime'] >= start_time) & (pressure_data['datetime'] <= end_time)]
    
    # Drop rows with NaN in the PM2.5 column
    PM_data_trimmed = PM_data_trimmed.dropna(subset=['pm2.5'])
        
    # Since data is taken every hour, ensure that we have enough data
    expected_data = (end_time - start_time).total_seconds() / 3600  # Blocking duration in hours
    len_PM = len(PM_data_trimmed) # Data coverge of PM_data in hours
    len_pr = len(pressure_data_trimmed)
    len_wi = len(wind_data_trimmed)
    len_te = len(temp_data_trimmed)
    len_ra = len(rain_data_trimmed)
        
    if len_PM/expected_data < 0.9:
        print(f" Plot has {np.round(100*len_PM/expected_data,2)}% PM2.5 data coverage.")
    if len_pr/expected_data < 0.9:
        print(f" Plot has {np.round(100*len_pr/expected_data,2)}% pressure data coverage.")
    if len_wi/expected_data < 0.9:
        print(f" Plot has {np.round(100*len_wi/expected_data,2)}% wind data coverage.")
    if len_te/expected_data < 0.9:
        print(f" Plot has {np.round(100*len_te/expected_data,2)}% temperature data coverage.")
    if len_ra/expected_data < 0.9:
        print(f" Plot has {np.round(100*len_ra/expected_data,2)}% rain data coverage.")
    
    # Merge PM_data_trimmed with block_data
    combined_data = pd.merge_asof(
            pressure_data_trimmed,
            PM_data_trimmed,
            left_on='datetime',
            right_on='datetime_start',
            direction='nearest')
    
    # Merge the result with wind_data_trimmed
    combined_data = pd.merge_asof(
            combined_data,
            wind_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest')
        
    combined_data = pd.merge_asof(
            combined_data,
            temp_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest')
        
    combined_data = pd.merge_asof(
            combined_data,
            rain_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
            )[['datetime', 'pressure', 'pm2.5', 'dir', 'speed', 'temp', 'rain' ]]
             
    # convert everything to arrays
    array = np.zeros((7,len(combined_data)))
            
    # Since the data is for every hour the index is the hour since start
    for hour in range(len(combined_data)):
        array[0,hour] =  hour
        array[1,hour], array[2,hour] = combined_data['pressure'][hour], combined_data['pm2.5'][hour]
        array[3,hour], array[4,hour] = combined_data['dir'][hour], combined_data['speed'][hour]
        array[5,hour], array[6,hour] = combined_data['temp'][hour], combined_data['rain'][hour]       
    
    if plot==True:
        time = array[0]/24 # Convert to days
        pressure = array[1]   
        pm25 = array[2]
        wind_dir = array[3]
        wind_speed = array[4]
        temp = array[5]
        rain = array[6]
        
        title = f'Data from {start_time} to {end_time}'
        
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
        axs[3].set_ylim(0,14)
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
        axs[5].set_xlabel('Time from start of period (days)')
        axs[5].legend()
        axs[5].grid(True)
        if save:
            plt.savefig(f"BachelorThesis/Figures/plot_{start_time}_to_{end_time}.pdf")
        plt.show()
    
    return array # Return list of all the datafiles


"""
This funtion make arrays of PM2.5 and the pressure, wind, temp, rain from the 
blocking list. This gives arrays stored in lists. This is without datetime 
althogh if wanted to only the start-end date can be extraced for each list 
element instead.
"""
def array_extra_blocking_list(PM_data, wind_data, temp_data, rain_data, blocking_list, 
                              cover=0.9, only_titles=False, info=False):
    """
    This function takes in the particle data, wind, rain, temp data and the pressure blocking data
    It returns a list of arrays for each blocking period with wind, rain, temp, pressure, PM2.5
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
    for i in range(len(blocking_list)):
        block_data = blocking_list[i]
        
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
        if coverage < cover:
            counter = counter + 1 
            continue
         
         # Store the dates if we want later on
        if only_titles == True:
                 title_list.append(f'Data from {start_time} to {end_time}') 
        

        combined_data = block_data.merge(
                        PM_data_trimmed,
                        how="left",
                        left_on="datetime",
                        right_on="datetime_start"
                    )

        # Fill missing datetime_start values with the datetime from block_data
        combined_data["datetime_start"] = combined_data["datetime_start"].fillna(combined_data["datetime"])
        
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
        print(f'From a total of {len(blocking_list)} high-pressure bocking events, {counter} plots were removed due to lack of PM\textsubscript{2.5} data since a filter of {round(cover*100)}\% was used. Thus resuting in {len(blocking_list)-counter} relevant high-pressure blocking events')
    
    return array_list # Return list of all the datafiles

def plot_extra_blocking_array(array, array_title=False, extrainfo=True, save=False):
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
    
    if not array_title:
        array_title = 'Plot Showing Data During a High Pressure Blocking'
    
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
        if save:
            plt.savefig(f"BachelorThesis/Figures/{array_title}.pdf")
        plt.show()


"""
The function down below is for extracting a certian period from <start> to <end>
and plotting all the data: pres, wind, temp, pm, rain. This is also displays
when there is a blockig in the background.
"""
def plot_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                      blocking_list, start_time, end_time, tempwind_plot=True, 
                      save=False, locationsave=False):
    """
    Plot PM_data, wind_data, temp_data, rain_data, pressure_data 
    over time wit hthe datafile format. Also uses shaded parts to highlight 
    periods of high pressure blocking.
    """

    # Convert start and end time to pandas datetime
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Filter data within the specified date range
    PM_data = PM_data.rename(columns={'datetime_start': 'datetime'})
    
    # Filter all datasets
    pressure_data = pressure_data[(pressure_data['datetime'] >= start_time) & (pressure_data['datetime'] <= end_time)]
    PM_data = PM_data[(PM_data['datetime'] >= start_time) & (PM_data['datetime'] <= end_time)]
    wind_data = wind_data[(wind_data['datetime'] >= start_time) & (wind_data['datetime'] <= end_time)]
    temp_data = temp_data[(temp_data['datetime'] >= start_time) & (temp_data['datetime'] <= end_time)]
    rain_data = rain_data[(rain_data['datetime'] >= start_time) & (rain_data['datetime'] <= end_time)]
    
    # Merge datasets
    merged_data = (pressure_data
                   .merge(PM_data, on='datetime', how='outer')
                   .merge(wind_data, on='datetime', how='outer')
                   .merge(temp_data, on='datetime', how='outer')
                   .merge(rain_data, on='datetime', how='outer'))
    
    # Sort and reset index
    merged_data = merged_data.sort_values(by='datetime').reset_index(drop=True)
    
    # Extract periods from blocking_list
    periods = []
    for datafile in blocking_list:
        start = min(datafile['datetime'])
        end = max(datafile['datetime'])  # Fix: Use max instead of min
        periods.append((start, end))
        
    if tempwind_plot == True:
        size  = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
    else: 
        size  = plt.subplots(3, 1, figsize=(5, 5), sharex=True)

    # Create figure and subplots
    fig, axs = size
    fig.suptitle(f'Data from {start_time.date()} to {end_time.date()}')
    
    if locationsave:
        fig.suptitle(f'Data during {start_time.year}, {locationsave}',
                     fontsize=13, fontname='serif', x=0.55)

    # Add shaded periods to all subplots
    for ax in axs:
        for start, end in periods:
            ax.axvspan(start, end, color='gray', alpha=0.3)  # Light gray shading

    # Plot Pressure
    axs[0].plot(merged_data['datetime'], merged_data['pressure'], label='Air Pressure', color='red')
    axs[0].set_ylabel('Air Pressure (hPa)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot PM2.5
    axs[1].plot(merged_data['datetime'], merged_data['pm2.5'], label='PM2.5', color='green')
    axs[1].set_ylabel('PM2.5 (µg/m³)')
    axs[1].set_ylim(0, 60)
    axs[1].legend()
    axs[1].grid(True)
    
    n = 2
    if tempwind_plot == True:
        n=5
        
        # Plot Wind Direction
        axs[2].scatter(merged_data['datetime'], merged_data['dir'], label='Wind Direction', color='orange', s=7)
        axs[2].set_ylabel('Wind Direction (°)')
        axs[2].set_yticks([0, 90, 180, 270, 360])
        axs[2].set_ylim(0, 360)
        axs[2].legend()
        axs[2].grid(True)
    
        # Plot Wind Speed
        axs[3].plot(merged_data['datetime'], merged_data['speed'], label='Wind Speed', color='teal')
        axs[3].set_ylabel('Wind Speed (m/s)')
        axs[3].set_ylim(0, 14)
        axs[3].legend()
        axs[3].grid(True)
        

        # Plot Temperature
        axs[4].plot(merged_data['datetime'], merged_data['temp'], label='Temperature', color='maroon')
        axs[4].set_ylabel('Temperature (°C)')
        axs[4].legend()
        axs[4].grid(True)

    # Plot Rainfall
    axs[n].plot(merged_data['datetime'], merged_data['rain'], label='Rainfall', color='blue')
    axs[n].set_ylabel('Rainfall (mm)')
    
    axs[n].set_xlabel('Date')
    axs[n].legend()
    axs[n].grid(True)
    axs[n].tick_params(axis='x', rotation=45)
    axs[n].set_xlim(start_time, end_time)

    plt.tight_layout()
    if save:
        plt.savefig(f'BachelorThesis/Figures/plot_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.pdf')
    plt.show() 

    if locationsave:
        plt.savefig(f'BachelorThesis/Figures/{locationsave}_plot_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.pdf')
    plt.show()       

"""
These functions sort the totdatalists into catatgories. 
"""
def sort_wind_dir(totdata_list, upperlim=False, lowerlim=False, pie=False, save=False,
                  pieinfo=False, sort=0.5):
    """
    This function filters a list of blocking arrays by wind direction based on a 50% threshold.
    It returns five lists:
        sort_wind_dir[0] -> NE (310° to 70°)
        sort_wind_dir[1] -> SE (70° to 190°)
        sort_wind_dir[2] -> W (190° to 310°)
        sort_wind_dir[3] -> Non-directional (if no category reaches 50%)
    """
    NE_totdata_list = []
    SE_totdata_list = []
    W_totdata_list = []
    Turning_totdata_list = []
    
    personalized_totdata_list = []

    # Loop through the arrays to sort by wind direction percentage
    for array in totdata_list:
        wind_dir_values = array[3]  # Extract wind direction values
        wind_speed_values = array[4]
        
        for i, speed in enumerate(wind_speed_values):
            if speed == 0:
                wind_dir_values[i] = np.nan
         
        total_values = len(wind_dir_values)

        # Count how many values fall into each category
        NE_count = np.sum((wind_dir_values > 310) | (wind_dir_values < 70))
        SE_count = np.sum((wind_dir_values > 70) & (wind_dir_values < 190))
        W_count = np.sum((wind_dir_values > 190) & (wind_dir_values < 310))

        # Compute percentage of values in each category
        NE_ratio = NE_count / total_values
        SE_ratio = SE_count / total_values
        W_ratio = W_count / total_values
        
        # Check if any category reaches the 50% threshold
        if NE_ratio >= sort:
            NE_totdata_list.append(array)
        elif SE_ratio >= sort:
            SE_totdata_list.append(array)
        elif W_ratio >= sort:
            W_totdata_list.append(array)
        else:
            Turning_totdata_list.append(array)  # If none reach 50%, add to non-directional

        # If upper and lower limits are provided, filter based on them
        if upperlim is not False and lowerlim is not False:
            valid_count = np.sum((wind_dir_values > lowerlim) & (wind_dir_values < upperlim))
            valid_ratio = valid_count / total_values
            if valid_ratio >= sort:
                personalized_totdata_list.append(array)
            
    # Pie Chart Visualization
        lenNE = len(NE_totdata_list) 
        lenSE = len(SE_totdata_list)
        lenW = len(W_totdata_list)
        lenTurning = len(Turning_totdata_list)        
        totlen = lenNE + lenSE + lenW + lenTurning
        
        partNE = len(NE_totdata_list) / totlen
        partSE = len(SE_totdata_list) / totlen
        partW = len(W_totdata_list) / totlen
        partTurning = len(Turning_totdata_list) / totlen
        
    if pie:
        # Prepare data for the pie chart
        sizes = [partNE, partSE, partW, partTurning]
        labels = ["NE (310° to 70°)", "SE (70° to 190°)", "W (190° to 310°)", "Turning direction"]
        colors = ["royalblue", "tomato", "seagreen", "gold", "lightgray"]
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
    if pieinfo:
        print(f'It is important to note that {round(100 * partNE,1)}\% of the winds came from the Northeast (310° to 70°), {round(100 * partSE,1)}\% from the Southeast (70° to 190°), {round(100 * partW,1)}\% from the West (190° to 310°) and {round(100 * partTurning,1)}\% from no specific direction.')        
        
    
    if upperlim:
        return personalized_totdata_list

    return NE_totdata_list, SE_totdata_list, W_totdata_list, Turning_totdata_list 

def sort_season(totdata_list, totdata_list_dates, pie=False, save=False,
                  pieinfo=False, uppermonthlim=False, lowermonthlim=False):
    """
    This function filters a list of blocking arrays by season.
    It returns five lists:
        sort_season[0] -> winter
        sort_season[1] -> spring
        sort_season[2] -> summer
        sort_season[3] -> autumn
    """
    winter_totdata_list = []
    spring_totdata_list = []
    summer_totdata_list = []
    autumn_totdata_list = []
    personalized_totdata_list = []
    
    # Loop through the ziped arraylists to sort by season d
    for array, date_str in zip(totdata_list, totdata_list_dates):
        
        matches = re.findall(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        start_month = int(matches[0][1])  # Extract the month from the first date
        end_month = int(matches[1][1])    # Extract the month from the second date
        month = (start_month + end_month) / 2
        
        if month in [12, 1, 2]:
          winter_totdata_list.append(array)  # Winter
        elif month in [3, 4, 5]:
           spring_totdata_list.append(array)  # Spring
        elif month in [6, 7, 8]:
           summer_totdata_list.append(array)  # Summer
        elif month in [9, 10, 11]:
           autumn_totdata_list.append(array)  # Autumn
           
        # If upper and lower limits are provided, filter based on them
        if uppermonthlim is not False and lowermonthlim is not False:
            if month >= lowermonthlim and month <= uppermonthlim:
                personalized_totdata_list.append(array)
            
    # Pie Chart Visualization
        lenWinter = len(winter_totdata_list) 
        lenSpring = len(spring_totdata_list)
        lenSummer = len(summer_totdata_list)
        lenAutumn = len(autumn_totdata_list)
        
        totlen = lenWinter + lenSpring + lenSummer + lenAutumn 
        
        partWinter = (lenWinter) / totlen
        partSpring= (lenSpring) / totlen
        partSummer = (lenSummer) / totlen
        partAutumn = (lenAutumn) / totlen
        
    if pie:
        # Prepare data for the pie chart
        sizes = [partWinter, partSpring, partSummer, partAutumn]
        labels = ["Winter", "Spring", "Summer", "Autumn"]
        colors = ["royalblue", "seagreen", "tomato", "gold"]
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
    if pieinfo:
        print(f'It is important to note that {round(100 * partWinter,1)}\% of the blockings occurred during the winter, {round(100 * partSpring,1)}\% during the spring, {round(100 * partSummer,1)}\% during the summer and {round(100 * partAutumn,1)}\% during the autumn.')        
    if uppermonthlim is not False and lowermonthlim is not False:
        return personalized_totdata_list

    return winter_totdata_list,spring_totdata_list, summer_totdata_list, autumn_totdata_list

def sort_pressure(totdata_list, pie=False, save=False, pieinfo=False, limits=[1020, 1025]):
    """This function sorts the list of arrays into three blocking categories based on mean pressure."""
    
    low_totdata_list = []
    medium_totdata_list = []
    high_totdata_list = []
        
    # Loop through the array list and classify by pressure
    for array in totdata_list:
        pressure = array[1]  # Assuming pressure is stored in index 1
        mean_pressure = np.mean(pressure)
        
        if mean_pressure <= limits[0]:
            low_totdata_list.append(array)  # Low blocking
        elif limits[0] < mean_pressure <= limits[1]:
            medium_totdata_list.append(array)  # Medium blocking
        elif limits[1] < mean_pressure:
            high_totdata_list.append(array)  # High blocking

    # Compute blocking category distribution
    lenLow, lenMedium, lenHigh = len(low_totdata_list), len(medium_totdata_list), len(high_totdata_list)
    totlen = lenLow + lenMedium + lenHigh

    if totlen > 0:
        partLow, partMedium, partHigh = lenLow / totlen, lenMedium / totlen, lenHigh / totlen
    else:
        partLow = partMedium = partHigh = 0  # Prevent division by zero

    # Pie Chart Visualization
    if pie:
        sizes = [partLow, partMedium, partHigh]
        labels = [
            f"Low Blocking (< {limits[0]} hPa)", 
            f"Medium Blocking ({limits[0]} - {limits[1]} hPa)", 
            f"High Blocking ({limits[1]} - {limits[2]} hPa)"
        ]
        colors = ["seagreen", "gold", "tomato"]
        colors = [mcolors.to_rgba(c, alpha=0.7) for c in colors]

        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        plt.axis('equal')
        plt.title('Distribution of Blocking Categories', fontsize=14)

        if save:
            plt.savefig("BachelorThesis/Figures/BlockingPieChart.pdf", bbox_inches="tight")
        plt.show()
        
    # Print summary in a single line with explicit pressure thresholds
    if pieinfo:
        print(f'It is important to note that {round(100 * partLow,1)}\% of the blockings occurred with a mean pressure below {limits[0]} hPa {round(100 * partMedium,1)}\% occurred between {limits[0]} and {limits[1]} hPa and {round(100 * partHigh,1)}\% occurred with a mean pressure over {limits[1]}hPa.')


    return low_totdata_list, medium_totdata_list, high_totdata_list


"""
These functions use statistics to evaluate PM25 during periods of high
pressure blocking.
"""
 
def plot_mean(totdata_list, daystoplot, minpoints=8, 
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
        plt.title(f'Number of datasets at {place}',
                 fontsize=13, fontname='serif', x=0.5)
        plt.xlabel('Time from start of blocking (days)')
        plt.ylabel('Number of datasets')
        plt.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5, label='Minimum number of datasets allowed')
        plt.yticks(np.arange(0, 201, 20))   
        plt.grid()
        plt.tight_layout()
        plt.legend()
        if infosave:
            plt.savefig(f"BachelorThesis/Figures/Meanplotinfo_{place}.pdf")
        plt.show()

    # Plot everything
    plt.figure(figsize=(5, 4))
    plt.title(f'Mean concentration of PM2.5, {place}',
                 fontsize=13, fontname='serif', x=0.5)
    
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

def plot_dir_mean(dir_totdata_list, daystoplot, minpoints=8, place='',
                  labels=["NE (310° to 70°)", "SE (70° to 190°)", "W (190° to 310°)", "No Specific"], 
                  pm_mean=False, pm_sigma=False, save=False):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each wind direction category in subplots.
    Only non-empty wind directions are plotted dynamically.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold", "orange"]  # Colors mapped to NE, SE, W, Turning, non

    # Filter out empty wind directions
    valid_data = [(totdata_list, label, color) for totdata_list, label, color in 
                  zip(dir_totdata_list, labels, colors) if len(totdata_list) > 0]
    
    # Create dynamic subplots based on available data
    fig, axes = plt.subplots(len(valid_data), 1, figsize=(5, 2.5* len(valid_data)), 
                             sharex=True, sharey=True, constrained_layout=True)
    plt.suptitle(f'Mean Concentration of PM2.5, {place}\n',
             fontsize=14, fontname='serif', x=0.5)
    
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
        ax.set_ylim(0,41)
        ax.legend()

    axes[-1].set_xlabel('Time from start of blocking (days)')

    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_dir_{place}.pdf")
    plt.show()

def plot_seasonal_mean(seasonal_totdata_list, daystoplot, minpoints=8, place='',
                  labels=["Winter", "Spring", "Summer", "Autumn"], 
                  pm_mean=False, pm_sigma=False, save=False):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each season in subplots.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "seagreen", "tomato", "gold"]  # Colors mapped 

    # Filter out empty wind directions
    valid_data = [(seasonal_totdata_list, label, color) for seasonal_totdata_list, label, color in 
                  zip(seasonal_totdata_list, labels, colors) if len(seasonal_totdata_list) > 0]
    
    # Create dynamic subplots based on available data    
    fig, axes = plt.subplots(len(valid_data), 1, figsize=(5, 2.5* len(valid_data)), 
                             sharex=True, sharey=True, constrained_layout=True)
    plt.suptitle(f'Mean Concentration of PM2.5, {place}\n',
             fontsize=14, fontname='serif', x=0.5)

    if len(valid_data) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, (seasonal_totdata_list, label, color) in zip(axes, valid_data):
        
        # Create an array to store all PM2.5 values
        PM_array = np.full((len(seasonal_totdata_list), timelen), np.nan)

        # Populate PM_array with available data
        for i, array in enumerate(seasonal_totdata_list):
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

        ax.set_title(f'{label}')
        ax.set_ylabel('PM2.5 [µg/m³]')
        ax.set_yticks(np.arange(0, 41, 5))  
        ax.set_ylim(0,41)
        ax.legend()

    axes[-1].set_xlabel('Time from start of blocking (days)')

    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_seasonal_{place}.pdf")
    plt.show()

def plot_pressure_mean(seasonal_totdata_list, daystoplot, minpoints=8, place='',
                  labels=["Weaker Pressure Blocking", "Medium Pressure Blocking", "Stronger Pressure Blocking"], 
                  pm_mean=False, pm_sigma=False, save=False):
    
    """
    This function computes and plots the mean PM2.5 concentration for each hour 
    over a specified number of days, categorized by pressure-based blocking levels.
    The data is visualized in separate subplots for each blocking category.
    """
    
    timelen = int(24 * daystoplot)  # Convert days to hours
    colors = ["royalblue", "seagreen", "tomato"]  # Colors for each blocking category

    # Filter out empty categories to avoid empty subplots
    valid_data = [(seasonal_totdata_list, label, color) for seasonal_totdata_list, label, color in 
                  zip(seasonal_totdata_list, labels, colors) if len(seasonal_totdata_list) > 0]
    
    # Create subplots dynamically based on the available data categories
    fig, axes = plt.subplots(len(valid_data), 1, figsize=(5, 2.5* len(valid_data)), 
                             sharex=True, sharey=True, constrained_layout=True)
    plt.suptitle(f'Mean Concentration of PM2.5, {place}\n',
             fontsize=14, fontname='serif', x=0.5)

    # Ensure axes is always iterable even if there's only one subplot
    if len(valid_data) == 1:
        axes = [axes]  

    for ax, (seasonal_totdata_list, label, color) in zip(axes, valid_data):
        
        # Initialize an array to store PM2.5 values
        PM_array = np.full((len(seasonal_totdata_list), timelen), np.nan)

        # Populate PM_array with available PM2.5 data, avoiding indexing errors
        for i, array in enumerate(seasonal_totdata_list):
            valid_len = min(len(array[2]), timelen)  
            PM_array[i, :valid_len] = array[2][:valid_len]

        # Compute mean and standard deviation while handling NaN values
        if PM_array.shape[0] == 0 or np.isnan(PM_array).all():
            mean = np.full(timelen, np.nan)
            sigma = np.full(timelen, np.nan)
        else:
            with np.errstate(all='ignore'):  # Suppress warnings for NaNs
                mean = np.nanmean(PM_array, axis=0)
                sigma = np.nanstd(PM_array, axis=0, ddof=0)

        t = np.arange(timelen) / 24  # Convert hours to days

        # Determine valid time indices with at least the minimum required data points
        valid_counts_per_hour = np.sum(~np.isnan(PM_array), axis=0)
        valid_indices = np.where(valid_counts_per_hour >= minpoints)[0]

        # Find the last valid time index (hmax) ensuring sufficient data points
        if len(valid_indices) == 0:
            non_nan_indices = np.where(~np.isnan(PM_array))[0]
            if len(non_nan_indices) == 0:
                continue  # Skip subplot if no valid data
            hmax = np.max(non_nan_indices)
        else:
            hmax = valid_indices[-1]
            
        # Plot mean PM2.5 concentration for non-blocking conditions if provided
        if pm_mean:
            ax.plot(t, pm_mean + t * 0, label='Mean during no blocking', c='gray')
            ax.fill_between(t, pm_mean + pm_sigma, pm_mean - pm_sigma, alpha=0.4, color='gray') 

        # Plot the mean PM2.5 concentration and standard deviation for each blocking category
        ax.plot(t[:hmax + 1], mean[:hmax + 1], label=f'Mean during {label}', color=color)
        ax.fill_between(t[:hmax + 1], mean[:hmax + 1] + sigma[:hmax + 1], 
                        mean[:hmax + 1] - sigma[:hmax + 1], alpha=0.3, color=color)

        # Add a horizontal line for the EU air quality standard (25 µg/m³)
        ax.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')

        # Configure y-axis with integer ticks and grid lines
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        ax.set_title(f'{label}')
        ax.set_ylabel('PM2.5 [µg/m³]')
        ax.set_yticks(np.arange(0, 41, 5))  
        ax.set_ylim(0, 41)
        ax.legend()

    # Label the x-axis only on the last subplot
    axes[-1].set_xlabel('Time from start of blocking (days)')

    # Save the figure if requested
    if save:
        plt.savefig(f"BachelorThesis/Figures/Meanplot_pressure_{place}.pdf")
        
    plt.show()



"""
These two functions make histograms showing the frequency of blocking per yerar
"""
def plot_blockingsdays_by_year(block_list, typ, save=False):
    """We want to show the number of blockings per year"""
    
    years = [] 
    
    # Make a loop to find all the relevant years
    for data in block_list:
        start, end = min(data['datetime']), max(data['datetime'])
        
        # Find the mean date and extract the year
        year = (start + (end - start) / 2).year
        
        if year not in years:
            years.append(year)  # Add the year to the list if it's unique
    
    # Make a dictionary with years as keys and values [0, 0, 0, 0] (for winter, spring, summer, autumn)
    blocking_seasonal = {year: [0, 0, 0, 0] for year in years} 
        
    blocking_strength = {year: [0, 0, 0] for year in years} 
        
    for data in block_list:
        start, end = min(data['datetime']), max(data['datetime'])
        duration = (end - start).days
        
        # Extract the mean pressure 
        mean_pressure = np.mean(data["pressure"])            
        # Find the date and month
        date = (start + (end - start) / 2)
        month = date.month
        year = date.year
        
        # Add to the winter or summer blocking duration
        if month in [12, 1, 2]:
            blocking_seasonal[year][0] += duration  # Winter
        elif month in [3, 4, 5]:
            blocking_seasonal[year][1] += duration  # Spring
        elif month in [6, 7, 8]:
            blocking_seasonal[year][2] += duration  # Summer
        elif month in [9, 10, 11]:
            blocking_seasonal[year][3] += duration  # Autumn
            
        if mean_pressure < 1020:
            blocking_strength[year][0] += duration  # weak
        elif mean_pressure < 1025 and mean_pressure > 1020:
            blocking_strength[year][1] += duration  # medium
        elif mean_pressure > 1025:
            blocking_strength[year][2] += duration  # strong    

        
    # Remove the first and last years since they are not full years
    blocking_seasonal.pop(min(blocking_seasonal))  # Remove the first year
    blocking_seasonal.pop(max(blocking_seasonal))  # Remove the last year
    
    blocking_strength.pop(min(blocking_strength))  # Remove the first year
    blocking_strength.pop(max(blocking_strength))  # Remove the last year
    
    # Extract the data as lists
    winter = [values[0] for values in blocking_seasonal.values()]  # Winter blocking
    spring = [values[1] for values in blocking_seasonal.values()]  # Spring blocking
    summer = [values[2] for values in blocking_seasonal.values()]  # Summer blocking
    autumn = [values[3] for values in blocking_seasonal.values()]  # Autumn blocking
    
    weak = [values[0] for values in blocking_strength.values()]  # weak blocking
    medium = [values[1] for values in blocking_strength.values()]  # medium blocking
    strong = [values[2] for values in blocking_strength.values()]  # strong blocking
   
    
    total = [values[0] + values[1] + values[2] + values[3] for values in blocking_seasonal.values()]  # Total blocking days
    
    years = list(blocking_seasonal.keys())  # Years list
   
    if typ == "season":
        # Create subplots: 2 rows and 4 columns
        fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    
        # Plot for seasons (left column)
        axes[0].plot(years, winter, label="Winter", color='b', linestyle='-', marker='o')
        axes[0].set_title("Winter")
        axes[0].legend()
        axes[0].grid()
    
        axes[1].plot(years, spring, label="Spring", color='g', linestyle='-', marker='^')
        axes[1].set_title("Spring")
        axes[1].legend()
        axes[1].grid()
    
        axes[2].plot(years, summer, label="Summer", color='r', linestyle='-', marker='s')
        axes[2].set_title("Summer")
        axes[2].legend()
        axes[2].grid()
    
        axes[3].plot(years, autumn, label="Autumn", color='orange', linestyle='-', marker='d')
        axes[3].set_title("Autumn")
        axes[3].legend()
        axes[3].grid()
    
        # Set the labels and title with improved font sizes
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Days of Blocking [days]", fontsize=12)        
        # Adjust x-axis ticks and label rotation
        plt.xticks(years[::10], rotation=45)  # Show only every fourth year
        # Adjust layout for better spacing
        plt.tight_layout()
        
    if typ == "strength":
        # Create subplots: 2 rows and 4 columns
        fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
    
        # Plot for seasons (left column)
        axes[0].plot(years, weak, label="Weak", color='b', linestyle='-', marker='o')
        axes[0].set_title("Weak")
        axes[0].legend()
        axes[0].grid()
    
        axes[1].plot(years, medium, label="Medium", color='g', linestyle='-', marker='^')
        axes[1].set_title("Medium")
        axes[1].legend()
        axes[1].grid()
    
        axes[2].plot(years, strong, label="Strong", color='r', linestyle='-', marker='s')
        axes[2].set_title("Strong")
        axes[2].legend()
        axes[2].grid()
    
        # Set the labels and title with improved font sizes
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Days of Blocking [days]", fontsize=12)        
        # Adjust x-axis ticks and label rotation
        plt.xticks(years[::10], rotation=45)  # Show only every fourth year
        # Adjust layout for better spacing
        plt.tight_layout()

    if typ == "tot":
        # Create a single subplot (1 row, 1 column)
        fig, ax = plt.subplots(1, 1, figsize=(8, 2.7), sharex=True)  # Adjust the figure size
    
        # Plot the total blocking days
        ax.plot(years, total, label="Total", color='black', linestyle='-', marker='o')
        ax.set_title("Total Blocking Days Per Year")  # Corrected title
        ax.legend()
        ax.grid(True)  # Add grid for better visibility
        
        # Set labels for x and y axes
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Days of Blocking [days]", fontsize=12)
        
        # Adjust x-axis ticks and label rotation
        ax.set_xticks(years[::4])  # Show every fourth year
        ax.set_xticklabels(years[::4], rotation=45)  # Rotate x-axis labels for better readability
        
        #plt.suptitle("Number of Blocking Days Per Year ", fontsize=14, fontname='serif', x=0.5)
        plt.tight_layout()
        
    if typ == "all":
        # Create a figure
        fig = plt.figure(figsize=(9, 11))

        # Create a GridSpec for the layout
        gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 1])  # The last row is twice as tall

        # Add subplots to the grid
        ax1 = fig.add_subplot(gs[0, 0])  # First column, first row
        ax2 = fig.add_subplot(gs[1, 0])  # First column, second row
        ax3 = fig.add_subplot(gs[2, 0])  # First column, third row
        ax4 = fig.add_subplot(gs[3, 0])  # First column, third row
        ax5 = fig.add_subplot(gs[0, 1])  # Second column, first row
        ax6 = fig.add_subplot(gs[1, 1])  # Second column, second row
        ax7 = fig.add_subplot(gs[2, 1])  # Second column, third row
        ax8 = fig.add_subplot(gs[4, :])  # Fourth row, spanning all columns

        # Plot the data for the first set of plots (seasons)
        ax1.plot(years, winter, label="Winter", color='b', linestyle='-', marker='s')
        ax1.set_title("Winter")
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel("Days", fontsize=12)
        ax1.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax1.grid(True)  # Add grid
        ax1.set_yticks(np.arange(0, max(winter), 20))
        
        ax2.plot(years, spring, label="Spring", color='g', linestyle='-', marker='s')
        ax2.set_title("Spring")
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Days", fontsize=12)
        ax2.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax2.grid(True)  # Add grid
        ax2.set_yticks(np.arange(0, max(spring), 20))
        
        ax3.plot(years, summer, label="Summer", color='r', linestyle='-', marker='s')
        ax3.set_title("Summer")
        ax3.set_xlabel("Year", fontsize=12)
        ax3.set_ylabel("Days", fontsize=12)
        ax3.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax3.grid(True)  # Add grid
        ax3.set_yticks(np.arange(0, max(summer), 20))

        ax4.plot(years, autumn, label="Autumn", color='orange', linestyle='-', marker='s')
        ax4.set_title("Autumn")
        ax4.set_xlabel("Year", fontsize=12)
        ax4.set_ylabel("Days", fontsize=12)
        ax4.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax4.grid(True)  # Add grid
        ax4.set_yticks(np.arange(0, max(autumn), 20))

        # Plot the data for the second set of plots (strength)
        ax5.plot(years, weak, label="Weak", color='b', linestyle='-', marker='s')
        ax5.set_title("Weak")
        ax5.set_xlabel("Year", fontsize=12)
        ax5.set_ylabel("Days", fontsize=12)
        ax5.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax5.grid(True)  # Add grid
        ax5.set_yticks(np.arange(0, max(weak), 20))

        ax6.plot(years, medium, label="Medium", color='g', linestyle='-', marker='s')
        ax6.set_title("Medium")
        ax6.set_xlabel("Year", fontsize=12)
        ax6.set_ylabel("Days", fontsize=12)
        ax6.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax6.grid(True)  # Add grid
        ax6.set_yticks(np.arange(0, max(medium), 20))

        ax7.plot(years, strong, label="Strong", color='r', linestyle='-', marker='s')
        ax7.set_title("Strong")
        ax7.set_xlabel("Year", fontsize=12)
        ax7.set_ylabel("Days", fontsize=12)
        ax7.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax7.grid(True)  # Add grid
        ax7.set_yticks(np.arange(0, max(strong), 20))

        # The large plot at the bottom
        ax8.plot(years, total, label="Total", color='black', linestyle='-', marker='s')
        ax8.set_title("Total Blocking Days Per Year")
        ax8.set_xlabel("Year", fontsize=12)
        ax8.set_ylabel("Days", fontsize=12)
        ax8.set_xticks(years[::4])  # Show every fourth year on the x-axis
        ax8.set_xticklabels(years[::4], rotation=45)  # Rotate the tick labels
        ax8.grid(True)  # Add grid
        ax8.set_yticks(np.arange(0, max(total), 20))

        plt.suptitle("Number of Blocking Days Per Year ", fontsize=14, fontname='serif', x=0.5)
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

     # Save the plot if needed
    if save:
            plt.savefig(f"BachelorThesis/Figures/blocking_days_per_year_{typ}.pdf")
        
        # Display the plot
    plt.show()

def plot_blockings_by_year(block_list, lim1, lim2, Histogram=False, save=False):
    """
    This function plots the number of blockings per year and the number of blockings 
    longer than 7 days for each year.
    """
    
    # Dictionary to store the number of blockings per year
    blockings_per_year = defaultdict(int)
    lim1_blockings_per_year = defaultdict(int)
    lim2_blockings_per_year = defaultdict(int)
    
    # Loop through the block list to count blockings per year and blockings > 7 days
    for data in block_list:
        start, end = min(data['datetime']), max(data['datetime'])  # Get start and end times
        year = start.year  # Extract the year from the start date
            
        duration = (end - start).days  # Calculate duration in days
        
        blockings_per_year[year] += 1  # Increment count of blockings for that year
        
        if duration > lim1:
            lim1_blockings_per_year[year] += 1  # Increment count for long blockings (over 7 days)
        if duration > lim2:
            lim2_blockings_per_year[year] += 1  # Increment count for long blockings (over 7 days)
       
    # Prepare data for plotting
    years = sorted(blockings_per_year.keys())  # Sorted list of years
    total_blockings = [blockings_per_year[year] for year in years]
    lim1_blockings_per_year = [lim1_blockings_per_year.get(year, 0) for year in years]  # Handle years with no long blockings
    lim2_blockings_per_year = [lim2_blockings_per_year.get(year, 0) for year in years]  # Handle years with no long blockings

    t = range(len(years))
    
    # Remove first and last year
    t=t[1:-1]
    total_blockings=total_blockings[1:-1]
    lim1_blockings_per_year=lim1_blockings_per_year[1:-1]
    lim2_blockings_per_year=lim2_blockings_per_year[1:-1]
    
    if Histogram:
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))
        # Bar plots for total blockings and long blockings
        ax.bar(t, total_blockings, label='Total Blockings', color='#D3D3D3', edgecolor='black', alpha=0.6)
        ax.bar(t, lim1_blockings_per_year, label=f'Blockings > {lim1} Days', color='red', edgecolor='black', alpha=0.9)
    
        
        # Labels and title
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Blockings', fontsize=12)
        ax.set_title(f'Number of Blockings Per Year and Blockings > {lim1} Days', 
                 fontsize=14, fontname='serif', x=0.5)
    
        # Set x-ticks every 4 years and rotate labels
        ax.set_xticks([i for i in range(0, len(years), 3)])  # Set ticks every 4th year
        ax.set_xticklabels(years[::3], rotation=45)  # Rotate the labels by 45 degrees
        
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
   
    else:
        fig, ax = plt.subplots(figsize=(9, 5))

        # Line plots for total blockings and long blockings
        ax.plot(t, total_blockings, label='Total Blockings', color='black', linestyle='-', marker='s', alpha=0.9)
        
        ax.plot(t, lim1_blockings_per_year, label=f'Blockings > {lim1} Days', color='green', linestyle='-', marker='o', alpha=0.9)
        ax.plot(t, lim2_blockings_per_year, label=f'Blockings > {lim2} Days', color='red', linestyle='-', marker='^', alpha=0.9)
        # Labels and title
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Blockings', fontsize=12)
        ax.set_title(f'Number of Blockings Per Year and Blockings', 
                     fontsize=14, fontname='serif', x=0.5)
        
        # Set x-ticks every 3 years and rotate labels
        ax.set_xticks([i for i in range(0, len(years), 5)])  
        ax.set_xticklabels(years[::5], rotation=45)
        
        plt.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    if save:
        plt.savefig("BachelorThesis/Figures/BlockingsPerYear.pdf")
    plt.show()

