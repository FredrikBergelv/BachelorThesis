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
import pymannkendall as mk



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
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)

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
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)

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
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)

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
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)

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
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)

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
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)

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
    if save=="pdf":
        plt.savefig(f"Presentation/Figures/yearly_{datatype}_{location}.pdf")
    if save=="png":
        plt.savefig(f"Presentation/Figures/yearly_{datatype}_{location}.png", dpi=400)
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
            plt.grid(True, axis='both', linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.legend()
            plt.show()
        datalist.append(streak_data)
    
    if info:
        print(f'A total of {len(datalist)} high pressure blocking events were found between {min(data["datetime"].dt.date)} and {max(data["datetime"].dt.date)}')
        
    return datalist  # Return a list of all the blocking data

"""
This function takes two blcoking lists and sorts them so the blokings line up 
with their respectivly dates. 
"""
def date_calibrate_blockinglists(blocking1, blocking2, timediff, info):
    """ 
    Takes two blcocking lists filled with pandas datatframe and one panda timediff
    returns two filtered dataframe
    """
    bl1 = blocking1
    bl2 = blocking2

    datelist_Malmö = pd.Series(index=range(len(bl2)), dtype='datetime64[ns]')
    datelist_Vavihill = pd.Series(index=range(len(bl1)), dtype='datetime64[ns]')

    for i, event in enumerate(bl2):
        start = min(event['datetime'])  
        datelist_Malmö[i] = pd.to_datetime(start)  
        
    for i, event in enumerate(bl1):
        start = min(event['datetime'])  
        datelist_Vavihill[i] = pd.to_datetime(start)  

    # Define time window (24 hours)
    time_window = timediff

    # Initialize sets to keep track of matched indices
    matched_malmo = set()
    matched_vavihill = set()

    # For each Vavihill event, check for Malmö events within ±24 hours
    for i_vav, date_vav in datelist_Vavihill.items():
        diffs = (datelist_Malmö - date_vav).abs()
        nearby = diffs <= time_window
        if nearby.any():
            # There is at least one Malmö event within ±24 hours — mark both as matched
            matched_vavihill.add(i_vav)
            matched_malmo.update(datelist_Malmö[nearby].index)

    # Now calculate unmatched indices
    all_malmo = set(datelist_Malmö.index)
    all_vavihill = set(datelist_Vavihill.index)

    unmatched_malmo = sorted(all_malmo - matched_malmo)
    unmatched_vavihill = sorted(all_vavihill - matched_vavihill)


    # First: create copies to avoid modifying original lists accidentally
    bl1_filtered = [event for i, event in enumerate(bl1) if i not in unmatched_vavihill]
    bl2_filtered = [event for i, event in enumerate(bl2) if i not in unmatched_malmo]

    # Now bl1_filtered and bl2_filtered contain only paired events!

    # Optional: if you want to overwrite original variables directly
    blocking1 = bl1_filtered
    blocking2 = bl2_filtered
    
    if info:
        num_events = len(bl1_filtered)
        if num_events > 0:
            start_date = min([min(event['datetime']) for event in bl1_filtered]).strftime('%B %d, %Y')
            end_date = max([max(event['datetime']) for event in bl1_filtered]).strftime('%B %d, %Y')
            print(f"\nAfter applying the high-pressure blocking detection method to the data from Vavihill and Malmö for the entire period, a total of {num_events} high-pressure blocking events were identified between {start_date} and {end_date}.")
    
    return blocking1, blocking2


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
        axs[0].grid(True, axis='both', linestyle='--', alpha=0.6)

            
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM$_{{2.5}}$ (µg/m³)')
        axs[1].legend()
        axs[1].set_ylim(0,60)
        axs[1].grid(True, axis='both', linestyle='--', alpha=0.6)

            
        # Plot Wind Direction
        axs[2].scatter(time, wind_dir, label='Wind Direction', color='orange', s=7)
        axs[2].set_ylabel('Wind Direction (°)')
        axs[2].legend()
        axs[2].set_yticks([0, 90, 180, 270, 365])
        axs[2].set_ylim(0,365)
        axs[2].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot Wind Speed
        axs[3].plot(time, wind_speed, label='Wind Speed', color='teal')
        axs[3].set_ylabel('Wind Speed (m/s)')
        axs[3].set_ylim(0,14)
        axs[3].legend()
        axs[3].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot temp
        axs[4].plot(time, temp, label='Temperatre', color='red')
        axs[4].set_ylabel('Temperature (°C)')
        axs[4].legend()
        axs[4].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot rain
        axs[5].plot(time, rain, label='Rain', color='darkblue')
        axs[5].set_ylabel('Rainfall (mm)')
        axs[5].set_xlabel('Time from start of period [days]')
        axs[5].legend()
        axs[5].grid(True, axis='both', linestyle='--', alpha=0.6)

        if save=="pdf":
            plt.savefig(f"Presentation/Figures/plot_{start_time}_to_{end_time}.pdf")
        if save=="pdf":
                plt.savefig(f"Presentation/Figures/plot_{start_time}_to_{end_time}.png", dpi=400)
        plt.show()
    
    return array # Return list of all the datafiles


"""
This funtion make arrays of PM2.5 and the pressure, wind, temp, rain from the 
blocking list. This gives arrays stored in lists. This is without datetime 
althogh if wanted to only the start-end date can be extraced for each list 
element instead.
"""
def array_blocking_list(PM_data, wind_data, rain_data, blocking_list, 
                              cover=0.9, info=False):
    """
    This function takes in the particle data, wind, rain, temp data and the pressure blocking data
    It returns a list of arrays for each blocking period with wind, rain, temp, pressure, PM2.5
    To get the array do list[i]=array"
    array[0]=hours, 
    array[1]=pressure, 
    array[2]=pm2.5, 
    array[3]=wind dir,  
    array[4]=wind speed, 
    array[5]=rain
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
            rain_data_trimmed,
            left_on='datetime',
            right_on='datetime',
            direction='nearest'
        )[['datetime', 'pressure', 'pm2.5', 'dir', 'speed', 'rain' ]]
          
        totdata_list.append(combined_data)
                
    # Store all blocking values in arrays
    array_list = []
     
    # loop trhrough entire list. 
    for i in range(len(totdata_list)):
             datafile = totdata_list[i]
             
             # convert everything to arrays
             array = np.zeros((6,len(datafile)))
            
            # Since the data is for every hour the index is the hour since start
             for hour in range(len(datafile)):
                    array[0,hour], array[1,hour], array[2,hour] = hour, datafile['pressure'][hour], datafile['pm2.5'][hour]
                    array[3,hour], array[4,hour] = datafile['dir'][hour], datafile['speed'][hour]
                    array[5,hour] = datafile['rain'][hour]
             array_list.append(array)
        
    if info == True:
        print(f'From a total of {len(blocking_list)} high-pressure bocking events, {counter} plots were removed due to lack of PM\textsubscript{2.5} data since a filter of {round(cover*100)}\% was used. Thus resuting in {len(blocking_list)-counter} relevant high-pressure blocking events')
    
    return array_list, title_list # Return list of all the datafiles

def plot_blocking_array(array, array_title=False, extrainfo=True, save=False):
    """
    Plots the blocking data with four subplots: Pressure, PM2.5, Wind Direction, and Wind Speed.
    """
    
    time = array[0]/24
    pressure = array[1]   
    pm25 = array[2]
    wind_dir = array[3]
    wind_speed = array[4]
    rain = array[5]
    
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
        axs[0].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM$_{{2.5}}$ [µg/m³]')
        axs[1].legend()
        axs[1].set_ylim(0,60)
        axs[1].grid(True, axis='both', linestyle='--', alpha=0.6)

        axs[1].set_xlabel('Time from start of blocking period [days]')
    else:
        # Create the figure and subplots
        fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(array_title, fontsize=16)
        
        # Plot Pressure
        axs[0].plot(time, pressure, label='Air Pressure', color='blue')
        axs[0].set_ylabel('Air Pressure [hPa]')
        axs[0].legend()
        axs[0].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot PM2.5
        axs[1].plot(time, pm25, label='PM2.5', color='green')
        axs[1].set_ylabel('PM$_{{2.5}}$ [µg/m³]')
        axs[1].legend()
        axs[1].set_ylim(0,60)
        axs[1].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot Wind Direction
        axs[2].scatter(time, wind_dir, label='Wind Direction', color='orange', s=7)
        axs[2].set_ylabel('Wind Direction [°]')
        axs[2].legend()
        axs[2].set_yticks([0, 90, 180, 270, 365])
        axs[2].set_ylim(0,365)
        axs[2].grid(True, axis='both', linestyle='--', alpha=0.6)

        
        # Plot Wind Speed
        axs[3].plot(time, wind_speed, label='Wind Speed', color='teal')
        axs[3].set_ylabel('Wind Speed [m/s]')
        axs[3].set_ylim(0,10)
        axs[3].legend()
        axs[3].grid(True, axis='both', linestyle='--', alpha=0.6)


        
        # Plot rain
        axs[4].plot(time, rain, label='Rain', color='darkblue')
        axs[4].set_ylabel('Rainfall [mm]')
        axs[4].set_xlabel('Time from start of blocking period [days]')
        axs[4].legend()
        axs[4].set_ylim(0,0.5)
        axs[4].grid(True, axis='both', linestyle='--', alpha=0.6)

        if save=="pdf":
            plt.savefig(f"Presentation/Figures/{array_title}.pdf")
        if save=="png":
            plt.savefig(f"Presentation/Figures/{array_title}.png", dpi=400)
        plt.show()



"""
The function down below is for extracting a certian period from <start> to <end>
and plotting all the data: pres, wind, temp, pm, rain. This is also displays
when there is a blockig in the background.
"""
def plot_period(PM_data, wind_data, rain_data, pressure_data,
                      blocking_list, start_time, end_time, wind_plot=True, 
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
    rain_data = rain_data[(rain_data['datetime'] >= start_time) & (rain_data['datetime'] <= end_time)]
    
    # Merge datasets
    merged_data = (pressure_data
                   .merge(PM_data, on='datetime', how='outer')
                   .merge(wind_data, on='datetime', how='outer')
                   .merge(rain_data, on='datetime', how='outer'))
    
    # Sort and reset index
    merged_data = merged_data.sort_values(by='datetime').reset_index(drop=True)
    
    # Extract periods from blocking_list
    periods = []
    for datafile in blocking_list:
        start = min(datafile['datetime'])
        end = max(datafile['datetime'])  # Fix: Use max instead of min
        periods.append((start, end))
        
    if wind_plot == True:
        size  = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
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
            
    if locationsave == "Vavihill":
        axs[0].text(0.95, 0.95, "(a)", transform= axs[0].transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        axs[1].text(0.95, 0.95, "(c)", transform=axs[1].transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        axs[2].text(0.95, 0.95, "(e)", transform=axs[2].transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    if locationsave == "Malmö":
        axs[0].text(0.95, 0.95, "(b)", transform= axs[0].transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        axs[1].text(0.95, 0.95, "(d)", transform=axs[1].transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        axs[2].text(0.95, 0.95, "(f)", transform=axs[2].transAxes, fontsize=12, fontname='serif', ha='right', va='top')

    # Plot Pressure
    axs[0].plot(merged_data['datetime'], merged_data['pressure'], label='Air Pressure', color='red')
    axs[0].set_ylabel('Air Pressure [hPa]')
    axs[0].legend()
    axs[0].grid(True, axis='both', linestyle='--', alpha=0.6)

    # Plot PM2.5
    axs[1].plot(merged_data['datetime'], merged_data['pm2.5'], label='PM2.5', color='green')
    axs[1].set_ylabel('PM$_{{2.5}}$ [µg/m³]')
    axs[1].set_ylim(0, 60)
    axs[1].legend()
    axs[1].grid(True, axis='both', linestyle='--', alpha=0.6)

    
    n = 2
    if wind_plot == True:
        n=4
        
        # Plot Wind Direction
        axs[2].scatter(merged_data['datetime'], merged_data['dir'], label='Wind Direction', color='orange', s=7)
        axs[2].set_ylabel('Wind Direction (°)')
        axs[2].set_yticks([0, 90, 180, 270, 360])
        axs[2].set_ylim(0, 360)
        axs[2].legend()
        axs[2].grid(True, axis='both', linestyle='--', alpha=0.6)

    
        # Plot Wind Speed
        axs[3].plot(merged_data['datetime'], merged_data['speed'], label='Wind Speed', color='teal')
        axs[3].set_ylabel('Wind Speed (m/s)')
        axs[3].set_ylim(0, 14)
        axs[3].legend()
        axs[3].grid(True, axis='both', linestyle='--', alpha=0.6)

        

    # Plot Rainfall
    axs[n].plot(merged_data['datetime'], merged_data['rain'], label='Rainfall', color='blue')
    axs[n].set_ylabel('Rainfall [mm]')
    
    axs[n].set_xlabel('Date')
    axs[n].legend(loc="upper left")
    axs[n].grid(True, axis='both', linestyle='--', alpha=0.6)

    axs[n].tick_params(axis='x', rotation=45)
    axs[n].set_xlim(start_time, end_time)

    plt.tight_layout()
    if save=="pdf":
        plt.savefig(f'Presentation/Figures/{locationsave}_plot_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.pdf')
    if save=="png":
        plt.savefig(f'Presentation/Figures/{locationsave}_plot_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.png', dpi=400)
    
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

        if save=="pdf":
            plt.savefig("Presentation/Figures/PieChart.pdf", bbox_inches="tight")
        if save=="png":
             plt.savefig("Presentation/Figures/PieChart.png", dpi=400, bbox_inches="tight")
        plt.show()
        
    # Print Summary
    if pieinfo:
        print(f'{round(100 * partNE,1)}\% of the winds came from the Northeast (310° to 70°), {round(100 * partSE,1)}\% from the Southeast (70° to 190°), {round(100 * partW,1)}\% from the West (190° to 310°) and {round(100 * partTurning,1)}\% from no specific direction.')        
        
    
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

        if save=="pdf":
            plt.savefig("Presentation/Figures/PieChart.pdf", bbox_inches="tight")
        if save=="png":
             plt.savefig("Presentation/Figures/PieChart.png", dpi=400, bbox_inches="tight")
        plt.show()
        
    # Print Summary
    if pieinfo:
        print(f'{round(100 * partWinter,1)}\% of the blocking events occurred during the winter, {round(100 * partSpring,1)}\% during the spring, {round(100 * partSummer,1)}\% during the summer and {round(100 * partAutumn,1)}\% during the autumn.')        
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

        if save=="pdf":
            plt.savefig("Presentation/Figures/PieChart.pdf", bbox_inches="tight")
        if save=="png":
             plt.savefig("Presentation/Figures/PieChart.png", dpi=400, bbox_inches="tight")
        plt.show()
        
    # Print summary in a single line with explicit pressure thresholds
    if pieinfo:
        print(f'{round(100 * partLow,1)}\% of the blocking events occurred with a mean pressure below {limits[0]} hPa {round(100 * partMedium,1)}\% occurred between {limits[0]} and {limits[1]} hPa and {round(100 * partHigh,1)}\% occurred with a mean pressure over {limits[1]}hPa.')


    return low_totdata_list, medium_totdata_list, high_totdata_list


"""
These functions use statistics to evaluate PM25 during periods of high
pressure blocking.
"""
 
def plot_mean(totdata_list1, totdata_list2, 
              daystoplot, minpoints=8, place1='', place2='',
              pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
              save=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour.
    You must specify how many days you wish to plot, you can add a wind title, 
    the number of datasets needed, plot info, etc.
    """
    timelen = int(24 * daystoplot)  # Initial length in hours

    # Create an array to store all the PM2.5 values
    PM_array1 = np.full((len(totdata_list1), timelen), np.nan)
    PM_array2 = np.full((len(totdata_list2), timelen), np.nan)

    
    # Populate the PM_array with data
    for i, array in enumerate(totdata_list1):
        valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
        PM_array1[i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for i, array in enumerate(totdata_list2):
         valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
         PM_array2[i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs
    mean1, sigma1 = np.nanmean(PM_array1, axis=0), np.nanstd(PM_array1, axis=0)
    mean2, sigma2 = np.nanmean(PM_array2, axis=0), np.nanstd(PM_array2, axis=0)

    t = np.arange(timelen) / 24  # Time axis in days   
    
    # Below we check the number of data points
    valid_counts_per_hour1 = np.sum(~np.isnan(PM_array1), axis=0)
    valid_counts_per_hour2 = np.sum(~np.isnan(PM_array2), axis=0)
    
    #create subfgure
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)  
    fig.suptitle(r'Mean concentration of PM$_{{2.5}}$',
                 fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.4, 1])  # Top row is twice as tall
    
    # Create subplots using GridSpec
    ax1 = fig.add_subplot(gs[0, 0])  # Large top-left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Large top-right plot
    ax3 = fig.add_subplot(gs[1, 0])  # Smaller bottom-left plot
    ax4 = fig.add_subplot(gs[1, 1])  # Smaller bottom-right plot
    
    tau1, slope1 = mk.original_test(mean1, 0.05)[4], mk.original_test(mean1, 0.05)[7]
    tau2, slope2 = mk.original_test(mean2, 0.05)[4], mk.original_test(mean2, 0.05)[7]
    
    # Add subplot labels (a), (b), (c), (d)
    ax1.text(0.95, 0.95, "(a)", transform=ax1.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax2.text(0.95, 0.95, "(b)", transform=ax2.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax3.text(0.95, 0.95, "(c)", transform=ax3.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax4.text(0.95, 0.95, "(d)", transform=ax4.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    # Plotting for ax1
    for i, points in enumerate(valid_counts_per_hour1):
        if points < minpoints:
            mean1[i] = np.nan
            sigma1[i] = np.nan
                
    ax1.plot(t, mean1, label=f'{place1}, $\\tau={tau1:.2f}$, sen-slope={slope1:.1e}', c='C0')
    ax1.plot(t, pm_mean1 + t * 0, label='Mean during no blocking', c='gray')
    ax1.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray') 
    ax1.fill_between(t, mean1 + sigma1, mean1 - sigma1, alpha=0.4, color='C0')
    ax1.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')
    ax1.set_xlabel('Time from start of blocking [days]')
    ax1.set_ylabel('PM2.5 [µg/m³]')
    ax1.set_ylim(0, 40)
    ax1.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plotting for ax3
    for i, points in enumerate(valid_counts_per_hour2):
        if points < minpoints:
            mean2[i] = np.nan
            sigma2[i] = np.nan
    
    ax2.plot(t, mean2, label=f'{place2}, $\\tau={tau2:.2f}$, sen-slope={slope2:.1e}', c='C0')
    ax2.plot(t, t * 0 + 25, c='r', linestyle='--')
    ax2.plot(t, pm_mean2 + t * 0, c='gray')
    ax2.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')
    ax2.fill_between(t, mean2 + sigma2, mean2 - sigma2, alpha=0.4, color='C0')
    ax2.set_xlabel('Time from start of blocking [days]')
    ax2.set_ylabel('PM$_{{2.5}}$ [µg/m³]')
    ax2.set_ylim(0, 40)
    ax2.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax2.legend()

    
    # Plotting for ax2
    maxval = max(max(valid_counts_per_hour1), max(valid_counts_per_hour2))
    ax3.plot(t, valid_counts_per_hour1, label=f'{place1}')
    #ax3.set_title(f'Number of datasets at {place1}', fontsize=13, fontname='serif', x=0.5)
    ax3.set_xlabel('Time from start of blocking [days]')
    ax3.set_ylabel('Number of events')
    ax3.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5, label='Minimum number of events allowed')
    ax3.set_yticks(np.arange(0, 201, 50))
    ax3.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left')
    
    
    # Plotting for ax4
    ax4.plot(t, valid_counts_per_hour2, label=f'{place2} ')
    #ax4.set_title(f'Number of datasets at {place2}', fontsize=13, fontname='serif', x=0.5)
    ax4.set_xlabel('Time from start of blocking [days]')
    ax4.set_ylabel('Number of events')
    ax4.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5)
    ax4.set_yticks(np.arange(0, 201, 50))
    ax4.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax4.legend(loc='center left')
    
    ax1.set_xlim(0,daystoplot)
    ax2.set_xlim(0,daystoplot)
    ax3.set_xlim(0,daystoplot)
    ax4.set_xlim(0,daystoplot)

    
    fig.tight_layout()
    fig.show()
    
    if save=="pdf":
        plt.savefig("Presentation/Figures/Meanplot.pdf")
    if save=="png":
        plt.savefig("Presentation/Figures/Meanplot.png", dpi=400)
        
    plt.show() 
    
def plot_dir_mean(dir_totdata_list1, dir_totdata_list2, daystoplot,  
                  minpoints=8, place1='', place2='', save=False,
                  pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False, 
                  info=False,
                  labels=["NE (310° to 70°)", "SE (70° to 190°)", "W (190° to 310°)", "No Specific"]):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each wind direction category in subplots.
    It displays plots side by side for place1 and place2.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold", "orange"]  # Colors mapped to NE, SE, W, Turning, non

    lenmax = dir_totdata_list1[0]+dir_totdata_list1[1]+dir_totdata_list1[2]+dir_totdata_list1[3]
    # Create an array to store all the PM2.5 values
    PM_array1 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]
    PM_array2 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]

    # Populate the PM_array with data
    for k, totodatatlist in enumerate(dir_totdata_list1):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array1[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for k, totodatatlist in enumerate(dir_totdata_list2):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array2[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs for each direction (place 1 and place 2)
    mean1 = [np.nanmean(PM_array1[k], axis=0) for k in range(4)]
    mean2 = [np.nanmean(PM_array2[k], axis=0) for k in range(4)]

    sigma1 = [np.nanstd(PM_array1[k], axis=0) for k in range(4)]
    sigma2 = [np.nanstd(PM_array2[k], axis=0) for k in range(4)]

    # Compute valid counts per hour
    valid_counts_per_hour1 = [np.sum(~np.isnan(PM_array1[k]), axis=0) for k in range(4)]
    valid_counts_per_hour2 = [np.sum(~np.isnan(PM_array2[k]), axis=0) for k in range(4)]

    # if points are lower tha nminpoints set to nan 
    for i in range(4):
        for j, counts in enumerate(valid_counts_per_hour1[i]):
            if counts < minpoints:
                mean1[i][j] = np.nan
                sigma1[i][j] = np.nan
        for j, counts in enumerate(valid_counts_per_hour2[i]):
            if counts < minpoints:
                mean2[i][j] = np.nan
                sigma2[i][j] = np.nan

    t = np.arange(timelen) / 24  # Time axis in days   
    
    #create subfgure
    scalingfactor = 1.1
    fig = plt.figure(figsize=(14*scalingfactor, 9*scalingfactor), constrained_layout=True)  
    fig.suptitle(r'Mean concentration of PM$_{{2.5}}$',
                 fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  
    
    # Create subplots using GridSpec
    ax11 = fig.add_subplot(gs[0, 0])  
    ax12 = fig.add_subplot(gs[1, 0])  
    ax13 = fig.add_subplot(gs[2, 0])  
    ax14 = fig.add_subplot(gs[3, 0])  
    ax21 = fig.add_subplot(gs[0, 1])  
    ax22 = fig.add_subplot(gs[1, 1])  
    ax23 = fig.add_subplot(gs[2, 1])  
    ax24 = fig.add_subplot(gs[3, 1])  
     
    #tau11, slope11 = mk.original_test(mean1[0], 0.05)[4], mk.original_test(mean1[0], 0.05)[7]
    tau12, slope12 = mk.original_test(mean1[1], 0.05)[4], mk.original_test(mean1[1], 0.05)[7]
    tau13, slope13 = mk.original_test(mean1[2], 0.05)[4], mk.original_test(mean1[2], 0.05)[7]
    tau14, slope14 = mk.original_test(mean1[3], 0.05)[4], mk.original_test(mean1[3], 0.05)[7]
    tau21, slope21 = mk.original_test(mean2[0], 0.05)[4], mk.original_test(mean2[0], 0.05)[7]
    tau22, slope22 = mk.original_test(mean2[1], 0.05)[4], mk.original_test(mean2[1], 0.05)[7]
    tau23, slope23 = mk.original_test(mean2[2], 0.05)[4], mk.original_test(mean2[2], 0.05)[7]
    tau24, slope24 = mk.original_test(mean2[3], 0.05)[4], mk.original_test(mean2[3], 0.05)[7]
    

    if info:
        first_nine_days = mean1[2][:9*24]
        tau, slope = mk.original_test(first_nine_days, 0.05)[4], mk.original_test(first_nine_days, 0.05)[7]
        print(f'tau for first nine days in SE for {place1} is {tau}')
    
    # Add subplot labels (a), (b), (c), (d)
    ax11.text(0.95, 0.95, "(a)", transform=ax11.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax12.text(0.95, 0.95, "(c)", transform=ax12.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax13.text(0.95, 0.95, "(e)", transform=ax13.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax14.text(0.95, 0.95, "(g)", transform=ax14.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax21.text(0.95, 0.95, "(b)", transform=ax21.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax22.text(0.95, 0.95, "(d)", transform=ax22.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax23.text(0.95, 0.95, "(f)", transform=ax23.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax24.text(0.95, 0.95, "(h)", transform=ax24.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    ax11.set_title('Direction: ' + labels[0])  # Setting the title for the first subplot
    #ax11.plot(t, mean1[0], label=f'{place1}, $\\tau={tau11:.2f}$, sen-slope={slope11:.1e}', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, mean1[0], label=f'{place1}, $\\tau$=NaN, sen-slope=NaN', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, pm_mean1 + t * 0, label='Mean during no blocking', c='gray')  # Plot the mean during no blocking
    ax11.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax11.fill_between(t, mean1[0] + sigma1[0], mean1[0] - sigma1[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    ax11.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')  # Plot the EU annual mean limit
    ax11.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax11.set_ylim(0, 40)  # Set the Y-axis limits
    ax11.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax11.legend()  # Display legend
    ax11.set_xticklabels([])
    
    ax12.set_title('Direction: ' + labels[1])  # Setting the title for the first subplot
    ax12.plot(t, mean1[1], label=f'{place1}, $\\tau={tau12:.2f}$, sen-slope={slope12:.1e}', color=colors[1])  # Plot the mean1 for place1
    ax12.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax12.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax12.fill_between(t, mean1[1] + sigma1[1], mean1[1] - sigma1[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    ax12.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax12.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax12.set_ylim(0, 40)  # Set the Y-axis limits
    ax12.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax12.legend()  # Display legend
    ax12.set_xticklabels([])

    
    ax13.set_title('Direction: ' + labels[2])  # Setting the title for the first subplot
    ax13.plot(t, mean1[2], label=f'{place1}, $\\tau={tau13:.2f}$, sen-slope={slope13:.1e}', color=colors[2])  # Plot the mean1 for place1
    ax13.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax13.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax13.fill_between(t, mean1[2] + sigma1[2], mean1[2] - sigma1[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    ax13.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax13.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax13.set_ylim(0, 40)  # Set the Y-axis limits
    ax13.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax13.legend()  # Display legend
    ax13.set_xticklabels([])

    
    ax14.set_title('Direction: ' + labels[3])  # Setting the title for the first subplot
    ax14.plot(t, mean1[3], label=f'{place1}, $\\tau={tau14:.2f}$, sen-slope={slope14:.1e}', color=colors[3])  # Plot the mean1 for place1
    ax14.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax14.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax14.fill_between(t, mean1[3] + sigma1[3], mean1[3] - sigma1[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    ax14.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax14.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax14.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax14.set_ylim(0, 40)  # Set the Y-axis limits
    ax14.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax14.legend()  # Display legend
    ax14.set_xticks(np.arange(0, daystoplot+1, 2))

    
    ax21.set_title('Direction: ' + labels[0])  # Setting the title for the first subplot
    ax21.plot(t, mean2[0], label=f'{place2}, $\\tau={tau21:.2f}$, sen-slope={slope21:.1e}', color=colors[0])  # Plot the mean2 for place1
    ax21.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax21.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax21.fill_between(t, mean2[0] + sigma2[0], mean2[0] - sigma2[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    ax21.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax21.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax21.set_ylim(0, 40)  # Set the Y-axis limits
    ax21.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax21.legend()  # Display legend
    ax21.set_xticklabels([])

    
    ax22.set_title('Direction: ' + labels[1])  # Setting the title for the first subplot
    ax22.plot(t, mean2[1], label=f'{place2}, $\\tau={tau22:.2f}$, sen-slope={slope22:.1e}', color=colors[1])  # Plot the mean2 for place1
    ax22.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax22.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax22.fill_between(t, mean2[1] + sigma2[1], mean2[1] - sigma2[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    ax22.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax22.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax22.set_ylim(0, 40)  # Set the Y-axis limits
    ax22.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax22.legend()  # Display legend
    ax22.set_xticklabels([])

    
    ax23.set_title('Direction: ' + labels[2])  # Setting the title for the first subplot
    ax23.plot(t, mean2[2], label=f'{place2}, $\\tau={tau23:.2f}$, sen-slope={slope23:.1e}', color=colors[2])  # Plot the mean2 for place1
    ax23.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax23.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax23.fill_between(t, mean2[2] + sigma2[2], mean2[2] - sigma2[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    ax23.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax23.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax23.set_ylim(0, 40)  # Set the Y-axis limits
    ax23.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax23.legend()  # Display legend
    ax23.set_xticklabels([])

    
    ax24.set_title('Direction: ' + labels[3])  # Setting the title for the first subplot
    ax24.plot(t, mean2[3], label=f'{place2}, $\\tau={tau24:.2f}$, sen-slope={slope24:.1e}', color=colors[3])  # Plot the mean2 for place1
    ax24.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax24.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax24.fill_between(t, mean2[3] + sigma2[3], mean2[3] - sigma2[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    ax24.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax24.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax24.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax24.set_ylim(0, 40)  # Set the Y-axis limits
    ax24.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax24.legend()  # Display legend
    ax24.set_xticks(np.arange(0, daystoplot+1, 2))
    
    ax11.set_xlim(0,daystoplot)
    ax12.set_xlim(0,daystoplot)
    ax12.set_xlim(0,daystoplot)
    ax13.set_xlim(0,daystoplot)
    ax14.set_xlim(0,daystoplot)   
    ax21.set_xlim(0,daystoplot)
    ax22.set_xlim(0,daystoplot)
    ax23.set_xlim(0,daystoplot)
    ax24.set_xlim(0,daystoplot)

    
    fig.tight_layout()
            
    if save=="pdf":
        plt.savefig("Presentation/Figures/Meanplot_dir.pdf")
    if save=="png":
        plt.savefig("Presentation/Figures/Meanplot_dir.png", dpi=400)
        
    plt.show()

def plot_seasonal_mean(seasonal_totdata_list1, seasonal_totdata_list2, daystoplot,  
                  minpoints=8, place1='', place2='', save=False,
                  pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
                  labels=["Winter", "Spring", "Summer", "Autumn"]):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each season category in subplots.
    It displays plots side by side for place1 and place2.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold", "orange"]  # Colors mapped to NE, SE, W, Turning, non

    lenmax = seasonal_totdata_list1[0]+seasonal_totdata_list1[1]+seasonal_totdata_list1[2]+seasonal_totdata_list1[3]
    
    # Create an array to store all the PM2.5 values
    PM_array1 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]
    PM_array2 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]

    # Populate the PM_array with data
    for k, totodatatlist in enumerate(seasonal_totdata_list1):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array1[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for k, totodatatlist in enumerate(seasonal_totdata_list2):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array2[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs for each direction (place 1 and place 2)
    mean1 = [np.nanmean(PM_array1[k], axis=0) for k in range(4)]
    mean2 = [np.nanmean(PM_array2[k], axis=0) for k in range(4)]

    sigma1 = [np.nanstd(PM_array1[k], axis=0) for k in range(4)]
    sigma2 = [np.nanstd(PM_array2[k], axis=0) for k in range(4)]

    # Compute valid counts per hour
    valid_counts_per_hour1 = [np.sum(~np.isnan(PM_array1[k]), axis=0) for k in range(4)]
    valid_counts_per_hour2 = [np.sum(~np.isnan(PM_array2[k]), axis=0) for k in range(4)]

    # if points are lower tha nminpoints set to nan 
    for i in range(4):
            for j, counts in enumerate(valid_counts_per_hour1[i]):
                if counts < minpoints:
                    mean1[i][j] = np.nan
                    sigma1[i][j] = np.nan
            for j, counts in enumerate(valid_counts_per_hour2[i]):
                if counts < minpoints:
                    mean2[i][j] = np.nan
                    sigma2[i][j] = np.nan
                    
    t = np.arange(timelen) / 24  # Time axis in days   
        
    #create subfgure
    scalingfactor = 1.1
    fig = plt.figure(figsize=(14*scalingfactor, 9*scalingfactor), constrained_layout=True)
    fig.suptitle(r'Mean concentration of PM$_{{2.5}}$',
                     fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  
        
    # Create subplots using GridSpec
    ax11 = fig.add_subplot(gs[0, 0])  
    ax12 = fig.add_subplot(gs[1, 0])  
    ax13 = fig.add_subplot(gs[2, 0])  
    ax14 = fig.add_subplot(gs[3, 0])  
    ax21 = fig.add_subplot(gs[0, 1])  
    ax22 = fig.add_subplot(gs[1, 1])  
    ax23 = fig.add_subplot(gs[2, 1])  
    ax24 = fig.add_subplot(gs[3, 1])  
    
    tau11, slope11 = mk.original_test(mean1[0], 0.05)[4], mk.original_test(mean1[0], 0.05)[7]
    tau12, slope12 = mk.original_test(mean1[1], 0.05)[4], mk.original_test(mean1[1], 0.05)[7]
    tau13, slope13 = mk.original_test(mean1[2], 0.05)[4], mk.original_test(mean1[2], 0.05)[7]
    tau14, slope14 = mk.original_test(mean1[3], 0.05)[4], mk.original_test(mean1[3], 0.05)[7]
    tau21, slope21 = mk.original_test(mean2[0], 0.05)[4], mk.original_test(mean2[0], 0.05)[7]
    tau22, slope22 = mk.original_test(mean2[1], 0.05)[4], mk.original_test(mean2[1], 0.05)[7]
    tau23, slope23 = mk.original_test(mean2[2], 0.05)[4], mk.original_test(mean2[2], 0.05)[7]
    tau24, slope24 = mk.original_test(mean2[3], 0.05)[4], mk.original_test(mean2[3], 0.05)[7]

    
       # Add subplot labels (a), (b), (c), (d)
    ax11.text(0.95, 0.95, "(a)", transform=ax11.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax12.text(0.95, 0.95, "(c)", transform=ax12.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax13.text(0.95, 0.95, "(e)", transform=ax13.transAxes, fontsize=12, fontname='serif', ha='right', va='top')        
    ax14.text(0.95, 0.95, "(g)", transform=ax14.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax21.text(0.95, 0.95, "(b)", transform=ax21.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax22.text(0.95, 0.95, "(d)", transform=ax22.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax23.text(0.95, 0.95, "(f)", transform=ax23.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax24.text(0.95, 0.95, "(h)", transform=ax24.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    ax11.set_title(labels[0])  # Setting the title for the first subplot
    ax11.plot(t, mean1[0], label=f'{place1}, $\\tau={tau11:.2f}$, sen-slope={slope11:.1e}', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, pm_mean1 + t * 0, label='Mean during no blocking', c='gray')  # Plot the mean during no blocking
    ax11.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax11.fill_between(t, mean1[0] + sigma1[0], mean1[0] - sigma1[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    ax11.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')  # Plot the EU annual mean limit
    ax11.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax11.set_ylim(0, 40)  # Set the Y-axis limits
    ax11.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax11.legend()  # Display legend
    ax11.set_xticklabels([])
        
    ax12.set_title(labels[1])  # Setting the title for the first subplot
    ax12.plot(t, mean1[1], label=f'{place1}, $\\tau={tau12:.2f}$, sen-slope={slope12:.1e}', color=colors[1])  # Plot the mean1 for place1
    ax12.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax12.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax12.fill_between(t, mean1[1] + sigma1[1], mean1[1] - sigma1[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    ax12.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax12.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax12.set_ylim(0, 40)  # Set the Y-axis limits
    ax12.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax12.legend()  # Display legend
    ax12.set_xticklabels([])

        
    ax13.set_title(labels[2])  # Setting the title for the first subplot
    ax13.plot(t, mean1[2], label=f'{place1}, $\\tau={tau13:.2f}$, sen-slope={slope13:.1e}', color=colors[2])  # Plot the mean1 for place1
    ax13.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax13.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax13.fill_between(t, mean1[2] + sigma1[2], mean1[2] - sigma1[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    ax13.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax13.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax13.set_ylim(0, 40)  # Set the Y-axis limits
    ax13.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax13.legend()  # Display legend
    ax13.set_xticklabels([])

        
    ax14.set_title(labels[3])  # Setting the title for the first subplot
    ax14.plot(t, mean1[3], label=f'{place1}, $\\tau={tau14:.2f}$, sen-slope={slope14:.1e}', color=colors[3])  # Plot the mean1 for place1
    ax14.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax14.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax14.fill_between(t, mean1[3] + sigma1[3], mean1[3] - sigma1[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    ax14.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax14.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax14.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax14.set_ylim(0, 40)  # Set the Y-axis limits
    ax14.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax14.legend()  # Display legend
    ax14.set_xticks(np.arange(0, daystoplot+1, 2))

        
    ax21.set_title(labels[0])  # Setting the title for the first subplot
    ax21.plot(t, mean2[0], label=f'{place2}, $\\tau={tau21:.2f}$, sen-slope={slope21:.1e}', color=colors[0])  # Plot the mean2 for place1
    ax21.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax21.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax21.fill_between(t, mean2[0] + sigma2[0], mean2[0] - sigma2[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    ax21.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax21.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax21.set_ylim(0, 40)  # Set the Y-axis limits
    ax21.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax21.legend()  # Display legend
    ax21.set_xticklabels([])

        
    ax22.set_title(labels[1])  # Setting the title for the first subplot
    ax22.plot(t, mean2[1], label=f'{place2}, $\\tau={tau22:.2f}$, sen-slope={slope22:.1e}', color=colors[1])  # Plot the mean2 for place1
    ax22.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax22.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax22.fill_between(t, mean2[1] + sigma2[1], mean2[1] - sigma2[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    ax22.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax22.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax22.set_ylim(0, 40)  # Set the Y-axis limits
    ax22.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax22.legend()  # Display legend
    ax22.set_xticklabels([])

        
    ax23.set_title(labels[2])  # Setting the title for the first subplot
    ax23.plot(t, mean2[2], label=f'{place2}, $\\tau={tau23:.2f}$, sen-slope={slope23:.1e}', color=colors[2])  # Plot the mean2 for place1
    ax23.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax23.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax23.fill_between(t, mean2[2] + sigma2[2], mean2[2] - sigma2[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    ax23.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax23.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax23.set_ylim(0, 40)  # Set the Y-axis limits
    ax23.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax23.legend()  # Display legend
    ax23.set_xticklabels([])

        
    ax24.set_title(labels[3])  # Setting the title for the first subplot
    ax24.plot(t, mean2[3], label=f'{place2}, $\\tau={tau24:.2f}$, sen-slope={slope24:.1e}', color=colors[3])  # Plot the mean2 for place1
    ax24.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax24.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax24.fill_between(t, mean2[3] + sigma2[3], mean2[3] - sigma2[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    ax24.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax24.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax24.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax24.set_ylim(0, 40)  # Set the Y-axis limits
    ax24.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax24.legend()  # Display legend
    ax24.set_xticks(np.arange(0, daystoplot+1, 2))

    ax11.set_xlim(0,daystoplot)
    ax12.set_xlim(0,daystoplot)
    ax12.set_xlim(0,daystoplot)
    ax13.set_xlim(0,daystoplot)
    ax14.set_xlim(0,daystoplot)   
    ax21.set_xlim(0,daystoplot)
    ax22.set_xlim(0,daystoplot)
    ax23.set_xlim(0,daystoplot)
    ax24.set_xlim(0,daystoplot)
        
    fig.tight_layout()
        
    if save=="pdf":
            plt.savefig("Presentation/Figures/Meanplot_seasonal.pdf")
    if save=="png":
            plt.savefig("Presentation/Figures/Meanplot_seasonal.png", dpi=400)
            
    plt.show()

def plot_pressure_mean(pressure_totdata_list1, pressure_totdata_list2, daystoplot,  
                  minpoints=8, place1='', place2='', save=False,
                  pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
                  labels=["Weaker Pressure Blocking", "Medium Pressure Blocking", "Stronger Pressure Blocking"], ):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each strength category in subplots.
    It displays plots side by side for place1 and place2.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "seagreen", "tomato"]  # Colors mapped to NE, SE, W, Turning, non

    lenmax = pressure_totdata_list1[0]+pressure_totdata_list1[1]+pressure_totdata_list1[2]
    # Create an array to store all the PM2.5 values
    PM_array1 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]
    PM_array2 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]

    # Populate the PM_array with data
    for k, totodatatlist in enumerate(pressure_totdata_list1):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array1[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for k, totodatatlist in enumerate(pressure_totdata_list2):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array2[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs for each direction (place 1 and place 2)
    mean1 = [np.nanmean(PM_array1[k], axis=0) for k in range(3)]
    mean2 = [np.nanmean(PM_array2[k], axis=0) for k in range(3)]

    sigma1 = [np.nanstd(PM_array1[k], axis=0) for k in range(3)]
    sigma2 = [np.nanstd(PM_array2[k], axis=0) for k in range(3)]

    # Compute valid counts per hour
    valid_counts_per_hour1 = [np.sum(~np.isnan(PM_array1[k]), axis=0) for k in range(3)]
    valid_counts_per_hour2 = [np.sum(~np.isnan(PM_array2[k]), axis=0) for k in range(3)]

    # if points are lower tha nminpoints set to nan 
    for i in range(3):
        for j, counts in enumerate(valid_counts_per_hour1[i]):
            if counts < minpoints:
                mean1[i][j] = np.nan
                sigma1[i][j] = np.nan
        for j, counts in enumerate(valid_counts_per_hour2[i]):
            if counts < minpoints:
                mean2[i][j] = np.nan
                sigma2[i][j] = np.nan

    t = np.arange(timelen) / 24  # Time axis in days   
    
    #create subfgure
    scalingfactor = 1.1
    fig = plt.figure(figsize=(14*scalingfactor, 9*3/4*scalingfactor), constrained_layout=True)  
    fig.suptitle(r'Mean concentration of PM$_{{2.5}}$',
                 fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  
    
    # Create subplots using GridSpec
    ax11 = fig.add_subplot(gs[0, 0])  
    ax12 = fig.add_subplot(gs[1, 0])  
    ax13 = fig.add_subplot(gs[2, 0])  
    ax21 = fig.add_subplot(gs[0, 1])  
    ax22 = fig.add_subplot(gs[1, 1])  
    ax23 = fig.add_subplot(gs[2, 1])  
    
    tau11, slope11 = mk.original_test(mean1[0], 0.05)[4], mk.original_test(mean1[0], 0.05)[7]
    tau12, slope12 = mk.original_test(mean1[1], 0.05)[4], mk.original_test(mean1[1], 0.05)[7]
    tau13, slope13 = mk.original_test(mean1[2], 0.05)[4], mk.original_test(mean1[2], 0.05)[7]
    tau21, slope21 = mk.original_test(mean2[0], 0.05)[4], mk.original_test(mean2[0], 0.05)[7]
    tau22, slope22 = mk.original_test(mean2[1], 0.05)[4], mk.original_test(mean2[1], 0.05)[7]
    tau23, slope23 = mk.original_test(mean2[2], 0.05)[4], mk.original_test(mean2[2], 0.05)[7]
    
    # Add subplot labels (a), (b), (c), (d)
    ax11.text(0.95, 0.95, "(a)", transform=ax11.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax12.text(0.95, 0.95, "(c)", transform=ax12.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax13.text(0.95, 0.95, "(e)", transform=ax13.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax21.text(0.95, 0.95, "(b)", transform=ax21.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax22.text(0.95, 0.95, "(d)", transform=ax22.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax23.text(0.95, 0.95, "(f)", transform=ax23.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    ax11.set_title(labels[0])  # Setting the title for the first subplot
    ax11.plot(t, mean1[0], label=f'{place1}, $\\tau={tau11:.2f}$, sen-slope={slope11:.1e}', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, pm_mean1 + t * 0, label='Mean during no blocking', c='gray')  # Plot the mean during no blocking
    ax11.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax11.fill_between(t, mean1[0] + sigma1[0], mean1[0] - sigma1[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    ax11.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')  # Plot the EU annual mean limit
    ax11.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax11.set_ylim(0, 40)  # Set the Y-axis limits
    ax11.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax11.legend()  # Display legend
    ax11.set_xticklabels([])
    
    ax12.set_title(labels[1])  # Setting the title for the first subplot
    ax12.plot(t, mean1[1], label=f'{place1}, $\\tau={tau12:.2f}$, sen-slope={slope12:.1e}', color=colors[1])  # Plot the mean1 for place1
    ax12.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax12.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax12.fill_between(t, mean1[1] + sigma1[1], mean1[1] - sigma1[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    ax12.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax12.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax12.set_ylim(0, 40)  # Set the Y-axis limits
    ax12.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax12.legend()  # Display legend
    ax12.set_xticklabels([])

    
    ax13.set_title(labels[2])  # Setting the title for the first subplot
    ax13.plot(t, mean1[2], label=f'{place1}, $\\tau={tau13:.2f}$, sen-slope={slope13:.1e}', color=colors[2])  # Plot the mean1 for place1
    ax13.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the mean during no blocking
    ax13.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax13.fill_between(t, mean1[2] + sigma1[2], mean1[2] - sigma1[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    ax13.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax13.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax13.set_ylim(0, 40)  # Set the Y-axis limits
    ax13.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax13.legend()  # Display legend
    ax13.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax13.set_xticks(np.arange(0, daystoplot+1, 2))

    
    ax21.set_title(labels[0])  # Setting the title for the first subplot
    ax21.plot(t, mean2[0], label=f'{place2}, $\\tau={tau21:.2f}$, sen-slope={slope21:.1e}', color=colors[0])  # Plot the mean2 for place1
    ax21.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax21.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax21.fill_between(t, mean2[0] + sigma2[0], mean2[0] - sigma2[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    ax21.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax21.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax21.set_ylim(0, 40)  # Set the Y-axis limits
    ax21.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax21.legend()  # Display legend
    ax21.set_xticklabels([])

    
    ax22.set_title(labels[1])  # Setting the title for the first subplot
    ax22.plot(t, mean2[1], label=f'{place2}, $\\tau={tau22:.2f}$, sen-slope={slope22:.1e}', color=colors[1])  # Plot the mean2 for place1
    ax22.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax22.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax22.fill_between(t, mean2[1] + sigma2[1], mean2[1] - sigma2[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    ax22.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax22.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax22.set_ylim(0, 40)  # Set the Y-axis limits
    ax22.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax22.legend()  # Display legend
    ax22.set_xticklabels([])

    
    ax23.set_title(labels[2])  # Setting the title for the first subplot
    ax23.plot(t, mean2[2], label=f'{place2}, $\\tau={tau23:.2f}$, sen-slope={slope23:.1e}', color=colors[2])  # Plot the mean2 for place1
    ax23.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the mean during no blocking
    ax23.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax23.fill_between(t, mean2[2] + sigma2[2], mean2[2] - sigma2[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    ax23.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax23.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax23.set_ylim(0, 40)  # Set the Y-axis limits
    ax23.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax23.legend()  # Display legend
    ax23.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax23.set_xticks(np.arange(0, daystoplot+1, 2))
    
    ax11.set_xlim(0,daystoplot)
    ax12.set_xlim(0,daystoplot)
    ax12.set_xlim(0,daystoplot)
    ax13.set_xlim(0,daystoplot)
    ax21.set_xlim(0,daystoplot)
    ax22.set_xlim(0,daystoplot)
    ax23.set_xlim(0,daystoplot)
    
    fig.tight_layout()
    
    if save=="pdf":
            plt.savefig("Presentation/Figures/Meanplot_pressure.pdf")
    if save=="png":
            plt.savefig("Presentation/Figures/Meanplot_pressure.png", dpi=400)    

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
        fig, ax = plt.subplots(1, 1, figsize=(9, 10), sharex=True)  # Adjust the figure size
    
        # Plot the total blocking days
        ax.plot(years, total, label="Total", color='black', linestyle='-', marker='o')
        ax.set_title("Total Blocking Days Per Year")  # Corrected title
        ax.legend()
        ax.grid(True, axis='both', linestyle='--', alpha=0.6)
  # Add grid for better visibility
        
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
        fig = plt.figure(figsize=(14, 9))

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
        
        # Add subplot labels (a), (b), (c), (d)
        ax1.text(0.95, 0.95, "(a)", transform=ax1.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax2.text(0.95, 0.95, "(b)", transform=ax2.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax3.text(0.95, 0.95, "(c)", transform=ax3.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax4.text(0.95, 0.95, "(d)", transform=ax4.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax5.text(0.95, 0.95, "(e)", transform=ax5.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax6.text(0.95, 0.95, "(f)", transform=ax6.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax7.text(0.95, 0.95, "(g)", transform=ax7.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        ax8.text(0.95, 0.95, "(h)", transform=ax8.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
        
        seasons = ["winter", "spring", "summer", "autumn", "strong", "weak", "medium", "total"]
        infostrings = {}
        
        seasons = ["winter", "spring", "summer", "autumn", "strong", "weak", "medium", "total"]
        infostrings = {}
        
        for season in seasons:
            # Get the p-value, tau, and slope for each season
            p_value = mk.original_test(eval(season), 0.05)[2]
            tau = mk.original_test(eval(season), 0.05)[4]
            slope = mk.original_test(eval(season), 0.05)[7]
            
            # Check if p-value is below 0.05 and format the string accordingly
            if p_value < 0.05:
                infostrings[season] = f"p={p_value:.1e}, $\\tau={tau:.2f}$, sen-slope={slope:.1e}"
            else:
                infostrings[season] = f"p={p_value:.3f} $\\nless$ 0.05"
                                                                                      

        # Plot the data for the first set of plots (seasons)
        ax1.plot(years, winter, label=infostrings["winter"], color='b', linestyle='-', marker='s')
        ax1.set_title("Winter")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Days")
        ax1.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax1.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax1.set_yticks(np.arange(0, max(winter), 20))
        
        ax2.plot(years, spring, label=infostrings["spring"], color='g', linestyle='-', marker='s')
        ax2.set_title("Spring")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Days")
        ax2.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax2.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax2.set_yticks(np.arange(0, max(spring), 20))
        
        ax3.plot(years, summer, label=infostrings["summer"], color='r', linestyle='-', marker='s')
        ax3.set_title("Summer")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Days")
        ax3.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax3.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax3.set_yticks(np.arange(0, max(summer), 20))

        ax4.plot(years, autumn, label=infostrings["autumn"], color='orange', linestyle='-', marker='s')
        ax4.set_title("Autumn")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Days")
        ax4.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax4.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax4.set_yticks(np.arange(0, max(autumn), 20))

        # Plot the data for the second set of plots (strength)
        ax5.plot(years, weak, label=infostrings["weak"], color='b', linestyle='-', marker='s')
        ax5.set_title("Weak")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Days")
        ax5.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax5.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax5.set_yticks(np.arange(0, max(weak), 20))

        ax6.plot(years, medium, label=infostrings["medium"], color='g', linestyle='-', marker='s')
        ax6.set_title("Medium")
        ax6.set_xlabel("Year")
        ax6.set_ylabel("Days")
        ax6.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax6.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax6.set_yticks(np.arange(0, max(medium), 20))

        ax7.plot(years, strong, label=infostrings["strong"], color='r', linestyle='-', marker='s')
        ax7.set_title("Strong")
        ax7.set_xlabel("Year")
        ax7.set_ylabel("Days")
        ax7.set_xticks(years[::12])  # Show every tenth year on the x-axis
        ax7.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax7.set_yticks(np.arange(0, max(strong), 20))

        # The large plot at the bottom
        ax8.plot(years, total, label=infostrings["total"], color='black', linestyle='-', marker='s')
        ax8.set_title("Total Blocking Days Per Year")
        ax8.set_xlabel("Year")
        ax8.set_ylabel("Days")
        ax8.set_xticks(years[::4])  # Show every fourth year on the x-axis
        ax8.set_xticklabels(years[::4], rotation=45)  # Rotate the tick labels
        ax8.set_yticks(np.arange(0, max(total), 40))  # Set major ticks every 40 units
        ax8.set_yticks(np.arange(0, max(total), 20), minor=True)  # Set minor ticks every 20 units
        ax8.grid(True, which="both", linestyle='--', alpha=0.6) # Apply grid for both major and minor ticks
        
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        ax3.legend(loc="upper left")
        ax4.legend(loc="upper left")
        ax5.legend(loc="upper left")
        ax6.legend(loc="upper left")
        ax7.legend(loc="upper left")
        ax8.legend(loc="lower left")
        
        ax1.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units
        ax2.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units
        ax3.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units
        ax4.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units
        ax5.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units
        ax6.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units
        ax7.set_yticks(np.arange(0, 81, 40))  # Set major ticks every 40 units

                            
        plt.suptitle("Number of Blocking Days Per Year ", fontsize=12, fontname='serif', x=0.5)
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

     # Save the plot if needed
    if save == "pdf":
            plt.savefig(f"Presentation/Figures/blocking_days_per_year_{typ}.pdf")
    if save == "png":
             plt.savefig(f"Presentation/Figures/blocking_days_per_year_{typ}.png", dpi=400)
        
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
    
    p1, tau1, slope1 = mk.original_test(lim1_blockings_per_year, 0.05)[2], mk.original_test(lim1_blockings_per_year, 0.05)[4], mk.original_test(lim1_blockings_per_year, 0.05)[7]
    p2, tau2, slope2 = mk.original_test(lim2_blockings_per_year, 0.05)[2], mk.original_test(lim2_blockings_per_year, 0.05)[4], mk.original_test(lim2_blockings_per_year, 0.05)[7]
    ptot, tautot, slopetot = mk.original_test(total_blockings, 0.05)[2], mk.original_test(total_blockings, 0.05)[4], mk.original_test(total_blockings, 0.05)[7]
    
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
        
        plt.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
   
    else:
        scalar = 0.85
        fig, ax = plt.subplots(figsize=(9*scalar, 5*scalar))

        # Line plots for total blockings and long blockings
        ax.plot(t, total_blockings, label=f'Total Blockings, p={ptot:.1e}, $\\tau={tautot:.2f}$, sen-slope={slopetot:.1e}', color='black', linestyle='-', marker='s', alpha=0.9)
        
        ax.plot(t, lim1_blockings_per_year, label=f'Blockings > {lim1} Days, p={p1:.1e}, $\\tau={tau1:.2f}$, sen-slope={slope1:.1e}', color='green', linestyle='-', marker='o', alpha=0.9)
        ax.plot(t, lim2_blockings_per_year, label=f'Blockings > {lim2} Days, p={p2:.1e}, $\\tau={tau2:.2f}$, sen-slope={slope2:.1e}', color='red', linestyle='-', marker='^', alpha=0.9)
        # Labels and title
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of events', fontsize=12)
        ax.set_title('Number of Blocking events Per Year and Blockings', 
                     fontsize=14, fontname='serif', x=0.5)
        
        # Set x-ticks every 3 years and rotate labels
        ax.set_xticks([i for i in range(0, len(years), 5)])  
        ax.set_xticklabels(years[::5], rotation=45)
        
        ax.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    if save=="pdf":
        plt.savefig("Presentation/Figures/BlockingsPerYear.pdf")
    if save=="png":
        plt.savefig("Presentation/Figures/BlockingsPerYear.png", dpi=400)
        
    plt.show()
    
#%%

import csv_data as csv
import time
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)
warnings.simplefilter("ignore", category=UserWarning)


info   = False          #<-------- CHANGE IF YOU WANT
imsave = "png"          # Can be pdf, png, False

press_lim   = 1014   # This is the pressure limit for classifying high pressure
dur_lim     = 5      # Minimum number of days for blocking
rain_lim    = 0.5    # Horly max rain rate
mindatasets = 8      # Minimum allowed of dattsets allowed when taking std and mean
daystoplot  = 14     # How long periods should the plots display
pm_coverege = 0.85   # How much PM2.5 coverge must the periods have
timediff = '5 hours' # This determines how far away two differnet blocking can be 


start_time = time.time()


###############################################################################

"""
Here we make the frequency plots
"""

blocking_list = find_blocking(csv.histogram_main['pressure'], 
                                   csv.histogram_main['rain'], 
                                   pressure_limit=press_lim, 
                                   duration_limit=dur_lim, 
                                   rain_limit=2/24) # This is avrege four 24 hours 


plot_blockings_by_year(blocking_list, lim1=7, lim2=10, save=imsave)


plot_blockingsdays_by_year(blocking_list, typ="all", save=imsave)



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


blocking_list_Vavihill = find_blocking(pressure_data, rain_data_Vavihill, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)


PM_data_Malmö   = csv.main['PM25']['Malmö'] 
rain_data_Malmö = csv.main['rain']["Malmö"]
wind_data_Malmö = csv.main['wind']["Malmö"]

blocking_list_Malmö = find_blocking(pressure_data, rain_data_Malmö, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)


blocking_list_Malmö, blocking_list_Vavihill = date_calibrate_blockinglists(
    blocking_list_Malmö, 
    blocking_list_Vavihill, 
    timediff, info=info)


plot_period(PM_data_Vavihill, wind_data_Vavihill, rain_data_Vavihill, 
                 pressure_data, blocking_list_Vavihill,
                 start_time='2001-01-01', 
                 end_time='2001-12-31',
                 wind_plot=False,
                 locationsave="Vavihill",
                 save=imsave)

plot_period(PM_data_Malmö, wind_data_Malmö, rain_data_Malmö, 
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
    
totdata_list_Vavihill, totdata_list_dates_Vavihill = array_blocking_list(
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


totdata_list_Malmö, totdata_list_dates_Malmö = array_blocking_list(
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


plot_mean(totdata_list1=totdata_list_Vavihill, totdata_list2=totdata_list_Malmö,
               daystoplot=daystoplot, minpoints=mindatasets, 
               place1='Vavihill', place2='Malmö', 
               pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
               pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö, 
               save=imsave)
    
if info: print(" \n ")
    
dir_totdata_list_Vavihill = sort_wind_dir(totdata_list_Vavihill, pieinfo=info)
dir_totdata_list_Malmö = sort_wind_dir(totdata_list_Malmö, pieinfo=info)

    
if info: print(" \n ")
    
seasonal_totdata_list_Vavihill = sort_season(totdata_list_Vavihill, 
                                                  totdata_list_dates_Vavihill, 
                                                  pieinfo=info)
seasonal_totdata_list_Malmö = sort_season(totdata_list_Malmö, 
                                               totdata_list_dates_Malmö, 
                                               pieinfo=info)
if info: print(" \n ")

pressure_totdata_list_Vavihill = sort_pressure(totdata_list_Vavihill, pieinfo=info)
pressure_totdata_list_Malmö = sort_pressure(totdata_list_Malmö, pieinfo=info)

if info: print(" \n ")


plot_dir_mean(dir_totdata_list1=dir_totdata_list_Vavihill, 
                   dir_totdata_list2=dir_totdata_list_Malmö, 
                   daystoplot=daystoplot,  
                   minpoints=8,
                   place1='Vavihill', place2='Malmö', save=imsave,
                   pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                   pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)



plot_pressure_mean(pressure_totdata_list1=pressure_totdata_list_Vavihill, 
                        pressure_totdata_list2=pressure_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=imsave,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)


plot_seasonal_mean(seasonal_totdata_list1=seasonal_totdata_list_Vavihill, 
                        seasonal_totdata_list2=seasonal_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=imsave,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)
    

if not info: print('3. The mean plots are now done')

if imsave: plt.close('all')
    
if not info: print(f"Elapsed time: {time.time() - start_time:.0f} seconds")

