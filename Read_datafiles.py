#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:04:20 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

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
    
    # Identify if next value is not high pressure or not (shift)
    # If the next value is not the same, add the value of True(1) cumulative 
    # This gives a unique streak_id for each streak depending on the limit
    data['streak_id'] = (data['highp'] != data['highp'].shift()).cumsum()
    
    # Group by streak_id and calculate the duration of each streak
    streaks = data.groupby('streak_id').agg(
        start_date = ('datetime', 'first'),  # start is the first datetime in each id
        end_date = ('datetime', 'last'),     # end is the last datetime in each id
        duration_days = ('datetime', lambda date: (date.max() - date.min()).days + 1), # Calculate the duration, in days
        highp = ('highp', 'max')             # Since all highp first/max, all give the same
    )
    # Filter for streaks with high pressure lasting at least the right number of days
    # We aslo filter the streaks which are over 100 days, due to problems whith combining data
    blocking = streaks[(streaks['highp'] == True) & (streaks['duration_days'] >= duration_limit) & (streaks['duration_days'] < 100)]
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
        print(f'All periods with an air pressure at least {pressure_limit} hPa during at least a {duration_limit} day period')
        print(blocking)
        print(f'A total of {len(datalist)} high pressure blockings were found between {min(data["datetime"])} and {max(data["datetime"])}')
        
    return datalist  # Return a list of all the blocking data

def particle_and_blocking_list(PM_data, SMHI_block_list, 
                               only_titles=False, plot=False, info=False):
    "This function takes in the particle data and the pressure blocking data"
    "It returns a list of datafiles for each blocking period with timestamp, pressure, PM2.5"
    "We can also plot if we want to"
        
    totdata_list = []
    
    title_list = []
    
    # Loop through each blocking period
    for i in range(len(SMHI_block_list)):
        
        block_data = SMHI_block_list[i]
        
        # Plotting the pressure data against the datetime for different locations
        start_time = block_data['datetime'].min()
        end_time = block_data['datetime'].max()
        
        # Filter both datasets to the overlapping time range
        PM_data_trimmed = PM_data[(PM_data['datetime_start'] >= start_time) & (PM_data['datetime_end'] <= end_time)]
        
        # Drop rows with NaN in the PM2.5 column
        PM_data_trimmed = PM_data_trimmed.dropna(subset=['pm2.5'])
        
        # If PM_data is empty do not plot
        if PM_data_trimmed.empty:
            if info == True:
                print(f"Skipping plot {start_time} to {end_time} due to no PM2.5 data coverage.")
            continue
        
        # SInce data is taken every hour, ensure that we have neogh data
        expected_data = (end_time - start_time).total_seconds() / 3600  # Blocking duration in hours
        actual_data = len(PM_data_trimmed) # Data coverge of PM_data in hours
        coverage = actual_data/expected_data
        if coverage < 0.9:
            if info == True:
                print(f"Skipping plot {start_time} to {end_time} due to only {100*np.round(coverage,2)}% PM2.5 data coverage.")
            continue
         
        # Store the dates if we want later on
        if only_titles == True:
                title_list.append(f'Data from {start_time} to {end_time}') 
                
        # Here we merge the datafiles based on the closest timestamp
        combined_data = pd.merge_asof(
            PM_data_trimmed, 
            block_data, 
            left_on='datetime_start', # Look at the datetime_start
            right_on='datetime',      # Look at the datetime
            direction='nearest'       # Find the closest Pressure and PM2.5 value to the date
        )[['datetime', 'pressure', 'pm2.5']]
        
        totdata_list.append(combined_data)
                
        if plot == True:
            plt.figure(figsize=(10, 6))
            plt.plot(block_data['datetime'], block_data['pressure'], label='Air Pressure [hPa]')
            plt.plot(PM_data_trimmed['datetime_start'], PM_data_trimmed['pm2.5'] + 1000, label='Vavihill PM2.5 + 1000 [µg/m³]')
            
            plt.title=(f'PM2.5 concentration and air pressure between {start_time} to {end_time}')
            plt.xlabel('Date and Time')
            plt.ylabel('')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()
            
        totdata_list.append(combined_data)
        
    if only_titles == True:
            return title_list   
        
    return totdata_list # Return list of all the datafiles

def array_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                       start_time, end_time, info=False, plot=False):
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
        plt.show()
    
    return array # Return list of all the datafiles

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



"""
PM_Vavihill = get_pm_data(Vavihill_PM25)
PM_Hallahus = get_pm_data(Hallahus_PM25)
PM_data = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)

wind_data = get_wind_data(Hörby_wind)
temp_data = get_temp_data(Hörby_temperature)
rain_data = get_rain_data(Hörby_rain)
pres_data = get_pressure_data(Helsingborg_pressure)


SMHI_block_list = find_blocking(pres_data, rain_data, 
                                     pressure_limit = 1015, 
                                     duration_limit = 5, 
                                     rain_limit = 2,
                                     info=False)
"""

#SMHI_data = get_pressure_data(Helsingborg, plot=True)
#PM_Vavihill = get_pm_data(Vavihill_PM25,True)
#PM_Hallahus = get_pm_data(Hallahus_PM25, True)





