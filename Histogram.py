#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:34:04 2025

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

def histogram(block_list, season):
    """We want to show the number of blockings per year"""
    
    years = [] 
    
    # Make a loop to find all the relevant years
    for data in block_list:
        start, end = min(data['datetime']), max(data['datetime'])
        
        # Find the mean date and extract the year
        year = (start + (end-start)/2 ).year
        
        if year not in years:
            years.append(year)  # Add the year to the list if it's unique
    
    # Make a dictionary with years as keys and values [0, 0] (for winter, summer)
    blocking = {year: [0, 0, 0, 0] for year in years} 
        
    for data in block_list:
        
        start, end = min(data['datetime']), max(data['datetime'])
        duration = (end - start).days
        
        # Find the date and month
        date = (start + (end - start) / 2)
        month = date.month
        year = date.year
        
        # Add to the winter or summer blocking duration
        if month == 12 or month == 1 or month == 2:
            blocking[year][0] += duration  # Winter
            
        elif month == 3 or month == 4 or month == 5:
            blocking[year][1] += duration  # Spring
            
        elif month == 6 or month == 7 or month == 8:
            blocking[year][2] += duration  # Summer
            
        elif month == 9 or month == 10 or month == 11:
            blocking[year][3] += duration  # Autumn
        
    # Remove the first and last years since they are not full years
    blocking.pop(min(blocking))  # Remove the first year
    blocking.pop(max(blocking))  # Remove the last year
   
    # Extract the data as lists
    winter = [values[0] for values in blocking.values()]  # Winter blocking
    spring = [values[1] for values in blocking.values()]  # Summer blocking
    summer = [values[2] for values in blocking.values()]  # Summer blocking
    autumn = [values[3] for values in blocking.values()]  # Summer blocking

    total = [values[0] + values[1] + values[2] + values[3] for values in blocking.values()]  # Total blocking days
    years = list(blocking.keys())  # Years list
    
    # Plotting
    plt.figure(figsize=(10, 6))  # Set a wider figure size for clarity

    plt.bar(years, total, label="Total", color='#D3D3D3', edgecolor='black', alpha=0.5)
    
    # Plot individual seasons
    if season == 'winter':
        plt.bar(years, winter, label="Winter", color='b', edgecolor='black', alpha=0.9)
    elif season == 'spring':
        plt.bar(years, spring, label="Spring", color='g', edgecolor='black', alpha=0.9)
    elif season == 'summer':
        plt.bar(years, summer, label="Summer", color='r', edgecolor='black', alpha=0.9)
    elif season == 'autumn':
        plt.bar(years, autumn, label="Autumn", color='orange', edgecolor='black', alpha=0.9)
    elif season == 'all':
        width = 0.2  # Bar width for side-by-side effect
        plt.bar([year - width*1.5 for year in years], winter, width=width, label="Winter", color='b', edgecolor='black', alpha=0.9)
        plt.bar([year - width/2 for year in years], spring, width=width, label="Spring", color='g', edgecolor='black', alpha=0.9)
        plt.bar([year + width/2 for year in years], summer, width=width, label="Summer", color='r', edgecolor='black', alpha=0.9)
        plt.bar([year + width*1.5 for year in years], autumn, width=width, label="Autumn", color='orange', edgecolor='black', alpha=0.9)
    else:
        raise ValueError(f'season must be "winter", "spring", "summer", "autumn", or "all", not {season}')
    
    # Labels and title with improved font sizes
    plt.xlabel("Year")
    plt.ylabel("Days of Blocking")
    plt.title("Number of Days Under High-Pressure Blocking Per Year")
    plt.xticks(years, rotation=45)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()  
    plt.show()




Hörby_data = read.get_rain_data(Hörby_rain)
Örja_data =  read.get_rain_data(Örja_rain)
rain_data = pd.concat([Örja_data, Hörby_data], axis=0)

Ängelholm_data = read.get_pressure_data(Ängelholm_pressure)
Helsingborg_data = read.get_pressure_data(Helsingborg_pressure)
pres_data = pd.concat([Ängelholm_data, Helsingborg_data], axis=0)


block_list = read.find_blocking(pres_data, rain_data, 
                                     pressure_limit = 1015, 
                                     duration_limit = 5, 
                                     rain_limit = 0.2,
                                     info=True)

histogram(block_list, 'summer')


