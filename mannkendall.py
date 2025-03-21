#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 21:40:00 2025

@author: fredrik
"""
import pymannkendall as mk
import numpy as np
import read_datafiles as read
import csv_data as csv


press_lim   = 1015 
dur_lim     = 5 
rain_lim    = 0.5
mindatasets = 8

location = 'Vavihill'


pressure_data = csv.main['pressure']
rain_data = csv.main['rain'] 
temp_data = csv.main['temperature'] 
wind_data = csv.main['wind'] 
PM_data = csv.main['PM25'][location]



blocking_list = read.find_blocking(pressure_data, 
                                   rain_data, 
                                     pressure_limit=press_lim, 
                                     duration_limit=dur_lim, 
                                     rain_limit=rain_lim)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=1, info=False)

totdata_list_dates = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=1, only_titles=True)


        

def mean_array(data_list, minpoints=mindatasets):
    """"
    This function populates the an array by taking the nan mean of the values
    in a givven list of array for each index. This also uses minpoints to ensure
    we have enough values for our data.
    """
    
    # Determine the length of the longest dataset
    timelen = max(len(array[2]) for array in data_list)
    
    # Initialize the array with Nan
    new_array = np.full((len(data_list), timelen), np.nan)

    # Populate the PM_array with the data from data_list
    for i, array in enumerate(data_list):
        valid_len = len(array[2])  # Length of the data for this particular array
        new_array[i, :valid_len] = array[2][:valid_len]  # Fill available values
    
    # Ensure that each time point (column) has at least `minpoints` valid data points
    for t in range(timelen):
        valid_count = np.count_nonzero(~np.isnan(new_array[:, t]))  # Count valid points for each time point
        if valid_count < minpoints:
            new_array[:, t] = np.nan  # Set that time point to NaN if not enough valid points

    # Removing columns  that only contain NaNs
    new_array = new_array[:, ~np.isnan(new_array).all(axis=0)]
    # Calculate the nan mean
    mean = np.nanmean(new_array, axis=0)
    return mean

def whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept):
    """ This function tells what info to print """
    
    samllresult = f"trend={trend}, p={np.round(p, 5)}, tau={np.round(tau, 3)}"
    
    largeresult = f"trend={trend}, h={h}, p={np.round(p, 5)}, z={np.round(z, 5)}, tau={np.round(tau, 3)}, s={np.round(s, 5)}, var_s={np.round(var_s, 5)}, slope={np.round(slope, 5)}, intercept={np.round(intercept, 5)}"
    
    return samllresult
    

# Populate the PM_array with data
mean_pm_data = mean_array(totdata_list)

# Apply the Mann-Kendall test
result = mk.original_test(mean_pm_data, 0.05)
trend,h,p,z,tau,s,var_s,slope,intercept = result

# Interpretation
print("---"+location+"---")
print(f"Tot mean: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
print("")


##### THIS IS FOR THE DIRECTION ####

dir_totdata_list = read.sort_wind_dir(totdata_list)

dirlabels=["NE (310° to 70°)", "SE (70° to 190°)", "W (190° to 310°)", "No Specific"]

title = 0 # This is the value where we print the title
for k, data in enumerate(dir_totdata_list):
    # Populate the PM_array with data
    mean_pm_data = mean_array(data)

    # Check if the result is empty
    if mean_pm_data.size == 0:
        title += 1 # Next one is title if we skip
        continue  # Skip if no data
    
    # Do the Mann_kendall test
    dirresult = mk.original_test(mean_pm_data, 0.05)
    trend,h,p,z,tau,s,var_s,slope,intercept = dirresult
    
    if k == title:
        print(f"Dir mean: {dirlabels[title]}: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
    else:
        print(f"          {dirlabels[k]}: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
print("")


##### THIS IS FOR THE SEASON ####

seasonal_totdata_list = read.sort_season(totdata_list, totdata_list_dates)

sealabels=["Winter", "Spring", "Summer", "Autumn"]

title = 0 # This is the value where we print the title
for k, data in enumerate(seasonal_totdata_list):
    # Populate the PM_array with data
    mean_pm_data = mean_array(data)
    if mean_pm_data.size == 0:
        title += 1 # Next one is title if we skip
        continue  # Skip if no data
        
    # Do the Mann_kendall test
    searesult = mk.original_test(mean_pm_data, 0.05)
    trend,h,p,z,tau,s,var_s,slope,intercept = searesult
    
    if k == title:
        print(f"Seasonal mean: {sealabels[title]}: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
    else:
        print(f"               {sealabels[k]}: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
print("")
        
        
##### THIS IS FOR THE PRESSURE ####
        
pressure_totdata_list = read.sort_pressure(totdata_list)

prlabels=["Weaker Pressure Blocking", "Medium Pressure Blocking", "Stronger Pressure Blocking", "None"]

title = 0 # This is the value where we print the title
for k, data in enumerate(pressure_totdata_list):
    # Populate the PM_array with data
    mean_pm_data = mean_array(data)
    if mean_pm_data.size == 0:
        title += 1 # Next one is title if we skip
        continue  # Skip if no data
        
     # Do the Mann_kendall test
    prresult = mk.original_test(mean_pm_data, 0.05)
    trend,h,p,z,tau,s,var_s,slope,intercept = prresult
    
    if k == title:
        print(f"Strength mean: {prlabels[title]}: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
    else:
        print(f"               {prlabels[k]}: {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)}")
print("")
