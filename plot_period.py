#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:04:20 2025

@author: fredrik
"""

import read_datafiles as read
import time as time 
import csv_data as csv


start_time = time.time()

pressure_data = csv.main['pressure']
rain_data = csv.main['rain'] 
temp_data = csv.main['temperature'] 
wind_data = csv.main['wind'] 
PM_data = csv.main['PM25']['Vavihill']


blocking_list = read.find_blocking(pressure_data, 
                                   rain_data, 
                                     pressure_limit=1015, 
                                     duration_limit=5, 
                                     rain_limit=5)

read.plot_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                       blocking_list,
                       start_time='2018-01-01', 
                       end_time='2018-12-31',
                       save=False)

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
