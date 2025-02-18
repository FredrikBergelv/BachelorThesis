#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:04:20 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import Read_datafiles as read
import time as time 

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
Vavihill_O3             = 'Vavihill_O3.csv'
Örja_rain               = 'Örja_rain.csv'

start_time = time.time()

PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_data = pd.concat([PM_Vavihill, PM_Hallahus])

wind_data = read.get_wind_data(Hörby_wind)
temp_data = read.get_temp_data(Hörby_temperature)
rain_data = read.get_rain_data(Hörby_rain)
pressure_data = read.get_pressure_data(Helsingborg_pressure)
SMHI_block_list = read.find_blocking(pressure_data, rain_data, 
                                     pressure_limit=1015, 
                                     duration_limit=5, 
                                     rain_limit=5)


read.plot_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data,
                  SMHI_block_list,
                  start_time='2018-01-01', 
                  end_time='2018-12-31',
                  save=False)

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
