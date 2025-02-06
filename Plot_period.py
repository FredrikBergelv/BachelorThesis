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

PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_data = pd.concat([PM_Vavihill, PM_Hallahus])

wind_data = read.get_wind_data(Hörby_wind)
temp_data = read.get_temp_data(Hörby_temperature)
rain_data = read.get_rain_data(Hörby_rain)
pressure_data = read.get_pressure_data(Helsingborg_pressure)


array = read.array_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data, 
                                start_time = '2001-01-01', 
                                end_time = '2001-02-01', plot = True)




"""
years = [(f"{year}-01-01", f"{year+1}-01-01") for year in range(1999, 2023)]

events = [
    ("2022-02-20", "2022-04-10"),
    ("2019-01-14", "2019-03-06"),
    ("2019-03-06", "2019-05-05"),
    ("2018-10-02", "2018-10-27"),
    ("2017-01-01", "2017-02-25"),
    ("2012-01-10", "2012-02-18"),
    ("2011-02-12", "2011-03-09"),
    ("2010-09-12", "2010-11-03"),
    ("2009-11-25", "2009-12-31"),
    ("2006-01-01", "2006-02-07"),
    ("2005-02-03", "2005-02-12"),
    ("2003-01-06", "2003-03-07"), # Exciting period
    ("2002-03-25", "2002-04-04"), # Exciting period
    ("2002-08-08", "2002-08-29"),
    ("2001-01-10", "2001-01-28"), # Exciting period
    ("2001-03-03", "2001-03-10"),
    ("2000-10-09", "2000-10-27")]


for time in events:
    array = read.array_extra_period(PM_data, wind_data, temp_data, rain_data, pressure_data, 
                                    start_time = time[0],
                                    end_time = time[1],
                                    plot = True)

"""



