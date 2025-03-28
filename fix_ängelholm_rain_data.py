#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:50:34 2025

@author: fredrik
"""
import pymannkendall as mk
import numpy as np
import read_datafiles as read
import csv_data as csv


press_lim   = 1014   # This is the pressure limit for classifying high pressure
dur_lim     = 5      # Minimum number of days for blocking
rain_lim    = 0.5    # Horly max rain rate
mindatasets = 8      # Minimum allowed of dattsets allowed when taking std and mean
pm_coverege = 0.95    # How much PM2.5 coverge must the periods have


raindata  = read.get_rain_data(csv.data["rain"]["Ängelholm"])

blocking_list = read.find_blocking(csv.histogram_main['pressure'], 
                                   raindata, 
                                   pressure_limit=press_lim, 
                                   duration_limit=dur_lim, 
                                   rain_limit=rain_lim*24)


read.plot_blockings_by_year(blocking_list, lim=7, save=False)
