#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:34:04 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import read_datafiles as read
import csv_data as csv
import time
from collections import defaultdict


start_time = time.time()

pressure_data = csv.main['pressure']
rain_data = csv.main['rain'] 



blocking_list = read.find_blocking(pressure_data, rain_data, 
                                     pressure_limit = 1015, 
                                     duration_limit = 5, 
                                     rain_limit = 0.2,
                                     info=False)


"""
histogram(block_list, 'summer', save=True)
histogram(block_list, 'autumn', save=True)
histogram(block_list, 'winter', save=True)
histogram(block_list, 'spring', save=True)
"""

read.plot_blockings_by_year(blocking_list, lim=7, save=False)



read.plot_blockingsdays_by_year(blocking_list, 'all')

print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


