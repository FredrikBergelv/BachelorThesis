#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:27:44 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import read_datafiles as read
import csv_data as csv
import time
import warnings


# Assuming Vavihill is your DataFrame
Vavihill = read.get_pm_data(csv.data["temperature"]["Lund"])


print(f" Start date {Vavihill["datetime_start"][0]}")
print(f" End date {Vavihill["datetime_start"][len(Vavihill)-1]}")


# 1. Length of the DataFrame in days (assuming hourly data)
length_in_days = len(Vavihill) / 24

# 2. Length where there are no NaN values
length_val = len(Vavihill.dropna()) / 24

# 3. Fraction of non-NaN values
fraction_non_nan = length_val / length_in_days

# Output the results
print(f"Length of DataFrame in days: {length_in_days:.0f}")
print(f"Length with no NaN values: {length_val:.0f}")
print(f"Fraction of non-NaN values: {100*fraction_non_nan:.0f}%")
