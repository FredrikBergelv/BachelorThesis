#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:35:47 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import read_datafiles as read
import csv_data as csv
import matplotlib.colors as mcolors
import time

# Read pressure data correctly
Helsingborg = read.get_pressure_data(csv.data['pressure']['Helsingborg'])
Ängelholm = read.get_pressure_data(csv.data['pressure']['Ängelholm'])
Sturup = read.get_pressure_data(csv.data['pressure']['Sturup'])

# Define time range
start_time = '2023-6-01'
end_time   = '2023-7-31'

# Convert datetime column to pandas datetime format
Helsingborg['datetime'] = pd.to_datetime(Helsingborg['datetime'])
Ängelholm['datetime'] = pd.to_datetime(Ängelholm['datetime'])
Sturup['datetime'] = pd.to_datetime(Sturup['datetime'])

# Filter data within the given time range
mask_h = (Helsingborg['datetime'] >= start_time) & (Helsingborg['datetime'] <= end_time)
mask_a = (Ängelholm['datetime'] >= start_time) & (Ängelholm['datetime'] <= end_time)
mask_s = (Sturup['datetime'] >= start_time) & (Sturup['datetime'] <= end_time)

Helsingborg = Helsingborg[mask_h]
Ängelholm = Ängelholm[mask_a]
Sturup = Sturup[mask_s]

# Plot filtered data
plt.figure(figsize=(8,5))
plt.plot(Helsingborg['datetime'], Helsingborg['pressure'], label='Helsingborg')
plt.plot(Ängelholm['datetime'], Ängelholm['pressure'], label='Ängelholm')
plt.plot(Sturup['datetime'], Sturup['pressure'], label='Sturup')

plt.xlabel('Date')
plt.ylabel('Pressure [hPa]')
plt.title('Pressure Data Over Time for Different Stations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig("BachelorThesis/Figures/Pressure_difference.pdf")
plt.show()

#%%%

diff_HeÄn = pd.merge(Helsingborg, Ängelholm, on='datetime', suffixes=('_Helsingborg', '_Ängelholm'))
diff_HeÄn['pressure_diff'] = np.abs(diff_HeÄn['pressure_Helsingborg'] - diff_HeÄn['pressure_Ängelholm'])

diff_HeSt = pd.merge(Helsingborg, Sturup, on='datetime', suffixes=('_Helsingborg', '_Sturup'))
diff_HeSt['pressure_diff'] = np.abs(diff_HeSt['pressure_Helsingborg'] - diff_HeSt['pressure_Sturup'])

diff_ÄnSt = pd.merge(Ängelholm, Sturup, on='datetime', suffixes=('_Ängelholm', '_Sturup'))
diff_ÄnSt['pressure_diff'] = np.abs(diff_ÄnSt['pressure_Ängelholm'] - diff_ÄnSt['pressure_Sturup'])

# Plot filtered data
plt.figure(figsize=(8,5))
plt.plot(diff_HeÄn['datetime'], diff_HeÄn['pressure_diff'], label='Helsingborg - Ängelholm')
plt.plot(diff_HeSt['datetime'], diff_HeSt['pressure_diff'], label='Helsingborg - Sturup')
plt.plot(diff_ÄnSt['datetime'], diff_ÄnSt['pressure_diff'], label='Ängelholm - Sturup')


plt.xlabel('Date')
plt.ylabel('Pressure [hPa]')
plt.title('Pressure Difference Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()