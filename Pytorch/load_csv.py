#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:19:41 2025

@author: fredrik
"""


import numpy as np
import read_datafiles as read
import csv_data as csv
import pandas as pd


location = 'Vavihill'


pressure_data = csv.main['pressure']
rain_data = csv.main['rain'] 
temp_data = csv.main['temperature'] 
wind_data = csv.main['wind'] 
PM_data = csv.main['PM25'][location]

# Rename 'datetime_start' in PM data to 'datetime' for consistency
PM_data = PM_data.rename(columns={'datetime_start': 'datetime'})

# Merge all dataframes on the 'datetime' column
merged_df = pressure_data.merge(rain_data, on='datetime', how='outer')\
                         .merge(temp_data, on='datetime', how='outer')\
                         .merge(wind_data, on='datetime', how='outer')

# Sort by datetime to ensure chronological order
merged_df = merged_df.sort_values(by='datetime')

# Drop rows with any NaN values
merged_df = merged_df.dropna()

# Drop the 'datetime_end' column if it exists
if 'datetime_end' in merged_df.columns:
    merged_df = merged_df.drop(columns=['datetime_end'])

# Move 'pressure' before 'datetime'
columns = list(merged_df.columns)
columns.remove('pressure')  # Remove 'pressure' from the list
columns.insert(1, 'pressure')  # Insert 'pressure' at the beginning
merged_df = merged_df[columns]  # Reorder columns



merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

# Group by date (without time)
daily_stats = merged_df.groupby(merged_df['datetime'].dt.date).agg({
    'pressure': 'mean',  # Mean pressure
    'rain': 'sum',       # Total rainfall
    'temp': 'max',       # Max temperature
    'speed': 'mean',     # Mean wind speed
}).reset_index()

# Rename columns for clarity
daily_stats.rename(columns={'datetime': 'date'}, inplace=True)

# Display result
print(daily_stats)



import torch

# Convert 'date' to datetime and sort by date
daily_stats['date'] = pd.to_datetime(daily_stats['date'])
daily_stats = daily_stats.sort_values(by='date')


# Convert features and labels to tensors
features = torch.tensor(daily_stats.iloc[:-1, 1:].values, dtype=torch.float64)  # All except last row (X)
targets = torch.tensor(daily_stats.iloc[1:, 1:].values, dtype=torch.float64)   # All except first row (y)
