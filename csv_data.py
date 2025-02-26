#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:31:19 2025

@author: fredrik
"""

import os
import pandas as pd
import read_datafiles as read

# Define the base directory for CSV files
CSV_DIR = "csv_files"

"""
Below we store all the data in a dictionary called `data` for referencing CSV files.
"""

data = {
    "PM25": {
        "Vavihill": os.path.join(CSV_DIR, "Vavihill_PM25.csv"),
        "Hallahus": os.path.join(CSV_DIR, "Hallahus_PM25.csv"),
        "Malmö": os.path.join(CSV_DIR, "Malmö_PM25.csv"),
        "Hornsgatan": os.path.join(CSV_DIR, "Hornsgatan_PM25.csv"),
        "Skellefteå": os.path.join(CSV_DIR, "Skellefteå_PM25.csv")
    },
    "Pressure": {
        "Ängelholm": os.path.join(CSV_DIR, "Ängelholm_pressure.csv"),
        "Sturup": os.path.join(CSV_DIR, "Sturup_pressure.csv"),
        "Helsingborg": os.path.join(CSV_DIR, "Helsingborg_pressure.csv"),
        "Stockholm": os.path.join(CSV_DIR, "smhi-Stockholm_pressure.csv")
    },
    "Wind": {
        "Helsingborg": os.path.join(CSV_DIR, "Helsingborg_wind.csv"),
        "Hörby": os.path.join(CSV_DIR, "Hörby_wind.csv"),
        "Sturup": os.path.join(CSV_DIR, "Sturup_wind.csv")
    },
    "Rain": {
        "Helsingborg": os.path.join(CSV_DIR, "Helsingborg_rain.csv"),
        "Hörby": os.path.join(CSV_DIR, "Hörby_rain.csv"),
        "Örja": os.path.join(CSV_DIR, "Örja_rain.csv")
    },
    "Temperature": {
        "Hörby": os.path.join(CSV_DIR, "Hörby_temperature.csv"),
        "Lund": os.path.join(CSV_DIR, "Lund_temperature.csv"),
        "Sturup": os.path.join(CSV_DIR, "Sturup_temperature.csv")
    },
    "Other": {
        "Vavihill_blackC": os.path.join(CSV_DIR, "Vavihill_blackC.csv"),
        "Vavihill_O3": os.path.join(CSV_DIR, "Vavihill_O3.csv")
    }
}

"""
Direct file path references for easier access.
"""
Stockholm_pressure = os.path.join(CSV_DIR, "smhi-Stockholm_pressure.csv")
Vavihill_blackC = os.path.join(CSV_DIR, "Vavihill_blackC.csv")
Sturup_temperature = os.path.join(CSV_DIR, "Sturup_temperature.csv")
Lund_temperature = os.path.join(CSV_DIR, "Lund_temperature.csv")
Sturup_wind = os.path.join(CSV_DIR, "Sturup_wind.csv")
Vavihill_PM25 = os.path.join(CSV_DIR, "Vavihill_PM25.csv")
Hallahus_PM25 = os.path.join(CSV_DIR, "Hallahus_PM25.csv")
Ängelholm_pressure = os.path.join(CSV_DIR, "Ängelholm_pressure.csv")
Sturup_pressure = os.path.join(CSV_DIR, "Sturup_pressure.csv")
Helsingborg_pressure = os.path.join(CSV_DIR, "Helsingborg_pressure.csv")
Helsingborg_wind = os.path.join(CSV_DIR, "Helsingborg_wind.csv")
Helsingborg_rain = os.path.join(CSV_DIR, "Helsingborg_rain.csv")
Hörby_wind = os.path.join(CSV_DIR, "Hörby_wind.csv")
Hörby_temperature = os.path.join(CSV_DIR, "Hörby_temperature.csv")
Hörby_rain = os.path.join(CSV_DIR, "Hörby_rain.csv")
Vavihill_O3 = os.path.join(CSV_DIR, "Vavihill_O3.csv")
Örja_rain = os.path.join(CSV_DIR, "Örja_rain.csv")
Malmö_PM25 = os.path.join(CSV_DIR, "Malmö_PM25.csv")

"""
Reading data using read_datafiles.py and processing it.
"""

# Read PM2.5 Data
PM_Malmö = read.get_pm_data(Malmö_PM25)

PM_Vavihill = read.get_pm_data(Vavihill_PM25)
PM_Hallahus = read.get_pm_data(Hallahus_PM25)
PM_Vavihill = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)
PM_Vavihill = PM_Vavihill.sort_values(by='datetime_start', ascending=False).drop_duplicates(
    subset=['datetime_start'], keep='first'
).sort_values(by='datetime_start').reset_index(drop=True)  # Reset index

# Read Pressure Data
Ängelholm_data = read.get_pressure_data(Ängelholm_pressure)
Helsingborg_data = read.get_pressure_data(Helsingborg_pressure)
pres_data = pd.concat([Helsingborg_data, Ängelholm_data], axis=0)  # Combine the data
pres_data = pres_data.sort_values(by='datetime', ascending=False).drop_duplicates(
    subset=['datetime'], keep='first'
).sort_values(by='datetime').reset_index(drop=True)  # Reset index

# Read Rain Data
Örja_data = read.get_rain_data(Örja_rain)
Hörby_data = read.get_rain_data(Hörby_rain)
rain_data = pd.concat([Hörby_data, Örja_data], axis=0)
rain_data = rain_data.sort_values(by='datetime', ascending=False).drop_duplicates(
    subset=['datetime'], keep='first'
).sort_values(by='datetime').reset_index(drop=True)  # Reset index

# Read Wind & Temperature Data
wind_data = read.get_wind_data(Hörby_wind)
temp_data = read.get_temp_data(Hörby_temperature)

"""
Store all processed data in `main` dictionary.
"""
main = {
    "PM25": {
        "Vavihill": PM_Vavihill,
        "Malmö": PM_Malmö
    },
    "pressure": pres_data,
    "wind": wind_data,
    "rain": rain_data,
    "temperature": temp_data
}
