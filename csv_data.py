#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:31:19 2025

@author: fredrik
"""

import os
import pandas as pd
import read_datafiles as read

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:31:19 2025

@author: fredrik
"""

import pandas as pd
import read_datafiles as read

# Define direct file paths

data = {
    "PM25": {
        "Vavihill": "csv_files/Vavihill_PM25.csv",
        "Hallahus": "csv_files/Hallahus_PM25.csv",
        "Malmö": "csv_files/Malmö_PM25.csv",
        "Hornsgatan": "csv_files/Hornsgatan_PM25.csv",
        "Skellefteå": "csv_files/Skellefteå_PM25.csv"
    },
    "pressure": {
        "Ängelholm": "csv_files/Ängelholm_pressure.csv",
        "Sturup": "csv_files/Sturup_pressure.csv",
        "Helsingborg": "csv_files/Helsingborg_pressure.csv",
        "Stockholm": "csv_files/smhi-Stockholm_pressure.csv"
    },
    "wind": {
        "Helsingborg": "csv_files/Helsingborg_wind.csv",
        "Hörby": "csv_files/Hörby_wind.csv",
        "Sturup": "csv_files/Sturup_wind.csv"
    },
    "rain": {
        "Helsingborg": "csv_files/Helsingborg_rain.csv",
        "Hörby": "csv_files/Hörby_rain.csv",
        "Örja": "csv_files/Örja_rain.csv"
    },
    "temperature": {
        "Hörby": "csv_files/Hörby_temperature.csv",
        "Lund": "csv_files/Lund_temperature.csv",
        "Sturup": "csv_files/Sturup_temperature.csv"
    },
    "other": {
        "Vavihill_blackC": "csv_files/Vavihill_blackC.csv",
        "Vavihill_O3": "csv_files/Vavihill_O3.csv"
    }
}

# Reading data using read_datafiles.py

PM_Malmö = read.get_pm_data(data["PM25"]["Malmö"])

PM_Vavihill = read.get_pm_data(data["PM25"]["Vavihill"])
PM_Hallahus = read.get_pm_data(data["PM25"]["Hallahus"])
PM_Vavihill = pd.concat([PM_Vavihill, PM_Hallahus], axis=0)
PM_Vavihill = PM_Vavihill.sort_values(by='datetime_start', ascending=False).drop_duplicates(
    subset=['datetime_start'], keep='first'
).sort_values(by='datetime_start').reset_index(drop=True)  # Reset index

# Read Pressure Data
Ängelholm_data = read.get_pressure_data(data["pressure"]["Ängelholm"])
Helsingborg_data = read.get_pressure_data(data["pressure"]["Helsingborg"])
pres_data = pd.concat([Helsingborg_data, Ängelholm_data], axis=0)
pres_data = pres_data.sort_values(by='datetime', ascending=False).drop_duplicates(
    subset=['datetime'], keep='first'
).sort_values(by='datetime').reset_index(drop=True)  # Reset index

# Read Rain Data
Örja_data = read.get_rain_data(data["rain"]["Örja"])
Hörby_data = read.get_rain_data(data["rain"]["Hörby"])
rain_data = pd.concat([Hörby_data, Örja_data], axis=0)
rain_data = rain_data.sort_values(by='datetime', ascending=False).drop_duplicates(
    subset=['datetime'], keep='first'
).sort_values(by='datetime').reset_index(drop=True)  # Reset index

# Read Wind & Temperature Data
wind_data = read.get_wind_data(data["wind"]["Hörby"])
temp_data = read.get_temp_data(data["temperature"]["Hörby"])

# Store all processed data in `main` dictionary
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
