#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 21:40:00 2025

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

location = 'Malmö'

pressure_data = csv.main['pressure']
temp_data = csv.main['temperature'] 
PM_data = csv.main['PM25'][location]

if location == "Malmö":
     rain_data = csv.main['rain']["Malmö"]
     wind_data = csv.main['wind']["Malmö"]
     
if location == "Vavihill":
      rain_data = csv.main['rain']["Hörby"]
      wind_data = csv.main['wind']["Hörby"]

blocking_list = read.find_blocking(pressure_data, 
                                   rain_data, 
                                     pressure_limit=press_lim, 
                                     duration_limit=dur_lim, 
                                     rain_limit=rain_lim)

totdata_list = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=pm_coverege, info=False)

totdata_list_dates = read.array_extra_blocking_list(PM_data, wind_data, 
                                              temp_data, rain_data, 
                                              blocking_list, 
                                              cover=pm_coverege, only_titles=True)

##############################################################################
def mean_array(data_list, minpoints=mindatasets):
    """"
    This function populates the an array by taking the nan mean of the values
    in a givven list of array for each index. This also uses minpoints to ensure
    we have enough values for our data.
    """
    
    # Determine the length of the longest dataset
    timelen = max(len(array[2]) for array in data_list)
    
    # Initialize the array with Nan
    new_array = np.full((len(data_list), timelen), np.nan)

    # Populate the PM_array with the data from data_list
    for i, array in enumerate(data_list):
        valid_len = len(array[2])  # Length of the data for this particular array
        new_array[i, :valid_len] = array[2][:valid_len]  # Fill available values
    
    # Ensure that each time point (column) has at least `minpoints` valid data points
    for t in range(timelen):
        valid_count = np.count_nonzero(~np.isnan(new_array[:, t]))  # Count valid points for each time point
        if valid_count < minpoints:
            new_array[:, t] = np.nan  # Set that time point to NaN if not enough valid points

    # Removing columns  that only contain NaNs
    new_array = new_array[:, ~np.isnan(new_array).all(axis=0)]
    # Calculate the nan mean
    mean = np.nanmean(new_array, axis=0)
    return mean

def whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept):
    samllresult = f"& {np.round(p, 5)} & {np.round(tau, 3)}"
    largeresult = f"trend={trend}, h={h}, p={np.round(p, 5)}, z={np.round(z, 5)}, tau={np.round(tau, 3)}, s={np.round(s, 5)}, var_s={np.round(var_s, 5)}, slope={np.round(slope, 5)}, intercept={np.round(intercept, 5)}"
    return samllresult

# Generate Mann-Kendall Test for both "Malmö" and "Vavihill"
def generate_latex_table(location, totdata_list, totdata_list_dates):
    # Calculate Mann-Kendall test results
    mean_pm_data = mean_array(totdata_list)
    result = mk.original_test(mean_pm_data, 0.05)
    trend,h,p,z,tau,s,var_s,slope,intercept = result

    # For the Direction category:
    dir_totdata_list = read.sort_wind_dir(totdata_list)
    dirlabels = ["NE (310° to 70°)", "SE (70° to 190°)", "W (190° to 310°)", "No Specific Direction"]

    direction_results = []
    for k, data in enumerate(dir_totdata_list):
        mean_pm_data = mean_array(data)
        if mean_pm_data.size == 0:
            continue
        dirresult = mk.original_test(mean_pm_data, 0.05)
        trend,h,p,z,tau,s,var_s,slope,intercept = dirresult
        direction_results.append((dirlabels[k], np.round(p, 5), np.round(tau, 3)))

    # For the Season category:
    seasonal_totdata_list = read.sort_season(totdata_list, totdata_list_dates)
    sealabels = ["Winter", "Spring", "Summer", "Autumn"]

    season_results = []
    for k, data in enumerate(seasonal_totdata_list):
        mean_pm_data = mean_array(data)
        if mean_pm_data.size == 0:
            continue
        searesult = mk.original_test(mean_pm_data, 0.05)
        trend,h,p,z,tau,s,var_s,slope,intercept = searesult
        season_results.append((sealabels[k], np.round(p, 5), np.round(tau, 3)))

    # For the Pressure category:
    pressure_totdata_list = read.sort_pressure(totdata_list)
    prlabels = ["Weaker", "Medium", "Stronger"]

    pressure_results = []
    for k, data in enumerate(pressure_totdata_list):
        mean_pm_data = mean_array(data)
        if mean_pm_data.size == 0:
            continue
        prresult = mk.original_test(mean_pm_data, 0.05)
        trend,h,p,z,tau,s,var_s,slope,intercept = prresult
        pressure_results.append((prlabels[k], np.round(p, 5), np.round(tau, 3)))

    # Generate the LaTeX table string with the total mean result included
    latex_table = f"""
    \\begin{{minipage}}{{0.45\\textwidth}}
        \\centering
        \\caption{{Mann-Kendall Test for {location}}}
        \\label{{tab:{location}}}
        \\begin{{tabular}}{{@{{}}cccc@{{}}}}
        \\toprule
        \\textbf{{Category}} & \\textbf{{Interpretation}} & \\textbf{{p-value}} & \\textbf{{Tau}} \\\\ \\midrule
    """

    # Add the total mean result for the entire dataset
    latex_table += f"    \\multirow{{1}}{{*}}{{\\textbf{{Total Mean}}}} & {whattoprint(trend,h,p,z,tau,s,var_s,slope,intercept)} \\\\ \\midrule"

    # Fill in the direction results
    latex_table += f"    \\multirow{{4}}{{*}}{{\\textbf{{Direction}}}}"
    for label, p_val, tau_val in direction_results:
        latex_table += f" & {label} & {p_val} & {tau_val} \\\\ "

    # Fill in the season results
    latex_table += f" \\midrule   \\multirow{{4}}{{*}}{{\\textbf{{Season}}}}"
    for label, p_val, tau_val in season_results:
        latex_table += f" & {label} & {p_val} & {tau_val} \\\\ "

    # Fill in the pressure results
    latex_table += f" \\midrule    \\multirow{{4}}{{*}}{{\\textbf{{Pressure}}}}"
    for label, p_val, tau_val in pressure_results:
        latex_table += f" & {label} & {p_val} & {tau_val} \\\\ "

    latex_table += """
        \\bottomrule
        \\end{tabular}
    \\end{minipage}
    """

    return latex_table

# Generate tables for both locations
latex_table_malmo = generate_latex_table("Malmö", totdata_list, totdata_list_dates)
latex_table_vavihill = generate_latex_table("Vavihill", totdata_list, totdata_list_dates)

# Create LaTeX code to display both tables side by side
final_latex = f"""
\\begin{{table}}[h!]
    \\centering
    {latex_table_malmo} \\hfill {latex_table_vavihill}
\\end{{table}}
"""

print(final_latex)
