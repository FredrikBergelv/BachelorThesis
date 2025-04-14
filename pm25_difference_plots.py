#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:51:51 2025

@author: fredrik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import read_datafiles as read
import csv_data as csv
import warnings
import matplotlib.gridspec as gridspec
import pymannkendall as mk

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)
warnings.simplefilter("ignore", category=UserWarning)
plt.close('all')


def plot_diffmean(totdata_list1, totdata_list2, 
                  daystoplot, minpoints=8, place1='', place2='',
                  pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
                  save=False):
    """
    This function takes the mean of the PM2.5 concentration for each hour.
    You must specify how many days you wish to plot, you can add a wind title, 
    the number of datasets needed, plot info, etc.
    """
    timelen = int(24 * daystoplot)  # Initial length in hours

    # Create an array to store all the PM2.5 values
    PM_array1 = np.full((len(totdata_list1), timelen), np.nan)
    PM_array2 = np.full((len(totdata_list2), timelen), np.nan)

    
    # Populate the PM_array with data
    for i, array in enumerate(totdata_list1):
        valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
        PM_array1[i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for i, array in enumerate(totdata_list2):
         valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
         PM_array2[i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs
    mean1, sigma1 = np.nanmean(PM_array1, axis=0), np.nanstd(PM_array1, axis=0)
    mean2, sigma2 = np.nanmean(PM_array2, axis=0), np.nanstd(PM_array2, axis=0)

    t = np.arange(timelen) / 24  # Time axis in days   
    
    # Below we check the number of data points
    valid_counts_per_hour1 = np.sum(~np.isnan(PM_array1), axis=0)
    valid_counts_per_hour2 = np.sum(~np.isnan(PM_array2), axis=0)
    
    #create subfgure
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)  
    fig.suptitle(r'Relative Mean Concentration of PM$_{{2.5}}$',
                 fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.4, 1])  # Top row is twice as tall
    
    # Create subplots using GridSpec
    ax1 = fig.add_subplot(gs[0, 0])  # Large top-left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Large top-right plot
    ax3 = fig.add_subplot(gs[1, 0])  # Smaller bottom-left plot
    ax4 = fig.add_subplot(gs[1, 1])  # Smaller bottom-right plot
    
    tau1, slope1 = mk.original_test(mean1, 0.05)[4], mk.original_test(mean1, 0.05)[7]
    tau2, slope2 = mk.original_test(mean2, 0.05)[4], mk.original_test(mean2, 0.05)[7]
    
    # Add subplot labels (a), (b), (c), (d)
    ax1.text(0.95, 0.95, "(a)", transform=ax1.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax2.text(0.95, 0.95, "(b)", transform=ax2.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax3.text(0.95, 0.95, "(c)", transform=ax3.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax4.text(0.95, 0.95, "(d)", transform=ax4.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    # Plotting for ax1
    for i, points in enumerate(valid_counts_per_hour1):
        if points < minpoints:
            mean1[i] = np.nan
            sigma1[i] = np.nan
                
    ax1.plot(t, mean1, label=f'{place1}, $\\tau={tau1:.2f}$, sen-slope={slope1:.1e}', c='C0')
    ax1.plot(t, pm_mean1 + t * 0, label='Mean at start of event ', c='gray')
    ax1.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray') 
    ax1.fill_between(t, mean1 + sigma1, mean1 - sigma1, alpha=0.4, color='C0')
    #ax1.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')
    ax1.set_xlabel('Time from start of blocking [days]')
    ax1.set_ylabel('Relative Mean Concentration [PM2.5 (µg/m³)]')
    ax1.set_ylim(-10, 30)
    ax1.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plotting for ax3
    for i, points in enumerate(valid_counts_per_hour2):
        if points < minpoints:
            mean2[i] = np.nan
            sigma2[i] = np.nan
    
    ax2.plot(t, mean2, label=f'{place2}, $\\tau={tau2:.2f}$, sen-slope={slope2:.1e}', c='C0')
    #ax2.plot(t, t * 0 + 25, c='r', linestyle='--')
    ax2.plot(t, pm_mean2 + t * 0, c='gray')
    ax2.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')
    ax2.fill_between(t, mean2 + sigma2, mean2 - sigma2, alpha=0.4, color='C0')
    ax2.set_xlabel('Time from start of blocking [days]')
    ax2.set_ylabel('PM$_{{2.5}}$ [µg/m³]')
    ax2.set_ylim(-10, 30)
    ax2.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax2.legend()

    
    # Plotting for ax2
    ax3.plot(t, valid_counts_per_hour1, label=f'Number of datasets, {place1}')
    #ax3.set_title(f'Number of datasets at {place1}', fontsize=13, fontname='serif', x=0.5)
    ax3.set_xlabel('Time from start of blocking [days]')
    ax3.set_ylabel('Number of datasets')
    ax3.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5, label='Minimum number of datasets allowed')
    ax3.set_yticks(np.arange(0, 201, 50))
    ax3.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left')

    
    # Plotting for ax4
    ax4.plot(t, valid_counts_per_hour2, label=f'Number of datasets, {place2} ')
    #ax4.set_title(f'Number of datasets at {place2}', fontsize=13, fontname='serif', x=0.5)
    ax4.set_xlabel('Time from start of blocking [days]')
    ax4.set_ylabel('Number of datasets')
    ax4.axhline(y=minpoints, color='red', linestyle='--', linewidth=1.5)
    ax4.set_yticks(np.arange(0, 201, 50))
    ax4.grid(True, axis='both', linestyle='--', alpha=0.6)
    ax4.legend(loc='center left')

    
    fig.tight_layout()
    fig.show()
    
    if save:
        plt.savefig("BachelorThesis/Figures/Meanplot.pdf")
    plt.show() 
    
def plot_dir_diffmean(dir_totdata_list1, dir_totdata_list2, daystoplot,  
                      minpoints=8, place1='', place2='', save=False,
                      pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
                      labels=["NE (310° to 70°)", "SE (70° to 190°)", "W (190° to 310°)", "No Specific"]):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each wind direction category in subplots.
    It displays plots side by side for place1 and place2.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold", "orange"]  # Colors mapped to NE, SE, W, Turning, non

    lenmax = dir_totdata_list1[0]+dir_totdata_list1[1]+dir_totdata_list1[2]+dir_totdata_list1[3]
    # Create an array to store all the PM2.5 values
    PM_array1 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]
    PM_array2 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]

    # Populate the PM_array with data
    for k, totodatatlist in enumerate(dir_totdata_list1):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array1[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for k, totodatatlist in enumerate(dir_totdata_list2):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array2[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs for each direction (place 1 and place 2)
    mean1 = [np.nanmean(PM_array1[k], axis=0) for k in range(4)]
    mean2 = [np.nanmean(PM_array2[k], axis=0) for k in range(4)]

    sigma1 = [np.nanstd(PM_array1[k], axis=0) for k in range(4)]
    sigma2 = [np.nanstd(PM_array2[k], axis=0) for k in range(4)]

    # Compute valid counts per hour
    valid_counts_per_hour1 = [np.sum(~np.isnan(PM_array1[k]), axis=0) for k in range(4)]
    valid_counts_per_hour2 = [np.sum(~np.isnan(PM_array2[k]), axis=0) for k in range(4)]

    # if points are lower tha nminpoints set to nan 
    for i in range(4):
        for j, counts in enumerate(valid_counts_per_hour1[i]):
            if counts < minpoints:
                mean1[i][j] = np.nan
                sigma1[i][j] = np.nan
        for j, counts in enumerate(valid_counts_per_hour2[i]):
            if counts < minpoints:
                mean2[i][j] = np.nan
                sigma2[i][j] = np.nan

    t = np.arange(timelen) / 24  # Time axis in days   
    
    #create subfgure
    scalingfactor = 1.1
    fig = plt.figure(figsize=(9*scalingfactor, 7.5*scalingfactor), constrained_layout=True)  
    fig.suptitle(r'Relative Mean Concentration of PM$_{{2.5}}$',
                 fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  
    
    # Create subplots using GridSpec
    ax11 = fig.add_subplot(gs[0, 0])  
    ax12 = fig.add_subplot(gs[1, 0])  
    ax13 = fig.add_subplot(gs[2, 0])  
    ax14 = fig.add_subplot(gs[3, 0])  
    ax21 = fig.add_subplot(gs[0, 1])  
    ax22 = fig.add_subplot(gs[1, 1])  
    ax23 = fig.add_subplot(gs[2, 1])  
    ax24 = fig.add_subplot(gs[3, 1])  
    
    tau11, slope11 = mk.original_test(mean1[0], 0.05)[4], mk.original_test(mean1[0], 0.05)[7]
    tau12, slope12 = mk.original_test(mean1[1], 0.05)[4], mk.original_test(mean1[1], 0.05)[7]
    tau13, slope13 = mk.original_test(mean1[2], 0.05)[4], mk.original_test(mean1[2], 0.05)[7]
    tau14, slope14 = mk.original_test(mean1[3], 0.05)[4], mk.original_test(mean1[3], 0.05)[7]
    tau21, slope21 = mk.original_test(mean2[0], 0.05)[4], mk.original_test(mean2[0], 0.05)[7]
    tau22, slope22 = mk.original_test(mean2[1], 0.05)[4], mk.original_test(mean2[1], 0.05)[7]
    tau23, slope23 = mk.original_test(mean2[2], 0.05)[4], mk.original_test(mean2[2], 0.05)[7]
    tau24, slope24 = mk.original_test(mean2[3], 0.05)[4], mk.original_test(mean2[3], 0.05)[7]

    
    # Add subplot labels (a), (b), (c), (d)
    ax11.text(0.95, 0.95, "(a)", transform=ax11.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax12.text(0.95, 0.95, "(c)", transform=ax12.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax13.text(0.95, 0.95, "(e)", transform=ax13.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax14.text(0.95, 0.95, "(g)", transform=ax14.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax21.text(0.95, 0.95, "(b)", transform=ax21.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax22.text(0.95, 0.95, "(d)", transform=ax22.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax23.text(0.95, 0.95, "(f)", transform=ax23.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax24.text(0.95, 0.95, "(h)", transform=ax24.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    ax11.set_title('Direction: ' + labels[0])  # Setting the title for the first subplot
    ax11.plot(t, mean1[0], label=f'{place1}, $\\tau={tau11:.2f}$, sen-slope={slope11:.1e}', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, pm_mean1 + t * 0, label='Mean at start of event ', c='gray')  # Plot the Mean at start of event 
    ax11.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax11.fill_between(t, mean1[0] + sigma1[0], mean1[0] - sigma1[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    #ax11.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')  # Plot the EU annual mean limit
    ax11.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax11.set_ylim(-10, 30)  # Set the Y-axis limits
    ax11.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax11.legend()  # Display legend
    ax11.set_xticklabels([])
    
    ax12.set_title('Direction: ' + labels[1])  # Setting the title for the first subplot
    ax12.plot(t, mean1[1], label=f'{place1}, $\\tau={tau12:.2f}$, sen-slope={slope12:.1e}', color=colors[1])  # Plot the mean1 for place1
    ax12.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax12.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax12.fill_between(t, mean1[1] + sigma1[1], mean1[1] - sigma1[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    #ax12.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax12.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax12.set_ylim(-10, 30)  # Set the Y-axis limits
    ax12.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax12.legend()  # Display legend
    ax12.set_xticklabels([])

    
    ax13.set_title('Direction: ' + labels[2])  # Setting the title for the first subplot
    ax13.plot(t, mean1[2], label=f'{place1}, $\\tau={tau13:.2f}$, sen-slope={slope13:.1e}', color=colors[2])  # Plot the mean1 for place1
    ax13.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax13.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax13.fill_between(t, mean1[2] + sigma1[2], mean1[2] - sigma1[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    #ax13.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax13.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax13.set_ylim(-10, 30)  # Set the Y-axis limits
    ax13.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax13.legend()  # Display legend
    ax13.set_xticklabels([])

    
    ax14.set_title('Direction: ' + labels[3])  # Setting the title for the first subplot
    ax14.plot(t, mean1[3], label=f'{place1}, $\\tau={tau14:.2f}$, sen-slope={slope14:.1e}', color=colors[3])  # Plot the mean1 for place1
    ax14.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax14.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax14.fill_between(t, mean1[3] + sigma1[3], mean1[3] - sigma1[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    #ax14.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax14.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax14.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax14.set_ylim(-10, 30)  # Set the Y-axis limits
    ax14.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax14.legend()  # Display legend
    ax14.set_xticks(np.arange(0, daystoplot+1, 2))

    
    ax21.set_title('Direction: ' + labels[0])  # Setting the title for the first subplot
    ax21.plot(t, mean2[0], label=f'{place2}, $\\tau={tau21:.2f}$, sen-slope={slope21:.1e}', color=colors[0])  # Plot the mean2 for place1
    ax21.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax21.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax21.fill_between(t, mean2[0] + sigma2[0], mean2[0] - sigma2[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    #ax21.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax21.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax21.set_ylim(-10, 30)  # Set the Y-axis limits
    ax21.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax21.legend()  # Display legend
    ax21.set_xticklabels([])

    
    ax22.set_title('Direction: ' + labels[1])  # Setting the title for the first subplot
    ax22.plot(t, mean2[1], label=f'{place2}, $\\tau={tau22:.2f}$, sen-slope={slope22:.1e}', color=colors[1])  # Plot the mean2 for place1
    ax22.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax22.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax22.fill_between(t, mean2[1] + sigma2[1], mean2[1] - sigma2[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    #ax22.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax22.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax22.set_ylim(-10, 30)  # Set the Y-axis limits
    ax22.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax22.legend()  # Display legend
    ax22.set_xticklabels([])

    
    ax23.set_title('Direction: ' + labels[2])  # Setting the title for the first subplot
    ax23.plot(t, mean2[2], label=f'{place2}, $\\tau={tau23:.2f}$, sen-slope={slope23:.1e}', color=colors[2])  # Plot the mean2 for place1
    ax23.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax23.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax23.fill_between(t, mean2[2] + sigma2[2], mean2[2] - sigma2[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    #ax23.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax23.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax23.set_ylim(-10, 30)  # Set the Y-axis limits
    ax23.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax23.legend()  # Display legend
    ax23.set_xticklabels([])

    
    ax24.set_title('Direction: ' + labels[3])  # Setting the title for the first subplot
    ax24.plot(t, mean2[3], label=f'{place2}, $\\tau={tau24:.2f}$, sen-slope={slope24:.1e}', color=colors[3])  # Plot the mean2 for place1
    ax24.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax24.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax24.fill_between(t, mean2[3] + sigma2[3], mean2[3] - sigma2[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    #ax24.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax24.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax24.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax24.set_ylim(-10, 30)  # Set the Y-axis limits
    ax24.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax24.legend()  # Display legend
    ax24.set_xticks(np.arange(0, daystoplot+1, 2))

    
    
    fig.tight_layout()
    
    if save:
        plt.savefig("BachelorThesis/Figures/Meanplot_dir.pdf")
    plt.show()

def plot_seasonal_diffmean(seasonal_totdata_list1, seasonal_totdata_list2, daystoplot,  
                           minpoints=8, place1='', place2='', save=False,
                           pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
                           labels=["Winter", "Spring", "Summer", "Autumn"]):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each season category in subplots.
    It displays plots side by side for place1 and place2.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "tomato", "seagreen", "gold", "orange"]  # Colors mapped to NE, SE, W, Turning, non

    lenmax = seasonal_totdata_list1[0]+seasonal_totdata_list1[1]+seasonal_totdata_list1[2]+seasonal_totdata_list1[3]
    
    # Create an array to store all the PM2.5 values
    PM_array1 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]
    PM_array2 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]

    # Populate the PM_array with data
    for k, totodatatlist in enumerate(seasonal_totdata_list1):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array1[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for k, totodatatlist in enumerate(seasonal_totdata_list2):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array2[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs for each direction (place 1 and place 2)
    mean1 = [np.nanmean(PM_array1[k], axis=0) for k in range(4)]
    mean2 = [np.nanmean(PM_array2[k], axis=0) for k in range(4)]

    sigma1 = [np.nanstd(PM_array1[k], axis=0) for k in range(4)]
    sigma2 = [np.nanstd(PM_array2[k], axis=0) for k in range(4)]

    # Compute valid counts per hour
    valid_counts_per_hour1 = [np.sum(~np.isnan(PM_array1[k]), axis=0) for k in range(4)]
    valid_counts_per_hour2 = [np.sum(~np.isnan(PM_array2[k]), axis=0) for k in range(4)]

    # if points are lower tha nminpoints set to nan 
    for i in range(4):
            for j, counts in enumerate(valid_counts_per_hour1[i]):
                if counts < minpoints:
                    mean1[i][j] = np.nan
                    sigma1[i][j] = np.nan
            for j, counts in enumerate(valid_counts_per_hour2[i]):
                if counts < minpoints:
                    mean2[i][j] = np.nan
                    sigma2[i][j] = np.nan
                    
    t = np.arange(timelen) / 24  # Time axis in days   
        
    #create subfgure
    scalingfactor = 1.1
    fig = plt.figure(figsize=(9*scalingfactor, 7.5*scalingfactor), constrained_layout=True)
    fig.suptitle(r'Relative Mean Concentration of PM$_{{2.5}}$',
                     fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])  
        
    # Create subplots using GridSpec
    ax11 = fig.add_subplot(gs[0, 0])  
    ax12 = fig.add_subplot(gs[1, 0])  
    ax13 = fig.add_subplot(gs[2, 0])  
    ax14 = fig.add_subplot(gs[3, 0])  
    ax21 = fig.add_subplot(gs[0, 1])  
    ax22 = fig.add_subplot(gs[1, 1])  
    ax23 = fig.add_subplot(gs[2, 1])  
    ax24 = fig.add_subplot(gs[3, 1])  
    
    tau11, slope11 = mk.original_test(mean1[0], 0.05)[4], mk.original_test(mean1[0], 0.05)[7]
    tau12, slope12 = mk.original_test(mean1[1], 0.05)[4], mk.original_test(mean1[1], 0.05)[7]
    tau13, slope13 = mk.original_test(mean1[2], 0.05)[4], mk.original_test(mean1[2], 0.05)[7]
    tau14, slope14 = mk.original_test(mean1[3], 0.05)[4], mk.original_test(mean1[3], 0.05)[7]
    tau21, slope21 = mk.original_test(mean2[0], 0.05)[4], mk.original_test(mean2[0], 0.05)[7]
    tau22, slope22 = mk.original_test(mean2[1], 0.05)[4], mk.original_test(mean2[1], 0.05)[7]
    tau23, slope23 = mk.original_test(mean2[2], 0.05)[4], mk.original_test(mean2[2], 0.05)[7]
    tau24, slope24 = mk.original_test(mean2[3], 0.05)[4], mk.original_test(mean2[3], 0.05)[7]

    
       # Add subplot labels (a), (b), (c), (d)
    ax11.text(0.95, 0.95, "(a)", transform=ax11.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax12.text(0.95, 0.95, "(c)", transform=ax12.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax13.text(0.95, 0.95, "(e)", transform=ax13.transAxes, fontsize=12, fontname='serif', ha='right', va='top')        
    ax14.text(0.95, 0.95, "(g)", transform=ax14.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax21.text(0.95, 0.95, "(b)", transform=ax21.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax22.text(0.95, 0.95, "(d)", transform=ax22.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax23.text(0.95, 0.95, "(f)", transform=ax23.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax24.text(0.95, 0.95, "(h)", transform=ax24.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    ax11.set_title(labels[0])  # Setting the title for the first subplot
    ax11.plot(t, mean1[0], label=f'{place1}, $\\tau={tau11:.2f}$, sen-slope={slope11:.1e}', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, pm_mean1 + t * 0, label='Mean at start of event ', c='gray')  # Plot the Mean at start of event 
    ax11.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax11.fill_between(t, mean1[0] + sigma1[0], mean1[0] - sigma1[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    #ax11.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')  # Plot the EU annual mean limit
    ax11.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax11.set_ylim(-10, 30)  # Set the Y-axis limits
    ax11.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax11.legend()  # Display legend
    ax11.set_xticklabels([])
        
    ax12.set_title(labels[1])  # Setting the title for the first subplot
    ax12.plot(t, mean1[1], label=f'{place1}, $\\tau={tau12:.2f}$, sen-slope={slope12:.1e}', color=colors[1])  # Plot the mean1 for place1
    ax12.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax12.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax12.fill_between(t, mean1[1] + sigma1[1], mean1[1] - sigma1[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    #ax12.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax12.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax12.set_ylim(-10, 30)  # Set the Y-axis limits
    ax12.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax12.legend()  # Display legend
    ax12.set_xticklabels([])

        
    ax13.set_title(labels[2])  # Setting the title for the first subplot
    ax13.plot(t, mean1[2], label=f'{place1}, $\\tau={tau13:.2f}$, sen-slope={slope13:.1e}', color=colors[2])  # Plot the mean1 for place1
    ax13.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax13.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax13.fill_between(t, mean1[2] + sigma1[2], mean1[2] - sigma1[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    #ax13.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax13.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax13.set_ylim(-10, 30)  # Set the Y-axis limits
    ax13.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax13.legend()  # Display legend
    ax13.set_xticklabels([])

        
    ax14.set_title(labels[3])  # Setting the title for the first subplot
    ax14.plot(t, mean1[3], label=f'{place1}, $\\tau={tau14:.2f}$, sen-slope={slope14:.1e}', color=colors[3])  # Plot the mean1 for place1
    ax14.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax14.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax14.fill_between(t, mean1[3] + sigma1[3], mean1[3] - sigma1[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    #ax14.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax14.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax14.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax14.set_ylim(-10, 30)  # Set the Y-axis limits
    ax14.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax14.legend()  # Display legend
    ax14.set_xticks(np.arange(0, daystoplot+1, 2))

        
    ax21.set_title(labels[0])  # Setting the title for the first subplot
    ax21.plot(t, mean2[0], label=f'{place2}, $\\tau={tau21:.2f}$, sen-slope={slope21:.1e}', color=colors[0])  # Plot the mean2 for place1
    ax21.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax21.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax21.fill_between(t, mean2[0] + sigma2[0], mean2[0] - sigma2[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    #ax21.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax21.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax21.set_ylim(-10, 30)  # Set the Y-axis limits
    ax21.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax21.legend()  # Display legend
    ax21.set_xticklabels([])

        
    ax22.set_title(labels[1])  # Setting the title for the first subplot
    ax22.plot(t, mean2[1], label=f'{place2}, $\\tau={tau22:.2f}$, sen-slope={slope22:.1e}', color=colors[1])  # Plot the mean2 for place1
    ax22.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax22.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax22.fill_between(t, mean2[1] + sigma2[1], mean2[1] - sigma2[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    #ax22.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax22.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax22.set_ylim(-10, 30)  # Set the Y-axis limits
    ax22.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax22.legend()  # Display legend
    ax22.set_xticklabels([])

        
    ax23.set_title(labels[2])  # Setting the title for the first subplot
    ax23.plot(t, mean2[2], label=f'{place2}, $\\tau={tau23:.2f}$, sen-slope={slope23:.1e}', color=colors[2])  # Plot the mean2 for place1
    ax23.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax23.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax23.fill_between(t, mean2[2] + sigma2[2], mean2[2] - sigma2[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    #ax23.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax23.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax23.set_ylim(-10, 30)  # Set the Y-axis limits
    ax23.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax23.legend()  # Display legend
    ax23.set_xticklabels([])

        
    ax24.set_title(labels[3])  # Setting the title for the first subplot
    ax24.plot(t, mean2[3], label=f'{place2}, $\\tau={tau24:.2f}$, sen-slope={slope24:.1e}', color=colors[3])  # Plot the mean2 for place1
    ax24.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax24.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax24.fill_between(t, mean2[3] + sigma2[3], mean2[3] - sigma2[3], alpha=0.4, color=colors[3])  # Confidence interval for place1
    #ax24.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax24.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax24.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax24.set_ylim(-10, 30)  # Set the Y-axis limits
    ax24.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax24.legend()  # Display legend
    ax24.set_xticks(np.arange(0, daystoplot+1, 2))

        
        
    fig.tight_layout()
        
    if save:
            plt.savefig("BachelorThesis/Figures/Meanplot_seasonal.pdf")
    plt.show()

def plot_pressure_diffmean(pressure_totdata_list1, pressure_totdata_list2, daystoplot,  
                           minpoints=8, place1='', place2='', save=False,
                           pm_mean1=False, pm_sigma1=False, pm_mean2=False, pm_sigma2=False,
                           labels=["Weaker Pressure Blocking", "Medium Pressure Blocking", "Stronger Pressure Blocking"], ):
    
    """
    This function takes the mean of the PM2.5 concentration for each hour 
    and plots it separately for each strength category in subplots.
    It displays plots side by side for place1 and place2.
    """
    timelen = int(24 * daystoplot) 
    colors = ["royalblue", "seagreen", "tomato"]  # Colors mapped to NE, SE, W, Turning, non

    lenmax = pressure_totdata_list1[0]+pressure_totdata_list1[1]+pressure_totdata_list1[2]
    # Create an array to store all the PM2.5 values
    PM_array1 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]
    PM_array2 = [np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan), np.full((len(lenmax), timelen), np.nan)]

    # Populate the PM_array with data
    for k, totodatatlist in enumerate(pressure_totdata_list1):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array1[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
        
    for k, totodatatlist in enumerate(pressure_totdata_list2):
        for i, array in enumerate(totodatatlist):
            valid_len = min(len(array[2]), timelen)  # Take the minimum to avoid index errors
            PM_array2[k][i, :valid_len] = array[2][:valid_len]  # Fill available values
      
    # Compute mean and standard deviation, ignoring NaNs for each direction (place 1 and place 2)
    mean1 = [np.nanmean(PM_array1[k], axis=0) for k in range(3)]
    mean2 = [np.nanmean(PM_array2[k], axis=0) for k in range(3)]

    sigma1 = [np.nanstd(PM_array1[k], axis=0) for k in range(3)]
    sigma2 = [np.nanstd(PM_array2[k], axis=0) for k in range(3)]

    # Compute valid counts per hour
    valid_counts_per_hour1 = [np.sum(~np.isnan(PM_array1[k]), axis=0) for k in range(3)]
    valid_counts_per_hour2 = [np.sum(~np.isnan(PM_array2[k]), axis=0) for k in range(3)]

    # if points are lower tha nminpoints set to nan 
    for i in range(3):
        for j, counts in enumerate(valid_counts_per_hour1[i]):
            if counts < minpoints:
                mean1[i][j] = np.nan
                sigma1[i][j] = np.nan
        for j, counts in enumerate(valid_counts_per_hour2[i]):
            if counts < minpoints:
                mean2[i][j] = np.nan
                sigma2[i][j] = np.nan

    t = np.arange(timelen) / 24  # Time axis in days   
    
    #create subfgure
    scalingfactor = 1.1
    fig = plt.figure(figsize=(9*scalingfactor, 7.5*3/4*scalingfactor), constrained_layout=True)  
    fig.suptitle(r'Relative Mean Concentration of PM$_{{2.5}}$',
                 fontsize=13, fontname='serif', x=0.5,)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  
    
    # Create subplots using GridSpec
    ax11 = fig.add_subplot(gs[0, 0])  
    ax12 = fig.add_subplot(gs[1, 0])  
    ax13 = fig.add_subplot(gs[2, 0])  
    ax21 = fig.add_subplot(gs[0, 1])  
    ax22 = fig.add_subplot(gs[1, 1])  
    ax23 = fig.add_subplot(gs[2, 1])  
    
    tau11, slope11 = mk.original_test(mean1[0], 0.05)[4], mk.original_test(mean1[0], 0.05)[7]
    tau12, slope12 = mk.original_test(mean1[1], 0.05)[4], mk.original_test(mean1[1], 0.05)[7]
    tau13, slope13 = mk.original_test(mean1[2], 0.05)[4], mk.original_test(mean1[2], 0.05)[7]
    tau21, slope21 = mk.original_test(mean2[0], 0.05)[4], mk.original_test(mean2[0], 0.05)[7]
    tau22, slope22 = mk.original_test(mean2[1], 0.05)[4], mk.original_test(mean2[1], 0.05)[7]
    tau23, slope23 = mk.original_test(mean2[2], 0.05)[4], mk.original_test(mean2[2], 0.05)[7]
    
    # Add subplot labels (a), (b), (c), (d)
    ax11.text(0.95, 0.95, "(a)", transform=ax11.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax12.text(0.95, 0.95, "(c)", transform=ax12.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax13.text(0.95, 0.95, "(e)", transform=ax13.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax21.text(0.95, 0.95, "(b)", transform=ax21.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax22.text(0.95, 0.95, "(d)", transform=ax22.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    ax23.text(0.95, 0.95, "(f)", transform=ax23.transAxes, fontsize=12, fontname='serif', ha='right', va='top')
    
    ax11.set_title(labels[0])  # Setting the title for the first subplot
    ax11.plot(t, mean1[0], label=f'{place1}, $\\tau={tau11:.2f}$, sen-slope={slope11:.1e}', color=colors[0])  # Plot the mean1 for place1
    ax11.plot(t, pm_mean1 + t * 0, label='Mean at start of event ', c='gray')  # Plot the Mean at start of event 
    ax11.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax11.fill_between(t, mean1[0] + sigma1[0], mean1[0] - sigma1[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    #ax11.plot(t, t * 0 + 25, label='EU annual mean limit', c='r', linestyle='--')  # Plot the EU annual mean limit
    ax11.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax11.set_ylim(-10, 30)  # Set the Y-axis limits
    ax11.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax11.legend()  # Display legend
    ax11.set_xticklabels([])
    
    ax12.set_title(labels[1])  # Setting the title for the first subplot
    ax12.plot(t, mean1[1], label=f'{place1}, $\\tau={tau12:.2f}$, sen-slope={slope12:.1e}', color=colors[1])  # Plot the mean1 for place1
    ax12.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax12.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax12.fill_between(t, mean1[1] + sigma1[1], mean1[1] - sigma1[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    #ax12.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax12.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax12.set_ylim(-10, 30)  # Set the Y-axis limits
    ax12.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax12.legend()  # Display legend
    ax12.set_xticklabels([])

    
    ax13.set_title(labels[2])  # Setting the title for the first subplot
    ax13.plot(t, mean1[2], label=f'{place1}, $\\tau={tau13:.2f}$, sen-slope={slope13:.1e}', color=colors[2])  # Plot the mean1 for place1
    ax13.plot(t, pm_mean1 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax13.fill_between(t, pm_mean1 + t * 0 + pm_sigma1, pm_mean1 + t * 0 - pm_sigma1, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax13.fill_between(t, mean1[2] + sigma1[2], mean1[2] - sigma1[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    #ax13.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax13.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax13.set_ylim(-10, 30)  # Set the Y-axis limits
    ax13.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax13.legend()  # Display legend
    ax13.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax13.set_xticks(np.arange(0, daystoplot+1, 2))

    
    ax21.set_title(labels[0])  # Setting the title for the first subplot
    ax21.plot(t, mean2[0], label=f'{place2}, $\\tau={tau21:.2f}$, sen-slope={slope21:.1e}', color=colors[0])  # Plot the mean2 for place1
    ax21.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax21.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax21.fill_between(t, mean2[0] + sigma2[0], mean2[0] - sigma2[0], alpha=0.4, color=colors[0])  # Confidence interval for place1
    #ax21.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax21.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax21.set_ylim(-10, 30)  # Set the Y-axis limits
    ax21.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax21.legend()  # Display legend
    ax21.set_xticklabels([])

    
    ax22.set_title(labels[1])  # Setting the title for the first subplot
    ax22.plot(t, mean2[1], label=f'{place2}, $\\tau={tau22:.2f}$, sen-slope={slope22:.1e}', color=colors[1])  # Plot the mean2 for place1
    ax22.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax22.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax22.fill_between(t, mean2[1] + sigma2[1], mean2[1] - sigma2[1], alpha=0.4, color=colors[1])  # Confidence interval for place1
    #ax22.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax22.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax22.set_ylim(-10, 30)  # Set the Y-axis limits
    ax22.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax22.legend()  # Display legend
    ax22.set_xticklabels([])

    
    ax23.set_title(labels[2])  # Setting the title for the first subplot
    ax23.plot(t, mean2[2], label=f'{place2}, $\\tau={tau23:.2f}$, sen-slope={slope23:.1e}', color=colors[2])  # Plot the mean2 for place1
    ax23.plot(t, pm_mean2 + t * 0, c='gray')  # Plot the Mean at start of event 
    ax23.fill_between(t, pm_mean2 + t * 0 + pm_sigma2, pm_mean2 + t * 0 - pm_sigma2, alpha=0.4, color='gray')  # Confidence interval for no blocking
    ax23.fill_between(t, mean2[2] + sigma2[2], mean2[2] - sigma2[2], alpha=0.4, color=colors[2])  # Confidence interval for place1
    #ax23.plot(t, t * 0 + 25, c='r', linestyle='--')  # Plot the EU annual mean limit
    ax23.set_ylabel('PM$_{{2.5}}$ [µg/m³]')  # Y-axis label
    ax23.set_ylim(-10, 30)  # Set the Y-axis limits
    ax23.grid(True, axis='both', linestyle='--', alpha=0.6)  # Enable grid with style
    ax23.legend()  # Display legend
    ax23.set_xlabel('Time from start of blocking [days]')  # X-axis label
    ax23.set_xticks(np.arange(0, daystoplot+1, 2))
    
    fig.tight_layout()
    
    if save:
        plt.savefig("BachelorThesis/Figures/Meanplot_pressure.pdf")
    plt.show()



def pm_mean_diff(totlist, starthours=12):
    """
    This function applies baseline correction to the PM2.5 data in each event.
    It subtracts the mean of the first `starthours` values from each PM2.5 time series.
    It also returns the average standard deviation of the baseline periods.
    """
            
    for i, event in enumerate(totlist):
            pm25_series = event[2]
            baseline = np.nanmean(pm25_series[0:starthours])
            
            corrected_pm25 = pm25_series - baseline
            totlist[i][2] = corrected_pm25
                
    return totlist
    

    


info = False          #<-------- CHANGE IF YOU WANT

press_lim   = 1014   # This is the pressure limit for classifying high pressure
dur_lim     = 5     # Minimum number of days for blocking
rain_lim    = 0.5    # Horly max rain rate
mindatasets = 8      # Minimum allowed of dattsets allowed when taking std and mean
daystoplot  = 14     # How long periods should the plots display
pm_coverege = 0.80   # How much PM2.5 coverge must the periods have


###############################################################################


pressure_data = csv.main['pressure']
temp_data = csv.main['temperature'] 

PM_data_Vavihill   = csv.main['PM25']['Vavihill'] 
rain_data_Vavihill = csv.main['rain']["Hörby"]
wind_data_Vavihill = csv.main['wind']["Hörby"]
         

blocking_list_Vavihill = read.find_blocking(pressure_data, rain_data_Vavihill, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)

    
totdata_list_Vavihill, totdata_list_dates_Vavihill = read.array_blocking_list(
                                                  PM_data_Vavihill, 
                                                  wind_data_Vavihill, 
                                                  rain_data_Vavihill, 
                                                  blocking_list_Vavihill, 
                                                  cover=pm_coverege, 
                                                  info=info)

totdata_list_Vavihill = pm_mean_diff(totdata_list_Vavihill)


block_datafile_Vavihill= pd.concat(blocking_list_Vavihill, ignore_index=True)
PM_without_blocking_Vavihill = PM_data_Vavihill[~PM_data_Vavihill['datetime_start'].isin(block_datafile_Vavihill['datetime'])]

pm_mean_Vavihill = 0
pm_sigma_Vavihill = np.nanstd(np.array(PM_without_blocking_Vavihill['pm2.5']))



PM_data_Malmö   = csv.main['PM25']['Malmö'] 
rain_data_Malmö = csv.main['rain']["Malmö"]
wind_data_Malmö = csv.main['wind']["Malmö"]

blocking_list_Malmö = read.find_blocking(pressure_data, rain_data_Malmö, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)

totdata_list_Malmö, totdata_list_dates_Malmö = read.array_blocking_list(
                                                  PM_data_Malmö, 
                                                  wind_data_Malmö, 
                                                  rain_data_Malmö, 
                                                  blocking_list_Malmö, 
                                                  cover=pm_coverege, 
                                                  info=info)

totdata_list_Malmö = pm_mean_diff(totdata_list_Malmö)

    

block_datafile_Malmö = pd.concat(blocking_list_Malmö, ignore_index=True)
PM_without_blocking_Malmlö = PM_data_Malmö[~PM_data_Malmö['datetime_start'].isin(block_datafile_Malmö['datetime'])]

pm_mean_Malmö = 0
pm_sigma_Malmö = np.nanstd(np.array(PM_without_blocking_Malmlö['pm2.5']))





plot_diffmean(totdata_list1=totdata_list_Vavihill, totdata_list2=totdata_list_Malmö,
               daystoplot=daystoplot, minpoints=mindatasets, 
               place1='Vavihill', place2='Malmö', 
               pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
               pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö, 
               save=False)
    
    
dir_totdata_list_Vavihill = read.sort_wind_dir(totdata_list_Vavihill, pieinfo=info)
dir_totdata_list_Malmö = read.sort_wind_dir(totdata_list_Malmö, pieinfo=info)

    
    
seasonal_totdata_list_Vavihill = read.sort_season(totdata_list_Vavihill, 
                                                  totdata_list_dates_Vavihill, 
                                                  pieinfo=info)
seasonal_totdata_list_Malmö = read.sort_season(totdata_list_Malmö, 
                                               totdata_list_dates_Malmö, 
                                               pieinfo=info)

pressure_totdata_list_Vavihill = read.sort_pressure(totdata_list_Vavihill, pieinfo=info)
pressure_totdata_list_Malmö = read.sort_pressure(totdata_list_Malmö, pieinfo=info)




plot_dir_diffmean(dir_totdata_list1=dir_totdata_list_Vavihill, 
                   dir_totdata_list2=dir_totdata_list_Malmö, 
                   daystoplot=daystoplot,  
                   minpoints=8,
                   place1='Vavihill', place2='Malmö', save=False,
                   pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                   pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)



plot_pressure_diffmean(pressure_totdata_list1=pressure_totdata_list_Vavihill, 
                        pressure_totdata_list2=pressure_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=False,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)


plot_seasonal_diffmean(seasonal_totdata_list1=seasonal_totdata_list_Vavihill, 
                        seasonal_totdata_list2=seasonal_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=False,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)
    
#%%

""" Here is the orignal plots """


PM_data_Vavihill   = csv.main['PM25']['Vavihill'] 


blocking_list_Vavihill = read.find_blocking(pressure_data, rain_data_Vavihill, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)
    
totdata_list_Vavihill, totdata_list_dates_Vavihill = read.array_blocking_list(
                                                  PM_data_Vavihill, 
                                                  wind_data_Vavihill, 
                                                  rain_data_Vavihill, 
                                                  blocking_list_Vavihill, 
                                                  cover=pm_coverege, 
                                                  info=info)

block_datafile_Vavihill = pd.concat(blocking_list_Vavihill, ignore_index=True)
PM_without_blocking_Vavihill = PM_data_Vavihill[~PM_data_Vavihill['datetime_start'].isin(block_datafile_Vavihill['datetime'])]

pm_mean_Vavihill= np.nanmean(np.array(PM_without_blocking_Vavihill['pm2.5']))
pm_sigma_Vavihill = np.nanstd(np.array(PM_without_blocking_Vavihill['pm2.5']))

PM_data_Malmö   = csv.main['PM25']['Malmö'] 

blocking_list_Malmö = read.find_blocking(pressure_data, rain_data_Malmö, 
                                       pressure_limit=press_lim, 
                                       duration_limit=dur_lim, 
                                       rain_limit=rain_lim,
                                       info=info)

totdata_list_Malmö, totdata_list_dates_Malmö = read.array_blocking_list(
                                                  PM_data_Malmö, 
                                                  wind_data_Malmö, 
                                                  rain_data_Malmö, 
                                                  blocking_list_Malmö, 
                                                  cover=pm_coverege, 
                                                  info=info)
    

block_datafile_Malmö = pd.concat(blocking_list_Malmö, ignore_index=True)
PM_without_blocking_Malmlö = PM_data_Malmö[~PM_data_Malmö['datetime_start'].isin(block_datafile_Malmö['datetime'])]

pm_mean_Malmö = np.nanmean(np.array(PM_without_blocking_Malmlö['pm2.5']))
pm_sigma_Malmö = np.nanstd(np.array(PM_without_blocking_Malmlö['pm2.5']))


read.plot_mean(totdata_list1=totdata_list_Vavihill, totdata_list2=totdata_list_Malmö,
               daystoplot=daystoplot, minpoints=mindatasets, 
               place1='Vavihill', place2='Malmö', 
               pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
               pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö, 
               save=True)
        
dir_totdata_list_Vavihill = read.sort_wind_dir(totdata_list_Vavihill, pieinfo=info)
dir_totdata_list_Malmö = read.sort_wind_dir(totdata_list_Malmö, pieinfo=info)

        
seasonal_totdata_list_Vavihill = read.sort_season(totdata_list_Vavihill, 
                                                  totdata_list_dates_Vavihill, 
                                                  pieinfo=info)
seasonal_totdata_list_Malmö = read.sort_season(totdata_list_Malmö, 
                                               totdata_list_dates_Malmö, 
                                               pieinfo=info)

pressure_totdata_list_Vavihill = read.sort_pressure(totdata_list_Vavihill, pieinfo=info)
pressure_totdata_list_Malmö = read.sort_pressure(totdata_list_Malmö, pieinfo=info)


read.plot_dir_mean(dir_totdata_list1=dir_totdata_list_Vavihill, 
                   dir_totdata_list2=dir_totdata_list_Malmö, 
                   daystoplot=daystoplot,  
                   minpoints=8,
                   place1='Vavihill', place2='Malmö', save=False,
                   pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                   pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)



read.plot_pressure_mean(pressure_totdata_list1=pressure_totdata_list_Vavihill, 
                        pressure_totdata_list2=pressure_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=False,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)


read.plot_seasonal_mean(seasonal_totdata_list1=seasonal_totdata_list_Vavihill, 
                        seasonal_totdata_list2=seasonal_totdata_list_Malmö, 
                        daystoplot=daystoplot,  
                        minpoints=8,
                        place1='Vavihill', place2='Malmö', save=False,
                        pm_mean1=pm_mean_Vavihill, pm_sigma1=pm_sigma_Vavihill,
                        pm_mean2=pm_mean_Malmö, pm_sigma2=pm_sigma_Malmö)
    
