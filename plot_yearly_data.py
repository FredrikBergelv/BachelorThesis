#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:04:20 2025

@author: Fredrik Bergelv
"""


import read_datafiles as read
import csv_data as csv





read.yearly_histogram(csv.data["PM25"]["Hornsgatan"], datatype='pm')