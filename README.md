# Bachelor Thesis: Analysis of PM₂.₅ Concentration During High-Pressure Blocking Events

This repository contains all scripts, data, and the LaTeX source code for my Bachelor thesis in environmental analysis. The work investigates the relationship between PM₂.₅ concentrations and high-pressure blocking events using data from the Swedish Meteorological and Hydrological Institute (SMHI) and air quality measurements from both urban and rural sites in Skåne County, Sweden.

---

## Thesis Introduction

It is common knowledge that Earth’s increasing temperature has many side effects. One such effect is the increase in frequency of extreme weather phenomena [2]. One such phenomenon, which lacks extensive research, is high-pressure blocking events. High-pressure blocking events is an anticyclone that covers an area for a prolonged period of time and often blocks other types of weather, hence the name. This results in clearer weather and more extreme temperatures [3]. However, an anticyclone is also associated with lower air movement and wind, causing the air to remain stagnant. This can lead to an accumulation of aerosols such as PM₂.₅ in the region [4].

To investigate the relationship between PM₂.₅ and high-pressure blocking, one must analyse periods of high-pressure blocking and examine the concentration of PM₂.₅ during these periods. The goal of this thesis is to analyse the concentration of PM₂.₅ during periods of high-pressure blocking by examining data from the Swedish Meteorological and Hydrological Institute (SMHI) and PM₂.₅ data from rural (Vavihill, Svalöv Skåne county) and urban (Malmö, Skåne county) areas.

## Repository Contents


## Requirements

The following Python packages are required to run the code:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.colors as mcolors
from collections import defaultdict
import matplotlib.gridspec as gridspec
import re
import pymannkendall as mk '''


## Data Summary
csv_data.py reads and loads all .csv data files into pandas DataFrames, preparing them for analysis.

read_datafiless.py contains core functions used for calculations, including preprocessing and data transformations.

making_all_plots_for_report.py recreates the figures used in the thesis and can be adapted for other stations or periods.

## Author
Fredrik Bergelv
Bachelor Student, Lund univeristy
fredrik.bergelv@live.se
