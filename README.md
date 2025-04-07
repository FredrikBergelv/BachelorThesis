# Bachelor Thesis: Analysis of PM₂.₅ Concentration During High-Pressure Blocking Events

This repository contains all scripts, data, and the LaTeX source code for my Bachelor thesis in environmental analysis. The work investigates the relationship between PM₂.₅ concentrations and high-pressure blocking events using data from the Swedish Meteorological and Hydrological Institute (SMHI) and air quality measurements from both urban and rural sites in Skåne County, Sweden.

---

## Thesis Introduction

It is common knowledge that Earth’s increasing temperature has many side effects. One such effect is the increase in frequency of extreme weather phenomena. One such phenomenon, which lacks extensive research, is high-pressure blocking events. A high-pressure blocking event is an anticyclone that covers an area for a prolonged period of time and often blocks other types of weather, hence the name. This results in clearer weather and more extreme temperatures. However, an anticyclone is also associated with lower air movement and wind, causing the air to remain stagnant. This can lead to an accumulation of aerosols such as PM₂.₅ in the region.

To investigate the relationship between PM₂.₅ and high-pressure blocking, one must analyze periods of high-pressure blocking and examine the concentration of PM₂.₅ during these periods. The goal of this thesis is to analyze the concentration of PM₂.₅ during periods of high-pressure blocking by examining data from the Swedish Meteorological and Hydrological Institute (SMHI) and PM₂.₅ data from rural (Vavihill, Svalöv, Skåne County) and urban (Malmö, Skåne County) areas.

## Repository Contents

### Files

- **BachelorThesis.tex**: LaTeX source file for the full written thesis.
- **compare_pressure.py**: Compares pressure data against PM₂.₅ patterns.
- **csv_data.py**: Reads all CSV data files into pandas DataFrames.
- **info_of_data.py**: Provides summaries and descriptive statistics of the data.
- **making_all_plots_for_report.py**: Generates plots used in the thesis; can be customized for different datasets.
- **mannkendall_result.py**: Performs Mann-Kendall trend analysis on the data.
- **plot_yearly_data.py**: Plots annual PM₂.₅ trends for different stations.
- **read_datafiles.py**: Contains core functions for calculations and data processing.
- **csv_files/**: Folder containing all relevant input data files (CSV).

### Functionality

- `csv_data.py` reads and loads all `.csv` data files into pandas DataFrames, preparing them for analysis.
- `read_datafiles.py` contains core functions used for calculations, including preprocessing and data transformations.
- `making_all_plots_for_report.py` recreates the figures used in the thesis and can be adapted for other stations or periods.

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
import pymannkendall as mk
```
## Author
Fredrik Bergelv

Bachelor Student, Lund University

fredrik.bergelv@live.se
