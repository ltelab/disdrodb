#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:55:48 2022

@author: ghiggi
"""
from disdrodb.L1.utils import regularize_dataset
import xarray as xr 
import numpy as np 

fpath = "/ltenas3/0_Data/DISDRODB/Processed/EPFL/TICINO_2018/L1/TICINO_2018_s61.nc"
ds = xr.open_dataset(fpath)

ds = ds[['sensor_status']]
method = "nearest"
tolerance = pd.Timedelta(value=30, unit="second")
ds_reindexed = regularize_dataset(ds, range_freq="30s", 
                                  tolerance=tolerance, method=method,
                                  fill_value=4)

ds['time'].values[0:10]
ds_reindexed['time'].values[0:10]

ds['sensor_status'].values[0:10]
ds_reindexed['sensor_status'].values[0:10]

np.unique(ds['sensor_status'].values)
np.unique(ds_reindexed['sensor_status'].values)

ds_reindexed['sensor_status']

df = ds_reindexed.to_dataframe()[['sensor_status']]


### --------------------------------------------------------------
### TOOL0
# https://stackoverflow.com/questions/59602653/data-availability-chart-in-python

import seaborn

import random
import pandas as pd
import plotly.express as px
from random import choices

# random data with a somewhat higher
# probability of 1 than 0 to mimic OPs data
random.seed(1)
vals=[0,1]
prob=[0.4, 0.6]
choices(vals, prob)

data=[]
for i in range(0,5):
    data.append([choices(vals, prob)[0] for c in range(0,10)])

# organize data in a pandas dataframe
df=pd.DataFrame(data).T
df.columns=['Balance Sheet', 'Closing Price', 'Weekly Report', 'Analyst Data', 'Annual Report']
drng=pd.date_range(pd.datetime(2080, 1, 1).strftime('%Y-%m-%d'), periods=df.shape[0]).tolist()
df['date']=[d.strftime('%Y-%m-%d') for d in drng]
dfm=pd.melt(df, id_vars=['date'], value_vars=df.columns[:-1])

# plotly express
fig = px.bar(dfm, x="date", y="variable", color='value',
             orientation='h',
             hover_data=["date"],
             height=600,
             color_continuous_scale=['firebrick', '#2ca02c'],
             title='Data Availabiltiy Plot',
             template='plotly_white',
            )

fig.update_layout(yaxis=dict(title=''), xaxis=dict(title='', showgrid=False, gridcolor='grey',
                  tickvals=[],
                            )
                 )
fig.show()


### --------------------------------------------------------------
### --------------------------------------------------------------
### TOOL1

import numpy as np
import matplotlib.pyplot as plt

category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = {
    'Question 1': [10, 15, 17, 32, 26],
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    
    
    category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color, align="center")

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()

### --------------------------------------------------------------
### TOOL2
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def date_range(start, end, interval=dt.timedelta(days=1)):
    output = []
    while start <= end:
        output.append(start)
        start += interval
    return output

# Generate a series of dates for plotting...
edate = date_range(dt.datetime(2012, 2, 1), 
                   dt.datetime(2012, 6, 15), 
                   dt.timedelta(days=5))
bdate = date_range(dt.datetime(2012, 1, 1), 
                   dt.datetime(2012, 5, 15), 
                   dt.timedelta(days=5))

# Now convert them to matplotlib's internal format...
edate, bdate = [mdates.date2num(item) for item in (edate, bdate)]
edate = edate[0:2]
bdate = bdate[0:2]
ypos = range(len(edate))
fig, ax = plt.subplots()

# Plot the data
ax.barh(ypos, edate - bdate, left=bdate, height=0.8, align='center')
ax.axis('tight')

# We need to tell matplotlib that these are dates...
ax.xaxis_date()

plt.show()

### --------------------------------------------------------------
# TOOL 3
import pandas
from matplotlib import pyplot
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MultipleLocator

df = pandas.read_csv('thing.csv')

vert = {"upright":1, "supine":2, "lying_left":3, "prone":4, "lying_right":5, "unknown":6}

fig, (ax1) = pyplot.subplots(1, 1, figsize=(15,5))

# first plot of positions over time
# get the computed body positions as bars
bars = []
colors = []
for i,row in df.iterrows():
    h = vert[row['body_position']]
    t = row['timestamp']
    
    rect = [(t - 59000, h - 0.4), (t - 59000, h + 0.4), (t, h + 0.4), (t, h - 0.4), (t - 59000, h - 0.4)]
    bars.append(rect)
    colors.append("C" + str(h-1))

bars = PolyCollection(bars, facecolors=colors)
ax1.add_collection(bars)

ax1.autoscale()
ax1.xaxis.set_major_locator(MultipleLocator(60000))
#ax1.set_xticklabels([''], rotation=90)
#ax1.xaxis.set_major_formatter(dates.DateFormatter("%B-%d\n%H:%M"))

ax1.set_yticks(sorted(vert.values()))
ax1.set_yticklabels(sorted(vert, key=vert.get))

pyplot.tight_layout()
pyplot.show()