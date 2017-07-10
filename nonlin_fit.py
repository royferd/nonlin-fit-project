# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:14:42 2017

@author: royferd
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.optimize import leastsq
import math


def fitfn(params, x, data, eps_data):
    c0 = params['offset']
    c1 = params['slope']
    
    model = c0 + c1*x
    
    return (data - model)/eps_data
    
def yfit(params, x):
    c0 = params['offset']
    c1 = params['slope']
    
    model = c0 + c1*x
    
    return model

    
# define the parameters with initial guesses (values) corresponding to the
# user-defined function. Define bounds in params with options [min] & [max].
# Fix a parameter with option [vary = False]    
params = Parameters()
params.add('offset', value=0.0)
params.add('slope', value=0.007)


# mean voltages and mean standard deviations. 1-3 are the old fluxgate sensors,
# 4-6 are the new fluxgate sensors.
x1 = []
dx1 = []
y1 = []
dy1 = []


# read in voltages
file = input('File name: ')
with open(file,'r') as infile:
    for line in infile:
        data = line.split()
        if len(data) != 0:
            x1.append(data[0])
            dx1.append(data[1])
            y1.append(data[2])
            dy1.append(data[3])

for i in range(len(x1)):
    x1[i] = float(x1[i])
    dx1[i] = float(dx1[i])
    y1[i] = float(y1[i])
    dy1[i] = float(dy1[i])
    
# Convert lists to arrays so that plt.errorbar() can read them 
x1_ary = np.array(x1)
dx1_ary = np.array(dx1)
y1_ary = np.array(y1)
dy1_ary = np.array(dy1)


out = minimize(fitfn, params, args=(x1_ary, y1_ary, dy1_ary))
new_params = out.params
out.params.pretty_print()

xfitpoints = 20 * len(x1_ary)
xrange = math.ceil(max(x1_ary) - min(y1_ary))
xstep = math.ceil(xrange/xfitpoints)

x1_fit = np.zeros(xfitpoints,dtype=np.float)
dx1_fit = np.zeros(xfitpoints,dtype=np.float)
y1_fit = np.zeros(xfitpoints,dtype=np.float)
dy1_fit = np.zeros(xfitpoints,dtype=np.float)

for i in range(xfitpoints):
    j = i*xstep + min(x1_ary)
    x1_fit[i] = j    
    y1_fit[i] = yfit(new_params,j)
    
    

#Plot data as open-faced red circles
plt.scatter(x1,y1,marker='o', facecolor = 'none', edgecolors='r')

# Plot errorbars 
# THE X ERROR BARS ARE THERE! they're just very small, you can see them if you set 'xerr = 40' or more)
plt.errorbar(x1_ary, y1_ary, xerr = dx1_ary, yerr = dy1, linestyle = 'None', ecolor='green')
plt.plot(x1_fit,y1_fit)

# Assign plot title and axis labels.
plt.title('PLOT TITLE')
plt.xlabel('x axis label (units)')
plt.ylabel('y axis label (units)')

# Show the plot
#plt.show()

plt.show()