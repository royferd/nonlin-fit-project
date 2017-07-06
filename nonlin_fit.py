# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:14:42 2017

@author: royferd
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.optimize import leastsq


def fitfn(params, x, data, eps_data):
    c0 = params['offset']
    c1 = params['slope']
    
    model = c0 + c1*x
    
    return (data - model)/eps_data
    
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

out = minimize(fitfn, params, args=(x1, y1, dy1))

