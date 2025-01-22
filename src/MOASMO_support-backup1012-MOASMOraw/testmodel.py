import sys
sys.path.append('/Users/guoqiang/Github/CTSM/ctsm_optz/MO-ASMO/src/')
import numpy as np
import sampling
import gp
import NSGA2
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from os import path
import seaborn as sn
import pickle
# import smt
from sklearn import preprocessing
import random
import re
from sklearn.metrics import mean_squared_error
import os
# import hydroeval as he
import warnings
warnings.filterwarnings("ignore")

d = np.load('/Users/guoqiang/Downloads/test.npz')
x=d['x']
y=d['y']
nOutput=d['nOutput']
xlb_single_value_scaled=d['xlb_single_value_scaled']
xub_single_value_scaled=d['xub_single_value_scaled']
alpha=d['alpha']
lb=d['lb']
ub=d['ub']
nu=d['nu']


# sm = gp.GPR_Matern(x, y.copy(), 13, nOutput, 216, xlb_single_value_scaled, xub_single_value_scaled,alpha=alpha, leng_sb=[lb,ub], nu=nu)
# a = 1

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
sm = RandomForestRegressor()
sm.fit(x, y)

# perform optimization using the surrogate model
# define hyper parameters
pop = 100
gen = 100
crossover_rate = 0.9
mu = 20
mum = 20
n_sample = 12
bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(sm, x.shape[1], nOutput, xlb_single_value_scaled, xub_single_value_scaled, pop, gen, crossover_rate, mu, mum)
D = NSGA2.crowding_distance(besty_sm)
idxr = D.argsort()[::-1][:n_sample]
x_resample = bestx_sm[idxr, :]
y_resample = besty_sm[idxr, :]

