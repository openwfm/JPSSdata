# sample data into mesh - Sugarloaf
# navigate to /share_home/jmandel/sugarloaf to access sample data
import netCDF4 as nc
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from JPSSD import retrieve_af_data, read_fire_mesh
from interpolation import *
import saveload as sl

fxlon,fxlat,bbox,time_esmf=read_fire_mesh('wrfout_d03_2018-09-03_15:00:00')

# cannot get starting time from wrfout
time = ("2018-08-15T00:00:00Z", "2018-09-02T00:00:00Z") # tuple, not array

data=retrieve_af_data(bbox,time)

print 'saving data'

sl.save((data,fxlon,fxlat),'data')

print 'run setup next'