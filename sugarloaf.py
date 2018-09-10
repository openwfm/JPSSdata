#sample data into mesh - Sugarloaf
import netCDF4 as nc
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy import interpolate

fig = plt.figure()
ax = fig.gca(projection='3d')

d = nc.Dataset('dataset')
fxlon = d.variables['FXLONG'][0,:,:]
fxlat = d.variables['FXLAT'][0,:,:]
data = d.variables['TIGN_G'][10,:,:]
d.close()

surf = ax.plot_surface(fxlon,fxlat,data,cmap=cm.coolwarm)
plt.show()