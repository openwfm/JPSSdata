# sample data into mesh - Sugarloaf
# navigate to /share_home/jmandel/sugarloaf to access sample data
import netCDF4 as nc
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend('agg')
from scipy import interpolate

fig = plt.figure()
ax = fig.gca(projection='3d')

d = nc.Dataset('wrfout_d03_2018-09-03_15:00:00')
fxlon = d.variables['FXLONG'][0,:,:] # boundary masking conditions previously calculated(0:409)
fxlat = d.variables['FXLAT'][0,:,:]
data = d.variables['TIGN_G'][10,:,:]

bbox = [(np.amin(fxlon),np.amin(fxlat)),(np.amin(fxlon),np.amax(fxlat)),
	(np.amax(fxlon),np.amin(fxlat)),(np.amax(fxlon),np.amax(fxlat))]
print bbox

d.close()

surf = ax.plot_surface(fxlon,fxlat,data,cmap=cm.coolwarm)
plt.show()