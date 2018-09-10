#sample data into mesh - Sugarloaf
import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy import interpolate

d = nc.Dataset('dataset')
fxlon = d.variables['FXLONG'][0,:,:]
fxlat = d.variables['FXLAT'][0,:,:]
data = d.variables['TIGN_G'][0,:,:]
d.close()

#lon,lat = np.meshgrid(fxlon,fxlat,sparse=True)

plt.plot(fxlon,fxlat)
plt.plot(data)
plt.show()