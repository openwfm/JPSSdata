# sample data into mesh - Sugarloaf
# navigate to /share_home/jmandel/sugarloaf to access sample data
import netCDF4 as nc
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from JPSSD import retrieve_af_data
from interpolation import *

fig = plt.figure()
ax = fig.gca(projection='3d')

file='wrfout_d03_2018-09-03_15:00:00'
print 'opening ' + file
d = nc.Dataset(file)
m,n = d.variables['XLONG'][0,:,:].shape
fm,fn = d.variables['FXLONG'][0,:,:].shape
fm=fm-fm/(m+1)    # dimensions corrected for extra strip
fn=fn-fn/(n+1)
fxlon = d.variables['FXLONG'][0,:fm,:fn] #  masking  extra strip
fxlat = d.variables['FXLAT'][0,:fm,:fn]
tign_g = d.variables['TIGN_G'][0,:fm,:fn]
time_esmf = ''.join(d.variables['Times'][:][0])  # date string as YYYY-MM-DD_hh:mm:ss 
d.close()

bbox = [fxlon.min(),fxlon.max(),fxlat.min(),fxlat.max()]
print 'min max longitude latitude %s'  % bbox
print 'time (ESMF) %s' % time_esmf

#surf = ax.plot_surface(fxlon,fxlat,tign_g,cmap=cm.coolwarm)
#plt.show()

# cannot get starting time from wrfout
time = ("2018-08-15T00:00:00Z", "2018-09-02T00:00:00Z") # tuple, not array

data=retrieve_af_data(bbox,time)

print 'run setup next'

if __name__ == "__main__":
    print 'you should import this like\nfrom sugarloaf import data'
    sys.exit(1)
