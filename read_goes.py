from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pyproj import Proj
import sys

file=sys.argv[1]
ncf=Dataset(file,'r')

# Seconds since 2000-01-01 12:00:00
add_seconds=ncf.variables['t'][0]
# Datetime of image scan
dt=datetime(2000, 1, 1, 12)+timedelta(seconds=float(add_seconds.data))
print 'dt = ' 
print dt

# Detections in RGB array
# Load the RGB arrays
R=ncf.variables['CMI_C07'][:].data
G=ncf.variables['CMI_C06'][:].data
B=ncf.variables['CMI_C05'][:].data
# Turn empty values into nans
R[R==-1]=np.nan
G[G==-1]=np.nan
B[B==-1]=np.nan
# Apply range limits for each channel (mostly important for Red channel)
R=np.maximum(R, 273)
R=np.minimum(R, 333)
G=np.maximum(G, 0)
G=np.minimum(G, 1)
B=np.maximum(B, 0)
B=np.minimum(B, .75)
# Normalize each channel by the appropriate range of values (again, mostly important for Red channel)
R=(R-273)/(333-273)
G=(G-0)/(1-0)
B=(B-0)/(.75-0)
# Apply the gamma correction to Red channel.
#   I was told gamma=0.4, but I get the right answer with gamma=2.5 (the reciprocal of 0.4)
R=np.power(R, 2.5)
# The final RGB array :)
RGB=np.dstack([R, G, B])
print 'RGB = '
print RGB

# Geolocation
# Satellite height
sat_h=ncf.variables['goes_imager_projection'].perspective_point_height
print 'sat_h = '
print sat_h
# Satellite longitude
sat_lon=ncf.variables['goes_imager_projection'].longitude_of_projection_origin
print 'sat_lon = '
print sat_lon
# Satellite sweep
sat_sweep=ncf.variables['goes_imager_projection'].sweep_angle_axis
print 'sat_sweep = '
print sat_sweep
# The projection x and y coordinates equals 
# the scanning angle (in radians) multiplied by the satellite height (http://proj4.org/projections/geos.html)
X=ncf.variables['x'][:] * sat_h
Y=ncf.variables['y'][:] * sat_h
print 'X ='
print X
print 'Y ='
print Y
'''
p=Proj(proj='utm', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
# Convert map points to latitude and longitude with the magic provided by Pyproj
XX, YY=np.meshgrid(X, Y)
lons, lats=p(XX, YY, inverse=True)
lats[np.isnan(R)]=np.nan
lons[np.isnan(R)]=np.nan
print 'lons ='
print lons
print 'lats ='
print lats
'''

# Visualize the way Brian Blaylock does
p=Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
# Convert map points to latitude and longitude with the magic provided by Pyproj
XX, YY=np.meshgrid(X, Y)
lons, lats=p(XX, YY, inverse=True)
lats[np.isnan(R)]=np.nan
lons[np.isnan(R)]=np.nan
# Make a new map object for the HRRR model domain map projection
mH = Basemap(resolution='i', projection='lcc', area_thresh=5000, \
             width=1800*3000, height=1060*3000, \
             lat_1=38.5, lat_2=38.5, \
             lat_0=38.5, lon_0=-97.5)
# Create a color tuple for pcolormesh
rgb = RGB[:,:-1,:] # Using one less column is very imporant, else your image will be scrambled! (This is the stange nature of pcolormesh)
colorTuple = rgb.reshape((rgb.shape[0] * rgb.shape[1]), 3) # flatten array, becuase that's what pcolormesh wants.
colorTuple = np.insert(colorTuple, 3, 1.0, axis=1) # adding an alpha channel will plot faster, according to stackoverflow. Not sure why.
# Now we can plot the GOES data on the HRRR map domain and projection
plt.figure(figsize=[10,8])
# The values of R are ignored becuase we plot the color in colorTuple, but pcolormesh still needs its shape.
newmap = mH.pcolormesh(lons, lats, R, color=colorTuple, linewidth=0, latlon=True)
newmap.set_array(None) # without this line the linewidth is set to zero, but the RGB colorTuple is ignored. I don't know why.
mH.drawcoastlines(color='w')
mH.drawcountries(color='w')
mH.drawstates(color='w')
plt.title('GOES-16 Fire Temperature', fontweight='semibold', fontsize=15)
plt.title('%s' % dt.strftime('%H:%M UTC %d %B %Y'), loc='right')
plt.show()

# location of South Sugarloaf fire
l = {'latitude': 41.812,
     'longitude': -116.324}
'''
bounds=(-116.655846, -115.5455, 41.243748, 42.069675)
# Draw zoomed map
mZ = Basemap(resolution='i', projection='cyl', area_thresh=50000,\
             llcrnrlon=bounds[0], llcrnrlat=bounds[2],\
             urcrnrlon=bounds[1], urcrnrlat=bounds[3],)
'''
mZ = Basemap(resolution='i', projection='cyl', area_thresh=50000,\
             llcrnrlon=l['longitude']-.75, llcrnrlat=l['latitude']-.75,\
             urcrnrlon=l['longitude']+.75, urcrnrlat=l['latitude']+.75,)
# Now we can plot the GOES data on a zoomed in map centered on the Sugarloaf wildfire
plt.figure(figsize=[8, 8])
newmap = mZ.pcolormesh(lons, lats, R, color=colorTuple, linewidth=0, latlon=True)
newmap.set_array(None) 
mZ.drawcoastlines(color='w')
mZ.drawcountries(color='w')
mZ.drawstates(color='w')
plt.title('GOES-16 Fire Temperature\n', fontweight='semibold', fontsize=15)
plt.title('%s' % dt.strftime('%H:%M UTC %d %B %Y'), loc='right')
plt.title('South Sugarloaf Fire', loc='left')
plt.show()
