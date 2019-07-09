import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from forecast import process_tign_g
from JPSSD import time_iso2num
from setup import process_detections
from infrared_perimeters import process_ignitions
import saveload as sl

# Realistic bounds
nx = ny = 201
bounds = (-113.85068, -111.89413, 39.677563, 41.156837)

# Creation of synthetic coordinates
lon,lat = np.meshgrid(np.linspace(bounds[0],bounds[1],nx),
                    np.linspace(bounds[2],bounds[3],ny))
ilon = .5*(bounds[0]+bounds[1])
ilat = .5*(bounds[2]+bounds[3])

# Scaling fire arrival time to be from 0 to 1000
lenlon = abs(bounds[1]-bounds[0])*.5
lenlat = abs(bounds[3]-bounds[2])*.5
clon = 5e5/(lenlon**2)
clat = 5e5/(lenlat**2)

# Creation of cone synthetic fire arrival time
tign_g = np.minimum(600,10+np.sqrt(clon*(lon-ilon)**2+clat*(lat-ilat)**2))
print 'min tign_g %gs, max tign_g %gs' % (tign_g.min(),tign_g.max())
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.contour(lon,lat,tign_g,20)
ax.scatter(-113.25, ilat, 150, color='r')
plt.show()

# Some more set-ups needed
dx = dy = 1000.
dt = 10
ctime = '2013-08-14_00:00:00'

# Forecast creation
data = process_tign_g(lon,lat,tign_g,bounds,ctime,dx,dy,wrfout_file='ideal',dt_for=dt,plot=True)
#points = [[-113.25], [ilat], ['2013-08-13T23:52:30']]
points = [[-113.25, -113.25, -113.25], [ilat], ['2013-08-13T23:52:30']]
data.update(process_ignitions(points, bounds))

# Process detections
isotime = ctime.replace('_','T')
ftime = time_iso2num(isotime)
time_num = (ftime-tign_g.max(),ftime)
sl.save((data,lon,lat,time_num),'data')
result = process_detections(data,lon,lat,time_num)
