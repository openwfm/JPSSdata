import numpy as np 

def nearest_index(lon,lat,lons,lats):
	dist=np.square(lons-lon)+np.square(lats-lat)
	ii,jj = np.unravel_index(dist.argmin(),dist.shape)
	return (ii,jj)

dx=0.05
N=100
x=np.arange(0,N,dx)
lons=np.repeat(x[np.newaxis,:],x.shape[0], axis=0)
lats=np.repeat(x[np.newaxis,:],x.shape[0], axis=0).transpose()
print 'dim_mesh=(%d,%d)' % lons.shape
lon=np.arange(0.023,99.976,0.4942)
lat=np.arange(0.023,99.976,0.4942)
print 'dim_data=(%d,%d)' % (lon.shape[0], lat.shape[0])
lonm=np.zeros(lon.shape)
latm=np.zeros(lat.shape)
for ll in range(0,len(lon)):
	(ii,jj)=nearest_index(lon[ll],lat[ll],lons,lats)
	lonm[ll]=lons[ii,jj]
	latm[ll]=lats[ii,jj]
	if ((lon[ll]-lonm[ll]) > dx) or ((lat[ll]-latm[ll]) > dx):
		print 'Bad interpolation!'
		print 'lon=%g' % lon[ll]
		print 'lat=%g' % lat[ll]
		print 'fxlon=%g' % lonm[ll]
		print 'fxlat=%g' % latm[ll]

print lonm
print latm
