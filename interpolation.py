import numpy as np 
from time import time
from datetime import datetime

global t_init 

def sort_dates(data):
	return sorted(data.items(),key=lambda x: datetime.strptime(x[1]['time_start_geo'][0:18],'%Y-%m-%dT%H:%M:%S'))

def nearest_index(lon,lat,lons,lats,bounds):
	if (lon>bounds[0]) and (lon<bounds[1]) and (lat>bounds[2]) and (lat<bounds[3]):
		dist=np.square(lons-lon)+np.square(lats-lat)
		ii,jj = np.unravel_index(dist.argmin(),dist.shape)
		return (ii,jj)
	else:
		return none

if __name__ == "__main__":
	t_init = time()
	dx=10
	N=2580
	k=5.765
	x=np.arange(0,N)
	lons=np.repeat(x[np.newaxis,:],x.shape[0], axis=0)
	lats=np.repeat(x[np.newaxis,:],x.shape[0], axis=0).transpose()
	bounds=[lons.min(),lons.max(),lats.min(),lats.max()]
	print 'dim_mesh=(%d,%d)' % lons.shape
	lon=np.arange(-k*N,k*N,dx)
	lat=np.arange(-k*N,k*N,dx)
	print 'dim_data=(%d,%d)' % (lon.shape[0], lat.shape[0])
	lonm=np.zeros(lon.shape)
	latm=np.zeros(lat.shape)
	for ll in range(0,len(lon)):
		inds=nearest_index(lon[ll],lat[ll],lons,lats,bounds)
		if inds is none:
			lonm[ll]=np.nan;
			latm[ll]=np.nan;
		else:
			ii=inds[0]
			jj=inds[1]
			lonm[ll]=lons[ii,jj]
			latm[ll]=lats[ii,jj]
			if ((lon[ll]-lonm[ll]) > dx) or ((lat[ll]-latm[ll]) > dx):
				print 'Interpolation error:'
				print 'lon=%g' % lon[ll]
				print 'lat=%g' % lat[ll]
				print 'fxlon=%g' % lonm[ll]
				print 'fxlat=%g' % latm[ll]
	t_final = time()
	print 'Elapsed time: %ss.' % str(t_final-t_init)
