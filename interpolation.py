import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from time import time
from datetime import datetime
from scipy import spatial

global t_init 

def sort_dates(data):
	return sorted(data.iteritems(), key=lambda x: x[1]['time_num'])

def nearest_index(lon,lat,lons,lats,bounds):
	if (lon>bounds[0]) and (lon<bounds[1]) and (lat>bounds[2]) and (lat<bounds[3]):
		dist=np.square(lons-lon)+np.square(lats-lat)
		ii,jj = np.unravel_index(dist.argmin(),dist.shape)
		return (ii,jj)
	else:
		return None

def nearest_indices(lon,lat,lons,lats,dub=np.inf):
	vlon=np.reshape(lon,np.prod(lon.shape))
	vlat=np.reshape(lat,np.prod(lat.shape))
	vlons=np.reshape(lons,np.prod(lons.shape))
	vlats=np.reshape(lats,np.prod(lats.shape))
	vlonlat=np.column_stack((vlon,vlat))
	vlonlats=np.column_stack((vlons,vlats))
	inds=spatial.cKDTree(vlonlats).query(vlonlat,distance_upper_bound=dub)[1]
	ii=(inds!=vlons.shape[0])
	ret=np.empty((vlon.shape[0],2))
	ret[:]=np.nan
	ret[ii]=vlonlats[inds[ii]]
	rlon=[x[0] for x in ret]
	rlat=[x[1] for x in ret]
	return (np.reshape(rlon,lon.shape),np.reshape(rlat,lat.shape))


if __name__ == "__main__":
	t_init = time()
	N=100
	(dx1,dx2)=(1,1)
	(dy1,dy2)=(3,3)
	x=np.arange(0,N,dx1)
	lons=np.repeat(x[np.newaxis,:],x.shape[0], axis=0)
	vlons=np.reshape(lons,np.prod(lons.shape))
	x=np.arange(0,N,dx2)
	lats=np.repeat(x[np.newaxis,:],x.shape[0], axis=0).transpose()
	vlats=np.reshape(lats,np.prod(lats.shape))
	vlonlats=np.column_stack((vlons,vlats))
	bounds=[lons.min(),lons.max(),lats.min(),lats.max()]
	print 'bounds'
	print bounds
	print 'dim_mesh=(%d,%d)' % lons.shape
	print vlonlats
	y=np.arange(-N*1.432,2.432*N,dy1)
	lon=np.repeat(y[np.newaxis,:],y.shape[0], axis=0)
	vlon=np.reshape(lon,np.prod(lon.shape))
	y=np.arange(-N*1.432,2.432*N,dy2)
	lat=np.repeat(y[np.newaxis,:],y.shape[0], axis=0)
	vlat=np.reshape(lat,np.prod(lat.shape))
	vlonlat=np.column_stack((vlon,vlat))
	print 'dim_data=(%d,%d)' % (lon.shape[0], lat.shape[0])
	print vlonlat
	vlonm=np.zeros(vlon.shape)
	vlatm=np.zeros(vlat.shape)
	for ll in range(0,len(vlon)):
		inds=nearest_index(vlon[ll],vlat[ll],lons,lats,bounds)
		if inds is None:
			vlonm[ll]=np.nan;
			vlatm[ll]=np.nan;
		else:
			ii=inds[0]
			jj=inds[1]
			vlonm[ll]=lons[ii,jj]
			vlatm[ll]=lats[ii,jj]
			if ((vlon[ll]-vlonm[ll]) > dx1) or ((vlat[ll]-vlatm[ll]) > dx2):
				print 'Interpolation error:'
				print 'lon=%g' % vlon[ll]
				print 'lat=%g' % vlat[ll]
				print 'fxlon=%g' % vlonm[ll]
				print 'fxlat=%g' % vlatm[ll]
	vlonlatm=np.column_stack((vlonm,vlatm))
	print 'result 1'
	print vlonlatm
	t_final = time()
	print 'Elapsed time: %ss.' % str(t_final-t_init)
	rx=np.sqrt(dx1**2+dx2**2)
	ry=np.sqrt(dy1**2+dy2**2)
	dub=max(rx,ry)
	(rlon,rlat)=nearest_indices(lon,lat,lons,lats,dub)
	rlon=np.reshape(rlon,vlon.shape[0])
	rlat=np.reshape(rlat,vlat.shape[0])
	vlonlatm2=np.column_stack((rlon,rlat))
	print 'result 2'
	print vlonlatm2
	t_ffinal = time()
	print 'Elapsed time: %ss.' % str(t_ffinal-t_final)
	print (np.isclose(vlonlatm,vlonlatm2) | np.isnan(vlonlatm) | np.isnan(vlonlatm2)).all()
