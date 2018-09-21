import numpy as np 
from time import time
from datetime import datetime
from scipy import spatial

global t_init 

def sort_dates(data):
	""" 
    Sorting a dictionary depending on the time number in seconds from January 1, 1970
        :param:
            data 		dictionary of granules where each granule has a time_num key
        :returns: An array of dictionaries ordered by time_num key (seconds from January 1, 1970)

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	return sorted(data.iteritems(), key=lambda x: x[1]['time_num'])

def nearest_euclidean(lon,lat,lons,lats,bounds):
	""" 
    Returns the longitude and latitude arrays interpolated using Euclidean distance
        :param:
            lon 		2D array of longitudes to look the nearest neighbours
            lat 		2D array of latitudes to look the nearest neighbours
            lons		2D array of longitudes interpolating to
            lats		2D array of latitudes interpolating to
            bounds		array of 4 bounding boundaries where to interpolate: [minlon maxlon minlat maxlat]
        :returns: A tuple with a 2D array of longitudes and 2D array of latitudes interpolated from (lon,lat) to (lons,lats)

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	vlon=np.reshape(lon,np.prod(lon.shape))
	vlat=np.reshape(lat,np.prod(lat.shape))
	rlon=np.zeros(vlon.shape)
	rlat=np.zeros(vlat.shape)
	for k in range(0,len(vlon)):
		if (vlon[k]>bounds[0]) and (vlon[k]<bounds[1]) and (vlat[k]>bounds[2]) and (vlat[k]<bounds[3]):
			dist=np.square(lons-vlon[k])+np.square(lats-vlat[k])
			ii,jj = np.unravel_index(dist.argmin(),dist.shape)
			rlon[k]=lons[ii,jj]
			rlat[k]=lats[ii,jj]
		else:
			rlon[k]=np.nan;
			rlat[k]=np.nan;	
	return (np.reshape(rlon,lon.shape),np.reshape(rlat,lat.shape))
	

def nearest_scipy(lon,lat,lons,lats,dub=np.inf):
	""" 
    Returns the longitude and latitude arrays interpolated using scipy.spatial.cKDTree function
        :param:
            lon 		2D array of longitudes to look the nearest neighbours
            lat 		2D array of latitudes to look the nearest neighbours
            lons		2D array of longitudes interpolating to
            lats		2D array of latitudes interpolating to
            dub			optional: distance upper bound to look for the nearest neighbours
        :returns: A tuple with a 2D array of longitudes and 2D array of latitudes interpolated from (lon,lat) to (lons,lats)

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	vlon=np.reshape(lon,np.prod(lon.shape))
	vlat=np.reshape(lat,np.prod(lat.shape))
	vlonlat=np.column_stack((vlon,vlat))
	vlons=np.reshape(lons,np.prod(lons.shape))
	vlats=np.reshape(lats,np.prod(lats.shape))
	vlonlats=np.column_stack((vlons,vlats))
	inds=spatial.cKDTree(vlonlats).query(vlonlat,distance_upper_bound=dub)[1]
	ii=(inds!=vlons.shape[0])
	ret=np.empty((vlon.shape[0],2))
	ret[:]=np.nan
	ret[ii]=vlonlats[inds[ii]]
	rlon=[x[0] for x in ret]
	rlat=[x[1] for x in ret]
	return (np.reshape(rlon,lon.shape),np.reshape(rlat,lat.shape))

def distance_upper_bound(dx,dy):
	""" 
    Computes the distance upper bound
        :param:
            dx 		array of two elements with fire mesh grid resolutions
            dy 		array of two elements with satellite grid resolutions
        :returns: distance upper bound to look for the nearest neighbours

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	rx=np.sqrt(dx[0]**2+dx[1]**2)
	ry=np.sqrt(dy[0]**2+dy[1]**2)
	return max(rx,ry)

if __name__ == "__main__":
	t_init = time()
	# Initialization of grids
	N=100
	(dx1,dx2)=(1,1)
	(dy1,dy2)=(3,3)
	x=np.arange(0,N,dx1)
	lons=np.repeat(x[np.newaxis,:],x.shape[0], axis=0)
	x=np.arange(0,N,dx2)
	lats=np.repeat(x[np.newaxis,:],x.shape[0], axis=0).transpose()
	bounds=[lons.min(),lons.max(),lats.min(),lats.max()]
	print 'bounds'
	print bounds
	print 'dim_mesh=(%d,%d)' % lons.shape
	y=np.arange(-N*1.432,2.432*N,dy1)
	lon=np.repeat(y[np.newaxis,:],y.shape[0], axis=0)
	y=np.arange(-N*1.432,2.432*N,dy2)
	lat=np.repeat(y[np.newaxis,:],y.shape[0], axis=0)
	print 'dim_data=(%d,%d)' % (lon.shape[0], lat.shape[0])

	# Result by Euclidean distance
	print '>>Euclidean approax<<'
	(rlon,rlat)=nearest_euclidean(lon,lat,lons,lats,bounds)
	rlon=np.reshape(rlon,np.prod(lon.shape))
	rlat=np.reshape(rlat,np.prod(lat.shape))
	vlonlatm=np.column_stack((rlon,rlat))
	print vlonlatm
	t_final = time()
	print 'Elapsed time: %ss.' % str(t_final-t_init)

	# Result by scipy.spatial.cKDTree function
	dub=distance_upper_bound([dx1,dx2],[dy1,dy2])
	(rlon,rlat)=nearest_scipy(lon,lat,lons,lats,dub)
	rlon=np.reshape(rlon,np.prod(lon.shape))
	rlat=np.reshape(rlat,np.prod(lat.shape))
	vlonlatm2=np.column_stack((rlon,rlat))
	print '>>cKDTree<<'
	print vlonlatm2
	t_ffinal = time()
	print 'Elapsed time: %ss.' % str(t_ffinal-t_final)

	# Comparison
	print 'Same results?'
	print (np.isclose(vlonlatm,vlonlatm2) | np.isnan(vlonlatm) | np.isnan(vlonlatm2)).all()

