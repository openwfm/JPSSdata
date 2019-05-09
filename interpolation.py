import warnings
warnings.filterwarnings("ignore")
import numpy as np
from time import time
from datetime import datetime
from scipy import spatial
import itertools
from random import randint

def sort_dates(data):
	"""
    Sorting a dictionary depending on the time number in seconds from January 1, 1970

    :param data: dictionary of granules where each granule has a time_num key
    :return: an array of dictionaries ordered by time_num key (seconds from January 1, 1970)

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	return sorted(data.iteritems(), key=lambda x: x[1]['time_num'])

def nearest_euclidean(lon,lat,lons,lats,bounds):
	"""
    Returns the longitude and latitude arrays interpolated using Euclidean distance

    :param lon: 2D array of longitudes to look the nearest neighbours
    :param lat: 2D array of latitudes to look the nearest neighbours
    :param lons: 2D array of longitudes interpolating to
    :param lats: 2D array of latitudes interpolating to
    :param bounds: array of 4 bounding boundaries where to interpolate: [minlon maxlon minlat maxlat]
    :return ret: tuple with a 2D array of longitudes and 2D array of latitudes interpolated from (lon,lat) to (lons,lats)

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
	ret=(np.reshape(rlon,lon.shape),np.reshape(rlat,lat.shape))
	return ret


def nearest_scipy(lon,lat,stree,bounds):
	"""
    Returns the coordinates interpolated from (lon,lat) to (lons,lats) and the value of the indices where NaN values

    :param lon:	2D array of longitudes to look the nearest neighbours
    :param lat:	2D array of latitudes to look the nearest neighbours
    :param lons: 2D array of longitudes interpolating to
    :param lats: 2D array of latitudes interpolating to
    :param dub: optional, distance upper bound to look for the nearest neighbours
    :return ret: A tuple with a 2D array of coordinates interpolated from (lon,lat) to (lons,lats) and the value of the indices where NaN values

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	vlon=np.reshape(lon,np.prod(lon.shape))
	vlat=np.reshape(lat,np.prod(lat.shape))
	vlonlat=np.column_stack((vlon,vlat))
	M=(vlon>bounds[0])*(vlon<bounds[1])*(vlat>bounds[2])*(vlat<bounds[3])
	vlonlat=vlonlat[M]
	inds=np.array(stree.query(vlonlat)[1])
	ret=(inds,M)
	return ret

def distance_upper_bound(dx,dy):
	"""
    Computes the distance upper bound

    :param dx: array of two elements with fire mesh grid resolutions
    :param dy: array of two elements with satellite grid resolutions
    :return d: distance upper bound to look for the nearest neighbours

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	rx=np.sqrt(dx[0]**2+dx[1]**2)
	ry=np.sqrt(dy[0]**2+dy[1]**2)
	d=max(rx,ry)
	return d

def neighbor_indices(indices,shape,d=2):
	"""
    Computes all the neighbor indices from an indice list

    :param indices: list of coordinates in a 1D array
    :param shape: array of two elements with satellite grid size
    :param d: optional, distance of the neighbours
    :return ind: Returns a numpy array with the indices and the neighbor indices in 1D array

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
    # Width and Length of the 2D grid
	w=shape[0]
	l=shape[1]
	# 2D indices of the 1D indices
	I=[[np.mod(ind,w),ind/w] for ind in indices]
	# All the combinations (x,y) for all the neighbor points from x-d to x+d and y-d to y+d
	N=np.concatenate([np.array(list(itertools.product(range(max(i[0]-d,0),min(i[0]+d+1,w)),range(max(i[1]-d,0),min(i[1]+d+1,l))))) for i in I])
	# Recompute the 1D indices of the 2D coordinates inside the 2D domain
	ret=np.array([x[0]+w*x[1] for x in N])
	# Sort them and take each indice once
	ind=sorted(np.unique(ret))
	return ind

def neighbor_indices_pixel(lons,lats,lon,lat,scan,track):
	"""
    Computes all the neighbor indices depending on the size of the pixel

    :param lons: one dimensional array of the fire mesh longitud coordinates
    :param lats: one dimensional array of the fire mesh latitude coordinates
    :param lon: one dimensional array of the fire detection longitud coordinates
    :param lat: one dimensional array of the fire detection latitude coordinates
    :param scan: one dimensional array of the fire detection along-scan sizes
    :param track: one dimensional array of the fire detection along-track sizes
    :return ll: returns a one dimensional logical array with the indices in the fire mesh including pixel neighbours

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-10-16
    """
	R=6378   # Earth radius
	km_lat=180/(np.pi*R)  # 1km in degrees latitude
	km_lon=km_lat/np.cos(lat*np.pi/180)  # 1 km in longitude
	# pixel sizes in degrees
	sqlat=km_lat*track/2
	sqlon=km_lon*scan/2
	# creating bounds for each fire detection (pixel vertexs)
	bounds=[[lon[k]-sqlon[k],lon[k]+sqlon[k],lat[k]-sqlat[k],lat[k]+sqlat[k]] for k in range(len(lat))]
	# creating a logical array of indices in the fire mesh with the intersection of all the cases
	ll=np.sum([np.array(np.logical_and(np.logical_and(np.logical_and(lons>b[0],lons<b[1]),lats>b[2]),lats<b[3])) for b in bounds],axis=0).astype(bool)
	if ll is False:
		ll = [False]*len(lons)
	return ll

def neighbor_indices_ellipse(lons,lats,lon,lat,scan,track,mm=1):
	"""
    Computes all the neighbor indices in an ellipse of radius the size of the pixel

    :param lons: one dimensional array of the fire mesh longitud coordinates
    :param lats: one dimensional array of the fire mesh latitude coordinates
    :param lon: one dimensional array of the fire detection longitud coordinates
    :param lat: one dimensional array of the fire detection latitude coordinates
    :param scan: one dimensional array of the fire detection along-scan sizes
    :param track: one dimensional array of the fire detection along-track sizes
    :param mm: constant of magnitude of the ellipse, default mm=1 (ellipse inscribed in the pixel)
    :return ll: returns a one dimensional logical array with the indices in the fire mesh including pixel neighbours

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-10-18
    """
	R=6378   # Earth radius
	km_lat=180/(np.pi*R)  # 1km in degrees latitude
	km_lon=km_lat/np.cos(lat*np.pi/180)  # 1 km in longitude
	# pixel sizes in degrees
	sqlat=km_lat*track/2
	sqlon=km_lon*scan/2
	# creating the coordinates in the center of the ellipse
	C=[[(np.array(lons)-lon[k])/sqlon[k],(np.array(lats)-lat[k])/sqlat[k]] for k in range(len(lat))]
	# creating a logical array of indices in the fire mesh with the intersection of all the cases
	ll=np.sum([np.array((np.square(c[0])+np.square(c[1]))<=mm) for c in C],axis=0).astype(bool)
	return ll

def neighbor_indices_ball(tree,indices,shape,d=2):
	"""
    Computes all the neighbor indices from an indice list in a grid of indices defined through a cKDTree

    :param indices: list of coordinates in a 1D array
    :param shape: array of two elements with satellite grid size
    :param d: optional, distance of the neighbours computed as: sqrt(2*d**2)
    :return ind: returns a numpy array with the indices and the neighbor indices in 1D array respect to the grid used in the tree

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
    # Width of the 2D grid
	w=shape[0]
	# 2D indices of the 1D indices
	I=[[np.mod(ind,w),ind/w] for ind in indices]
	# Radius to look for
	radius=np.sqrt(2*d**2)
	# Request all the neighbors in a radius "radius"
	N=tree.query_ball_point(I,r=radius)
	# Return an unique and sorted array of 1D indices (indices pointing to the grid used for the tree)
	ind=sorted(np.unique(np.concatenate(N)))
	return ind

if __name__ == "__main__":
	t_init = time()
	# Initialization of grids
	N=100
	(dx1,dx2)=(1,1)
	(dy1,dy2)=(3,3)
	x=np.arange(0,N,dx1)
	lons=np.repeat(x[np.newaxis,:],x.shape[0], axis=0)
	x=np.arange(0,N,dx2)
	lats=np.repeat(x[np.newaxis,:],x.shape[0], axis=0).T
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
	vlons=np.reshape(lons,np.prod(lons.shape))
	vlats=np.reshape(lats,np.prod(lats.shape))
	vlonlats=np.column_stack((vlons,vlats))
	print vlonlats
	stree=spatial.cKDTree(vlonlats)
	(inds,K)=nearest_scipy(lon,lat,stree,bounds)
	vlonlatm2=np.empty((np.prod(lon.shape),2))
	vlonlatm2[:]=np.nan
	vlons=np.reshape(lons,np.prod(lons.shape))
	vlats=np.reshape(lats,np.prod(lats.shape))
	vlonlats=np.column_stack((vlons,vlats))
	vlonlatm2[K]=vlonlats[inds]
	print '>>cKDTree<<'
	print vlonlatm2
	t_ffinal = time()
	print 'Elapsed time: %ss.' % str(t_ffinal-t_final)

	# Comparison
	print 'Same results?'
	print (np.isclose(vlonlatm,vlonlatm2) | np.isnan(vlonlatm) | np.isnan(vlonlatm2)).all()

	# Testing neighbor indices
	N=100
	shape=(250,250)
	ind=sorted(np.unique([randint(0,np.prod(shape)-1) for i in range(0,N)]))
	#ind=[0,3,shape[0]/2+shape[1]/2*shape[0],np.prod(shape)-1]
	t_init = time()
	ne=neighbor_indices(ind,shape,d=8)
	t_final = time()
	print '1D neighbors:'
	print 'elapsed time: %ss.' % str(t_final-t_init)
	grid=np.array(list(itertools.product(np.array(range(0,shape[0])),np.array(range(0,shape[1])))))
	tree=spatial.cKDTree(grid)
	t_init = time()
	kk=neighbor_indices_ball(tree,ind,shape,d=8)
	t_final = time()
	nse=[x[0]+x[1]*shape[0] for x in grid[kk]]
	print '1D neighbors scipy:'
	print 'elapsed time: %ss.' % str(t_final-t_init)
	print 'Difference'
	print np.setdiff1d(ne,nse)
	print np.setdiff1d(nse,ne)
