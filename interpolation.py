import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from time import time
from datetime import datetime
from scipy import spatial
import itertools

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
	

def nearest_scipy(lon,lat,stree,bounds,dub=np.inf):
	""" 
    Returns the coordinates interpolated from (lon,lat) to (lons,lats) and the value of the indices where NaN values
        :param:
            lon 		2D array of longitudes to look the nearest neighbours
            lat 		2D array of latitudes to look the nearest neighbours
            lons		2D array of longitudes interpolating to
            lats		2D array of latitudes interpolating to
            dub			optional: distance upper bound to look for the nearest neighbours
        :returns: A tuple with a 2D array of coordinates interpolated from (lon,lat) to (lons,lats) and the value of the indices where NaN values

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
	vlon=np.reshape(lon,np.prod(lon.shape))
	vlat=np.reshape(lat,np.prod(lat.shape))
	vlonlat=np.column_stack((vlon,vlat))
	M=(vlon>bounds[0])*(vlon<bounds[1])*(vlat>bounds[2])*(vlat<bounds[3])
	vlonlat=vlonlat[M]
	inds=np.array(stree.query(vlonlat,distance_upper_bound=dub)[1])
	return (inds,M)

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

def neighbor_indices(indices,shape,d=2):
	""" 
    Computes all the neighbor indices from an indice list
        :param:
           	indices 	list of coordinates in a 1D array
            shape 		array of two elements with satellite grid size
            d 			optional: distance of the neighbours
        :returns: Returns a numpy array with the indices and the neighbor indices in 1D array

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
    # Width and Length of the 2D grid
	w=shape[0]
	l=shape[1]
	# 2D indices of the 1D indices
	I=[[ind/l,np.mod(ind,l)] for ind in indices]
	# All the combinations (x,y) for all the neighbor points from x-d to x+d and y-d to y+d
	N=np.concatenate([np.array(list(itertools.product(range(max(i[0]-d,0),min(i[0]+d+1,w)),range(max(i[1]-d,0),min(i[1]+d+1,l))))) for i in I])
	# Recompute the 1D indices of the 2D coordinates inside the 2D domain
	ret=np.array([x[1]+w*x[0] for x in N])
	# Sort them and take each indice once
	return sorted(np.unique(ret))

def neighbor_indices_new(indices,shape,d=2):
	""" 
    Computes all the neighbor indices from an indice list
        :param:
           	indices 	list of coordinates in a 1D array
            shape 		array of two elements with satellite grid size
            d 			optional: distance of the neighbours
        :returns: Returns a numpy array with the indices and the neighbor indices in 1D array

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-20
    """
    # Width and Length of the 2D grid
	w=shape[0]
	l=shape[1]
	# 2D indices of the 1D indices
	I=np.unravel_index(indices,dims=shape)
	# Indices of the neighbor
	ii=[np.array(range(max(i-d,0),min(i+d+1,w))) for i in I[0]]
	jj=[np.array(range(max(j-d,0),min(j+d+1,l))) for j in I[1]]
	# Union between consecutive group of indices
	Ni=[ np.unique(np.concatenate(ii[0:k])) for k in range(1,len(ii)) ]
	Ni.insert(0,np.array([]))
	Nj=[ np.unique(np.concatenate(jj[0:k])) for k in range(1,len(jj)) ]
	Nj.insert(0,np.array([]))
	# Compute combinations (x,y) for all the neighbor points from x-d to x+d and y-d to y+d avoiding some repetitions
	N=[ np.array(list(set(list(itertools.product(np.setdiff1d(ii[k],Ni[k]),jj[k]))) | set(list(itertools.product(ii[k],np.setdiff1d(jj[k],Nj[k])))))) for k in range(0,len(ii)) ]
	# Recompute the 1D indices of the 2D coordinates inside the 2D domain
	ret=np.array([xx[1]+w*xx[0] for x in N for xx in x])
	# Sort them and take each indice once
	return sorted(np.unique(ret))


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
	dub=distance_upper_bound([dx1,dx2],[dy1,dy2])
	vlons=np.reshape(lons,np.prod(lons.shape))
	vlats=np.reshape(lats,np.prod(lats.shape))
	vlonlats=np.column_stack((vlons,vlats))
	stree=spatial.cKDTree(vlonlats)
	(inds,K)=nearest_scipy(lon,lat,stree,bounds,dub)
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
	shape=(15,10)
	ind=[0,2,23,shape[0]/2+shape[1]/2*(shape[0]-1),np.prod(shape)-1]
	print '1D indices:'
	print ind
	t_init = time()
	ne=neighbor_indices(ind,shape,d=8)
	t_final = time()
	print '1D neighbours back:'
	print ne
	print 'elapsed time: %ss.' % str(t_final-t_init)
	t_init = time()
	nne=neighbor_indices_new(ind,shape,d=8)
	t_final = time()
	print '1D neighbours new:'
	print nne
	print 'elapsed time: %ss.' % str(t_final-t_init)
	print 'Difference'
	print np.setdiff1d(ne,nne)

