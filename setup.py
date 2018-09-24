import warnings
warnings.filterwarnings("ignore")
import scipy.io as sio
import pdb
import saveload as sl
from interpolation import sort_dates,nearest_scipy,distance_upper_bound,neighbor_indices
import time
import numpy as np
import sys
from scipy import spatial

print 'Loading data'
data,fxlon,fxlat,time_num=sl.load('data')

maxsize=400

print 'mesh shape %s %s' % fxlon.shape
coarsening=np.int(1+np.max(fxlon.shape)/maxsize)
print 'maximum size is %s, coarsening %s' % (maxsize, coarsening)
fxlon = fxlon[0::coarsening,0::coarsening]
fxlat = fxlat[0::coarsening,0::coarsening]
print 'coarsened  %s %s' % fxlon.shape

bounds=[fxlon.min(),fxlon.max(),fxlat.min(),fxlat.max()]
dx1=fxlon[0,1]-fxlon[0,0]
dx2=fxlat[1,0]-fxlat[0,0]
vfxlon=np.reshape(fxlon,np.prod(fxlon.shape))
vfxlat=np.reshape(fxlat,np.prod(fxlat.shape))
vfgrid=np.column_stack((vfxlon,vfxlat))
print 'Setting up interpolation'
stree=spatial.cKDTree(vfgrid)

# Sort dictionary by time_num into an array of tuples (key, dictionary of values) 
print 'Sort the granules by dates'
sdata=sort_dates(data)
tt=[ dd[1]['time_num'] for dd in sdata ]  # array of times
print 'Sorted?'
stt=sorted(tt)
print tt==stt

# Max and min time_num
maxt=time_num[1]
mint=time_num[0]
# time_scale_num = time_num
time_scale_num = [mint-0.5*(maxt-mint),maxt+2*(maxt-mint)]

# Creating the resulting arrays
U=np.empty(np.prod(fxlon.shape))
U[:]=time_scale_num[1]
L=np.empty(np.prod(fxlon.shape))
L[:]= time_scale_num[0]
T=np.empty(np.prod(fxlon.shape))
T[:]= time_scale_num[1]

# For granules in order increasing in time
GG=len(sdata)
for gran in range(0,GG):
	t_init = time.time()
	print 'Loading data of granule %d/%d' % (gran,GG)
	# Load granule lon, lat, fire arrays and time number
	slon=sdata[gran][1]['lon'] 
	slat=sdata[gran][1]['lat']
	ti=sdata[gran][1]['time_num']
	fire=sdata[gran][1]['fire']
	print 'Interpolation to fire grid'
	sys.stdout.flush()
	# Compute a distance upper bound
	dy1=slon[0,1]-slon[0,0]
	dy2=slat[1,0]-slat[0,0]
	dub=distance_upper_bound([dx1,dx2],[dy1,dy2])
	# Interpolate all the granule coordinates in bounds in the wrfout fire mesh
	# ff: The wrfout fire mesh indices where the pixels are interpolated to
	# gg: Mask of the pixel coordinates in the granule which are inside the bounds
	(ff,gg)=nearest_scipy(slon,slat,stree,bounds,dub)
	print 'Computing fire points'
	# 1D array of labels of fire detection product in the granule
	vfire=np.reshape(fire,np.prod(fire.shape))
	# 1D array of labels of fire detection product in the granule in the bounds
	gfire=vfire[gg]
	# Mask of pixels where there is fire detection
	fi=(gfire>6)
	print 'fire pixels: %s' % fi.sum()
	# If some fire pixel detect
	if fi.any():
		# Set U at the fire mesh node to the minimum of current U and the time of the granule
		U[ff[fi]]=np.minimum(U[ff[fi]],ti)
		print 'Update mask'
		# Compute the neighbor indices
		ii=neighbor_indices(ff[fi],fxlon.shape,d=8) # could use a larger d
		# Set mask T near the fire mesh node
		T[ii]=np.minimum(T[ii],ti)
	print 'Computing the ground points'
	# Mask of ground pixels (only category 5: land)
	la=(gfire==5)
	print 'land pixels: %s' % la.sum()
	if la.any():
		# Land coordinates
		land=ff[la]
		# Find the jj where ti<T (the mask T was not set at the fire node)
		jj=(ti<T[land])
		# Set L to the maximum of current L and the time of the granule
		L[land[jj]]=np.maximum(L[land[jj]],ti)
	t_final = time.time()
	print 'Elapsed time: %ss.' % str(t_final-t_init)

print "L<U: %s" % (L<U).sum()
print "L=U: %s" % (L==U).sum()
print "L>U: %s" % (L>U).sum()

print 'Saving results'
# Result
U=np.reshape(U - time_scale_num[0],fxlon.shape)
L=np.reshape(L - time_scale_num[0],fxlon.shape)
T=np.reshape(T - time_scale_num[0],fxlon.shape)

print 'U L R are shifted so that zero there is time_scale_num[0] = %s' % time_scale_num[0]
sl.save((U,L,T),'result')

result = {'U':U, 'L':L, 'T':T, 'fxlon': fxlon, 'fxlat': fxlat, 
          'time_num':time_num, 'time_scale_num' : time_scale_num}

sio.savemat('result.mat', mdict=result)

print 'to visualize, do in Matlab:'
print 'load result.mat'
print "mesh(fxlon,fxlat,U); title('U')"
print "mesh(fxlon,fxlat,L); title('L')"
