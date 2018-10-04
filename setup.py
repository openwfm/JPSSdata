import warnings
warnings.filterwarnings("ignore")
import scipy.io as sio
import pdb
import saveload as sl
from interpolation import sort_dates,nearest_scipy,distance_upper_bound,neighbor_indices_ball
import time
import numpy as np
import sys
from scipy import spatial
import itertools

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
vfxlon=np.reshape(fxlon,np.prod(fxlon.shape))
vfxlat=np.reshape(fxlat,np.prod(fxlat.shape))
vfgrid=np.column_stack((vfxlon,vfxlat))
print 'Setting up interpolation'
stree=spatial.cKDTree(vfgrid)
vfind=np.array(list(itertools.product(np.array(range(0,fxlon.shape[0])),np.array(range(0,fxlon.shape[1])))))
itree=spatial.cKDTree(vfind)

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
time_scale_num=[mint-0.5*(maxt-mint),maxt+2*(maxt-mint)]

# Creating the resulting arrays
U=np.empty(np.prod(fxlon.shape))
U[:]=time_scale_num[1]
L=np.empty(np.prod(fxlon.shape))
L[:]=time_scale_num[0]
T=np.empty(np.prod(fxlon.shape))
T[:]=time_scale_num[1]

# For granules in order increasing in time
GG=len(sdata)
for gran in range(0,GG):
	t_init = time.time()
	print 'Loading data of granule %d/%d' % (gran+1,GG)
	# Load granule lon, lat, fire arrays and time number
	slon=sdata[gran][1]['lon'] 
	slat=sdata[gran][1]['lat']
	ti=sdata[gran][1]['time_num']
	fire=sdata[gran][1]['fire']
	print 'Interpolation to fire grid'
	sys.stdout.flush()
	# Interpolate all the granule coordinates in bounds in the wrfout fire mesh
	# gg: mask in the granule of g-points = pixel coordinates inside the fire mesh
	# ff: the closed points in fire mesh indexed by g-points
	(ff,gg)=nearest_scipy(slon,slat,stree,bounds) ## indices to flattened granule array
	vfire=np.reshape(fire,np.prod(fire.shape)) ## flaten the fire detection array
	gfire=vfire[gg]   # the part withing the fire mesh bounds
	fi=gfire >= 8  # where fire detected - nominal or high confidence 
	nofi=np.logical_or(gfire == 3, gfire == 5) # where no fire detected
	unkn=np.logical_not(np.logical_or(fi,nofi)) # where unknown
	print 'fire detected    %s' % fi.sum()
	print 'no fire detected %s' % nofi.sum()
	print 'unknown          %s' % unkn.sum()
	if fi.any():   # at fire points
		U[ff[fi]]=ti   # set U to granule time where fire detected
		kk=neighbor_indices_ball(itree,ff[fi],fxlon.shape) 
		ii=sorted(np.unique([x[0]+x[1]*fxlon.shape[0] for x in vfind[kk]]))
		T[ii]=ti       # update mask
	if nofi.any(): # set L at no-fire points and not masked
		jj=np.logical_and(nofi,ti<T[ff])
		L[ff[jj]]=ti
		print 'L set at %s points' % jj.sum()
	t_final = time.time()
	print 'elapsed time: %ss.' % str(t_final-t_init)

print "L<U: %s" % (L<U).sum()
print "L=U: %s" % (L==U).sum()
print "L>U: %s" % (L>U).sum()
print "average U-L %s" % ((U-L).sum()/np.prod(U.shape))
print np.histogram((U-L)/(24*3600))

print 'Saving results'
# Result
U=np.reshape(U-time_scale_num[0],fxlon.shape)
L=np.reshape(L-time_scale_num[0],fxlon.shape)
T=np.reshape(T-time_scale_num[0],fxlon.shape)

print 'U L R are shifted so that zero there is time_scale_num[0] = %s' % time_scale_num[0]
sl.save((U,L,T),'result')

result = {'U':U, 'L':L, 'T':T, 'fxlon': np.transpose(fxlon), 'fxlat': np.transpose(fxlat), 
          'time_num':time_num, 'time_scale_num' : time_scale_num, 'time_num_granules' : tt}

sio.savemat('result.mat', mdict=result)

print 'To visualize, run in Matlab the script plot_results.m'
print 'Multigrid using in fire_interpolation the script jpss_mg.m'
