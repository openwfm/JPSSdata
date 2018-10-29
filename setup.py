import warnings
warnings.filterwarnings("ignore")
import scipy.io as sio
import pdb
import saveload as sl
from interpolation import *
import time
import numpy as np
import sys
from scipy import spatial
import itertools

# setup.py settings
maxsize=400 # Max size of the fire mesh
ut=1 # Upper bound technique, ut=1: Center of the pixel -- ut=2: Ellipse inscribed in the pixel
lt=1 # Lower bound technique, lt=1: Center of the pixel -- lt=2: Ellipse inscribed in the pixel (very slow)
mt=3 # Mask technique, mt=1: Ball -- mt=2: Pixel -- mt=3: Ellipse
if mt<2:
	dist=8 # If mt=1 (ball neighbours), radius of the balls is R=sqrt(2*dist^2)
elif mt>2:
	mm=2 # If mt=3 (ellipse neighbours), larger ellipses constant: (x/a)^2+(x/b)^2<=mm
pen=False # Creating heterogeneous penalty depending on the confidence level

print 'Loading data'
data,fxlon,fxlat,time_num=sl.load('data')

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
DD=np.prod(fxlon.shape)
U=np.empty(DD)
U[:]=time_scale_num[1]
L=np.empty(DD)
L[:]=time_scale_num[0]
T=np.empty(DD)
T[:]=time_scale_num[1]
if pen:
	LP=np.zeros(DD)
	UP=np.zeros(DD)

# For granules in order increasing in time
GG=len(sdata)
for gran in range(GG):
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
		# indices with fire mask larger than 7
		afi=gfire >= 7
		# creating the fire labels of the elements larger than 7
		gafire=gfire[afi]
		# indices from the previous elements which are larger than 8
		flc=gafire >= 8 # fire large confidence indexes
		# Set upper bounds
		if ut==1:
			iu=ff[fi]
			if pen:
				conf=sdata[gran][1]['conf_fire'][flc]
				LP[iu]=conf
		elif ut==2:
			# taking lon, lat, scan and track of the fire detections which are >=8
			lon=sdata[gran][1]['lon_fire'][flc]
			lat=sdata[gran][1]['lat_fire'][flc]
			scan=sdata[gran][1]['scan_fire'][flc]
			track=sdata[gran][1]['track_fire'][flc]
			# creating the indices for all the pixel neighbours of the upper bound
			iu=neighbor_indices_ellipse(vfxlon,vfxlat,lon,lat,scan,track)
		else:
			print 'ERROR: invalid ut option.'
			sys.exit()
		U[iu]=ti   # set U to granule time where fire detected
		# Set mask
		if mt==1:
			kk=neighbor_indices_ball(itree,ff[fi],fxlon.shape,dist) 
			im=sorted(np.unique([x[0]+x[1]*fxlon.shape[0] for x in vfind[kk]]))
		elif mt==2:
			# creating the indices for all the pixel neighbours
			im=neighbor_indices_pixel(vfxlon,vfxlat,lon,lat,scan,track)
		elif mt==3:
			# creating the indices for all the pixel neighbours
			im=neighbor_indices_ellipse(vfxlon,vfxlat,lon,lat,scan,track,mm)
		else:
			print 'ERROR: invalid mt option.'
			sys.exit()		
		T[im]=ti       # update mask

	if nofi.any(): # set L at no-fire points and not masked
		if lt==1:
			jj=np.logical_and(nofi,ti<T[ff])
			il=ff[jj]
		elif lt==2:
			# taking lon, lat, scan and track of the ground detections
			lon=sdata[gran][1]['lon_nofire']
			lat=sdata[gran][1]['lat_nofire']
			scan=sdata[gran][1]['scan_nofire']
			track=sdata[gran][1]['track_nofire']
			# creating the indices for all the pixel neighbours
			nofi=neighbor_indices_pixel(vfxlon,vfxlat,lon,lat,scan,track)
			il=np.logical_and(nofi,ti<T)
		else:
			print 'ERROR: invalid lt option.'
			sys.exit()	
		L[il]=ti
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
U=np.transpose(np.reshape(U-time_scale_num[0],fxlon.shape))
L=np.transpose(np.reshape(L-time_scale_num[0],fxlon.shape))
T=np.transpose(np.reshape(T-time_scale_num[0],fxlon.shape))

print 'U L R are shifted so that zero there is time_scale_num[0] = %s' % time_scale_num[0]
sl.save((U,L,T),'result')

if pen:
	result = {'U':U, 'L':L, 'T':T, 'fxlon': fxlon, 'fxlat': fxlat, 
	'time_num':time_num, 'time_scale_num' : time_scale_num, 
	'time_num_granules' : tt, 'UP':UP, 'LP':LP}
else:
	result = {'U':U, 'L':L, 'T':T, 'fxlon': fxlon, 'fxlat': fxlat, 
	'time_num':time_num, 'time_scale_num' : time_scale_num, 
	'time_num_granules' : tt}

sio.savemat('result.mat', mdict=result)

print 'To visualize, run in Matlab the script plot_results.m'
print 'Multigrid using in fire_interpolation the script jpss_mg.m'
