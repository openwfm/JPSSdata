import warnings
warnings.filterwarnings("ignore")
import scipy.io as sio
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import saveload as sl
from utils import Dict
from interpolation import sort_dates, nearest_scipy, neighbor_indices_ball, neighbor_indices_pixel, neighbor_indices_ellipse
import os, sys, time, itertools

def process_detections(data,fxlon,fxlat,time_num):
	"""
	Process detections to obtain upper and lower bounds

	:param data: data obtained from JPSSD
	:param fxlon: longitude coordinates of the fire mesh (from wrfout)
	:param fxlat: latitude coordinates of the fire mesh (from wrfout)
	:param time_num: numerical value of the starting and ending time
	:return result: upper and lower bounds with some parameters

	Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
	Angel Farguell (angel.farguell@gmail.com), 2019-04-01
	"""

	# process satellite settings
	maxsize=500 # Max size of the fire mesh
	ut=1 # Upper bound technique, ut=1: Center of the pixel -- ut=2: Ellipse inscribed in the pixel
	lt=1 # Lower bound technique, lt=1: Center of the pixel -- lt=2: Ellipse inscribed in the pixel (very slow)
	mt=2 # Mask technique, mt=1: Ball -- mt=2: Pixel -- mt=3: Ellipse
	dist=8 # If mt=1 (ball neighbours), radius of the balls is R=sqrt(2*dist^2)
	mm=5 # If mt=3 (ellipse neighbours), larger ellipses constant: (x/a)^2+(x/b)^2<=mm
	confl=10. # Minimum confidence level for the pixels
	confa=False # Histogram plot of the confidence level distribution
	confm=True # Store confidence of each fire and ground detection
	conf_nofire=70. # In absence of nofire confidence, value for nofire confidence (satellite data)
	burn=False # Using or not the burned scar product

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
	if confm:
		C=np.zeros(DD)
		Cg=np.zeros(DD)

	# Confidence analysis
	confanalysis=Dict({'f7': np.array([]),'f8': np.array([]), 'f9': np.array([])})

	# For granules in order increasing in time
	GG=len(sdata)
	for gran in range(GG):
		t_init = time.time()
		print 'Loading data of granule %d/%d' % (gran+1,GG)
		print 'Granule name: %s' % sdata[gran][0]
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
		fi=gfire >= 7  # where fire detected - low, nominal or high confidence (all the fire data in the granule)
		ffi=ff[fi] # indices in the fire mesh where the fire detections are
		nofi=np.logical_or(gfire == 3, gfire == 5) # where no fire detected
		unkn=np.logical_not(np.logical_or(fi,nofi)) # where unknown
		print 'fire detected    %s' % fi.sum()
		print 'no fire detected %s' % nofi.sum()
		print 'unknown          %s' % unkn.sum()
		if fi.any():   # at fire points
			rfire=gfire[gfire>=7]
			conf=sdata[gran][1]['conf_fire'] # confidence of the fire detections
			confanalysis.f7=np.concatenate((confanalysis.f7,conf[rfire==7]))
			confanalysis.f8=np.concatenate((confanalysis.f8,conf[rfire==8]))
			confanalysis.f9=np.concatenate((confanalysis.f9,conf[rfire==9]))
			flc=conf>=confl # fire large confidence indexes
			ffa=U[ffi][flc]>ti # first fire arrival

			if ut>1 or mt>1:
				# taking lon, lat, scan and track of the fire detections which fire large confidence indexes
				lon=sdata[gran][1]['lon_fire'][flc][ffa]
				lat=sdata[gran][1]['lat_fire'][flc][ffa]
				scan=sdata[gran][1]['scan_fire'][flc][ffa]
				track=sdata[gran][1]['track_fire'][flc][ffa]

			# Set upper bounds
			if ut==1:
				# indices with high confidence
				iu=ffi[flc][ffa]
			elif ut==2:
				# creating the indices for all the pixel neighbours of the upper bound
				iu=neighbor_indices_ellipse(vfxlon,vfxlat,lon,lat,scan,track)
			else:
				print 'ERROR: invalid ut option.'
				sys.exit()
			mu = U[iu] > ti # only upper bounds did not set yet
			if confm:
				if ut==1:
					C[iu[mu]]=conf[flc][ffa][mu]
				else:
					print 'ERROR: ut=2 and confm=True not implemented!'
					sys.exit(1)
			print 'U set at %s points' % mu.sum()
			if ut==1:
				U[iu[mu]]=ti # set U to granule time where fire detected and not detected before
			else:
				U[iu][mu]=ti # set U to granule time where fire detected and not detected before

			# Set mask
			if mt==1:
				# creating the indices for all the pixel neighbours of the upper bound indices
				kk=neighbor_indices_ball(itree,np.unique(ffi[flc]),fxlon.shape,dist)
				im=np.array(sorted(np.unique([x[0]+x[1]*fxlon.shape[0] for x in vfind[kk]])))
			elif mt==2:
				# creating the indices for all the pixel neighbours of the upper bound indices
				im=neighbor_indices_pixel(vfxlon,vfxlat,lon,lat,scan,track)
			elif mt==3:
				# creating the indices for all the pixel neighbours of the upper bound indices
				im=neighbor_indices_ellipse(vfxlon,vfxlat,lon,lat,scan,track,mm)
			else:
				print 'ERROR: invalid mt option.'
				sys.exit()
			if mt > 1:
				ind = np.where(im)[0]
				mmt = ind[ti < T[im]] # only mask did not set yet
				print 'T set at %s points' % mmt.shape
				T[mmt]=ti # update mask T
			else:
				print 'T set at %s points' % im[T[im] > ti].shape
				T[im[T[im] > ti]]=ti # update mask T

		# Set mask from burned scar data
		if burn:
			if 'burned' in sdata[gran][1].keys():
				# if burned scar exists, set the mask in the burned scar pixels
				burned=sdata[gran][1]['burned']
				bm=ff[np.reshape(burned,np.prod(burned.shape))[gg]]
				T[bm]=ti

		if nofi.any(): # set L at no-fire points and not masked
			if lt==1:
				# indices of clear ground
				jj=np.logical_and(nofi,ti<T[ff])
				il=ff[jj]
			elif lt==2:
				# taking lon, lat, scan and track of the ground detections
				lon=sdata[gran][1]['lon_nofire']
				lat=sdata[gran][1]['lat_nofire']
				scan=sdata[gran][1]['scan_nofire']
				track=sdata[gran][1]['track_nofire']
				# creating the indices for all the pixel neighbours of lower bound indices
				nofi=neighbor_indices_pixel(vfxlon,vfxlat,lon,lat,scan,track)
				il=np.logical_and(nofi,ti<T)
			else:
				print 'ERROR: invalid lt option.'
				sys.exit()
			if confm:
				if lt==1:
					mask_nofi = gg[np.logical_or(vfire == 3, vfire == 5)]
					try:
						# get nofire confidence if we have it
						confg=sdata[gran][1]['conf_nofire'][mask_nofi]
					except:
						# if not, define confidence from conf_nofire value
						confg=conf_nofire*np.ones(nofi.sum())
					Cg[il]=confg[(ti<T[ff])[nofi]]
				else:
					print 'ERROR: lt=2 and confm=True not implemented!'
					sys.exit(1)
			L[il]=ti # set L to granule time where fire detected
			print 'L set at %s points' % jj.sum()
		t_final = time.time()
		print 'elapsed time: %ss.' % str(t_final-t_init)

	print "L<U: %s" % (L<U).sum()
	print "L=U: %s" % (L==U).sum()
	print "L>U: %s" % (L>U).sum()
	print "average U-L %s" % ((U-L).sum()/np.prod(U.shape))
	print np.histogram((U-L)/(24*3600))

	if (L>U).sum() > 0:
		print "Inconsistency in the data, removing lower bounds..."
		L[L>U]=time_scale_num[0]
		print "L<U: %s" % (L<U).sum()
		print "L=U: %s" % (L==U).sum()
		print "L>U: %s" % (L>U).sum()
		print "average U-L %s" % ((U-L).sum()/np.prod(U.shape))
		print np.histogram((U-L)/(24*3600))

	print 'Confidence analysis'
	if confa:
		plt.subplot(1,3,1)
		plt.hist(x=confanalysis.f7,bins='auto',color='#ff0000',alpha=0.7, rwidth=0.85)
		plt.xlabel('Confidence')
		plt.ylabel('Frequency')
		plt.title('Fire label 7: %d' % len(confanalysis.f7))
		plt.subplot(1,3,2)
		plt.hist(x=confanalysis.f8,bins='auto',color='#00ff00',alpha=0.7, rwidth=0.85)
		plt.xlabel('Confidence')
		plt.ylabel('Frequency')
		plt.title('Fire label 8: %d' % len(confanalysis.f8))
		plt.subplot(1,3,3)
		plt.hist(x=confanalysis.f9,bins='auto',color='#0000ff',alpha=0.7, rwidth=0.85)
		plt.xlabel('Confidence')
		plt.ylabel('Frequency')
		plt.title('Fire label 9: %d' % len(confanalysis.f9))
		plt.show()

	print 'Saving results'
	# Result
	U=np.transpose(np.reshape(U-time_scale_num[0],fxlon.shape))
	L=np.transpose(np.reshape(L-time_scale_num[0],fxlon.shape))
	T=np.transpose(np.reshape(T-time_scale_num[0],fxlon.shape))

	print 'U L R are shifted so that zero there is time_scale_num[0] = %s' % time_scale_num[0]

	if confm:
		C=np.transpose(np.reshape(C,fxlon.shape))
		Cg=np.transpose(np.reshape(Cg,fxlon.shape))
		result = {'U':U, 'L':L, 'T':T, 'fxlon': fxlon, 'fxlat': fxlat,
		'time_num':time_num, 'time_scale_num' : time_scale_num,
		'time_num_granules' : tt, 'C':C, 'Cg': Cg}
	else:
		result = {'U':U, 'L':L, 'T':T, 'fxlon': fxlon, 'fxlat': fxlat,
		'time_num':time_num, 'time_scale_num' : time_scale_num,
		'time_num_granules' : tt}

	sio.savemat('result.mat', mdict=result)

	print 'To visualize, run in Matlab the script plot_results.m'
	print 'Multigrid using in fire_interpolation the script jpss_mg.m'

	result['fxlon'] = np.transpose(result['fxlon'])
	result['fxlat'] = np.transpose(result['fxlat'])

	return result

if __name__ == "__main__":

	sat_file = 'data'

	if os.path.isfile(sat_file) and os.access(sat_file,os.R_OK):
		print 'Loading the data...'
		data,fxlon,fxlat,time_num=sl.load('data')
	else:
		print 'Error: file %s not exist or not readable' % sat_file
		sys.exit(1)

	process_detections(data,fxlon,fxlat,time_num)

