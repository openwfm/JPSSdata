from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D
import saveload as sl
from interpolation import sort_dates
import sys

def create_pixels(lons,lats,widths,heights,alphas,color,label):
	"""
    Plot of pixels using centers (lons,lats), with sizes (widhts,heights), angle alphas, and color color.
    
    :param lons: array of longitude coordinates at the center of the pixels
    :param lats: array of latitude coordinates at the center of the pixels
    :param widths: array of widhts of the pixels
    :param heights: array of heights of the pixels
    :param alphas: array of angles of the pixels
    :param color: tuple with RGB color values
    :return col: collection of rectangles with the pixels
                 
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-12-28
    """

	# Create pixels
	pixels=[]
	for x, y, h, w, a in zip(lons, lats, heights, widths, alphas):
	    xpos = x - w/2 # The x position will be half the width from the center
	    ypos = y - h/2 # same for the y position, but with height
	    rect = Rectangle( (xpos, ypos), w, h, a) # Create a rectangle
	    pixels.append(rect) # Add the rectangle patch to our list

	# Create a collection from the rectangles
	col = PatchCollection(pixels)
	# set the alpha for all rectangles
	col.set_alpha(0.5)
	# Set the colors using the colormap
	col.set_facecolor( [color]*len(lons) )
	# No lines
	col.set_linewidth( 0 )

	return col

def center_pixels_plot(g,bounds):
	"""
    Center pixels plot: generates a plot of the center of the pixels.
    
    :param g: granule dictionary from read_*_files functions in JPSSD.py
    :param bounds: array with the coordinates of the bounding box of the case
    :return: 2D plot of the center of the pixels
                 
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-12-28
    """

	fig=plt.figure()
	ax=fig.add_subplot(111)

    # Size of the center of the pixels
	size=100

	# Ground pixels
	lons=np.array(g.lon_nofire)
	lats=np.array(g.lat_nofire)
	color=(0,0.59765625,0)
	plt.scatter(lons,lats,size,marker='.',color=color)

	# Fire pixels
	lons=np.array(g.lon_fire)
	lats=np.array(g.lat_fire)
	color=(0.59765625,0,0)
	plt.scatter(lons,lats,size,marker='.',color=color)

	ax.set_xlim(bounds[0],bounds[1])
	ax.set_ylim(bounds[2],bounds[3])

def pixels_plot(g,bounds):
	"""
    Regular pixels plot: generates a plot of the pixels using a regular grid by distances.
    
    :param g: granule dictionary from read_*_files functions in JPSSD.py
    :param bounds: array with the coordinates of the bounding box of the case
    :return: 2D plot of the pixels
                 
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-12-28
    """

	lon=g.lon
	lat=g.lat
	fire=g.fire

	lonh0=lon[:,0:-1]
	lonh1=lon[:,1:]
	lonv0=lon[0:-1,:]
	lonv1=lon[1:,:]
	dlonh=lonh0-lonh1
	dlonh=np.hstack((dlonh,dlonh[:,[-1]]))
	dlonv=lonv0-lonv1
	dlonv=np.vstack((dlonv,dlonv[-1]))

	lath0=lat[:,0:-1]
	lath1=lat[:,1:]
	latv0=lat[0:-1,:]
	latv1=lat[1:,:]
	dlath=lath0-lath1
	dlath=np.hstack((dlath,dlath[:,[-1]]))
	dlatv=latv0-latv1
	dlatv=np.vstack((dlatv,dlatv[-1]))

	dh=np.sqrt(dlonh**2+dlath**2)
	dv=np.sqrt(dlonv**2+dlatv**2)

	# Mean between the distances
	w=dh
	w[:,1:-1]=np.hstack([(dh[:,[k]]+dh[:,[k+1]])/2 for k in range(dh.shape[1]-2)])
	h=dv
	h[1:-1,:]=np.vstack([(dv[k]+dv[k+1])/2 for k in range(dv.shape[0]-2)])

	'''
	# Maximum between the distances
	w=dh
	w[:,1:-1]=np.hstack([np.maximum(dh[:,[k]],dh[:,[k+1]]) for k in range(dh.shape[1]-2)])
	h=dv
	h[1:-1,:]=np.vstack([np.maximum(dv[k],dv[k+1]) for k in range(dv.shape[0]-2)])
	'''

	angle=np.arctan(dlath/dlonh)*180/np.pi

	plot=False
	if plot:
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(lon,lat,w)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(lon,lat,h)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(lon,lat,angle)

	flon=np.reshape(lon,np.prod(lon.shape))
	flat=np.reshape(lat,np.prod(lat.shape))
	fil=np.logical_and(np.logical_and(np.logical_and(flon>bounds[0],flon<bounds[1]),flat>bounds[2]),flat<bounds[3])
	lon=flon[fil]
	lat=flat[fil]

	ff=np.reshape(fire,np.prod(fire.shape))
	ffg=ff[fil]
	nofi=np.logical_or(ffg == 3, ffg == 5)
	fi=np.array(ffg > 6)

	width=np.reshape(w,np.prod(w.shape))[fil]
	height=np.reshape(h,np.prod(h.shape))[fil]
	alpha=np.reshape(angle,np.prod(angle.shape))[fil]

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Ground pixels
	lons=lon[nofi]
	lats=lat[nofi]
	widths=width[nofi]
	heights=height[nofi]
	alphas=alpha[nofi]
	color=(0,0.59765625,0)
	col_nofire=create_pixels(lons,lats,widths,heights,alphas,color,'Ground')
	ax.add_collection(col_nofire)

	# Fire pixels
	lons=lon[fi]
	lats=lat[fi]
	widths=width[fi]
	heights=height[fi]
	alphas=alpha[fi]
	color=(0.59765625,0,0)
	col_fire=create_pixels(lons,lats,widths,heights,alphas,color,'Fire')
	ax.add_collection(col_fire)

	# Set axis
	ax.set_xlim(bounds[0],bounds[1])
	ax.set_ylim(bounds[2],bounds[3])

	# Color map
	#colors=[(0,0.59765625,0),(0.59765625,0,0)]
	#cmap=mp.colors.ListedColormap(colors)

	#cb.set_ticklabels(colors)

def perror():
	print "Error: python %s pixel_plot_type" % sys.argv[0] 
	print "  - Center pixels plot: 0."
	print "  - Regular pixels plot: 1."

if __name__ == "__main__":
	if len(sys.argv)!=2:
		perror()
		sys.exit(1)
	else:
		if sys.argv[1] not in ['0','1']:
			perror()
			print "Invalid value given: %s" % sys.argv[1]
			sys.exit(1)

	try:
		# Upload data
		print 'Loading data...'
		data,fxlon,fxlat,time_num=sl.load('data')
	except Exception:
		print "Error: No data file in the directory. First run a case using case.py."
		sys.exit(1)

	granules=sort_dates(data)
	bounds=[fxlon.min(),fxlon.max(),fxlat.min(),fxlat.max()]

	print 'Plotting data...'
	# Plot pixels
	if sys.argv[1] is '0':
		for g in granules:
			center_pixels_plot(g[1],bounds)
			plt.title('Granule %s' % g[0])
	else:
		for g in granules:
			pixels_plot(g[1],bounds)
			plt.title('Granule %s' % g[0])

	plt.show()