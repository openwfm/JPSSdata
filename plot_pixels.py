from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from JPSSD import read_viirs_files
from mpl_toolkits.mplot3d import Axes3D
import sys

def create_pixels(lons,lats,widths,heights,alphas,color):
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

def angles(lons,lats,ad):
	"""
    Approximation of the angles the pixels should be rotated
    
    :param lons: array of longitude coordinates at the center of the pixels
    :param lats: array of latitude coordinates at the center of the pixels
    :param ad: boolean variable to determine the direction of the angles
    :return alphas: array of angles of the pixels
                 
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-12-28
    """

	if ad:
		dx=lons[0:-1]-lons[1:]
		if dx[0]<0:
			dx[0]=dx[dx>0][0]
		dy=lats[0:-1]-lats[1:]
		if dy[0]<0:
			dy[0]=dy[dy>0][0]
		dlons=np.array([dx[k] if dx[k]>=0 else dx[k-1] for k in range(len(dx))])
		dlats=np.array([dy[k] if dy[k]>=0 else dy[k-1] for k in range(len(dy))])
		alphas=np.arctan(dlats/dlons)*180/np.pi
	else:
		dx=lons[0:-1]-lons[1:]
		if dx[0]<0:
			dx[0]=dx[dx>0][0]
		dy=lats[0:-1]-lats[1:]
		if dy[0]>0:
			dy[0]=dy[dy<0][0]
		dlons=np.array([dx[k] if dx[k]>=0 else dx[k-1] for k in range(len(dx))])
		dlats=np.array([dy[k] if dy[k]<=0 else dy[k-1] for k in range(len(dy))])
		alphas=np.arctan(dlats/dlons)*180/np.pi
	return alphas

def irregular_pixels_plot(g,bounds):
	"""
    Irregular pixels plot: generates a plot of the pixels using along-scan and track dimensions.
    
    :param g: granule dictionary from read_*_files functions in JPSSD.py
    :param bounds: array with the coordinates of the bounding box of the case
    :return: 2D plot of the pixels
                 
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-12-28
    """

	# Angle direction
	(ii,jj)=np.array(g.lat.shape)/2
	if g.lat[ii,jj]-g.lat[ii,jj+1]>0:
		ad=True # Angles should be positive
	else:
		ad=False # Angles should be negative

	fig = plt.figure()
	ax = fig.add_subplot(111)

	R=6378  # Earth radius

	# Ground pixels
	lons=g.lon_nofire
	lats=g.lat_nofire
	tracks=g.track_nofire
	scans=g.scan_nofire
	km_lats=[180/(np.pi*R)]*len(lons)  # 1km in degrees latitude
	km_lons=km_lats/np.cos(lats*np.pi/180)
	widths=km_lons*scans
	heights=km_lats*tracks
	alphas=angles(lons,lats,ad)
	color=(0,0.59765625,0)
	col_nofire=create_pixels(lons,lats,widths,heights,alphas,color)
	ax.add_collection(col_nofire)

	# Fire pixels
	lons=g.lon_fire
	lats=g.lat_fire
	tracks=g.track_fire
	scans=g.scan_fire
	km_lats=[180/(np.pi*R)]*len(lons)  # 1km in degrees latitude
	km_lons=km_lats/np.cos(lats*np.pi/180)
	widths=km_lons*scans
	heights=km_lats*tracks
	alphas=[np.mean(alphas)]*len(lons)
	color=(0.59765625,0,0)
	col_fire=create_pixels(lons,lats,widths,heights,alphas,color)
	ax.add_collection(col_fire)

	# Set axis
	ax.set_xlim(bounds[0],bounds[1])
	ax.set_ylim(bounds[2],bounds[3])

	plt.show()

def regular_pixels_plot(g,bounds):
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
	w[:,1:-1]=np.hstack([np.maximum(dh[:,[k]],dh[:,[k+1]]) for k in range(dh.shape[1]-2)])
	h=dv
	h[1:-1,:]=np.vstack([np.maximum(dv[k],dv[k+1]) for k in range(dv.shape[0]-2)])

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
	col_nofire=create_pixels(lons,lats,widths,heights,alphas,color)
	ax.add_collection(col_nofire)

	# Fire pixels
	lons=lon[fi]
	lats=lat[fi]
	widths=width[fi]
	heights=height[fi]
	alphas=alpha[fi]
	color=(0.59765625,0,0)
	col_fire=create_pixels(lons,lats,widths,heights,alphas,color)
	ax.add_collection(col_fire)

	# Set axis
	ax.set_xlim(bounds[0],bounds[1])
	ax.set_ylim(bounds[2],bounds[3])

	plt.show()

if __name__ == "__main__":
	if len(sys.argv)!=2:
		print "Error: python %s pixel_plot_type" % sys.argv[0] 
		print "  - Irregular pixels plot: 0."
		print "  - Regular pixels plot: 1."
		sys.exit(1)
	else:
		if sys.argv[1] not in ['0','1']:
			print "Error: python %s pixel_plot_type" % sys.argv[0] 
			print "  - Irregular pixels plot: 0."
			print "  - Regular pixels plot: 1."
			print sys.argv[1]
			sys.exit(1)

	# Upload data
	files=Dict({})
	files.geo='VNP03MODLL.A2018320.2036.001.2018321040917.h5'
	files.fire='VNP14.A2018320.2036.001.2018321030603.nc'
	bounds=[-122.042564, -120.96382, 39.34862, 40.174423]
	g=read_viirs_files(files,bounds)
	# Plot pixels
	if int(sys.argv[1]):
		regular_pixels_plot(g,bounds)
	else:
		irregular_pixels_plot(g,bounds)
