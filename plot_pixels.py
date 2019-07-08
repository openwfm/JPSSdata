from utils import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import saveload as sl
from interpolation import sort_dates
import StringIO
import sys
import os

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


def basemap_scatter_mercator(g,bounds,bmap):
	"""
	Scatter plot of fire and ground pixels in a png file using a basemap with mercator projection

	:param g: granule dictionary from read_*_files functions in JPSSD.py
	:param bounds: array with the coordinates of the bounding box of the case
	:param map: basemap mercator map where to plot the center of the pixels
	:return png: string with png plot
	:return float_bounds: coordinates of the borders of the png file

	Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
	Angel Farguell (angel.farguell@gmail.com) 2018-04-05
	"""

	size = 50
	# Satellite pixels
	flon = g.lon.ravel()
	flat = g.lat.ravel()
	mask = g.fire.ravel()

	fil = np.logical_and(np.logical_and(np.logical_and(flon >= bounds[0], flon <= bounds[1]), flat >= bounds[2]), flat <= bounds[3])

	categories = (np.array(mask[fil] == 3), np.array(mask[fil] == 5),
				np.array(mask[fil] == 7), np.array(mask[fil] == 8),
				np.array(mask[fil] == 9))
	alphas = (.4,.4,.5,.6,.7)
	labels = ('Water','Ground','Fire low','Fire nominal','Fire high')

	lon = []
	lat = []
	val = []
	for i,cat in enumerate(categories):
		lon.append(flon[fil][cat])
		lat.append(flat[fil][cat])
		val.append(np.ones(lon[i].shape)*i)

	N = len(categories)
	fig = plt.figure(frameon=False,figsize=(12,8),dpi=50)
	plt.axis('off')
	colors = [(0,0,.5),(0,.5,0),(1,1,0),(1,.65,0),(.5,0,0)]
	cmap = LinearSegmentedColormap.from_list('fire_detections', colors, N=N)
	for i in range(N):
		bmap.scatter(lon[i],lat[i],size,c=val[i],latlon=True,marker='.',cmap=cmap,vmin=-.5,vmax=N-.5,alpha=alphas[i],linewidths=0)

	str_io = StringIO.StringIO()
	plt.savefig(str_io,bbox_inches='tight',format='png',pad_inches=0,transparent=True)

	#colorbar
	orientation = 'vertical'
	size_in = 2
	cb_label = ''

	kwargs = { 'norm': mpl.colors.Normalize(-.5,N-.5),
			   'orientation': orientation,
			   'spacing': 'proportional',
			   'ticks': range(0,N),
			   'cmap': cmap}

	# build figure according to requested orientation
	hsize, wsize = (size_in,size_in*0.5) if orientation == 'vertical' else (size_in*0.5,size_in)
	fig = plt.figure(figsize=(wsize,hsize))

	# proportions that work with axis title (horizontal not tested)
	ax = fig.add_axes([.5,.03,.12,.8]) if orientation=='vertical' else fig.add_axes([0.03,.4,.8,.12])

	# construct the colorbar and modify properties
	cb = mpl.colorbar.ColorbarBase(ax,**kwargs)
	cb.ax.tick_params(length=0)
	cb.ax.set_yticklabels(labels)
	cb.set_label(cb_label,color='0',fontsize=8,labelpad=-50)

	# move ticks to left side
	ax.yaxis.set_ticks_position('left')
	for tick_lbl in ax.get_yticklabels():
		tick_lbl.set_color('0')
		tick_lbl.set_fontsize(5)

	#plt.show()
	plt.close()

	png = str_io.getvalue()

	numpy_bounds = [ (bounds[0],bounds[2]),(bounds[1],bounds[2]),(bounds[1],bounds[3]),(bounds[0],bounds[3]) ]
	float_bounds = [ (float(x), float(y)) for x,y in numpy_bounds ]
	return png, float_bounds


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
	size=80

	# Ground pixels
	lons=np.array(g.lon_nofire)
	lats=np.array(g.lat_nofire)
	color=(0,0.59765625,0)
	plt.scatter(lons,lats,size,marker='.',color=color,edgecolors='k')

	# Fire pixels
	lons=np.array(g.lon_fire)
	lats=np.array(g.lat_fire)
	color=(0.59765625,0,0)
	plt.scatter(lons,lats,size,marker='.',color=color,edgecolors='k')

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
	fil=np.logical_and(np.logical_and(np.logical_and(flon >= bounds[0], flon <= bounds[1]), flat >= bounds[2]), flat <= bounds[3])
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


def create_kml(kml_data,kml_path):
	"""
	Creation of KML file with png files from basemap_scatter_mercator function

	:param kml_data: dictionary with granules to plot information: name, png_file, bounds, time
	:param kml_path: path to kml to generate

	Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
	Angel Farguell (angel.farguell@gmail.com) 2018-04-05
	"""

	with open(kml_path,'w') as kml:
		kml.write("""<?xml version="1.0" encoding="UTF-8"?>\n""")
		kml.write("""<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n""")
		kml.write("""<Document><name>Observed Data</name>""")
		kml.write("""
		<Folder>
			<name>Overlays</name>""")
		for idx,data in enumerate(kml_data):
			name = data['name']
			time = data['time']
			png = data['png_file']
			bounds = data['bounds']
			coord = (str(bounds[0]),str(bounds[2]),
				str(bounds[1]),str(bounds[2]),
				str(bounds[1]),str(bounds[3]),
				str(bounds[0]),str(bounds[3]),
				str(bounds[0]),str(bounds[2]))
			kml.write("""
			<GroundOverlay>
				<name>%s</name>
				<Icon><href>%s</href></Icon>
				<gx:TimeStamp><when>%s</when></gx:TimeStamp>""" % (
				name, png, time))
			kml.write("""
				<gx:LatLonQuad>
					<coordinates>
						%s,%s,0
						%s,%s,0
						%s,%s,0
						%s,%s,0
						%s,%s,0
					</coordinates>
				</gx:LatLonQuad>""" % coord)
			kml.write("""
			</GroundOverlay>""")
		kml.write("""
		</Folder>""")
		kml.write("""</Document>\n</kml>\n""")

	print 'Created file %s' % kml_path


def perror():
	print "Error: python %s pixel_plot_type" % sys.argv[0]
	print "  - Center pixels in basemap: 0"
	print "  - Center pixels plot: 1"
	print "  - Regular pixels plot: 2"

if __name__ == "__main__":
	if len(sys.argv)!=2:
		perror()
		sys.exit(1)
	else:
		if sys.argv[1] not in ['0','1','2']:
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
		m = Basemap(projection='merc',llcrnrlat=bounds[2], urcrnrlat=bounds[3], llcrnrlon=bounds[0], urcrnrlon=bounds[1])
		kmld = []
		for idx, g in enumerate(granules):
			raster_png_data,corner_coords = basemap_scatter_mercator(g[1],bounds,m)
			bounds = (corner_coords[0][0],corner_coords[1][0],corner_coords[0][1],corner_coords[2][1])
			pngfile = g[0]+'.png'
			timestamp = g[1].acq_date + 'T' + g[1].acq_time[0:2] + ':' + g[1].acq_time[2:4] + 'Z'
			with open(pngfile, 'w') as f:
				f.write(raster_png_data)
			print '> File %s saved.' % g[0]
			kmld.append(Dict({'name': g[0], 'png_file': pngfile, 'bounds': bounds, 'time': timestamp}))
		create_kml(kmld,'./doc.kml')
		os.system('zip -r granules.kmz doc.kml *.png')
		print 'Created file granules.kmz'
		os.system('rm doc.kml *_A*_*.png')
	elif sys.argv[1] is '1':
		for g in granules:
			center_pixels_plot(g[1],bounds)
			plt.title('Granule %s' % g[0])
		plt.show()
	else:
		for g in granules:
			pixels_plot(g[1],bounds)
			plt.title('Granule %s' % g[0])
		plt.show()
