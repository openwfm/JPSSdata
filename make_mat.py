import saveload as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import spatial
import scipy.io as sio
from time import time
import sys

t_init = time()

print('Loading data...')
data,fxlon,fxlat,time_num=sl.load('data')
t1 = time()
print 'elapsed time: ',abs(t_init-t1),'s'
prefixes={'MOD': 'MOD14', 'MYD': 'MYD14', 'VNP': 'NPP_VAF_L2'}

# constants for geotransform
res = 0.01   # resolution
rot = 0.0    # rotation (not currently supported)

for gran in list(data):
	print '>> Processing granule: ',gran
	tg1 = time()
	# Creating file_name
	splitted=gran.split('_')
	prefix=splitted[0]
	date=splitted[1][3:8]+splitted[2]+'00'
	file_name = prefixes[prefix]+'.'+date+'.tif.mat'

	# Creating meshgrid where to interpolate the data
	bounds = [data[gran].lon.min(),data[gran].lon.max(),data[gran].lat.min(),data[gran].lat.max()]
	lons_interp = np.arange(bounds[0],bounds[1],res)
	lats_interp = np.arange(bounds[2],bounds[3],res)
	lons_interp,lats_interp = np.meshgrid(lons_interp,lats_interp)

	# Creating geotransform array with necessary geolocation information
	geotransform = [bounds[0],res,rot,bounds[3],rot,res]

	# Approximation of the radius for the tree
	dlon = abs(data[gran].lon[1,1]-data[gran].lon[0,0])
	dlat = abs(data[gran].lat[1,1]-data[gran].lat[0,0])
	radius = max(dlon,dlat)
	print 'radius =',radius

	# Flatten granule data into 1d arrays
	lons = np.reshape(data[gran].lon,np.prod(data[gran].lon.shape))
	lats = np.reshape(data[gran].lat,np.prod(data[gran].lat.shape))
	fires = np.reshape(data[gran].fire,np.prod(data[gran].fire.shape)).astype(np.int8)

	# Making tree
	print '> Making the tree...'
	t1 = time()
	tree = spatial.cKDTree(np.column_stack((lons,lats)))
	t2 = time()
	print 'elapsed time: ',abs(t2-t1),'s'
	glons = np.reshape(lons_interp,np.prod(lons_interp.shape))
	glats = np.reshape(lats_interp,np.prod(lats_interp.shape))
	print '> Interpolating the data...'
	t1 = time()
	indexes = np.array(tree.query_ball_point(np.column_stack((glons,glats)),radius))
	t2 = time()
	print 'elapsed time: ',abs(t2-t1),'s'
	filtered_indexes = np.array([index[0] if len(index) > 0 else np.nan for index in indexes])
	fire1d = [fires[int(ii)] if not np.isnan(ii) else 0 for ii in filtered_indexes]
	fires_interp = np.reshape(fire1d,lons_interp.shape)

	plot = False
	if plot:
		print '> Plotting the data...'
		#number of elements in arrays to plot
		nums = 100
		nums1 = 50
		fig1, (ax1,ax2) = plt.subplots(nrows = 2, ncols =1)
		ax1.pcolor(lons_interp[0::nums1],lats_interp[0::nums1],fires_interp[0::nums1])
		ax2.scatter(lons[0::nums],lats[0::nums],c=fires[0::nums],edgecolors='face')
		plt.show()

	print '> Saving the data...'
	t1 = time()
	result = {'data': fires_interp.astype(np.int8),'geotransform':geotransform}
	sio.savemat(file_name,mdict = result)
	t2 = time()
	print 'elapsed time: ',abs(t2-t1),'s'

	print '> File saved as ',file_name
	tg2 = time()
	print '>> Elapsed time for the granule: ',abs(tg2-tg1),'s'
	sys.exit()

t_final = time()
print '>> Total elapsed time: ',abs(t_final-t_init),'s <<'



