# General python for any case
from JPSSD import *
from interpolation import sort_dates
from setup import process_satellite_detections
from svm import preprocess_data_svm, SVM3
from contline import get_contour_verts
from contour2kml import contour2kml
import saveload as sl
from scipy.io import loadmat
import datetime as dt
import sys
from time import time

if len(sys.argv) != 4:
	print 'Error: python %s wrfout start_time days' % sys.argv[0]
	print '	* wrfout - string, wrfout file of WRF-SFIRE simulation'
	print '	* start_time - string, YYYYMMDDHHMMSS where: '
	print '		YYYY - year'
	print '		MM - month'
	print '		DD - day'
	print '		HH - hour'
	print '		MM - minute'
	print '		SS - second'
	print '	* days - float, number of days of simulation (can be less than a day)'
	sys.exit(0)

t_init = time()

satellite_file = 'data'
fire_file = 'fire_detections.kml'
ground_file = 'nofire.kml'
bounds_file = 'result.mat'
svm_file = 'svm.mat'
contour_file = 'perimeters_svm.kml'

print ''
if os.path.isfile(bounds_file) and os.access(bounds_file,os.R_OK):
	#print '>> Reading the fire mesh <<'
	#sys.stdout.flush()
	#fxlon,fxlat,bbox,time_esmf=read_fire_mesh(sys.argv[1])

	#print ''
	print '>> File %s already created! Skipping all satellite processing <<' % bounds_file
	print 'Loading from %s...' % bounds_file
	result = loadmat(bounds_file)
	# Taking necessary variables from result dictionary
	scale = result['time_scale_num'][0]
	time_num_granules = result['time_num_granules'][0]
else:
	if os.path.isfile(satellite_file) and os.access(satellite_file,os.R_OK):
		print '>> File %s already created! Skipping satellite retrieval <<' % satellite_file
		print 'Loading from %s...' % satellite_file
		data,fxlon,fxlat,time_num=sl.load(satellite_file)
		bbox = [fxlon.min(),fxlon.max(),fxlat.min(),fxlat.max()]
	else:
		print '>> Reading the fire mesh <<'
		sys.stdout.flush()
		fxlon,fxlat,bbox,time_esmf=read_fire_mesh(sys.argv[1])
		# converting times to ISO
		dti=dt.datetime.strptime(sys.argv[2],'%Y%m%d%H%M%S')
		time_start_iso='%d-%02d-%02dT%02d:%02d:%02dZ' % (dti.year,dti.month,dti.day,dti.hour,dti.minute,dti.second)
		dtf=dti+dt.timedelta(days=float(sys.argv[3]))
		time_final_iso='%d-%02d-%02dT%02d:%02d:%02dZ' % (dtf.year,dtf.month,dtf.day,dtf.hour,dtf.minute,dtf.second)
		time_iso=(time_start_iso,time_final_iso)

		print ''
		print '>> Retrieving satellite data <<'
		sys.stdout.flush()
		data=retrieve_af_data(bbox,time_iso)

		print ''
		print '>> Saving satellite data file (data) <<'
		sys.stdout.flush()
		time_num=map(time_iso2num,time_iso)
		sl.save((data,fxlon,fxlat,time_num),satellite_file)
		print 'data file saved correctly!'

	print ''
	ffile = (os.path.isfile(fire_file) and os.access(fire_file,os.R_OK))
	gfile = (os.path.isfile(ground_file) and os.access(ground_file,os.R_OK))
	if (not ffile) or (not gfile):
		print '>> Generating KML of fire and ground detections <<'
		sys.stdout.flush()
		# sort the granules by dates
		sdata=sort_dates(data)
		tt=[ dd[1]['time_num'] for dd in sdata ]  # array of times
	if ffile:
		print '>> File %s already created! <<' % fire_file
	else:
		# writting fire detections file
		print 'writting KML with fire detections'
		keys=['latitude','longitude','brightness','scan','track','acq_date','acq_time','satellite','instrument','confidence','bright_t31','frp','scan_angle']
		dkeys=['lat_fire','lon_fire','brig_fire','scan_fire','track_fire','acq_date','acq_time','sat_fire','instrument','conf_fire','t31_fire','frp_fire','scan_angle_fire']
		prods={'AF':'Active Fires','FRP':'Fire Radiative Power'}
		N=[len(d[1]['lat_fire']) for d in sdata]
		json=sdata2json(sdata,keys,dkeys,N)
		json2kml(json,fire_file,bbox,prods)
	if gfile:
		print ''
		print '>> File %s already created! <<' % ground_file
	else:
		# writting ground detections file
		print 'writting KML with ground'
		keys=['latitude','longitude','scan','track','acq_date','acq_time','satellite','instrument','scan_angle']
		dkeys=['lat_nofire','lon_nofire','scan_nofire','track_nofire','acq_date','acq_time','sat_fire','instrument','scan_angle_nofire']
		prods={'NF':'No Fire'}
		N=[len(d[1]['lat_nofire']) for d in sdata]
		json=sdata2json(sdata,keys,dkeys,N)
		json2kml(json,ground_file,bbox,prods)

	print ''
	print '>> Processing satellite data <<'
	sys.stdout.flush()
	result = process_satellite_detections(data,fxlon,fxlat,time_num)
	# Taking necessary variables from result dictionary
	scale = result['time_scale_num']
	time_num_granules = result['time_num_granules']

lon = result['fxlon']
lat = result['fxlat']
U = np.array(result['U']).astype(float)
L = np.array(result['L']).astype(float)
T = np.array(result['T']).astype(float)

print ''
print '>> Preprocessing the data <<'
sys.stdout.flush()
X,y=preprocess_data_svm(lon,lat,U,L,T,scale,time_num_granules)

print ''
print '>> Running Support Vector Machine <<'
sys.stdout.flush()
C = 10.
kgam = 10.
F = SVM3(X,y,C=C,kgam=kgam,fire_grid=(lon,lat))

print ''
print '>> Saving the results <<'
sys.stdout.flush()
tscale = 24*3600
# Creating the dictionary with the results
svm = {'dxlon': lon, 'dxlat': lat, 'U': U/tscale, 'L': L/tscale,
        'fxlon': F[0], 'fxlat': F[1], 'fmc_g': F[2],
        'tscale': tscale, 'time_num_granules': time_num_granules,
        'time_scale_num': scale}
# Save resulting file
sio.savemat('svm.mat', mdict=svm)
print 'The results are saved in svm.mat file'

print ''
print '>> Computing contour lines of the fire arrival time <<'
print 'Computing the contours...'
# Scale fire arrival time
fmc_g = F[2]*tscale+scale[0]
# Granules numeric times
data = get_contour_verts(F[0], F[1], fmc_g, time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6)
print 'Creating the KML file...'
# Creating the KML file
contour2kml(data,'perimeters_svm.kml')
print 'The resulting contour lines are saved in perimeters_svm.kml file'

print ''
print '>> DONE <<'
t_final = time()
print 'Elapsed time for all the process: %ss.' % str(abs(t_final-t_init))
sys.exit()
