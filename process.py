# process.py
#
# DESCRIPTION
# Driver python code to estimate fire arrival time using Active Fire Satellite Data
#
# INPUTS
# In the existence of a 'data' satellite granules file and/or 'results.mat' bounds file, any input is necessary.
# Otherwise:
# 	wrfout - path to a simulation wrfout file (containing FXLON and FXLAT coordinates).
#	start_time - date string with format: YYYYMMDDHHMMSS
# 	days - length of the simulation in decimal days
#
# OVERFLOW
# 	1) Methods from JPSSD.py file:
# 		*) Find granules overlaping fire domain and time interval.
#		*) Download Active Satellite Data.
#		*) Read Active Satellite Data files.
#		*) Save satellite granule information in 'data' file.
#	2) Methods from interpolation.py and JPSSD.py files:
#		*) Write KML file 'fire_detections.kml' with fire detection pixels.
#		*) Write KML file 'nofire.kml' with saved ground detection pixels.
#	3) Method process_detections from setup.py file:
#		*) Sort all the granules from all the sources in time order.
#		*) Construct upper and lower bounds using a mask to prevent ground after fire.
#		*) Save results in 'results.mat' file.
#	4) Methods preprocess_data_svm and SVM3 from svm.py file:
#		*) Preprocess bounds as an input of Support Vector Machine method.
#		*) Run Support Vector Machine method.
#		*) Save results in svm.mat file.
#	5) Methods from contline.py and contour2kml.py files:
#		*) Construct a smooth contour line representation of the fire arrival time.
#		*) Write the contour lines in a KML file called 'perimeters_svm.kml'.
#
# OUTPUTS
#	- 'data': binary file containing satellite granules information.
#	- 'result.mat': matlab file containing upper and lower bounds (U and L) from satellite data.
# 	- 'svm.mat': matlab file containing the solution to the Support Vector Machine execution.
#				 Contains estimation of the fire arrival time in tign_g variable.
#	- 'fire_detections.kml': KML file with fire satellite detection pixels.
#	- 'nofire.kml': KML file with saved ground satellite detection pixels.
#	- 'perimeters_svm.kml': KML file with perimeters from estimation of the fire arrival time using SVM.
#
# Developed in Python 2.7.15 :: Anaconda, Inc.
# Angel Farguell (angel.farguell@gmail.com), 2019-04-29
#---------------------------------------------------------------------------------------------------------------------
from JPSSD import read_fire_mesh, retrieve_af_data, sdata2json, json2kml, time_iso2num
from interpolation import sort_dates
from setup import process_detections
from infrared_perimeters import process_ignitions, process_infrared_perimeters
from svm import preprocess_data_svm, SVM3
from contline import get_contour_verts
from contour2kml import contour2kml
import saveload as sl
from scipy.io import loadmat, savemat
import numpy as np
import datetime as dt
import sys
import os
from time import time

# if ignitions are known: ([lons],[lats],[dates]) where lons and lats in degrees and dates in ESMF format
# examples: igns = ([100],[45],['2015-05-15T20:09:00']) or igns = ([100,105],[45,39],['2015-05-15T20:09:00','2015-05-15T23:09:00'])
igns = None
# if infrared perimeters: path to KML files
# examples: perim_path = './pioneer_perim'
perim_path = ''

satellite_file = 'data'
fire_file = 'fire_detections.kml'
ground_file = 'nofire.kml'
bounds_file = 'result.mat'
svm_file = 'svm.mat'
contour_file = 'perimeters_svm.kml'

def exist(path):
	return (os.path.isfile(path) and os.access(path,os.R_OK))

satellite_exists = exist(satellite_file)
fire_exists = exist(fire_file)
ground_exists = exist(ground_file)
bounds_exists = exist(bounds_file)

if len(sys.argv) != 4 and (not bounds_exists) and (not satellite_exists):
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
	print 'or link an existent file %s or %s' % (satellite_file,bounds_file)
	sys.exit(0)

t_init = time()

print ''
if bounds_exists:
	print '>> File %s already created! Skipping all satellite processing <<' % bounds_file
	print 'Loading from %s...' % bounds_file
	result = loadmat(bounds_file)
	# Taking necessary variables from result dictionary
	scale = result['time_scale_num'][0]
	time_num_granules = result['time_num_granules'][0]
	time_num_interval = result['time_num'][0]
	lon = np.array(result['fxlon']).astype(float)
	lat = np.array(result['fxlat']).astype(float)
else:
	if satellite_exists:
		print '>> File %s already created! Skipping satellite retrieval <<' % satellite_file
		print 'Loading from %s...' % satellite_file
		data,fxlon,fxlat,time_num = sl.load(satellite_file)
		bbox = [fxlon.min(),fxlon.max(),fxlat.min(),fxlat.max()]
	else:
		print '>> Reading the fire mesh <<'
		sys.stdout.flush()
		fxlon,fxlat,bbox,time_esmf = read_fire_mesh(sys.argv[1])
		# converting times to ISO
		dti = dt.datetime.strptime(sys.argv[2],'%Y%m%d%H%M%S')
		time_start_iso = '%d-%02d-%02dT%02d:%02d:%02dZ' % (dti.year,dti.month,dti.day,dti.hour,dti.minute,dti.second)
		dtf = dti+dt.timedelta(days=float(sys.argv[3]))
		time_final_iso = '%d-%02d-%02dT%02d:%02d:%02dZ' % (dtf.year,dtf.month,dtf.day,dtf.hour,dtf.minute,dtf.second)
		time_iso = (time_start_iso,time_final_iso)

		print ''
		print '>> Retrieving satellite data <<'
		sys.stdout.flush()
		data = retrieve_af_data(bbox,time_iso)
		if igns:
			data.update(process_ignitions(igns,bounds=bbox))
		if perim_path:
			data.update(process_infrared_perimeters(perim_path,bounds=bbox))

		if data:
			print ''
			print '>> Saving satellite data file (data) <<'
			sys.stdout.flush()
			time_num = map(time_iso2num,time_iso)
			sl.save((data,fxlon,fxlat,time_num),satellite_file)
			print 'data file saved correctly!'
		else:
			print ''
			print 'ERROR: No data obtained...'
			sys.exit(1)

	print ''
	if (not fire_exists) or (not ground_exists):
		print '>> Generating KML of fire and ground detections <<'
		sys.stdout.flush()
		# sort the granules by dates
		sdata=sort_dates(data)
	if fire_exists:
		print '>> File %s already created! <<' % fire_file
	else:
		# writting fire detections file
		print 'writting KML with fire detections'
		keys = ['latitude','longitude','brightness','scan','track','acq_date','acq_time','satellite','instrument','confidence','bright_t31','frp','scan_angle']
		dkeys = ['lat_fire','lon_fire','brig_fire','scan_fire','track_fire','acq_date','acq_time','sat_fire','instrument','conf_fire','t31_fire','frp_fire','scan_angle_fire']
		prods = {'AF':'Active Fires','FRP':'Fire Radiative Power'}
		N = [len(d[1]['lat_fire']) if 'lat_fire' in d[1] else 0 for d in sdata]
		json = sdata2json(sdata,keys,dkeys,N)
		json2kml(json,fire_file,bbox,prods)
	if ground_exists:
		print ''
		print '>> File %s already created! <<' % ground_file
	else:
		# writting ground detections file
		print 'writting KML with ground'
		keys = ['latitude','longitude','scan','track','acq_date','acq_time','satellite','instrument','scan_angle']
		dkeys = ['lat_nofire','lon_nofire','scan_nofire','track_nofire','acq_date','acq_time','sat_fire','instrument','scan_angle_nofire']
		prods = {'NF':'No Fire'}
		N = [len(d[1]['lat_nofire']) if 'lat_nofire' in d[1] else 0 for d in sdata]
		json = sdata2json(sdata,keys,dkeys,N)
		json2kml(json,ground_file,bbox,prods)

	print ''
	print '>> Processing satellite data <<'
	sys.stdout.flush()
	result = process_detections(data,fxlon,fxlat,time_num)
	# Taking necessary variables from result dictionary
	scale = result['time_scale_num']
	time_num_granules = result['time_num_granules']
	time_num_interval = result['time_num']
	lon = np.array(result['fxlon']).astype(float)
	lat = np.array(result['fxlat']).astype(float)

U = np.array(result['U']).astype(float)
L = np.array(result['L']).astype(float)
T = np.array(result['T']).astype(float)

if 'C' in result.keys():
	conf = np.array(result['C'])
else:
	conf = None

print ''
print '>> Preprocessing the data <<'
sys.stdout.flush()
X,y,c = preprocess_data_svm(lon,lat,U,L,T,scale,time_num_granules,C=conf)

print ''
print '>> Running Support Vector Machine <<'
sys.stdout.flush()
C = 10.
kgam = 10.
F = SVM3(X,y,C=C,kgam=kgam,fire_grid=(lon,lat))

print ''
print '>> Saving the results <<'
sys.stdout.flush()
tscale = 24*3600 # scale from seconds to days
# Fire arrival time in seconds from the begining of the simulation
tign_g = F[2]*float(tscale)+scale[0]-time_num_interval[0]
# Creating the dictionary with the results
svm = {'dxlon': lon, 'dxlat': lat, 'U': U/tscale, 'L': L/tscale,
        'fxlon': F[0], 'fxlat': F[1], 'Z': F[2],
        'tign_g': tign_g,
        'tscale': tscale, 'time_num_granules': time_num_granules,
        'time_scale_num': scale, 'time_num': time_num_interval}
# Save resulting file
savemat(svm_file, mdict=svm)
print 'The results are saved in svm.mat file'

print ''
print '>> Computing contour lines of the fire arrival time <<'
print 'Computing the contours...'
# Fire arrival time in seconds from an old date
Z = F[2]*tscale+scale[0]
# Granules numeric times
contour_data = get_contour_verts(F[0], F[1], Z, time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6)
print 'Creating the KML file...'
# Creating the KML file
contour2kml(contour_data,contour_file)
print 'The resulting contour lines are saved in perimeters_svm.kml file'

print ''
print '>> DONE <<'
t_final = time()
print 'Elapsed time for all the process: %ss.' % str(abs(t_final-t_init))
sys.exit()
