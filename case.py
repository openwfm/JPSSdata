# General python for any case
from JPSSD import *
from interpolation import sort_dates
from setup import process_satellite_detections
from svm import preprocess_data_svm, SVM3
from contline import get_contour_verts
from contour2kml import contour2kml
import saveload as sl
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

print ''
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
print '>> Generating KML of fire and ground detections<<'
sys.stdout.flush()
csv=False # CSV file of fire detections
opt='granules' # KML folders sorted by (pixels, granules or dates)
# sort the granules by dates
sdata=sort_dates(data)
tt=[ dd[1]['time_num'] for dd in sdata ]  # array of times
print 'writting KML with fire detections'
keys=['latitude','longitude','brightness','scan','track','acq_date','acq_time','satellite','instrument','confidence','bright_t31','frp','scan_angle']
dkeys=['lat_fire','lon_fire','brig_fire','scan_fire','track_fire','acq_date','acq_time','sat_fire','instrument','conf_fire','t31_fire','frp_fire','scan_angle_fire']
prods={'AF':'Active Fires','FRP':'Fire Radiative Power'}
if csv or opt != 'granules':
	N=[len(data[d]['lat_fire']) for d in data]
	json=data2json(data,keys,dkeys,N)
	if csv:
		write_csv(json,bbox)
	json2kml(json,'nofire.kml',bbox,prods,opt=opt)
if opt == 'granules':
	N=[len(d[1]['lat_fire']) for d in sdata]
	json=sdata2json(sdata,keys,dkeys,N)
	json2kml(json,'fire_detections.kml',bbox,prods)
print 'writting KML with ground'
keys=['latitude','longitude','scan','track','acq_date','acq_time','satellite','instrument','scan_angle']
dkeys=['lat_nofire','lon_nofire','scan_nofire','track_nofire','acq_date','acq_time','sat_fire','instrument','scan_angle_nofire']
prods={'NF':'No Fire'}
if opt != 'granules':
	N=[len(data[d]['lat_nofire']) for d in data]
	json=data2json(data,keys,dkeys,N)
	json2kml(json,'nofire.kml',bbox,prods,opt=opt)
else:
	N=[len(d[1]['lat_nofire']) for d in sdata]
	json=sdata2json(sdata,keys,dkeys,N)
	json2kml(json,'nofire.kml',bbox,prods)

print ''
print '>> Saving satellite data file (data) <<'
sys.stdout.flush()
time_num=map(time_iso2num,time_iso)
sl.save((data,fxlon,fxlat,time_num),'data')
print 'data file saved correctly!'

print ''
print '>> Processing satellite data <<'
sys.stdout.flush()
result=process_satellite_detections(data,fxlon,fxlat,time_num)

print ''
print '>> Preprocessing the data <<'
# Taking necessary variables from result dictionary
scale = result['time_scale_num']
time_num_granules = result['time_num_granules']
lon = result['fxlon']
lat = result['fxlat']
U = np.array(result['U']).astype(float)
L = np.array(result['L']).astype(float)
T = np.array(result['T']).astype(float)
sys.stdout.flush()
X,y=preprocess_data_svm(lon,lat,U,L,T,scale,time_num_granules)

print ''
print '>> Running Support Vector Machine <<'
sys.stdout.flush()
C = 10.
kgam = 10.
F = SVM3(X,y,C=C,kgam=kgam,fire_grid=(fxlon,fxlat))

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
data = get_contour_verts(F[0], F[1], fmc_g, time_num_granules, contour_dt_hours=.05, contour_dt_init=.05, contour_dt_final=.05)
print 'Creating the KML file...'
# Creating the KML file
contour2kml(data,'perimeters_svm.kml')
print 'The resulting contour lines are saved in perimeters_svm.kml file'

print ''
print '>> DONE <<'
t_final = time()
print 'Elapsed time for all the process: %ss.' % str(abs(t_final-t_init))
sys.exit()
