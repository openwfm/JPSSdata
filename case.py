# General python for any case
from JPSSD import retrieve_af_data, read_fire_mesh, time_iso2num, data2json, write_csv, json2kml
import saveload as sl
import datetime as dt
import sys

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
	print '	* days - integer, number of days of simulation'
	sys.exit(0)

fxlon,fxlat,bbox,time_esmf=read_fire_mesh(sys.argv[1])

dti=dt.datetime.strptime(sys.argv[2],'%Y%m%d%H%M%S')
time_start_iso='%d-%02d-%02dT%02d:%02d:%02dZ' % (dti.year,dti.month,dti.day,dti.hour,dti.minute,dti.second)
dtf=dti+dt.timedelta(days=int(sys.argv[3]))
time_final_iso='%d-%02d-%02dT%02d:%02d:%02dZ' % (dtf.year,dtf.month,dtf.day,dtf.hour,dtf.minute,dtf.second)

# cannot get starting time from wrfout
time_iso=(time_start_iso,time_final_iso) # tuple, not array

data=retrieve_af_data(bbox,time_iso)

print 'writting fire detections in CSV and KML'

keys=['latitude','longitude','brightness','scan','track','acq_date','acq_time','satellite','instrument','confidence','bright_t31','frp','scan_angle']
dkeys=['lat_fire','lon_fire','brig_fire','scan_fire','track_fire','acq_date','acq_time','sat_fire','instrument','conf_fire','t31_fire','frp_fire','scan_angle_fire']
N=[len(data[d]['lat_fire']) for d in data]
json=data2json(data,keys,dkeys,N)
write_csv(json,bbox)
prods={'AF':'Active Fires','FRP':'Fire Radiative Power'}
json2kml(json,'fire_detections.kml',bbox,prods)

print 'writting KML with ground'

keys=['latitude','longitude','scan','track','acq_date','acq_time','satellite','instrument','scan_angle']
dkeys=['lat_nofire','lon_nofire','scan_nofire','track_nofire','acq_date','acq_time','sat_fire','instrument','scan_angle_nofire']
N=[len(data[d]['lat_nofire']) for d in data]
json=data2json(data,keys,dkeys,N)
prods={'NF':'No Fire'}
json2kml(json,'nofire.kml',bbox,prods)

print 'saving data'

sl.save((data,fxlon,fxlat,map(time_iso2num,time_iso)),'data')

print 'run setup next'
