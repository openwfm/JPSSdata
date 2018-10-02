# General python for any case
from JPSSD import retrieve_af_data, read_fire_mesh, time_iso2num, write_csv
from interpolation import sort_dates
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
time_start_iso='%d-%02d-%02dT%02d:%02d:%02d-06:00' % (dti.year,dti.month,dti.day,dti.hour,dti.minute,dti.second)
dtf=dti+dt.timedelta(days=int(sys.argv[3]))
time_final_iso='%d-%02d-%02dT%02d:%02d:%02d-06:00' % (dtf.year,dtf.month,dtf.day,dtf.hour,dtf.minute,dtf.second)

# cannot get starting time from wrfout
time_iso=(time_start_iso,time_final_iso) # tuple, not array

data=retrieve_af_data(bbox,time_iso)

print 'writting CSV detections'
write_csv(data,bbox)

print 'saving data'

sl.save((data,fxlon,fxlat,map(time_iso2num,time_iso)),'data')

print 'run setup next'
