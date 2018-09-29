# sample data into mesh - Sugarloaf
# navigate to /share_home/jmandel/sugarloaf to access sample data
from JPSSD import retrieve_af_data, read_fire_mesh, time_iso2num, write_csv
import saveload as sl

fxlon,fxlat,bbox,time_esmf=read_fire_mesh('wrfout_d03_2018-09-03_15:00:00')

# cannot get starting time from wrfout
time_iso = ("2018-08-15T00:00:00Z", "2018-09-02T00:00:00Z") # tuple, not array

data=retrieve_af_data(bbox,time_iso)

print 'writting CSV detections'
write_csv(data)

print 'saving data'

sl.save((data,fxlon,fxlat,map(time_iso2num,time_iso)),'data')

print 'run setup next'
