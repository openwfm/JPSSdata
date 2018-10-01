# sample data into mesh - Las Conchas
# navigate to /glade/u/home/angelfc/project/lasconchas/simulation_large to access sample data
from JPSSD import retrieve_af_data, read_fire_mesh, time_iso2num, write_csv
import saveload as sl

fxlon,fxlat,bbox,time_esmf=read_fire_mesh('wrfout_d04_2011-06-26_12:00:00')

# cannot get starting time from wrfout
time_iso = ("2011-06-25T00:00:00Z", "2011-07-04T00:00:00Z") # tuple, not array

data=retrieve_af_data(bbox,time_iso)

print 'writting CSV detections'
write_csv(data,bbox)

print 'saving data'

sl.save((data,fxlon,fxlat,map(time_iso2num,time_iso)),'data')

print 'run setup next'
