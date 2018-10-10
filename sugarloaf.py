# sample data into mesh - Sugarloaf
# navigate to /share_home/jmandel/sugarloaf to access sample data
from JPSSD import retrieve_af_data, read_fire_mesh, time_iso2num, write_csv
import saveload as sl

fxlon,fxlat,bbox,time_esmf=read_fire_mesh('wrfout_d03_2018-09-03_15:00:00')

d = nc.Dataset('wrfout_d03_2018-09-03_15:00:00')
fxlon = d.variables['FXLONG'][0,:,:] # boundary masking conditions previously calculated(0:409)
fxlat = d.variables['FXLAT'][0,:,:]
data = d.variables['TIGN_G'][10,:,:]

bbox = [(np.amin(fxlon),np.amin(fxlat)),(np.amin(fxlon),np.amax(fxlat)),
	(np.amax(fxlon),np.amin(fxlat)),(np.amax(fxlon),np.amax(fxlat))]
print bbox

d.close()

data=retrieve_af_data(bbox,time_iso)

print 'writting CSV detections'
write_csv(data,bbox)

print 'saving data'

sl.save((data,fxlon,fxlat,map(time_iso2num,time_iso)),'data')

print 'run setup next'
