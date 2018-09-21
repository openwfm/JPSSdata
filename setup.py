import pdb
import saveload as sl
from interpolation import sort_dates,nearest_scipy

data,fxlon,fxlat=sl.load('data')

# Sort dictionary by time_num into an array of dictionaries
sdata=sort_dates(data)
tt=[ dd[1]['time_num'] for dd in sdata ]  # array of times
print 'Sorted?'
stt=sorted(tt)
print tt==stt

# Grid interpolation
slon=sdata[10][1]['lon'] # example of granule
slat=sdata[10][1]['lat']
(rlon,rlat)=nearest_scipy(slon,slat,fxlon,fxlat)
print rlon
print rlat

