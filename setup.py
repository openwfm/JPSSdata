import pdb
import saveload as sl
from interpolation import sort_dates,nearest_scipy,distance_upper_bound
import time

global t_init

print 'loading data'
data,fxlon,fxlat=sl.load('data')
dx1=fxlon[0,1]-fxlon[0,0]
dx2=fxlat[1,0]-fxlat[0,0]

# Sort dictionary by time_num into an array of tuples (key, dictionary of values) 
sdata=sort_dates(data)
tt=[ dd[1]['time_num'] for dd in sdata ]  # array of times
print 'Sorted?'
stt=sorted(tt)
print tt==stt

# Grid interpolation
slon=sdata[10][1]['lon'] # example of granule
slat=sdata[10][1]['lat']
dy1=slon[0,1]-slon[0,0]
dy2=slat[1,0]-slat[0,0]
dub=distance_upper_bound([dx1,dx2],[dy1,dy2])
print 'distance upper bound'
print dub
t_init = time.time()
(rlon,rlat)=nearest_scipy(slon,slat,fxlon,fxlat,dub)
t_final = time.time()
print 'Elapsed time: %ss.' % str(t_final-t_init)
print rlon
print rlat
