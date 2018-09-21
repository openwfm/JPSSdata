import pdb
# from sugarloaf import data

# Sort dictionary by time_start_geo in an ordered array of dictionaries
sdata=sort_dates(data)
tt=[ dd[1]['time_num'] for dd in sdata ]
print 'Sorted?'
stt=sorted(tt)
print tt==stt

# Grid interpolation
slon=sdata[10][1]['lon'] # example of granule
slat=sdata[10][1]['lat']
(rlon,rlat)=nearest_scipy(slon,slat,fxlon,fxlat)
print rlon
print rlat

