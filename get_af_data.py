from JPSSD import retrieve_af_data
import sys
if len(sys.argv) != 7:
    print 'usage: ./get_data.sh lonmin lonmax latmin latmax timemin timemax'
    print 'example:'
    print './get_data.sh -132.86966 -102.0868788 44.002495 66.281204 2012-09-11T00:00:00Z 2012-09-12T00:00:00Z'
    sys.exit(1)
bbox=[]
for s in sys.argv[1:5]:
    print s
    bbox.append(float(s))
time = (sys.argv[5],sys.argv[6])
print 'bounding box is %s' % bbox 
print 'time interval is %s %s' % time 
retrieve_af_data(bbox,time)
print 'Retrieved data written to Matlab file out.mat'
