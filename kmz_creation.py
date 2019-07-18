import sys, os
import datetime as dt
import saveload as sl
from utils import Dict
from JPSSD import retrieve_af_data
from interpolation import sort_dates
from mpl_toolkits.basemap import Basemap
from plot_pixels import basemap_scatter_mercator, create_kml

def exist(path):
	return (os.path.isfile(path) and os.access(path,os.R_OK))

print ''
print '>> Reading the arguments <<'
dti = dt.datetime.strptime(sys.argv[1],'%Y%m%d%H%M%S')
time_start_iso = '%d-%02d-%02dT%02d:%02d:%02dZ' % (dti.year,dti.month,dti.day,dti.hour,dti.minute,dti.second)
dtf = dti+dt.timedelta(days=float(sys.argv[2]))
time_final_iso = '%d-%02d-%02dT%02d:%02d:%02dZ' % (dtf.year,dtf.month,dtf.day,dtf.hour,dtf.minute,dtf.second)
time_iso = (time_start_iso,time_final_iso)

bbox = (float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]))

print ''
print '>> Retrieving satellite data <<'
sys.stdout.flush()
data = retrieve_af_data(bbox,time_iso)
sl.save((data,time_iso,bbox),'sat_data')
# sort the granules by dates
sdata=sort_dates(data)

print ''
print '>> Creating KMZ file <<'
# creating KMZ overlay of each information
# create the Basemap to plot into
bmap = Basemap(projection='merc',llcrnrlat=bbox[2], urcrnrlat=bbox[3], llcrnrlon=bbox[0], urcrnrlon=bbox[1])
# initialize array
kmld = []
# for each observed information
for idx, g in enumerate(sdata):
    # create png name
    pngfile = g[0]+'.png'
    # create timestamp for KML
    timestamp = g[1].acq_date + 'T' + g[1].acq_time[0:2] + ':' + g[1].acq_time[2:4] + 'Z'
    if not exist(pngfile):
        # plot a scatter basemap
        raster_png_data,corner_coords = basemap_scatter_mercator(g[1],bbox,bmap)
        # compute bounds
        bounds = (corner_coords[0][0],corner_coords[1][0],corner_coords[0][1],corner_coords[2][1])
        # write PNG file
        with open(pngfile, 'w') as f:
            f.write(raster_png_data)
        print '> File %s saved.' % g[0]
    else:
        print '> File %s already created.' % g[0]
    # append dictionary information for the KML creation
    kmld.append(Dict({'name': g[0], 'png_file': pngfile, 'bounds': bbox, 'time': timestamp}))
# create KML
create_kml(kmld,'./doc.kml')
# create KMZ with all the PNGs included
os.system('zip -r %s doc.kml *_A*_*.png' % 'googlearth.kmz')
print 'Created file googlearth.kmz'

print ''
print '>> DONE <<'
sys.exit()
