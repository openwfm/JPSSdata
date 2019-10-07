from wrf2geotiff import readncproj, readllproj, create_geotiff
from contline import get_contour_verts
from contour2shp import contour2shp
from scipy.io import loadmat
import numpy as np

time = ['2019-08-01_00:00:00']

svm = loadmat('svm.mat')

if 'fxlon_interp' in svm:
    fxlon = svm['fxlon_interp']
    fxlat = svm['fxlat_interp']
    tign_g = svm['tign_g_interp']
else:
    fxlon = svm['fxlon']
    fxlat = svm['fxlat']
    tign_g = svm['tign_g']

tign_g[tign_g==tign_g.max()] = np.nan

print('Creating tign_g GeoTIFF file...')
sname = 'TIGN_G'
data = [tign_g]
ndv = -9999.0
csr,geotransform = readllproj(fxlon,fxlat)
create_geotiff(sname, data, ndv, geotransform, csr, time)

print('Computing the contours...')
time_num_interval = svm['time_num'][0]
time_num_granules = svm['time_num_granules'][0]
data = get_contour_verts(fxlon, fxlat, tign_g+time_num_interval[0], time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6)

print('Creating tign_g shape files...')
contour2shp(data, '.', geotransform, csr)
