import numpy as np
from scipy.io import loadmat
from contline import get_contour_verts
from contour2kml import contour2kml

print 'Loading the data...'
# Load all the data in svm.mat
svm=loadmat('svm.mat')
xx=np.array(svm['fxlon'])
yy=np.array(svm['fxlat'])
tscale=svm['tscale'][0]
time_scale_num=svm['time_scale_num'][0]
zz=svm['fmc_g']*tscale+time_scale_num[0]

print 'Computing the contours...'
# Granules numeric times
time_num_granules = svm['time_num_granules'][0]
data = get_contour_verts(xx, yy, zz, time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6)

print 'Creating the KML file...'
# Creating the KML file
contour2kml(data,'perimeters_svm.kml')

print 'perimeters_svm.kml generated'