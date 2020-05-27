import numpy as np
from scipy.io import loadmat
from contline import get_contour_verts
from contour2kml import contour2kml
import os
import sys

matlab_file = 'svm.mat'
if os.path.isfile(matlab_file) and os.access(matlab_file,os.R_OK):
    print 'Loading the data...'
    svm=loadmat('svm.mat')
else:
    print 'Error: file %s not exist or not readable' % matlab_file
    sys.exit(1)

# Reading the variables in the file
xx=np.array(svm['fxlon'])
yy=np.array(svm['fxlat'])
tscale=svm['tscale'][0]
time_scale_num=svm['time_scale_num'][0]
zz=np.array(svm['Z'])*tscale+time_scale_num[0]
#zn=zz.ravel()
#Z=np.reshape(zn,zz.shape,'F')

print 'Computing the contours...'
# Granules numeric times
time_num_granules = svm['time_num_granules'][0]
data = get_contour_verts(xx, yy, zz, time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6)

print 'Creating the KML file...'
# Creating the KML file
contour2kml(data,'perimeters_svm.kml')

print 'perimeters_svm.kml generated'
