import warnings
warnings.filterwarnings("ignore")
# following https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import *
import datetime
import time
from JPSSD import time_num2iso
from contour2kml import contour2kml
import sys

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            # read all the vertices
            paths.append(pp.vertices)
        contours.append(paths)

    return contours

if __name__ == "__main__":
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
    time_num_granules=svm['time_num_granules'][0]
    #sys.exit()
    # Datetimes for the first and last granule
    dt1=datetime.datetime.fromtimestamp(time_num_granules[0])
    dt2=datetime.datetime.fromtimestamp(time_num_granules[-1])
    # Starting 6 hours before the first granule and finishing 6 hours after the last granule
    dti=dt1-datetime.timedelta(hours=6)
    dtf=dt2+datetime.timedelta(hours=6)
    # Number of periods of 6 hours we need to compute rounded
    d=dtf-dti
    hours=d.total_seconds()/3600
    contour_dt_hours = 6
    M=int(np.round((hours+1)/contour_dt_hours ))
    # Datetimes where we are going to compute the levels
    dts=[dti+datetime.timedelta(hours=k*contour_dt_hours) for k in range(1,M)]
    levels=[time.mktime(t.timetuple()) for t in dts]
    # Computing the contours
    cn=plt.contour(xx,yy,zz,levels=levels)
    # Scaling the time components as in detections
    time_num=np.array(cn.levels)
    time_iso=[time_num2iso(t) for t in time_num]
    # Generating the coordinates for the contours
    contours=get_contour_verts(cn)

    # Plotting or not the contour lines
    plot=True
    if plot:
        print 'contours are collections of line, each line consisting of poins with x and y coordinates'
        for c in contours:
            for cc in c:
                xx=[x[0] for x in cc]
                yy=[y[1] for y in cc]
                plt.scatter(xx,yy)
        plt.show()

    print 'Creating the dictionary...'
    # Creating an array of dictionaries for each perimeter
    conts=[Dict({'text':time_iso[k],
        'LineStyle':{
            'color':'ff081388',
            'width':'2.5',
        },
        'PolyStyle':{
            'color':'66000086',
            'colorMode':'random'
        },
        'time_begin':time_iso[k],
        'polygons': contours[k] }) for k in range(0,len(contours))]

    # Creating a dictionary to store the KML file information
    data=Dict({'name':'contours.kml',
        'folder_name':'Perimeters'})
    data.update({'contours': conts})

    print 'Creating the KML file...'
    # Creating the KML file
    contour2kml(data,'perimeters.kml')

    print 'perimeters.kml generated'

