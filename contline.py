# following https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from utils import *
import datetime
import time
from JPSSD import time_num2iso
from contour2kml import contour2kml
from scipy.ndimage import gaussian_filter
import os
import sys

def get_contour_verts(xx, yy, zz, time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6, gauss_filter=True, plot_contours=False, col_repr=False, levels_gran=False):
    fig = plt.figure()
    # Computing the levels
    # Datetimes for the first and last granule
    dt1=datetime.datetime.fromtimestamp(time_num_granules[0])
    dt2=datetime.datetime.fromtimestamp(time_num_granules[-1])
    # Starting 6 hours before the first granule and finishing 6 hours after the last granule
    dti=dt1-datetime.timedelta(hours=contour_dt_init)
    dtf=dt2+datetime.timedelta(hours=contour_dt_final)
    # Number of periods of 6 hours we need to compute rounded
    d=dtf-dti
    hours=d.total_seconds()/3600
    M=int(np.round((hours+1)/contour_dt_hours ))
    # Datetimes where we are going to compute the levels
    dts=[dti+datetime.timedelta(hours=k*contour_dt_hours) for k in range(1,M)]
    if levels_gran:
        levels=time_num_granules
    else:
        levels=[time.mktime(t.timetuple()) for t in dts]

    # Scaling the time components as in detections
    time_num=np.array(levels)
    time_iso=[time_num2iso(t) for t in time_num]

    # Computing and generating the coordinates for the contours
    contours = []
    if gauss_filter:
        # for each level
        for level in levels:
            # copy the fire arrival time
            Z = np.array(zz)
            # distinguish either in or out the perimeter
            Z[Z < level] = 0
            Z[Z >= level] = 1
            # smooth the perimeter using a gaussian filter
            sigma = 2.
            ZZ = gaussian_filter(Z,sigma)
            # find the contour line in between
            cn = plt.contour(xx,yy,ZZ,levels=0.5)
            # contour line
            cc = cn.collections[0]
            # initialize the path
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                # read all the vertices
                paths.append(pp.vertices)
            contours.append(paths)
    else:
        # computing all the contours
        cn = plt.contour(xx,yy,zz,levels=levels)
        # for each contour line
        for cc in cn.collections:
            # initialize the path
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                # read all the vertices
                paths.append(pp.vertices)
            contours.append(paths)

    # Plotting or not the contour lines
    if plot_contours:
        print 'contours are collections of line, each line consisting of poins with x and y coordinates'
        for c in contours:
            for cc in c:
                xx=[x[0] for x in cc]
                yy=[y[1] for y in cc]
                plt.scatter(xx,yy)
        plt.show()

    if col_repr:
        import matplotlib.colors as colors
        col = np.flip(np.divide([(230, 25, 75, 150), (245, 130, 48, 150), (255, 255, 25, 150),
                                (210, 245, 60, 150), (60, 180, 75, 150), (70, 240, 240, 150),
                                (0, 0, 128, 150), (145, 30, 180, 150), (240, 50, 230, 150),
                                (128, 128, 128, 150)],255.),0)
        cm = colors.LinearSegmentedColormap.from_list('BuRd',col,len(contours))
        cols = ['%02x%02x%02x%02x' % tuple(255*np.flip(c)) for c in cm(range(cm.N))]

        # Creating an array of dictionaries for each perimeter
        conts=[Dict({'text':time_iso[k],
            'LineStyle':{
                'color': cols[k],
                'width':'2.5',
            },
            'PolyStyle':{
                'color':'66000086',
                'colorMode':'random'
            },
            'time_begin':time_iso[k],
            'polygons': contours[k] }) for k in range(len(contours))]
    else:
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
            'polygons': contours[k] }) for k in range(len(contours))]

    # Creating a dictionary to store the KML file information
    data=Dict({'name':'contours.kml',
        'folder_name':'Perimeters'})
    data.update({'contours': conts})

    return data

if __name__ == "__main__":
    result_file = 'result.mat'
    mgout_file = 'mgout.mat'
    if os.path.isfile(result_file) and os.access(result_file,os.R_OK) and os.path.isfile(mgout_file) and os.access(mgout_file,os.R_OK):
        print 'Loading the data...'
        result=loadmat(result_file)
        mgout=loadmat(mgout_file)
    else:
        print 'Error: file %s or %s not exist or not readable' % (result_file,mgout_file)
        sys.exit(1)

    # Indexing the coordinates into the same as the multigrid solution
    xind=mgout['sm'][0]-1
    yind=mgout['sn'][0]-1
    x=np.array(result['fxlon'])
    xx=x[np.ix_(xind,yind)]
    y=np.array(result['fxlat'])
    yy=y[np.ix_(xind,yind)]
    tscale=mgout['tscale'][0]
    time_scale_num=mgout['time_scale_num'][0]
    zz=mgout['a']*tscale+time_scale_num[0]

    print 'Computing the contours...'
    # Granules numeric times
    time_num_granules = result['time_num_granules'][0]
    data = get_contour_verts(xx, yy, zz, time_num_granules, contour_dt_hours=24, gauss_filter=False)

    print 'Creating the KML file...'
    # Creating the KML file
    contour2kml(data,'perimeters.kml')

    print 'perimeters.kml generated'

