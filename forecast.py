import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as colors
import numpy as np
from datetime import timedelta
from JPSSD import time_iso2num, time_iso2datetime, time_datetime2iso
from utils import Dict
import re, glob, sys, os

def process_tign_g(lon,lat,tign_g,ctime,scale,time_num,epsilon=5,plot=False):
    """
    Process forecast from lon, lat, and tign_g as 3D data points

    :param lon: array of longitudes
    :param lat: array of latitudes
    :param tign_g: array of fire arrival time
    :param ctime: time char in wrfout of the last fire arrival time
    :param scale: numerical time scale in seconds of start and end of simulation
    :param time_num: numerical time interval in seconds of start and end of simulation
    :param epsilon: optional, epsilon in seconds to define the separation of artificial data
    :param plot: optional, plotting results

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) and James Haley, 2019-06-05
    """

    # confidences
    conf_ground = 25.
    conf_fire = 25.
    
    # ctime transformations
    ctime_iso = ctime.replace('_','T')
    ctime_datetime = time_iso2datetime(ctime_iso)
    tscale = 3600.*24.

    # tign_g as data points smaller than maximum tign_g
    t1d = np.ravel(tign_g)
    etime = t1d.max()
    mask = t1d < etime
    x1d = np.ravel(lon)[mask]
    y1d = np.ravel(lat)[mask]
    t1d = t1d[mask]

    # compute time as days from start of the simulation
    tt = (np.array([time_iso2num(time_datetime2iso(
                        ctime_datetime-timedelta(seconds=float(etime-T)))) 
                            for T in t1d])-scale[0])/tscale

    # define forecast artificial data points
    fground = np.c_[x1d.ravel(),y1d.ravel(),tt.ravel() - epsilon/tscale]
    ffire = np.c_[x1d.ravel(),y1d.ravel(),tt.ravel() + epsilon/tscale]
    
    # coarsening
    maxsize = 5e3
    coarsening = np.int(1+len(ffire)/maxsize)
    fground = fground[::coarsening,:]
    ffire = ffire[::coarsening,:]
    
    # define output data
    X = np.concatenate((fground,ffire))
    y = np.concatenate((-np.ones(len(fground)),np.ones(len(ffire))))
    c = np.concatenate((conf_ground*np.ones(len(fground)),conf_fire*np.ones(len(ffire))))

    # plot results
    if plot:
        from contline import get_contour_verts
        from contour2kml import contour2kml
        tt = np.array([time_iso2num(time_datetime2iso(
                        ctime_datetime-timedelta(seconds=float(etime-T)))) 
                            for T in np.ravel(tign_g)])
        data = get_contour_verts(lon, lat, np.reshape(tt,tign_g.shape), time_num, contour_dt_hours=6, contour_dt_init=24, contour_dt_final=12)
        contour2kml(data,'perimeters_forecast.kml')
        col = [(0, .5, 0), (.5, 0, 0)]
        cm_GR = colors.LinearSegmentedColormap.from_list('GrRd',col,N=2)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                    c=y, cmap=cm_GR, s=1, alpha=.5, 
                    vmin=y.min(), vmax=y.max())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Fire arrival time [days]")
        plt.show()

    return X,y,c


def process_tign_g_slices(lon,lat,tign_g,bounds,ctime,dx,dy,wrfout_file='',dt_for=3600.,plot=False):
    """
    Process forecast as slices on time from lon, lat, and tign_g

    :param lon: array of longitudes
    :param lat: array of latitudes
    :param tign_g: array of fire arrival time
    :param bounds: coordinate bounding box filtering to
    :param ctime: time char in wrfout of the last fire arrival time
    :param dx,dy: data resolution in meters
    :param dt_for: optional, temporal resolution to get the fire arrival time from in seconds
    :param wrfout_file: optional, to get the name of the file in the dictionary

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) and James Haley, 2019-06-05
    """

    if plot:
        fig = plt.figure()

    # prefix of the elements in the dictionary
    prefix = "FOR"
    # confidences
    conf_fire = 50
    conf_nofire = 10
    # margin percentage
    margin = .5
    # scan and track dimensions of the observation (in km)
    scan = dx/1000.
    track = dy/1000.
    # initializing dictionary
    forecast = Dict({})

    # ctime transformations
    ctime_iso = ctime.replace('_','T')
    ctime_datetime = time_iso2datetime(ctime_iso)
    # mask coordinates to bounding box
    mask = np.logical_and(np.logical_and(np.logical_and(lon >= bounds[0], lon <= bounds[1]), lat >= bounds[2]), lat <= bounds[3])
    # create a square subset of fire arrival time less than the maximum
    mtign = np.logical_and(mask, tign_g < tign_g.max())
    mlon = lon[mtign]
    mlat = lat[mtign]
    mlenlon = margin*(mlon.max()-mlon.min())
    mlenlat = margin*(mlat.max()-mlat.min())
    sbounds = (mlon.min()-mlenlon, mlon.max()+mlenlon, mlat.min()-mlenlat, mlat.max()+mlenlat)
    smask = np.logical_and(np.logical_and(np.logical_and(lon >= sbounds[0], lon <= sbounds[1]), lat >= sbounds[2]), lat <= sbounds[3])

    # times to get fire arrival time from
    TT = np.arange(tign_g.min()-3*dt_for,tign_g.max(),dt_for)[0:-1]
    for T in TT:
        # fire arrival time to datetime
        time_datetime = ctime_datetime-timedelta(seconds=float(tign_g.max()-T)) # there is an error of about 10 seconds (wrfout not ending at exact time of simulation)
        # create ISO time of the date
        time_iso = time_datetime2iso(time_datetime)
        # create numerical time from the ISO time
        time_num = time_iso2num(time_iso)
        # create time stamp
        time_data = '_A%04d%03d_%02d%02d_%02d' % (time_datetime.year, time_datetime.timetuple().tm_yday,
                                                time_datetime.hour, time_datetime.minute, time_datetime.second)
        # create acquisition date
        acq_date = '%04d-%02d-%02d' % (time_datetime.year, time_datetime.month, time_datetime.day)
        # create acquisition time
        acq_time = '%02d%02d' % (time_datetime.hour, time_datetime.minute)

        print 'Retrieving forecast from %s' % time_iso

        # masks
        fire = tign_g <= T
        nofire = np.logical_and(tign_g > T, np.logical_or(tign_g != tign_g.max(), smask))
        # coordinates masked
        lon_fire = lon[np.logical_and(mask,fire)]
        lat_fire = lat[np.logical_and(mask,fire)]
        lon_nofire = lon[np.logical_and(mask,nofire)]
        lat_nofire = lat[np.logical_and(mask,nofire)]
        # create general arrays
        lons = np.concatenate((lon_nofire,lon_fire))
        lats = np.concatenate((lat_nofire,lat_fire))
        fires = np.concatenate((5*np.ones(lon_nofire.shape),9*np.ones(lon_fire.shape)))

        # plot results
        if plot:
            plt.ion()
            plt.cla()
            plt.plot(lons[fires==5],lats[fires==5],'g.')
            plt.plot(lons[fires==9],lats[fires==9],'r.')
            plt.pause(.0001)
            plt.show()

        # update perimeters dictionary
        forecast.update({prefix + time_data: Dict({'file': wrfout_file, 'time_tign_g': T, 'lon': lons, 'lat': lats,
                            'fire': fires, 'conf_fire': np.array(conf_fire*np.ones(lons[fires==9].shape)),
                            'lon_fire': lons[fires==9], 'lat_fire': lats[fires==9], 'lon_nofire': lons[fires==5], 'lat_nofire': lats[fires==5],
                            'scan_fire': scan*np.ones(lons[fires==9].shape), 'track_fire': track*np.ones(lons[fires==9].shape),
                            'conf_nofire': np.array(conf_nofire*np.ones(lons[fires==5].shape)),
                            'scan_nofire': scan*np.ones(lons[fires==5].shape), 'track_nofire': track*np.ones(lons[fires==9].shape),
                            'time_iso': time_iso, 'time_num': time_num, 'acq_date': acq_date, 'acq_time': acq_time})})

    return forecast

def read_forecast_wrfout(wrfout_file):
    """
    Read forecast from a wrfout.

    :param wrfout_file: path to wrfout file

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-06-05
    """

    # read file
    import netCDF4 as nc
    try:
        data = nc.Dataset(wrfout_file)
    except Exception as e:
        print 'Warning: No netcdf file %s in the path' % wrfout_file
        return []
    # current time
    ctime = ''.join(data['Times'][0])
    # getting rid of strip
    atmlenx = len(data.dimensions['west_east'])
    atmleny = len(data.dimensions['south_north'])
    staglenx = len(data.dimensions['west_east_stag'])
    stagleny = len(data.dimensions['south_north_stag'])
    dimlenx = len(data.dimensions['west_east_subgrid'])
    dimleny = len(data.dimensions['south_north_subgrid'])
    ratiox = dimlenx/max(staglenx,atmlenx+1)
    ratioy = dimleny/max(stagleny,atmleny+1)
    lenx = dimlenx-ratiox
    leny = dimleny-ratioy
    # coordinates
    lon = data['FXLONG'][0][0:lenx,0:leny]
    lat = data['FXLAT'][0][0:lenx,0:leny]
    # fire arrival time
    tign_g = data['TIGN_G'][0][0:lenx,0:leny]
    # resolutions
    DX = data.getncattr('DX')
    DY = data.getncattr('DY')
    sx = data.dimensions['west_east_subgrid'].size/data.dimensions['west_east_stag'].size
    sy = data.dimensions['south_north_subgrid'].size/data.dimensions['south_north_stag'].size
    dx = DX/sx
    dy = DY/sy
    # close netcdf file
    data.close()
    return lon,lat,tign_g,ctime,dx,dy

def process_forecast_slides_wrfout(wrfout_file,bounds,plot=False):
    """
    Process forecast from a wrfout.

    :param wrfout_file: path to wrfout file
    :param bounds: coordinate bounding box filtering to

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-06-05
    """

    # read forecast from wrfout
    lon,lat,tign_g,ctime,dx,dy = read_forecast_wrfout(wrfout_file)
    # create forecast
    forecast = process_tign_g_slices(lon,lat,tign_g,bounds,ctime,dx,dy,wrfout_file=wrfout_file,plot=plot)

    return forecast

def process_forecast_wrfout(wrfout_file,scale,time_num,epsilon=5,plot=False):
    """
    Process forecast from a wrfout.

    :param wrfout_file: path to wrfout file
    :param scale: numerical time in seconds of start and end of simulation

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-06-05
    """

    # read forecast from wrfout
    lon,lat,tign_g,ctime,dx,dy = read_forecast_wrfout(wrfout_file)      
    # create forecast
    X,y,c = process_tign_g(lon,lat,tign_g,ctime,scale,time_num,epsilon=epsilon,plot=plot)
    print 'shape X_f: ', X.shape
    print 'shape y_f: ', y.shape
    print 'shape c_f: ', c.shape
    print 'len fire: ', len(X[y==1])
    print 'len ground: ', len(X[y==-1])

    return X,y,c

if __name__ == "__main__":
    import saveload as sl
    real = True
    cloud = True

    if real:
        plot = True
        if cloud:
            time_num = [1376114400.0, 1376546400.0]
            scale = [1375898400.0, 1377410400.0]
            dst = './patch/wrfout_patch'
            X,y,c = process_forecast_wrfout(dst,scale,time_num,plot=plot)
            sl.save(dict({'X': X, 'y': y, 'c': c}),'forecast')
        else:
            bounds = (-113.85068, -111.89413, 39.677563, 41.156837)
            dst = './patch/wrfout_patch'
            f = process_forecast_slides_wrfout(dst,bounds,plot=plot)
            sl.save(f,'forecast')
    else:
        from infrared_perimeters import process_ignitions
        from setup import process_detections
        dst = 'ideal_test'
        plot = False
        ideal = sl.load(dst)
        kk = 4
        data = process_tign_g_slices(ideal['lon'][::kk,::kk],ideal['lat'][::kk,::kk],ideal['tign_g'][::kk,::kk],ideal['bounds'],ideal['ctime'],ideal['dx'],ideal['dy'],wrfout_file='ideal',dt_for=ideal['dt'],plot=plot)
        if 'point' in ideal.keys():
            p = [[ideal['point'][0]],[ideal['point'][1]],[ideal['point'][2]]]
            data.update(process_ignitions(p,ideal['bounds']))
        elif 'points' in ideal.keys():
            p = [[point[0] for point in ideal['points']],[point[1] for point in ideal['points']],[point[2] for point in ideal['points']]]
            data.update(process_ignitions(p,ideal['bounds']))
        etime = time_iso2num(ideal['ctime'].replace('_','T'))
        time_num_int = (etime-ideal['tign_g'].max(),etime)
        sl.save((data,ideal['lon'],ideal['lat'],time_num_int),'data')
        process_detections(data,ideal['lon'],ideal['lat'],time_num_int)

