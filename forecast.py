import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import netCDF4 as nc
from datetime import timedelta
from JPSSD import time_iso2num, time_iso2datetime, time_datetime2iso
from utils import Dict
import saveload as sl
import re, glob, sys, os


def process_tign_g(lon,lat,tign_g,bounds,ctime,scan,track,wrfout_file=''):
    """
    Process forecast from lon, lat, and tign_g

    :param lon: array of longitudes
    :param lat: array of latitudes
    :param tign_g: array of fire arrival time
    :param bounds: coordinate bounding box filtering to

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-06-05
    """

    # prefix of the elements in the dictionary
    prefix = "FOR"
    # confidences
    conf_fire = 70
    # margin percentage
    margin = .1
    # initializing dictionary
    forecast = Dict({})

    # ctime transformations
    ctime_iso = ctime.replace('_','T')
    ctime_datetime = time_iso2datetime(ctime_iso)
    # mask coordinates to bounding box
    mask = np.logical_and(np.logical_and(np.logical_and(lon>bounds[0],lon<bounds[1]),lat>bounds[2]),lat<bounds[3])
    # create a square subset of fire arrival time less than the maximum
    mtign = np.logical_and(mask,tign_g < tign_g.max())
    mlon = lon[mtign]
    mlat = lat[mtign]
    mlen = margin*(mlon.max()-mlon.min())
    sbounds = (mlon.min()-mlen, mlon.max()+mlen, mlat.min()-mlen, mlat.max()+mlen)
    smask = np.logical_and(np.logical_and(np.logical_and(lon>sbounds[0],lon<sbounds[1]),lat>sbounds[2]),lat<sbounds[3])

    # times to get fire arrival time from
    dt_forecast = 600 # in seconds
    TT = np.arange(tign_g.min(),tign_g.max(),dt_forecast)[0:-1]
    for T in TT:
        # fire arrival time to datetime
        time_datetime = ctime_datetime-timedelta(seconds=float(tign_g.max()-T)) # there is an error of about 10 seconds (wrfout not ending at exact time of simulation)
        # create ISO time of the date
        time_iso = time_datetime2iso(time_datetime)
        # create numerical time from the ISO time
        time_num = time_iso2num(time_iso)
        # create time stamp
        time_data = '_A%04d%03d_%02d%02d' % (time_datetime.year, time_datetime.timetuple().tm_yday,
                                                time_datetime.hour, time_datetime.minute)
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
            plt.plot(lons[fires==5],lats[fires==5],'g.')
            plt.plot(lons[fires==9],lats[fires==9],'r.')
            plt.show()
            plt.pause(.001)
            plt.cla()

        # update perimeters dictionary
        forecast.update({prefix + time_data: Dict({'file': wrfout_file, 'time_tign_g': T, 'lon': lons, 'lat': lats,
                            'fire': fires, 'conf_fire': np.array(conf_fire*np.ones(lons[fires==9].shape)),
                            'lon_fire': lons[fires==9], 'lat_fire': lats[fires==9], 'lon_nofire': lons[fires==5], 'lat_nofire': lats[fires==5],
                            'scan_fire': scan*np.ones(lons[fires==9].shape), 'track_fire': track*np.ones(lons[fires==9].shape),
                            'scan_nofire': scan*np.ones(lons[fires==5].shape), 'track_nofire': track*np.ones(lons[fires==9].shape),
                            'time_iso': time_iso, 'time_num': time_num, 'acq_date': acq_date, 'acq_time': acq_time})})

    return forecast

def process_forecast_wrfout(wrfout_file,bounds,plot=False):
    """
    Process forecast from a wrfout.

    :param dst: path to wrfout file
    :param bounds: coordinate bounding box filtering to

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-06-05
    """

    # read file
    try:
        data = nc.Dataset(wrfout_file)
    except Exception as e:
        print 'Warning: No netcdf file %s in the path' % wrfout_file
        return []
    # current time
    ctime = ''.join(data['Times'][-1])
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
    dx = data.getncattr('DX')
    dy = data.getncattr('DY')
    # scan and track dimensions of the observation (in km)
    scan = dx/1000.
    track = dy/1000.
    # create forecast
    forecast = process_tign_g(lon,lat,tign_g,bounds,ctime,scan,track,wrfout_file=wrfout_file)
    # close netcdf file
    data.close()

    return forecast


if __name__ == "__main__":
    plot = True
    bounds = (-113.85068, -111.89413, 39.677563, 41.156837)
    dst = './patch/wrfout_patch'

    f = process_forecast_wrfout(dst,bounds,plot=plot)
    sl.save(f,'forecast')
