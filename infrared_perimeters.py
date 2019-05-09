import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from JPSSD import time_iso2num, time_iso2datetime
from utils import Dict
import saveload as sl
import re, glob, sys, os

def process_ignitions(igns,bounds=None):
    prefix = "IGN"
    ignitions = dict({})
    scan = 1.
    track = 1.
    for lon, lat, time_iso in zip(igns[0],igns[1],igns[2]):
        try:
            lon = np.array(lon)
            lat = np.array(lat)
            mask = np.logical_and(np.logical_and(np.logical_and(lon>bounds[0],lon<bounds[1]),lat>bounds[2]),lat<bounds[3])
            lons = lon[mask]
            lats = lat[mask]
            time_num = time_iso2num(time_iso)
            time_datetime = time_iso2datetime(time_iso)
            time_data = '_A%04d%03d_%02d%02d' % (time_datetime.year, time_datetime.timetuple().tm_yday,
                                        time_datetime.hour, time_datetime.minute)
            acq_date = '%04d-%02d-%02d' % (time_datetime.year, time_datetime.month, time_datetime.day)
            acq_time = '%02d%02d' % (time_datetime.hour, time_datetime.minute)
        except Exception as e:
            print 'Error: bad ignition %s specified.' % igns
            print 'Exception: %s.' % e
            sys.exit(1)
        ignitions.update({prefix + time_data: Dict({'lon': lons, 'lat': lats,
                                'fire': np.array(9*np.ones(lons.shape)), 'conf_fire': np.array(100*np.ones(lons.shape)),
                                'lon_fire': lons, 'lat_fire': lats, 'scan_fire': scan*np.ones(lons.shape),
                                'track_fire': track*np.ones(lons.shape), 'time_iso': time_iso, 'time_num': time_num,
                                'acq_date': acq_date, 'acq_time': acq_time})})
    return ignitions

def process_infrared_perimeters(dst,bounds=None,maxp=500,plot=False):
    files = glob.glob(osp.join(dst, '*.kml'))
    prefix = "PER"
    perimeters = Dict({})
    scan = .01
    track = .01
    if files:
        for file in files:
            print 'Retrieving perimeters from %s' % file
            try:
                f = open(file,"r")
                f_str = ''.join(f.readlines())
                f.close()
            except Exception as e:
                print 'Error: exception when opening file %s, %s' % (file,e.message)
                sys.exit(1)
            try:
                name = re.findall(r'<name>(.*?)</name>',f_str,re.DOTALL)[0]
                date = re.match(r'.*([0-9]{2}-[0-9]{2}-[0-9]{4} [0-9]{4})',name).groups()[0]
                time_iso = date[6:10]+'-'+date[0:2]+'-'+date[3:5]+'T'+date[11:13]+':'+date[13:15]+':00'
                time_num = time_iso2num(time_iso)
                time_datetime = time_iso2datetime(time_iso)
                time_data = '_A%04d%03d_%02d%02d' % (time_datetime.year, time_datetime.timetuple().tm_yday,
                                                    time_datetime.hour, time_datetime.minute)
                acq_date = '%04d-%02d-%02d' % (time_datetime.year, time_datetime.month, time_datetime.day)
                acq_time = '%02d%02d' % (time_datetime.hour, time_datetime.minute)
                polygons = re.findall(r'<Polygon>(.*?)</Polygon>',f_str,re.DOTALL)
                buckets = [re.split('\r\n\s+',re.findall(r'<coordinates>(.*?)</coordinates>',p,re.DOTALL)[0])[1:] for p in polygons]
                coordinates = [np.array(re.split(',',b)[0:2]).astype(float) for bucket in buckets for b in bucket]
                if len(coordinates) > maxp:
                    coarse = len(coordinates)/maxp
                    if coarse > 0:
                        coordinates = coordinates[1::coarse]
            except Exception as e:
                print 'Error: file %s has not proper structure.' % file
                print 'Exception: %s.' % e
                sys.exit(1)
            lon = np.array([coord[0] for coord in coordinates])
            lat = np.array([coord[1] for coord in coordinates])
            mask = np.logical_and(np.logical_and(np.logical_and(lon>bounds[0],lon<bounds[1]),lat>bounds[2]),lat<bounds[3])
            lons = lon[mask]
            lats = lat[mask]
            if plot:
                plt.plot(lons,lats,'*')
            perimeters.update({prefix + time_data: Dict({'file': file, 'lon': lons, 'lat': lats,
                            'fire': np.array(9*np.ones(lons.shape)), 'conf_fire': np.array(100*np.ones(lons.shape)),
                            'lon_fire': lons, 'lat_fire': lats, 'scan_fire': scan*np.ones(lons.shape),
                            'track_fire': track*np.ones(lons.shape), 'time_iso': time_iso, 'time_num': time_num,
                            'acq_date': acq_date, 'acq_time': acq_time})})
        if plot:
            plt.show()
    else:
        print 'Warning: No KML files in the path specified'
        perimeters = []
    return perimeters


if __name__ == "__main__":
    plot = False
    dst = './pioneer_perim'

    p = process_infrared_perimeters(dst,plot)
    print p
