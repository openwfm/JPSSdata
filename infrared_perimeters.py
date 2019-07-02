from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from JPSSD import time_iso2num, time_iso2datetime
from utils import Dict
import saveload as sl
import re, glob, sys, os


def process_ignitions(igns,bounds):
    """
    Process ignitions the same way than satellite data.

    :param igns: ([lons],[lats],[dates]) where lons and lats in degrees and dates in ESMF format
    :param bounds: coordinate bounding box filtering to
    :return ignitions: dictionary with all the information from each ignition similar to satellite data

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-05-29
    """

    # prefix of the elements in the dictionary
    prefix = "IGN"
    # initializing dictionary
    ignitions = dict({})
    # scan and track dimensions of the observation (in km)
    scan = 1.
    track = 1.
    # confidences
    conf_fire = 100

    # for each ignition
    for lon, lat, time_iso in zip(igns[0],igns[1],igns[2]):
        try:
            # take coordinates
            lon = np.array(lon)
            lat = np.array(lat)
            # look if coordinates in bounding box
            mask = np.logical_and(np.logical_and(np.logical_and(lon>bounds[0],lon<bounds[1]),lat>bounds[2]),lat<bounds[3])
            if not mask.sum():
                break
            lons = lon[mask]
            lats = lat[mask]
            # get time elements
            time_num = time_iso2num(time_iso)
            time_datetime = time_iso2datetime(time_iso)
            time_data = '_A%04d%03d_%02d%02d_%02d' % (time_datetime.year, time_datetime.timetuple().tm_yday,
                                        time_datetime.hour, time_datetime.minute, time_datetime.second)
            acq_date = '%04d-%02d-%02d' % (time_datetime.year, time_datetime.month, time_datetime.day)
            acq_time = '%02d%02d' % (time_datetime.hour, time_datetime.minute)
        except Exception as e:
            print 'Error: bad ignition %s specified.' % igns
            print 'Exception: %s.' % e
            sys.exit(1)

        # no nofire detection
        lon_nofire = np.array([])
        lat_nofire = np.array([])
        conf_nofire = np.array([])

        # update ignitions dictionary
        ignitions.update({prefix + time_data: Dict({'lon': lons, 'lat': lats,
                                'fire': np.array(9*np.ones(lons.shape)), 'conf_fire': np.array(conf_fire*np.ones(lons.shape)),
                                'lon_fire': lons, 'lat_fire': lats, 'lon_nofire': lon_nofire, 'lat_nofire': lat_nofire,
                                'scan_fire': scan*np.ones(lons.shape), 'track_fire': track*np.ones(lons.shape),
                                'conf_nofire' : conf_nofire, 'scan_nofire': scan*np.ones(lon_nofire.shape),
                                'track_nofire': track*np.ones(lon_nofire.shape), 'time_iso': time_iso,
                                'time_num': time_num, 'acq_date': acq_date, 'acq_time': acq_time})})
    return ignitions


def process_infrared_perimeters(dst,bounds,maxp=1000,ngrid=50,plot=False):
    """
    Process infrared perimeters the same way than satellite data.

    :param dst: path to kml perimeter files
    :param bounds: coordinate bounding box filtering to
    :param maxp: optional, maximum number of points for each perimeter
    :param ngrid: optional, number of nodes for the grid of in/out nodes at each direction
    :param plot: optional, boolean to plot or not the result at each perimeter iteration
    :return perimeters: dictionary with all the information from each perimeter similar to satellite data

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-05-29
    """

    # list of kml files in 'dst' path
    files = glob.glob(osp.join(dst, '*.kml'))
    # prefix of the elements in the dictionary
    prefix = "PER"
    # initializing dictionary
    perimeters = Dict({})
    # scan and track dimensions of the observation (in km)
    scan = .05
    track = .05
    # confidences
    conf_fire = 100
    conf_nofire = 100

    # Creating grid where to evaluate in/out of the perimeter
    [X,Y] = np.meshgrid(np.linspace(bounds[0],bounds[1],ngrid),np.linspace(bounds[2],bounds[3],ngrid))
    XX = np.c_[(X.ravel(),Y.ravel())]

    # if any file
    if files:
        # for each file
        for file in files:
            print 'Retrieving perimeters from %s' % file
            try:
                # open the file
                f = open(file,"r")
                # read all the lines of the file
                f_str = ''.join(f.readlines())
                # close the file
                f.close()
            except Exception as e:
                print 'Error: exception when opening file %s, %s' % (file,e.message)
                sys.exit(1)
            try:
                # Read name and get time elements
                # read name of the file
                name = re.findall(r'<name>(.*?)</name>',f_str,re.DOTALL)[0]
                # read date of the perimeter
                date = re.match(r'.*([0-9]+)-([0-9]+)-([0-9]+) ([0-9]{2})([0-9]{2})',name).groups()
                date = (date[2],date[0],date[1],date[3],date[4])
                # create ISO time of the date
                time_iso = '%04d-%02d-%02dT%02d:%02d:00' % tuple([ int(d) for d in date ])
                # create numerical time from the ISO time
                time_num = time_iso2num(time_iso)
                # create datetime element from the ISO time
                time_datetime = time_iso2datetime(time_iso)
                # create time stamp
                time_data = '_A%04d%03d_%02d%02d' % (time_datetime.year, time_datetime.timetuple().tm_yday,
                                                    time_datetime.hour, time_datetime.minute)
                # create acquisition date
                acq_date = '%04d-%02d-%02d' % (time_datetime.year, time_datetime.month, time_datetime.day)
                # create acquisition time
                acq_time = '%02d%02d' % (time_datetime.hour, time_datetime.minute)

                # Get the coordinates of all the perimeters
                # regex of the polygons (or perimeters)
                polygons = re.findall(r'<Polygon>(.*?)</Polygon>',f_str,re.DOTALL)
                # for each polygon, regex of the coordinates
                buckets = [re.split('\r\n\s+',re.findall(r'<coordinates>(.*?)</coordinates>',p,re.DOTALL)[0])[1:] for p in polygons]
                # array of arrays with each polygon coordinates
                coordinates = [[np.array(re.split(',',b)[0:2]).astype(float) for b in bucket] for bucket in buckets]
            except Exception as e:
                print 'Error: file %s has not proper structure.' % file
                print 'Exception: %s.' % e
                sys.exit(1)

            # Plot perimeter
            if plot:
                plt.ion()
                plt.plot([coord[0] for coordinate in coordinates for coord in coordinate],[coord[1] for coordinate in coordinates for coord in coordinate],'bx')

            # Create upper and lower bound coordinates depending on in/out polygons
            # compute path elements for each polygon
            paths = [Path(coord) for coord in coordinates]
            # compute mask of coordinates inside polygon for each polygon
            masks = [path.contains_points(XX) for path in paths]
            # logical or for all the masks
            inmask = np.logical_or.reduce(masks)
            # upper and lower bounds arifitial from in/out polygon
            up_arti = XX[inmask]
            lon_fire = np.array([up[0] for up in up_arti])
            lat_fire = np.array([up[1] for up in up_arti])
            low_arti = XX[~inmask]
            lon_nofire = np.array([low[0] for low in low_arti])
            lat_nofire = np.array([low[1] for low in low_arti])

            # take a coarsening of the perimeters
            for k,coord in enumerate(coordinates):
                if len(coord) > maxp:
                    coarse = len(coord)/maxp
                    if coarse > 0:
                        coordinates[k] = [coord[ind] for ind in np.concatenate(([0],range(len(coord))[coarse:-coarse:coarse]))]

            # append perimeter nodes
            lon_fire = np.append(lon_fire,np.array([coord[0] for coordinate in coordinates for coord in coordinate]))
            lat_fire = np.append(lat_fire,np.array([coord[1] for coordinate in coordinates for coord in coordinate]))

            # create general arrays
            lon = np.concatenate((lon_nofire,lon_fire))
            lat = np.concatenate((lat_nofire,lat_fire))
            fire = np.concatenate((5*np.ones(lon_nofire.shape),9*np.ones(lon_fire.shape)))

            # mask in bounds
            mask = np.logical_and(np.logical_and(np.logical_and(lon>bounds[0],lon<bounds[1]),lat>bounds[2]),lat<bounds[3])
            if not mask.sum():
                break
            lons = lon[mask]
            lats = lat[mask]
            fires = fire[mask]

            # plot results
            if plot:
                plt.plot(lons[fires==5],lats[fires==5],'g.')
                plt.plot(lons[fires==9],lats[fires==9],'r.')
                plt.show()
                plt.pause(.001)
                plt.cla()

            # update perimeters dictionary
            perimeters.update({prefix + time_data: Dict({'file': file, 'lon': lons, 'lat': lats,
                            'fire': fires, 'conf_fire': np.array(conf_fire*np.ones(lons[fires==9].shape)),
                            'lon_fire': lons[fires==9], 'lat_fire': lats[fires==9], 'lon_nofire': lons[fires==5], 'lat_nofire': lats[fires==5],
                            'scan_fire': scan*np.ones(lons[fires==9].shape), 'track_fire': track*np.ones(lons[fires==9].shape),
                            'conf_nofire': np.array(conf_nofire*np.ones(lons[fires==5].shape)),
                            'scan_nofire': scan*np.ones(lons[fires==5].shape), 'track_nofire': track*np.ones(lons[fires==9].shape),
                            'time_iso': time_iso, 'time_num': time_num, 'acq_date': acq_date, 'acq_time': acq_time})})
    else:
        print 'Warning: No KML files in the path specified'
        perimeters = []
    return perimeters


if __name__ == "__main__":
    plot = True
    #bounds = (-115.97282409667969, -115.28449249267578, 43.808258056640625, 44.302913665771484)
    #dst = './pioneer/perim'
    bounds = (-113.85068, -111.89413, 39.677563, 41.156837)
    dst = './patch/perim'

    p = process_infrared_perimeters(dst,bounds,plot=plot)
    sl.save(p,'perimeters')
