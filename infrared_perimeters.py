from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from JPSSD import time_iso2num, time_iso2datetime
from utils import Dict
import saveload as sl
import re, glob, sys, os

def process_paths(outer_coords,inner_coords,min_points=50):
    outer_paths = []
    inner_paths = []
    for n,outer in enumerate(outer_coords):
        outer = np.array(outer)
        if len(outer[:,0]) > min_points:
            path = Path(outer)
            outer_paths.append(path)
            inners = np.array(inner_coords[n])
            if len(inners.shape) > 2:
                in_paths = []
                for inner in inners:
                    if len(inner) > min_points:
                        path = Path(inner)
                        in_paths.append(path)    
            else:
                if len(inners) > min_points:
                    in_paths = Path(inners)
                else:
                    in_paths = []
            inner_paths.append(in_paths)
    return outer_paths,inner_paths

def process_ignitions(igns,bounds,time=None):
    """
    Process ignitions the same way than satellite data.

    :param igns: ([lons],[lats],[dates]) where lons and lats in degrees and dates in ESMF format
    :param bounds: coordinate bounding box filtering to
    :return ignitions: dictionary with all the information from each ignition similar to satellite data

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-05-29
    """

    # Time interval
    if time:
        interval_datetime = map(time_iso2datetime,time)

    # prefix of the elements in the dictionary
    prefix = "IGN"
    # initializing dictionary
    ignitions = dict({})
    # scan and track dimensions of the observation (in km)
    scan = 1.
    track = 1.
    # confidences
    conf_fire = 100.

    # for each ignition
    for lon, lat, time_iso in zip(igns[0],igns[1],igns[2]):
        try:
            # take coordinates
            lon = np.array(lon)
            lat = np.array(lat)
            # look if coordinates in bounding box
            mask = np.logical_and(np.logical_and(np.logical_and(lon >= bounds[0], lon <= bounds[1]),lat >= bounds[2]),lat <= bounds[3])
            if not mask.sum():
                continue
            lons = lon[mask]
            lats = lat[mask]
            # get time elements
            time_num = time_iso2num(time_iso)
            time_datetime = time_iso2datetime(time_iso)
            # skip if not inside interval (only if time is determined)
            if time and (time_datetime < interval_datetime[0] or time_datetime > interval_datetime[1]):
                print 'Perimeter from %s skipped, not inside the simulation interval!' % file
                continue
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


def process_infrared_perimeters(dst,bounds,time=None,maxp=500,ngrid=50,plot=False,gen_polys=False):
    """
    Process infrared perimeters the same way than satellite data.

    :param dst: path to kml perimeter files
    :param bounds: coordinate bounding box filtering to
    :param time: optional, time interval in ISO
    :param maxp: optional, maximum number of points for each perimeter
    :param ngrid: optional, number of nodes for the grid of in/out nodes at each direction
    :param plot: optional, boolean to plot or not the result at each perimeter iteration
    :return perimeters: dictionary with all the information from each perimeter similar to satellite data

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-05-29
    """

    # Time interval
    if time:
        interval_datetime = map(time_iso2datetime,time)

    # list of kml files in 'dst' path
    files = glob.glob(osp.join(dst, '*.kml'))
    # prefix of the elements in the dictionary
    prefix = "PER"
    # initializing dictionary
    perimeters = Dict({})
    # scan and track dimensions of the observation (in km)
    scan = .5
    track = .5
    # confidences
    conf_fire = 100.
    conf_nofire = 70.

    # Creating grid where to evaluate in/out of the perimeter
    [X,Y] = np.meshgrid(np.linspace(bounds[0],bounds[1],ngrid),np.linspace(bounds[2],bounds[3],ngrid))
    XX = np.c_[np.ravel(X),np.ravel(Y)]

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
                date = re.match(r'.* ([0-9]+)-([0-9]+)-([0-9]+) ([0-9]{2})([0-9]{2})',name).groups()
                date = (date[2],date[0],date[1],date[3],date[4])
                # create ISO time of the date
                time_iso = '%04d-%02d-%02dT%02d:%02d:00' % tuple([ int(d) for d in date ])
                # create numerical time from the ISO time
                time_num = time_iso2num(time_iso)
                # create datetime element from the ISO time
                time_datetime = time_iso2datetime(time_iso)
                # skip if not inside interval (only if time is determined)
                if time and (time_datetime < interval_datetime[0] or time_datetime > interval_datetime[1]):
                    print 'Perimeter from %s skipped, not inside the simulation interval!' % file
                    continue
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
                # for each polygon, outer boundary
                outer_buckets = [re.findall(r'<outerBoundaryIs>(.*?)</outerBoundaryIs>',p,re.DOTALL)[0] for p in polygons]
                # for each outer polygon, regex of the coordinates
                buckets = [re.split(r'\s',re.findall(r'<coordinates>(.*?)</coordinates>',p,re.DOTALL)[0])[1:] for p in outer_buckets]
                # array of arrays with each outer polygon coordinates
                outer_coordinates = [[np.array(re.split(',',b)[0:2]).astype(float) for b in bucket if b is not ''] for bucket in buckets]
                # for each polygon, inner boundary
                inner_buckets = [re.findall(r'<innerBoundaryIs>(.*?)</innerBoundaryIs>',p,re.DOTALL)[0] if re.findall(r'<innerBoundaryIs>(.*?)</innerBoundaryIs>',p,re.DOTALL) else '' for p in polygons]
                # for each inner polygon, regex of the coordinates
                buckets = [re.split(r'\s',re.findall(r'<coordinates>(.*?)</coordinates>',p,re.DOTALL)[0])[1:] if p != '' else '' for p in inner_buckets]
                # array of arrays with each inner polygon coordinates
                inner_coordinates = [[np.array(re.split(',',b)[0:2]).astype(float) for b in bucket if b is not ''] for bucket in buckets]
            except Exception as e:
                print 'Error: file %s has not proper structure.' % file
                print 'Exception: %s.' % e
                sys.exit(1)

            # Plot perimeter
            if plot:
                plt.ion()
                for outer in outer_coordinates:
                    if len(outer):
                        x = np.array(outer)[:,0]
                        y = np.array(outer)[:,1]
                        plt.plot(x,y,'gx')
                for inner in inner_coordinates:
                    if len(inner):
                        x = np.array(outer)[:,0]
                        y = np.array(outer)[:,1]
                        plt.plot(x,y,'rx')

            # Create paths for each polygon (outer and inner)
            outer_paths,inner_paths = process_paths(outer_coordinates,inner_coordinates)
            if len(outer_paths):
                # compute mask of coordinates inside outer polygons
                outer_mask = np.logical_or.reduce([path.contains_points(XX) for path in outer_paths if path])
                # compute mask of coordinates inside inner polygons
                inner_mask = np.logical_or.reduce([path.contains_points(XX) for path in inner_paths if path])
                # mask inside outer polygons and outside inner polygons
                inmask = outer_mask * ~inner_mask
                # upper and lower bounds arifitial from in/out polygon
                up_arti = XX[inmask]
                lon_fire = np.array([up[0] for up in up_arti])
                lat_fire = np.array([up[1] for up in up_arti])
                low_arti = XX[~inmask]
                lon_nofire = np.array([low[0] for low in low_arti])
                lat_nofire = np.array([low[1] for low in low_arti])

                # take a coarsening of the outer polygons
                coordinates = []
                for k,coord in enumerate(outer_coordinates):
                    if len(coord) > maxp:
                        coarse = len(coord)/maxp
                        if coarse > 0:
                            coordinates += [coord[ind] for ind in np.concatenate(([0],range(len(coord))[coarse:-coarse:coarse]))]

                if coordinates:
                    # append perimeter nodes
                    lon_fire = np.append(lon_fire,np.array(coordinates)[:,0])
                    lat_fire = np.append(lat_fire,np.array(coordinates)[:,1])

                # create general arrays
                lon = np.concatenate((lon_nofire,lon_fire))
                lat = np.concatenate((lat_nofire,lat_fire))
                fire = np.concatenate((5*np.ones(lon_nofire.shape),9*np.ones(lon_fire.shape)))

                # mask in bounds
                mask = np.logical_and(np.logical_and(np.logical_and(lon >= bounds[0],lon <= bounds[1]),lat >= bounds[2]),lat <= bounds[3])
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
                    plt.pause(.5)
                    plt.cla()
            else:
                lons = np.array([])
                lats = np.array([])
                fires = np.array([])

            if gen_polys:
                from shapely.geometry import Polygon
                from shapely.ops import transform
                from functools import partial
                import pyproj

                proj4_wgs84 = '+proj=latlong +ellps=WGS84 +datum=WGS84 +units=degree +no_defs'
                proj4_moll = '+proj=moll +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
                proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                                pyproj.Proj(proj4_moll))
                polys = []
                for n,outer in enumerate(outer_coordinates):
                    x = np.array(outer)[:,0]
                    y = np.array(outer)[:,1]
                    shape = Polygon([(l[0],l[1]) for l in zip(x,y)]).buffer(0)
                    inner = np.array(inner_coordinates[n])
                    if len(inner):
                        if len(inner.shape) > 2:
                            for h in inner_coordinates[n]:
                                xh = np.array(h)[:,0]
                                yh = np.array(h)[:,1]
                        else:
                            xh = np.array(inner)[:,0]
                            yh = np.array(inner)[:,1]
                        shapeh = Polygon([(l[0],l[1]) for l in zip(xh,yh)]).buffer(0)
                        shape.difference(shapeh)
                    poly = transform(proj,shape)
                    polys.append(poly)

            idn = prefix + time_data
            if idn in perimeters.keys():
                idn = idn + '_2'
            # update perimeters dictionary
            perimeters.update({idn: Dict({'file': file, 'lon': lons, 'lat': lats,
                            'fire': fires, 'conf_fire': np.array(conf_fire*np.ones(lons[fires==9].shape)),
                            'lon_fire': lons[fires==9], 'lat_fire': lats[fires==9], 'lon_nofire': lons[fires==5], 'lat_nofire': lats[fires==5],
                            'scan_fire': scan*np.ones(lons[fires==9].shape), 'track_fire': track*np.ones(lons[fires==9].shape),
                            'conf_nofire': np.array(conf_nofire*np.ones(lons[fires==5].shape)),
                            'scan_nofire': scan*np.ones(lons[fires==5].shape), 'track_nofire': track*np.ones(lons[fires==9].shape),
                            'time_iso': time_iso, 'time_num': time_num, 'acq_date': acq_date, 'acq_time': acq_time})})
            if gen_polys:
                perimeters[idn].update({'polys': polys})
    else:
        print 'Warning: No KML files in the path specified'
        perimeters = []
    return perimeters


if __name__ == "__main__":
    # Experiment to do
    exp = 4
    # Plot perimeters as created
    plot = True

    # Defining options
    def pioneer():
        bounds = (-115.97282409667969, -115.28449249267578, 43.808258056640625, 44.302913665771484)
        time_iso = ('2016-07-18T00:00:00Z', '2016-08-31T00:00:00Z')
        igns = None
        perims = './pioneer/perim'
        return bounds, time_iso, igns, perims
    def patch():
        bounds = (-113.85068, -111.89413, 39.677563, 41.156837)
        time_iso = ('2013-08-10T00:00:00Z', '2013-08-15T00:00:00Z')
        igns = ([-112.676039],[40.339372],['2013-08-10T20:00:00Z'])
        perims = './patch/perim'
        return bounds, time_iso, igns, perims
    def saddleridge():
        bounds = (-118.60684204101562, -118.35965728759766, 34.226539611816406, 34.43047332763672)
        time_iso = ('2019-10-10T00:00:00Z', '2019-10-15T00:00:00Z')
        igns = None
        perims = './saddleridge/perim'
        return bounds, time_iso, igns, perims
    def polecreek():
        bounds = (-111.93914, -111.311035, 39.75985, 40.239746)
        time_iso = ('2018-09-09T00:00:00Z', '2018-09-23T00:00:00Z')
        igns = None
        perims = './polecreek/perim'
        return bounds, time_iso, igns, perims

    # Creating the options
    options = {1: pioneer, 2: patch, 3: saddleridge, 4: polecreek}

    # Defining the option depending on the experiment
    bounds, time_iso, igns, perims = options[exp]()

    # Processing infrared perimeters
    p = process_infrared_perimeters(perims,bounds,time=time_iso,plot=plot,gen_polys=True)
    # Processing ignitions if defined
    if igns:
        p.update(process_ignitions(igns,bounds,time=time_iso))
    # Saving results
    sl.save(p,'perimeters')

