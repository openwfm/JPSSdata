import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import requests
import urlparse
import os
import sys
import re
import glob
import netCDF4 as nc
from cmr import CollectionQuery, GranuleQuery
from pyhdf.SD import SD, SDC
from utils import *
import scipy.io as sio
import h5py
from datetime import datetime
import time
import pandas as pd

def search_api(sname,bbox,time,maxg=50,platform="",version=""):
    """API search of the different satellite granules
        :param:
            sname       short name 
            bbox        polygon with the search bounding box
            time        time interval (init_time,final_time)
            maxg        max number of granules to process
            platform    string with the platform
            version     string with the version
        :returns:
            granules    dictionary with the metadata of the search

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    api = GranuleQuery()
    if not version:    
        if not platform:
            search = api.parameters(
                                short_name=sname,
                                downloadable=True,
                                polygon=bbox,
                                temporal=time
                                )
        else:
            search = api.parameters(
                                short_name=sname,
                                platform=platform,
                                downloadable=True,
                                polygon=bbox,
                                temporal=time
                                )
    else:
        if not platform:
            search = api.parameters(
                                short_name=sname,
                                downloadable=True,
                                polygon=bbox,
                                temporal=time,
                                version=version
                                )
        else:
            search = api.parameters(
                                short_name=sname,
                                platform=platform,
                                downloadable=True,
                                polygon=bbox,
                                temporal=time,
                                version=version
                                )
    sh=search.hits()
    print "%s gets %s hits in this range" % (sname,sh)
    if sh>maxg:
        print "The number of hits %s is larger than the limit %s." % (sh,maxg)
        print "Use a reduced bounding box or a reduced time interval."
        granules = []
    else:
        granules = api.get(sh)
    return granules

def get_meta(bbox,time,maxg=50):
    """ 
    Get all the meta data from the API for all the necessary products
        :param:
            bbox        polygon with the search bounding box
            time        time interval (init_time,final_time)
            maxg        max number of granules to process
        :returns:
            granules    dictionary with the metadata of all the products

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    granules=Dict([]);
    #MOD14: MODIS Terra fire data
    granules.MOD14=search_api("MOD14",bbox,time,maxg,"Terra")
    #MOD03: MODIS Terra geolocation data
    granules.MOD03=search_api("MOD03",bbox,time,maxg,"Terra","6")
    #MYD14: MODIS Aqua fire data
    granules.MYD14=search_api("MYD14",bbox,time,maxg,"Aqua")
    #MYD03: MODIS Aqua geolocation data
    granules.MYD03=search_api("MYD03",bbox,time,maxg,"Aqua","6")
    #VNP14: VIIRS fire data, res 750m
    granules.VNP14=search_api("VNP14",bbox,time,maxg)
    #VNP03MODLL: VIIRS geolocation data, res 750m
    granules.VNP03=search_api("VNP03MODLL",bbox,time,maxg)
    #VNP14hi: VIIRS fire data, res 375m
    #granules.VNP14hi=search("VNP14IMGTDL_NRT",bbox,time,maxg)
    return granules

def group_files(path,reg):
    """ 
    Agrupate the geolocation (03) and fire (14) files of a specific product in a path
        :param:
            path    path to the geolocation (03) and fire (14) files
            reg     string with the first 3 characters specifying the product (MOD, MYD or VNP)
        :returns: 
            files   list of pairs with geolocation (03) and fire (14) file names in the path of the specific product

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    files=[Dict({'geo':k}) for k in glob.glob(path+'/'+reg+'03*')]
    filesf=glob.glob(path+'/'+reg+'14*')
    if len(filesf)>0:
        for f in filesf:
            mf=re.split("/",f)
            if mf is not None:
                m=mf[-1].split('.')
                if m is not None:
                    for k,g in enumerate(files):
                        mmf=re.split("/",g.geo)
                        mm=mmf[-1].split('.')
                        if mm[0][1]==m[0][1] and mm[1]+'.'+mm[2]==m[1]+'.'+m[2]:
                            files[k].fire=f 
    return files

def group_all(path):
    """ 
    Combine all the geolocation (03) and fire (14) files in a path
        :param:
            path    path to the geolocation (03) and fire (14) files
        :returns: 
            files   list of products with a list of pairs with geolocation (03) and fire (14) file names in the path

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    # MOD files
    modf=group_files(path,'MOD')
    # MYD files
    mydf=group_files(path,'MYD')
    # VIIRS files
    vif=group_files(path,'VNP')
    files=[modf,mydf,vif]
    return files

def read_modis_files(files):
    """ 
    Read the geolocation (03) and fire (14) files for MODIS products (MOD or MYD)
        :param: files  pair with geolocation (03) and fire (14) file names for MODIS products (MOD or MYD)
        :returns: ret  dictionary with Latitude, Longitude and fire mask arrays read
    
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """

    hdfg=SD(files.geo,SDC.READ)
    hdff=SD(files.fire,SDC.READ)
    lat_obj=hdfg.select('Latitude')
    lon_obj=hdfg.select('Longitude')    
    fire_obj=hdff.select('fire mask')
    lat_fire_obj=hdff.select('FP_latitude')
    lon_fire_obj=hdff.select('FP_longitude')
    brig_fire_obj=hdff.select('FP_T21')
    sample_fire_obj=hdff.select('FP_sample')
    conf_fire_obj=hdff.select('FP_confidence')
    t31_fire_obj=hdff.select('FP_T31')
    frp_fire_obj=hdff.select('FP_power')
    ret=Dict([])
    ret.lat=lat_obj.get()
    ret.lon=lon_obj.get()
    ret.fire=fire_obj.get()
    nf=np.logical_or(ret.fire == 3, ret.fire == 5)
    try:
        ret.lat_fire=lat_fire_obj.get()
    except:
        ret.lat_fire=np.array([])
    try:
        ret.lon_fire=lon_fire_obj.get()
    except:
        ret.lon_fire=np.array([])
    try:   
        ret.brig_fire=brig_fire_obj.get()
    except:
        ret.brig_fire=np.array([])
    ret.sat_fire=hdff.Satellite 
    try:
        ret.conf_fire=conf_fire_obj.get()
    except:
        ret.conf_fire=np.array([])
    try:
        ret.t31_fire=t31_fire_obj.get()
    except:
        ret.t31_fire=np.array([])
    try:
        ret.frp_fire=frp_fire_obj.get()
    except:
        ret.frp_fire=np.array([])
    try:
        sf=sample_fire_obj.get()
        # Satellite information
        N=1354 # Number of columns (maxim number of sample)
        h=705. # Altitude of the satellite in km
        p=1. # Nadir pixel resolution in km
        ret.scan_fire,ret.track_fire=pixel_dim(sf,N,h,p)
    except:
        ret.scan_fire=np.array([])
        ret.track_fire=np.array([])

    '''
    s=hdff.attributes()['CoreMetadata.0']
    clon=np.mean(np.array(s.split('GRINGPOINTLONGITUDE')[1].split('(')[1].split(')')[0].split(',')).astype(float))
    clat=np.mean(np.array(s.split('GRINGPOINTLATITUDE')[1].split('(')[1].split(')')[0].split(',')).astype(float))
    ret.scan_angle_fire,ret.scan_fire,ret.track_fire=pixel_scan(coords,center,h)
    '''
    return ret

def read_viirs_files(files):
    """ 
    Read the geolocation (03) and fire (14) files for VIIRS products (VNP)
        :param:
            files   pair with geolocation (03) and fire (14) file names for VIIRS products (VNP)
        :returns:
            ret     dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    h5g=h5py.File(files.geo,'r')
    ret=Dict([])
    ret.lat=np.array(h5g['HDFEOS']['SWATHS']['VNP_750M_GEOLOCATION']['Geolocation Fields']['Latitude'])
    ret.lon=np.array(h5g['HDFEOS']['SWATHS']['VNP_750M_GEOLOCATION']['Geolocation Fields']['Longitude'])
    ncf=nc.Dataset(files.fire,'r')
    ret.fire=np.array(ncf.variables['fire mask'][:])
    ret.lat_fire=np.array(ncf.variables['FP_latitude'][:])
    ret.lon_fire=np.array(ncf.variables['FP_longitude'][:])
    ret.brig_fire=np.array(ncf.variables['FP_T13'][:])
    sf=np.array(ncf.variables['FP_sample'][:])
    # Satellite information
    N=3200 # Number of columns (maxim number of sample)
    h=828. # Altitude of the satellite in km
    p=(0.75+0.75/2+0.75/3)/3 # Nadir pixel resolution in km (mean in 3 different sections)
    ret.scan_fire,ret.track_fire=pixel_dim(sf,N,h,p)
    ret.sat_fire=ncf.SatelliteInstrument
    ret.conf_fire=np.array(ncf.variables['FP_confidence'][:])
    ret.t31_fire=np.array(ncf.variables['FP_T15'][:])
    ret.frp_fire=np.array(ncf.variables['FP_power'][:])
    return ret

def read_data(files,file_metadata):
    """ 
    Read all the geolocation (03) and fire (14) files
        :param files:           list of products with a list of pairs with geolocation (03) and fire (14) file names in the path
        :param file_metadata:   dictionary with file names as key and granules metadata as values
        :returns:
            data            dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) and Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17

    VIIRS file names according to https://lpdaac.usgs.gov/sites/default/files/public/product_documentation/vnp14_user_guide_v1.3.pdf
    VNP14IMG.AYYYYDDD.HHMM.vvv.yyyydddhhmmss.nc
    VNP14.AYYYYDDD.HHMM.vvv.yyyydddhhmmss.nc
    Where:
    YYYYDDD =	year and	Julian	day	(001-366) of	data	acquisition
    HHMM =	hour	and	minute	of	data	acquisition	(approximate	beginning	time)
    vvv =	version	number
    yyyyddd =	year	and	Julian	day	of	data	processing
    hhmmss =	hour,	minute,	and	second	of	data	processing
    """
    print "read_data files=%s" %  files
    data=Dict([])
    for f in files:
        print "read_data f=%s" % f
        lf = len(f)
        if lf != 2:
            print 'ERROR: read_data got %s files using %s' % (lf,f)
            continue
        f0=os.path.basename(f.geo)
        f1=os.path.basename(f.fire)
        prefix = f0[:3] 
        print 'prefix %s' % prefix
        if prefix != f1[:3]:
            print 'ERROR: the prefix of %s %s must coincide' % (f0,f1)
            continue 
        m=f.geo.split('/')
        mm=m[-1].split('.')
        key=mm[1]+'_'+mm[2]
        id = prefix + '_' + key
        print "id " + id
        if prefix=="MOD" or prefix=="MYD":
            item=read_modis_files(f)
            item.instrument="MODIS"
        elif prefix=="VNP":
            item=read_viirs_files(f)
            item.instrument="VIIRS"
        elif prefix=="OR":
            item=read_goes_files(f) # add function "read_goes_files()"
            item.instrument="GOES"
        else:
            print 'ERROR: the prefix of %s %s must be MOD, MYD, or VNP' % (f0,f1)
            continue 
        if (f0 in file_metadata.keys()) and (f1 in file_metadata.keys()):
            # connect the file back to metadata
            item.time_start_geo_iso=file_metadata[f0]["time_start"]
            item.time_num=time_iso2num(item.time_start_geo_iso)
            dt=datetime.strptime(item.time_start_geo_iso[0:18],'%Y-%m-%dT%H:%M:%S')
            item.acq_date='%02d-%02d-%02d' % (dt.year,dt.month,dt.day)
            item.acq_time='%02d%02d' % (dt.hour,dt.minute)
            item.time_start_fire_iso=file_metadata[f1]["time_start"]
            item.time_end_geo_iso=file_metadata[f0]["time_end"]
            item.time_end_fire_iso=file_metadata[f1]["time_end"]
            item.file_geo=f0
            item.file_fire=f1
            item.prefix=prefix
            item.name=key
            data.update({id:item})
        else:
            print 'WARNING: file %s or %s not found in downloaded metadata, ignoring both' % (f0, f1)
            continue

    return data

def download(granules):
    """
    Download files as listed in the granules metadata
        :param: 
            granules        list of products with a list of pairs with geolocation (03) and fire (14) file names in the path  
        :returns: 
            file_metadata   dictionary with file names as key and granules metadata as values
    
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    file_metadata = {} 
    for granule in granules:
        #print json.dumps(granule,indent=4, separators=(',', ': ')) 
        url = granule['links'][0]['href']
        filename=os.path.basename(urlparse.urlsplit(url).path)
        file_metadata[filename]=granule

        # to store as object in memory (maybe not completely downloaded until accessed?)
        # with requests.Session() as s:
        #    data.append(s.get(url))

        # download -  a minimal code without various error checking and corrective actions
        # see wrfxpy/src/ingest/downloader.py
        if os.path.isfile(filename):
            print 'file %s already downloaded' % filename
            continue
        try:
            chunk_size = 1024*1024
            s = 0
            print 'downloading %s as %s' % (url,filename)
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                content_size = int(r.headers['Content-Length'])
                print 'downloading %s as %s size %sB' % (url, filename, content_size)
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)
                        s =  s + len(chunk)
                        print('downloaded %sB  of %sB' % (s, content_size))
            else: 
                print 'cannot connect to %s' % url
                print 'web request status code %s' % r.status_code
                print 'Make sure you have file ~/.netrc permission 600 with the content'
                print 'machine urs.earthdata.nasa.gov\nlogin yourusername\npassword yourpassword' 
                sys.exit(1)
        except Exception as e:
            print 'download failed with error %s' % e 
    return file_metadata
     
def retrieve_af_data(bbox,time):
    """
    Retrieve the data in a bounding box coordinates and time interval and save it in a Matlab structure inside the out.mat Matlab file
        :param: 
            bbox    polygon with the search bounding box
            time    time interval (init_time,final_time)      
        :returns: 
            out.mat Matlab file with all the data in a Matlab structure

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) and Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """

    # Define settings
    lonmin,lonmax,latmin,latmax = bbox
    bbox = [(lonmin,latmax),(lonmin,latmin),(lonmax,latmin),(lonmax,latmax),(lonmin,latmax)]
    maxg = 100

    print "bbox"
    print bbox
    print "time:"
    print time
    print "maxg:"
    print maxg

    # Get data
    granules=get_meta(bbox,time,maxg)
    #print 'medatada found:\n' + json.dumps(granules,indent=4, separators=(',', ': ')) 

    file_metadata = {}
    for k,g in granules.items():
        print 'Downloading %s files' % k
        file_metadata.update(download(g))
        #print "download g:"
        #print g

    print "download complete"

    # Group all files downloaded
    files=group_all(".")
    #print "group all files:"
    #print files

    # Generate data dictionary
    data=Dict({})
    data.update(read_data(files[0],file_metadata))
    data.update(read_data(files[1],file_metadata))
    data.update(read_data(files[2],file_metadata))

    return data

def read_fire_mesh(filename):
    """
    Read necessary variables in the fire mesh of the wrfout file filename
        :param: 
            filename    wrfout file    
        :returns: 
            fxlon,fxlat:    lon and lat coordinates in the fire mesh
            bbox:           bounding box
            time_esmf:      simulation times in ESMF format

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    print 'opening ' + filename
    d = nc.Dataset(filename)
    m,n = d.variables['XLONG'][0,:,:].shape
    fm,fn = d.variables['FXLONG'][0,:,:].shape
    fm=fm-fm/(m+1)    # dimensions corrected for extra strip
    fn=fn-fn/(n+1)
    fxlon = d.variables['FXLONG'][0,:fm,:fn] #  masking  extra strip
    fxlat = d.variables['FXLAT'][0,:fm,:fn]
    tign_g = d.variables['TIGN_G'][0,:fm,:fn]
    time_esmf = ''.join(d.variables['Times'][:][0])  # date string as YYYY-MM-DD_hh:mm:ss
    d.close()
    bbox = [fxlon.min(),fxlon.max(),fxlat.min(),fxlat.max()]
    print 'min max longitude latitude %s'  % bbox
    print 'time (ESMF) %s' % time_esmf

    plot = False
    if plot:
        plot_3D(fxlon,fxlat,tign_g)
    
    return fxlon,fxlat,bbox,time_esmf

def write_csv(data,bounds):
    """ 
    Write fire detections from data dictionary to a CSV file
        :param:
            data    dictionary with Latitude, Longitude and fire mask arrays and metadata information
        :returns: write a fire_detections.csv file

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    d={'latitude': np.concatenate([data[d]['lat_fire'] for d in list(data)]), 
    'longitude': np.concatenate([data[d]['lon_fire'] for d in list(data)]), 
    'brightness': np.concatenate([data[d]['brig_fire'] for d in list(data)]),
    'scan': np.concatenate([data[d]['scan_fire'] for d in list(data)]),
    'track': np.concatenate([data[d]['track_fire'] for d in list(data)]),
    'acq_date': np.concatenate([[data[d]['acq_date']]*len(data[d]['lat_fire']) for d in list(data)]), 
    'acq_time': np.concatenate([[data[d]['acq_time']]*len(data[d]['lat_fire']) for d in list(data)]), 
    'satellite': np.concatenate([[data[d]['sat_fire']]*len(data[d]['lat_fire']) for d in list(data)]), 
    'instrument': np.concatenate([[data[d]['instrument']]*len(data[d]['lat_fire']) for d in list(data)]), 
    'confidence': np.concatenate([data[d]['conf_fire'] for d in list(data)]), 
    'bright_t31': np.concatenate([data[d]['t31_fire'] for d in list(data)]),
    'frp': np.concatenate([data[d]['frp_fire'] for d in list(data)]) }
    df=pd.DataFrame(data=d)
    df=df[(df['longitude']>bounds[0]) & (df['longitude']<bounds[1]) & (df['latitude']>bounds[2]) & (df['latitude']<bounds[3])]
    df.to_csv('fire_detections.csv', encoding='utf-8', index=False)

def plot_3D(xx,yy,zz):
    """
    Plot surface of (xx,yy,zz) data
        :param: 
            xx,yy   x and y arrays
            zz      values at the (x,y) points
        :returns: A plot show of the 3D data

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-09-21
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,zz,cmap=cm.coolwarm)
    plt.show()

def time_iso2num(time_iso):
    """
    Transform an iso time string to a time integer number of seconds since January 1, 1970
        :param: 
            time_iso        string iso date
        :returns: Integer number of seconds since January 1, 1970
    
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    time_datetime=datetime.strptime(time_iso[0:18],'%Y-%m-%dT%H:%M:%S')
    # seconds since January 1, 1970
    return time.mktime(time_datetime.timetuple())

def time_num2iso(time_num):
    """
    Transform a time integer number of seconds since January 1, 1970 to an iso time string
        :param: 
            time_num       Integer number of seconds since January 1, 1970
        :returns: String iso date
    
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-10-01
    """
    dt=datetime.fromtimestamp(time_num)
    # seconds since January 1, 1970
    return '%02d-%02d-%02dT%02d:%02d:%02dZ' % (dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)

def pixel_dim(sample,N,h,p):
    """
    Computes pixel dimensions (along-scan and track pixel sizes)
        :param: 
            sample      Array of integers with the column number (sample variable in files)
            N           Scalar, total number of pixels in each row of the image swath
            h           Scalar, altitude of the satellite in km
            p           Scalar, pixel nadir resolution in km 
        :returns: 
            scan        Along-scan pixel size in km
            track       Along-track pixel size in km
                 
    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH. 
    Angel Farguell (angel.farguell@gmail.com) 2018-10-01
    """
    Re=6378.137 # approximation of the radius of the Earth in km
    s=np.arctan(p/h) # trigonometry (deg/sample)
    r=Re+h
    theta=s*(sample-0.5*(N+1))
    scan=Re*s*(np.cos(theta)/np.sqrt((Re/r)**2-np.square(np.sin(theta)))-1)
    track=r*s*(np.cos(theta)-np.sqrt((Re/r)**2-np.square(np.sin(theta))))
    return (scan,track)

def pixel_scan(lons,lats,center,h):
    Re=6378.137 # approximation of the radius of the Earth in km
    a=np.square(np.sin((center[1]-lats)/2))+np.cos(center[0])*np.cos(lats)*np.square(np.sin((center[0]-lons)/2))
    print a
    c=2*atan2(np.sqrt(a),np.sqrt(1-a))
    print c
    d=Re*c
    print d
    
    scan=Re*s*(np.cos(theta)/np.sqrt((Re/r)**2-np.square(np.sin(theta)))-1)
    track=r*s*(np.cos(theta)-np.sqrt((Re/r)**2-np.square(np.sin(theta))))
    return (theta,scan,track)


if __name__ == "__main__":
    bbox=[-132.86966,-102.0868788,44.002495,66.281204]
    time = ("2012-09-11T00:00:00Z", "2012-09-12T00:00:00Z")
    data=retrieve_af_data(bbox,time)
    # Save the data dictionary into a matlab structure file out.mat
    sio.savemat('out.mat', mdict=data)

