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
import datetime
import time
import pandas as pd
import matplotlib.colors as colors
from itertools import groupby
from subprocess import check_output, call

def search_api(sname,bbox,time,maxg=50,platform="",version=""):
    """
    API search of the different satellite granules

    :param sname: short name
    :param bbox: polygon with the search bounding box
    :param time: time interval (init_time,final_time)
    :param maxg: max number of granules to process
    :param platform: string with the platform
    :param version: string with the version
    :return granules: dictionary with the metadata of the search

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

def search_archive(url,prod,time,grans):
    """
    Archive search of the different satellite granules

    :param url: base url of the archive domain
    :param prod: string of product with version, for instance: '5000/VNP09'
    :param time: time interval (init_time,final_time)
    :param grans: granules of the geolocation metadata
    :return granules: dictionary with the metadata of the search

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-01-03
    """
    ids=['.'.join(k['producer_granule_id'].split('.')[1:3]) for k in grans] # satellite ids in bounding box
    granules=[]
    dts=[datetime.datetime.strptime(d,'%Y-%m-%dT%H:%M:%SZ') for d in time]
    delta=dts[1]-dts[0]
    nh=int(delta.total_seconds()/3600)
    dates=[dts[0]+datetime.timedelta(seconds=3600*k) for k in range(1,nh+1)]
    fold=np.unique(['%d/%03d' % (date.timetuple().tm_year,date.timetuple().tm_yday) for date in dates])
    urls=[url+'/'+prod+'/'+f for f in fold]
    for u in urls:
        try:
            js=requests.get(u+'.json').json()
            for j in js:
                arg=np.argwhere(np.array(ids)=='.'.join(j['name'].split('.')[1:3]))
                if arg.size!=0:
                    ar=arg[0][0]
                    g=Dict(j)
                    g.links=[{'href':u+'/'+g.name}]
                    g.time_start=grans[ar]["time_start"]
                    g.time_end=grans[ar]["time_end"]
                    granules.append(g)
        except Exception as e:
            "warning: some JSON request failed"
    print "%s gets %s hits in this range" % (prod.split('/')[-1],len(granules))
    return granules

def get_meta(bbox,time,maxg=50,burned=False):
    """
    Get all the meta data from the API for all the necessary products

    :param bbox: polygon with the search bounding box
    :param time: time interval (init_time,final_time)
    :param maxg: max number of granules to process
    :return granules: dictionary with the metadata of all the products

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    granules=Dict([]);
    #MOD14: MODIS Terra fire data
    granules.MOD14=search_api("MOD14",bbox,time,maxg,"Terra")
    #MOD03: MODIS Terra geolocation data
    granules.MOD03=search_api("MOD03",bbox,time,maxg,"Terra","6")
    #MOD09: MODIS Atmospherically corrected surface reflectance
    #granules.MOD09=search_api("MOD09",bbox,time,maxg,"Terra","6")
    #MYD14: MODIS Aqua fire data
    granules.MYD14=search_api("MYD14",bbox,time,maxg,"Aqua")
    #MYD03: MODIS Aqua geolocation data
    granules.MYD03=search_api("MYD03",bbox,time,maxg,"Aqua","6")
    #MOD09: MODIS Atmospherically corrected surface reflectance
    #granules.MYD09=search_api("MYD09",bbox,time,maxg,"Terra","6")
    #VNP14: VIIRS fire data, res 750m
    granules.VNP14=search_api("VNP14",bbox,time,maxg)
    #VNP03MODLL: VIIRS geolocation data, res 750m
    granules.VNP03=search_api("VNP03MODLL",bbox,time,maxg)
    #VNP14hi: VIIRS fire data, res 375m
    #granules.VNP14hi=search_api("VNP14IMGTDL_NRT",bbox,time,maxg)
    if burned:
        #VNP09: VIIRS Atmospherically corrected surface reflectance
        url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData" # base url
        prod="5000/VNP09" # product
        granules.VNP09=search_archive(url,prod,time,granules.VNP03)
    return granules

def group_files(path,reg):
    """
    Agrupate the geolocation (03) and fire (14) files of a specific product in a path

    :param path: path to the geolocation (03) and fire (14) files
    :param reg: string with the first 3 characters specifying the product (MOD, MYD or VNP)
    :return files: list of pairs with geolocation (03) and fire (14) file names in the path of the specific product

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    files=[Dict({'geo':k}) for k in glob.glob(path+'/'+reg+'03*')]
    filesf=glob.glob(path+'/'+reg+'14*')
    filesr=glob.glob(path+'/'+reg+'09*')
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
    if len(filesr)>0:
        for f in filesr:
            mf=re.split("/",f)
            if mf is not None:
                m=mf[-1].split('.')
                if m is not None:
                    for k,g in enumerate(files):
                        mmf=re.split("/",g.geo)
                        mm=mmf[-1].split('.')
                        if mm[0][1]==m[0][1] and mm[1]+'.'+mm[2]==m[1]+'.'+m[2]:
                            files[k].ref=f
    return files

def group_all(path):
    """
    Combine all the geolocation (03) and fire (14) files in a path

    :param path: path to the geolocation (03) and fire (14) files
    :return files: dictionary of products with a list of pairs with geolocation (03) and fire (14) file names in the path

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    files=Dict({})
    # MOD files
    modf=group_files(path,'MOD')
    # MYD files
    mydf=group_files(path,'MYD')
    # VIIRS files
    vif=group_files(path,'VNP')
    files.MOD=modf
    files.MYD=mydf
    files.VNP=vif
    return files

def read_modis_files(files,bounds):
    """
    Read the geolocation (03) and fire (14) files for MODIS products (MOD or MYD)

    :param files: pair with geolocation (03) and fire (14) file names for MODIS products (MOD or MYD)
    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return ret: dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    ret=Dict([])
    # Satellite information
    N=1354 # Number of columns (maxim number of sample)
    h=705. # Altitude of the satellite in km
    p=1. # Nadir pixel resolution in km
    # Reading MODIS files
    hdfg=SD(files.geo,SDC.READ)
    hdff=SD(files.fire,SDC.READ)
    # Creating all the objects
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
    # Geolocation and mask information
    ret.lat=lat_obj.get()
    ret.lon=lon_obj.get()
    ret.fire=fire_obj.get()
    # Fire detected information
    try:
        flats=lat_fire_obj.get()
    except:
        flats=np.array([])
    try:
        flons=lon_fire_obj.get()
    except:
        flons=np.array([])
    fll=np.logical_and(np.logical_and(np.logical_and(flons >= bounds[0], flons <= bounds[1]), flats >= bounds[2]), flats <= bounds[3])
    ret.lat_fire=flats[fll]
    ret.lon_fire=flons[fll]
    try:
        ret.brig_fire=brig_fire_obj.get()[fll]
    except:
        ret.brig_fire=np.array([])
    ret.sat_fire=hdff.Satellite
    try:
        ret.conf_fire=conf_fire_obj.get()[fll]
    except:
        ret.conf_fire=np.array([])
    try:
        ret.t31_fire=t31_fire_obj.get()[fll]
    except:
        ret.t31_fire=np.array([])
    try:
        ret.frp_fire=frp_fire_obj.get()[fll]
    except:
        ret.frp_fire=np.array([])
    try:
        sf=sample_fire_obj.get()[fll]
    except:
        sf=np.array([])
    ret.scan_angle_fire,ret.scan_fire,ret.track_fire=pixel_dim(sf,N,h,p)
    # No fire data
    lats=np.reshape(ret.lat,np.prod(ret.lat.shape))
    lons=np.reshape(ret.lon,np.prod(ret.lon.shape))
    ll=np.logical_and(np.logical_and(np.logical_and(lons >= bounds[0], lons <= bounds[1]), lats >= bounds[2]), lats <= bounds[3])
    lats=lats[ll]
    lons=lons[ll]
    fire=np.reshape(ret.fire,np.prod(ret.fire.shape))
    fire=fire[ll]
    nf=np.logical_or(fire == 3, fire == 5)
    ret.lat_nofire=lats[nf]
    ret.lon_nofire=lons[nf]
    sample=np.array([range(0,ret.lat.shape[1])]*ret.lat.shape[0])
    sample=np.reshape(sample,np.prod(sample.shape))
    sample=sample[ll]
    sfn=sample[nf]
    ret.scan_angle_nofire,ret.scan_nofire,ret.track_nofire=pixel_dim(sfn,N,h,p)
    # Close files
    hdfg.end()
    hdff.end()
    return ret

def read_viirs_files(files,bounds):
    """
    Read the geolocation (03) and fire (14) files for VIIRS products (VNP)

    :param files: pair with geolocation (03) and fire (14) file names for VIIRS products (VNP)
    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return ret: dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    ret=Dict([])
    # Satellite information
    N=3200 # Number of columns (maxim number of sample)
    h=828. # Altitude of the satellite in km
    alpha=np.array([0,31.59,44.68,56.06])/180*np.pi
    #p=(0.75+0.75/2+0.75/3)/3 # Nadir pixel resolution in km (mean in 3 different sections)
    p=np.array([0.75,0.75/2,0.75/3])
    # Reading VIIRS files
    h5g=h5py.File(files.geo,'r')
    ncf=nc.Dataset(files.fire,'r')
    # Geolocation and mask information
    ret.lat=np.array(h5g['HDFEOS']['SWATHS']['VNP_750M_GEOLOCATION']['Geolocation Fields']['Latitude'])
    ret.lon=np.array(h5g['HDFEOS']['SWATHS']['VNP_750M_GEOLOCATION']['Geolocation Fields']['Longitude'])
    ret.fire=np.array(ncf.variables['fire mask'][:])
    # Fire detected information
    flats=np.array(ncf.variables['FP_latitude'][:])
    flons=np.array(ncf.variables['FP_longitude'][:])
    fll=np.logical_and(np.logical_and(np.logical_and(flons >= bounds[0], flons <= bounds[1]), flats >= bounds[2]),flats <= bounds[3])
    ret.lat_fire=flats[fll]
    ret.lon_fire=flons[fll]
    ret.brig_fire=np.array(ncf.variables['FP_T13'][:])[fll]
    ret.sat_fire=ncf.SatelliteInstrument
    ret.conf_fire=np.array(ncf.variables['FP_confidence'][:])[fll]
    ret.t31_fire=np.array(ncf.variables['FP_T15'][:])[fll]
    ret.frp_fire=np.array(ncf.variables['FP_power'][:])[fll]
    sf=np.array(ncf.variables['FP_sample'][:])[fll]
    ret.scan_angle_fire,ret.scan_fire,ret.track_fire=pixel_dim(sf,N,h,p,alpha)
    # No fire data
    lats=np.reshape(ret.lat,np.prod(ret.lat.shape))
    lons=np.reshape(ret.lon,np.prod(ret.lon.shape))
    ll=np.logical_and(np.logical_and(np.logical_and(lons >= bounds[0], lons <= bounds[1]), lats >= bounds[2]), lats <= bounds[3])
    lats=lats[ll]
    lons=lons[ll]
    fire=np.reshape(ret.fire,np.prod(ret.fire.shape))
    fire=fire[ll]
    nf=np.logical_or(fire == 3, fire == 5)
    ret.lat_nofire=lats[nf]
    ret.lon_nofire=lons[nf]
    sample=np.array([range(0,ret.lat.shape[1])]*ret.lat.shape[0])
    sample=np.reshape(sample,np.prod(sample.shape))
    sample=sample[ll]
    sfn=sample[nf]
    ret.scan_angle_nofire,ret.scan_nofire,ret.track_nofire=pixel_dim(sfn,N,h,p,alpha)
    # Reflectance data for burned scar algorithm
    if 'ref' in files.keys():
        # Read reflectance data
        hdf=SD(files.ref,SDC.READ)
        # Bands data
        M7=hdf.select('750m Surface Reflectance Band M7') # 0.86 nm
        M8=hdf.select('750m Surface Reflectance Band M8') # 1.24 nm
        M10=hdf.select('750m Surface Reflectance Band M10') # 1.61 nm
        M11=hdf.select('750m Surface Reflectance Band M11') # 2.25 nm
        bands=Dict({})
        bands.M7=M7.get()*1e-4
        bands.M8=M8.get()*1e-4
        bands.M10=M10.get()*1e-4
        bands.M11=M11.get()*1e-4
        # Burned scar mask using the burned scar granule algorithm
        ret.burned=burned_algorithm(bands)
        # Close reflectance file
        hdf.end()
    # Close files
    h5g.close()
    ncf.close()
    return ret

def read_viirs375_files(path,bounds):
    """
    Read the geolocation and fire information from VIIRS CSV files (fire_archive_*.csv and/or fire_nrt_*.csv)

    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return ret: dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-10-23
    """
    # Opening files if they exist
    f1=glob.glob(path+'/fire_archive_*.csv')
    f2=glob.glob(path+'/fire_nrt_*.csv')
    if len(f1)==1:
        df1=pd.read_csv(f1[0])
        if len(f2)==1:
            df2=pd.read_csv(f2[0])
            dfs=pd.concat([df1,df2],sort=True,ignore_index=True)
        else:
            dfs=df1
    else:
        if len(f2)==1:
            dfs=pd.read_csv(f2[0])
        else:
            return {}

    ret=Dict({})
    # In the case something exists, read all the information from the CSV files
    dfs=dfs[(dfs['longitude']>bounds[0]) & (dfs['longitude']<bounds[1]) & (dfs['latitude']>bounds[2]) & (dfs['latitude']<bounds[3])]
    date=np.array(dfs['acq_date'])
    time=np.array(dfs['acq_time'])
    dfs['time']=np.array(['%s_%04d' % (date[k],time[k]) for k in range(len(date))])
    dfs['time']=pd.to_datetime(dfs['time'], format='%Y-%m-%d_%H%M')
    dfs['datetime']=dfs['time']
    dfs=dfs.set_index('time')
    for group_name, df in dfs.groupby(pd.TimeGrouper("D")):
        items=Dict([])
        items.lat=np.array(df['latitude'])
        items.lon=np.array(df['longitude'])
        conf=np.array(df['confidence'])
        firemask=np.zeros(conf.shape)
        conf_fire=np.zeros(conf.shape)
        firemask[conf=='l']=7
        conf_fire[conf=='l']=30.
        firemask[conf=='n']=8
        conf_fire[conf=='n']=60.
        firemask[conf=='h']=9
        conf_fire[conf=='h']=90.
        items.fire=firemask.astype(int)
        items.lat_fire=items.lat
        items.lon_fire=items.lon
        items.brig_fire=np.array(df['bright_ti4'])
        items.sat_fire='Suomi NPP'
        items.conf_fire=conf_fire
        items.t31_fire=np.array(df['bright_ti5'])
        items.frp_fire=np.array(df['frp'])
        items.scan_fire=np.array(df['scan'])
        items.track_fire=np.array(df['track'])
        items.scan_angle_fire=np.ones(items.scan_fire.shape)*np.nan
        items.lat_nofire=np.array([])
        items.lon_nofire=np.array([])
        items.scan_angle_nofire=np.array([])
        items.scan_nofire=np.array([])
        items.track_nofire=np.array([])
        items.instrument=df['instrument'][0]
        dt=df['datetime'][0]
        items.time_start_geo_iso='%02d-%02d-%02dT%02d:%02d:%02dZ' % (dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
        items.time_num=time_iso2num(items.time_start_geo_iso)
        items.acq_date='%02d-%02d-%02d' % (dt.year,dt.month,dt.day)
        items.acq_time='%02d%02d' % (dt.hour,dt.minute)
        items.time_start_fire_iso=items.time_start_geo_iso
        items.time_end_geo_iso=items.time_start_geo_iso
        items.time_end_fire_iso=items.time_start_geo_iso
        items.file_geo=f1+f2
        items.file_fire=items.file_geo
        tt=df['datetime'][0].timetuple()
        id='VNPH_A%04d%03d_%02d%02d' % (tt.tm_year,tt.tm_yday,tt.tm_hour,tt.tm_min)
        items.prefix='VNPH'
        items.name='A%04d%03d_%02d%02d' % (tt.tm_year,tt.tm_yday,tt.tm_hour,tt.tm_min)
        ret.update({id: items})
    return ret

def read_goes_files(files):
    """
    Read the files for GOES products - geolocation and fire data already included (OS)

     remove :param files: pair with geolocation (03) and fire (14) file names for MODIS products (MOD or MYD)
    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return ret: dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on WINDOWS10.
    Lauren Hearn (lauren@robotlauren.com), 2018-10-16
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
    N=2500 # Number of columns (maxim number of sample)
    h=35786. # Altitude of the satellite in km
    p=2. # Nadir pixel resolution in km
    ret.scan_fire,ret.track_fire=pixel_dim(sf,N,h,p)
    ret.sat_fire=ncf.SatelliteInstrument
    ret.conf_fire=np.array(ncf.variables['FP_confidence'][:])
    ret.t31_fire=np.array(ncf.variables['FP_T15'][:])
    ret.frp_fire=np.array(ncf.variables['FP_power'][:])
    return ret

def read_data(files,file_metadata,bounds):
    """
    Read all the geolocation (03) and fire (14) files and if necessary, the reflectance (09) files

    MODIS file names according to https://lpdaac.usgs.gov/sites/default/files/public/product_documentation/archive/mod14_v5_user_guide.pdf
    MOD14.AYYYYDDD.HHMM.vvv.yyyydddhhmmss.hdf
    MYD14.AYYYYDDD.HHMM.vvv.yyyydddhhmmss.hdf
    Where:
    YYYYDDD =   year and    Julian  day (001-366) of    data    acquisition
    HHMM =  hour    and minute  of  data    acquisition (approximate    beginning   time)
    vvv =   version number
    yyyyddd =   year    and Julian  day of  data    processing
    hhmmss =    hour,   minute, and second  of  data    processing

    VIIRS file names according to https://lpdaac.usgs.gov/sites/default/files/public/product_documentation/vnp14_user_guide_v1.3.pdf
    VNP14IMG.AYYYYDDD.HHMM.vvv.yyyydddhhmmss.nc
    VNP14.AYYYYDDD.HHMM.vvv.yyyydddhhmmss.nc
    Where:
    YYYYDDD =   year and    Julian  day (001-366) of    data    acquisition
    HHMM =  hour    and minute  of  data    acquisition (approximate    beginning   time)
    vvv =   version number
    yyyyddd =   year    and Julian  day of  data    processing
    hhmmss =    hour,   minute, and second  of  data    processing

    :param files: list of products with a list of pairs with geolocation (03) and fire (14) file names in the path
    :param file_metadata: dictionary with file names as key and granules metadata as values
    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return data: dictionary with Latitude, Longitude and fire mask arrays read

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) and Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    print "read_data files=%s" %  files
    data=Dict([])
    if files=='VIIRS375':
        data.update(read_viirs375_files('.',bounds))
    else:
        for f in files:
            print "read_data f=%s" % f
            if 'geo' in f.keys():
                f0=os.path.basename(f.geo)
            else:
                print 'ERROR: read_data cannot read files=%s, not geo file' % f
                sys.exit(1)
            if 'fire' in f.keys():
                f1=os.path.basename(f.fire)
            else:
                print 'ERROR: read_data cannot read files=%s, not geo file' % f
                sys.exit(1)
            boo=True
            if 'ref' in f.keys():
                f2=os.path.basename(f.ref)
                boo=False
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
                item=read_modis_files(f,bounds)
                item.instrument="MODIS"
            elif prefix=="VNP":
                item=read_viirs_files(f,bounds)
                item.instrument="VIIRS"
            elif prefix=="OR":
                item=read_goes_files(f)
                item.instrument="GOES"
            else:
                print 'ERROR: the prefix of %s %s must be MOD, MYD, or VNP' % (f0,f1)
                continue
            if not boo:
                if f2 in file_metadata.keys():
                    boo=True
            if (f0 in file_metadata.keys()) and (f1 in file_metadata.keys()) and boo:
                # connect the file back to metadata
                item.time_start_geo_iso=file_metadata[f0]["time_start"]
                item.time_num=time_iso2num(item.time_start_geo_iso)
                dt=datetime.datetime.strptime(item.time_start_geo_iso[0:18],'%Y-%m-%dT%H:%M:%S')
                item.acq_date='%02d-%02d-%02d' % (dt.year,dt.month,dt.day)
                item.acq_time='%02d%02d' % (dt.hour,dt.minute)
                item.time_start_fire_iso=file_metadata[f1]["time_start"]
                item.time_end_geo_iso=file_metadata[f0]["time_end"]
                item.time_end_fire_iso=file_metadata[f1]["time_end"]
                item.file_geo=f0
                item.file_fire=f1
                if 'ref' in f.keys():
                    item.file_ref=f2
                    item.time_start_ref_iso=file_metadata[f2]["time_start"]
                    item.time_end_ref_iso=file_metadata[f2]["time_end"]
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

    :param granules: list of products with a list of pairs with geolocation (03) and fire (14) file names in the path
    :return file_metadata: dictionary with file names as key and granules metadata as values

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

def download_GOES16(time):
    """
    Download the GOES16 data through rclone application

    :param time: tupple with the start and end times
    :return bucket: dictionary of all the data downloaded and all the GOES16 data downloaded in the same directory level

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) 2018-10-12
    """
    bucket=Dict({})
    dts=[datetime.datetime.strptime(d,'%Y-%m-%dT%H:%M:%SZ') for d in time]
    delta=dts[1]-dts[0]
    nh=int(delta.total_seconds()/3600)
    dates=[dts[0]+datetime.timedelta(seconds=3600*k) for k in range(1,nh+1)]
    for date in dates:
        tt=date.timetuple()
        argT='%d/%03d/%02d' % (tt.tm_year,tt.tm_yday,tt.tm_hour)
        try:
            args=['rclone','copyto','goes16aws:noaa-goes16/ABI-L2-MCMIPC/'+argT,'.','-L']
            print 'running: '+' '.join(args)
            res=call(args)
            print 'goes16aws:noaa-goes16/ABI-L2-MCMIPC/'+argT+' downloaded.'
            args=['rclone','ls','goes16aws:noaa-goes16/ABI-L2-MCMIPC/'+argT,'-L']
            out=check_output(args)
            bucket.update({argT : [o.split(' ')[2] for o in out.split('\n')[:-1]]})
        except Exception as e:
            print 'download failed with error %s' % e
    return bucket

def retrieve_af_data(bbox,time,burned=False):
    """
    Retrieve the data in a bounding box coordinates and time interval and save it in a Matlab structure inside the out.mat Matlab file

    :param bbox: polygon with the search bounding box
    :param time: time interval (init_time,final_time)
    :return data: dictonary with all the data and out.mat Matlab file with a Matlab structure of the dictionary

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) and Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """

    # Define settings
    lonmin,lonmax,latmin,latmax = bbox
    bounds=bbox
    bbox = [(lonmin,latmax),(lonmin,latmin),(lonmax,latmin),(lonmax,latmax),(lonmin,latmax)]
    maxg = 1000

    print "bbox"
    print bbox
    print "time:"
    print time
    print "maxg:"
    print maxg

    # Get data
    granules=get_meta(bbox,time,maxg,burned=burned)
    #print 'medatada found:\n' + json.dumps(granules,indent=4, separators=(',', ': '))

    # Eliminating the NRT data (repeated always)
    nrt_elimination(granules)

    file_metadata = {}
    for k,g in granules.items():
        print 'Downloading %s files' % k
        sys.stdout.flush()
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
    data.update(read_data(files.MOD,file_metadata,bounds))
    data.update(read_data(files.MYD,file_metadata,bounds))
    data.update(read_data(files.VNP,file_metadata,bounds))
    #data.update(read_data('VIIRS375','',bounds))

    return data

def nrt_elimination(granules):
    """
    Cleaning all the NRT data which is repeated

    :param granules: Dictionary of granules products to clean up
    :return: It will update the granules dictionary

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) and Jan Mandel (jan.mandel@ucdenver.edu) 2018-11-30
    """

    if 'MOD14' in granules:
    	nlist=[g for g in granules['MOD14'] if g['data_center']=='LPDAAC_ECS']
   	granules['MOD14']=nlist
    if 'MYD14' in granules:
    	nlist=[g for g in granules['MYD14'] if g['data_center']=='LPDAAC_ECS']
    	granules['MYD14']=nlist


def read_fire_mesh(filename):
    """
    Read necessary variables in the fire mesh of the wrfout file filename

    :param filename: wrfout file
    :return fxlon: lon coordinates in the fire mesh
    :return fxlat: lat coordinates in the fire mesh
    :return bbox: bounding box
    :return time_esmf: simulation times in ESMF format

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

def data2json(data,keys,dkeys,N):
    """
    Create a json dictionary from data dictionary

    :param data: dictionary with Latitude, Longitude and fire mask arrays and metadata information
    :param keys: keys which are going to be included into the json
    :param dkeys: keys in the data dictionary which correspond to the previous keys (same order)
    :param N: number of entries in each instance of the json dictionary (used for the variables with only one entry in the data dictionary)
    :return ret: dictionary with all the fire detection information to create the KML file

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    ret=Dict({})
    for i,k in enumerate(keys):
        if isinstance(data[list(data)[0]][dkeys[i]],(list, tuple, np.ndarray)):
            dd=[np.reshape(data[d][dkeys[i]],np.prod(data[d][dkeys[i]].shape)) for d in list(data)]
            ret.update({k : np.concatenate(dd)})
        else:
            dd=[[data[d[1]][dkeys[i]]]*N[d[0]] for d in enumerate(list(data))]
            ret.update({k : np.concatenate(dd)})

    return ret

def sdata2json(sdata,keys,dkeys,N):
    """
    Create a json dictionary from sorted array of data dictionaries

    :param sdata: sorted array of data dictionaries with Latitude, Longitude and fire mask arrays and metadata information
    :param keys: keys which are going to be included into the json
    :param dkeys: keys in the data dictionary which correspond to the previous keys (same order)
    :param N: number of entries in each instance of the json dictionary (used for the variables with only one entry in the data dictionary)
    :return ret: dictionary with all the fire detection information to create the KML file

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-12-03
    """
    ret=Dict({'granules': [d[0] for d in sdata]})
    for i,k in enumerate(keys):
        dd = [d[1][dkeys[i]] if dkeys[i] in d[1] else None for d in sdata]
        if dd:
            if np.any([isinstance(d,(list, tuple, np.ndarray)) for d in dd]):
                out = [d if d is not None else np.array([]) for d in dd]
                ret.update({k : dd})
            else:
                out = [[d[1][1][dkeys[i]]]*N[d[0]] if dkeys[i] in d[1][1] else [] for d in enumerate(sdata)]
                ret.update({k : out})

    return ret


def write_csv(d,bounds):
    """
    Write fire detections from data dictionary d to a CSV file

    :param d: dictionary with Latitude, Longitude and fire mask arrays and metadata information
    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return: fire_detections.csv file with all the detections

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2018-09-17
    """
    df=pd.DataFrame(data=d)
    df=df[(df['longitude']>bounds[0]) & (df['longitude']<bounds[1]) & (df['latitude']>bounds[2]) & (df['latitude']<bounds[3])]
    df.to_csv('fire_detections.csv', encoding='utf-8', index=False)

def plot_3D(xx,yy,zz):
    """
    Plot surface of (xx,yy,zz) data

    :param xx: x arrays
    :param yy: y arrays
    :param zz: values at the (x,y) points
    :return: a plot show of the 3D data

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
    Transform an iso time string to a time integer number of seconds since December 31 1969 at 17:00:00

    :param time_iso: string iso date
    :return s: integer number of seconds since December 31 1969 at 17:00:00

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    time_datetime=datetime.datetime.strptime(time_iso[0:19],'%Y-%m-%dT%H:%M:%S')
    # seconds since December 31 1969 at 17:00:00
    s=time.mktime(time_datetime.timetuple())
    return s

def time_iso2datetime(time_iso):
    """
    Transform an iso time string to a datetime element

    :param time_iso: string iso date
    :return time_datetime: datetime element

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    time_datetime=datetime.datetime.strptime(time_iso[0:19],'%Y-%m-%dT%H:%M:%S')
    return time_datetime

def time_datetime2iso(time_datetime):
    """
    Transform a datetime element to iso time string

    :param time_datetime: datetime element
    :return time_iso: string iso date

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) 2018-10-01
    """
    time_iso='%02d-%02d-%02dT%02d:%02d:%02dZ' % (time_datetime.year,time_datetime.month,
                                                    time_datetime.day,time_datetime.hour,
                                                    time_datetime.minute,time_datetime.second)
    return time_iso

def time_num2iso(time_num):
    """
    Transform a time integer number of seconds since December 31 1969 at 17:00:00 to an iso time string

    :param time_num: integer number of seconds since December 31 1969 at 17:00:00
    :return date: time string in ISO date

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) 2018-10-01
    """
    dt=datetime.datetime.fromtimestamp(time_num)
    # seconds since December 31 1969 at 17:00:00
    date='%02d-%02d-%02dT%02d:%02d:%02dZ' % (dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
    return date

def pixel_dim(sample,N,h,p,a=None):
    """
    Computes pixel dimensions (along-scan and track pixel sizes)

    :param sample: array of integers with the column number (sample variable in files)
    :param N: scalar, total number of pixels in each row of the image swath
    :param h: scalar, altitude of the satellite in km
    :param p: scalar, pixel nadir resolution in km
    :param a: array of floats of the size of p with the angles where the resolution change
    :return theta: scan angle in radiands
    :return scan: along-scan pixel size in km
    :return track: along-track pixel size in km

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) 2018-10-01
    """
    Re=6378 # approximation of the radius of the Earth in km
    r=Re+h
    M=(N-1)*0.5
    s=np.arctan(p/h) # trigonometry (deg/sample)
    if isinstance(p,(list, tuple, np.ndarray)):
        Ns=np.array([int((a[k]-a[k-1])/s[k-1]) for k in range(1,len(a)-1)])
        Ns=np.append(Ns,int(M-Ns.sum()))
        theta=s[0]*(sample-M)
        scan=Re*s[0]*(np.cos(theta)/np.sqrt((Re/r)**2-np.square(np.sin(theta)))-1)
        track=r*s[0]*(np.cos(theta)-np.sqrt((Re/r)**2-np.square(np.sin(theta))))
        for k in range(1,len(Ns)):
            p=sample<=M-Ns[0:k].sum()
            theta[p]=s[k]*(sample[p]-(M-Ns[0:k].sum()))-(s[0:k]*Ns[0:k]).sum()
            scan[p]=Re*np.mean(s)*(np.cos(theta[p])/np.sqrt((Re/r)**2-np.square(np.sin(theta[p])))-1)
            track[p]=r*np.mean(s)*(np.cos(theta[p])-np.sqrt((Re/r)**2-np.square(np.sin(theta[p]))))
            p=sample>=M+Ns[0:k].sum()
            theta[p]=s[k]*(sample[p]-(M+Ns[0:k].sum()))+(s[0:k]*Ns[0:k]).sum()
            scan[p]=Re*np.mean(s)*(np.cos(theta[p])/np.sqrt((Re/r)**2-np.square(np.sin(theta[p])))-1)
            track[p]=r*np.mean(s)*(np.cos(theta[p])-np.sqrt((Re/r)**2-np.square(np.sin(theta[p]))))
    else:
        theta=s*(sample-M)
        scan=Re*s*(np.cos(theta)/np.sqrt((Re/r)**2-np.square(np.sin(theta)))-1)
        track=r*s*(np.cos(theta)-np.sqrt((Re/r)**2-np.square(np.sin(theta))))
    return (theta,scan,track)

def copyto(partial_path,kml):
    """
    Copy information in partial_path to kml

    :param partial_path: path to a partial KML file
    :param kml: KML object where to write to
    :return: information from partial_path into kml

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """
    with open(partial_path,'r') as part:
        for line in part:
            kml.write(line)

def json2kml(d,kml_path,bounds,prods,opt='granule'):
    """
    Creates a KML file kml_path from a dictionary d

    :param d: dictionary with all the fire detection information to create the KML file
    :param kml_path: path in where the KML file is going to be written
    :param bounds: spatial bounds tuple (lonmin,lonmax,latmin,latmax)
    :return: a KML file

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Jan Mandel (jan.mandel@ucdenver.edu) 2018-09-17
    """

    if 'latitude' in d:
        if not isinstance(d['latitude'][0],(list, tuple, np.ndarray)):
            if opt=='dates':
                ind=[i[0] for i in sorted(enumerate(d.acq_date), key=lambda x:x[1])]
                L=[len(list(grp)) for k, grp in groupby(d['acq_date'][ind])]
                L.insert(0,0)
                ll=[sum(L[0:k+1]) for k in range(len(L))]
                for v in list(d):
                    sort=[d[v][i] for i in ind]
                    d[v]=[sort[ll[k]:ll[k+1]] for k in range(len(ll)-1)]
            else:
                for v in list(d):
                    d[v]=[d[v]]
                opt='pixels'

        frp_style={-1:'modis_frp_no_data',40:'modis_frp_gt_400'}
        for i in range(0,40):
            frp_style[i]='modis_frp_%s_to_%s' % (i*10, i*10+10)

        with open(kml_path,'w') as kml:

            copyto('kmls/partial1.kml',kml)

            # set some constants
            r = 6378   # Earth radius
            km_lat = 180/(np.pi*r)  # 1km in degrees latitude
            ND = len(d['latitude'])

            for prod in prods:

                kml.write('<Folder>\n')
                kml.write('<name>%s</name>\n' % prods[prod])

                if prod == 'FRP':
                    copyto('kmls/partial2.kml',kml)

                if prod == 'TF':
                    col = np.flip(np.divide([(230, 25, 75, 150), (245, 130, 48, 150), (255, 255, 25, 150),
                            (210, 245, 60, 150), (60, 180, 75, 150), (70, 240, 240, 150),
                            (0, 0, 128, 150), (145, 30, 180, 150), (240, 50, 230, 150),
                            (128, 128, 128, 150)],255.),0)
                    cm = colors.LinearSegmentedColormap.from_list('BuRd',col,ND)
                    cols = ['%02x%02x%02x%02x' % tuple(255*np.flip(c)) for c in cm(range(cm.N))]
                    t_range = range(ND-1,-1,-1)
                else:
                    t_range = range(ND)

                for t in t_range:
                    lats=np.array(d['latitude'][t]).astype(float)
                    lons=np.array(d['longitude'][t]).astype(float)
                    ll=np.logical_and(np.logical_and(np.logical_and(lons >= bounds[0], lons <= bounds[1]), lats >= bounds[2]), lats <= bounds[3])
                    latitude=lats[ll]
                    longitude=lons[ll]
                    NN=len(latitude)
                    if NN > 0:
                        acq_date=np.array(d['acq_date'][t])[ll]
                        acq_time=np.array(d['acq_time'][t])[ll]
                        try:
                            satellite=np.array(d['satellite'][t])[ll]
                        except:
                            satellite=np.array(['Not available']*NN)
                        try:
                            instrument=np.array(d['instrument'][t])[ll]
                        except:
                            instrument=np.array(['Not available']*NN)
                        try:
                            confidence=np.array(d['confidence'][t])[ll].astype(float)
                        except:
                            confidence=np.array(np.zeros(NN)).astype(float)
                        try:
                            frps=np.array(d['frp'][t])[ll].astype(float)
                        except:
                            frps=np.array(np.zeros(NN)).astype(float)
                        try:
                            angles=np.array(d['scan_angle'][t])[ll].astype(float)
                        except:
                            angles=np.array(['Not available']*NN)
                        try:
                            scans=np.array(d['scan'][t])[ll].astype(float)
                        except:
                            scans=np.ones(NN).astype(float)
                        try:
                            tracks=np.array(d['track'][t])[ll].astype(float)
                        except:
                            tracks=np.ones(NN).astype(float)

                        kml.write('<Folder>\n')
                        if opt=='date':
                            kml.write('<name>%s</name>\n' % acq_date[0])
                        elif opt=='granule':
                            kml.write('<name>%s</name>\n' % d['granules'][t])
                        else:
                            kml.write('<name>Pixels</name>\n')

                        for p in range(NN):
                            lat=latitude[p]
                            lon=longitude[p]
                            conf=confidence[p]
                            frp=frps[p]
                            angle=angles[p]
                            scan=scans[p]
                            track=tracks[p]
                            timestamp=acq_date[p] + 'T' + acq_time[p][0:2] + ':' + acq_time[p][2:4] + 'Z'
                            timedescr=acq_date[p] + ' ' + acq_time[p][0:2] + ':' + acq_time[p][2:4] + ' UTC'

                            if prod == 'NF':
                                kml.write('<Placemark>\n<name>Ground detection square</name>\n')
                                kml.write('<description>\nlongitude:   %s\n' % lon
                                                      +  'latitude:    %s\n' % lat
                                                      +  'time:        %s\n' % timedescr
                                                      +  'satellite:   %s\n' % satellite[p]
                                                      +  'instrument:  %s\n' % instrument[p]
                                                      +  'scan angle:  %s\n' % angle
                                                      +  'along-scan:  %s\n' % scan
                                                      +  'along-track: %s\n' % track
                                        + '</description>\n')
                            else:
                                kml.write('<Placemark>\n<name>Fire detection square</name>\n')
                                kml.write('<description>\nlongitude:   %s\n' % lon
                                                      +  'latitude:    %s\n' % lat
                                                      +  'time:        %s\n' % timedescr
                                                      +  'satellite:   %s\n' % satellite[p]
                                                      +  'instrument:  %s\n' % instrument[p]
                                                      +  'confidence:  %s\n' % conf
                                                      +  'FRP:         %s\n' % frp
                                                      +  'scan angle:  %s\n' % angle
                                                      +  'along-scan:  %s\n' % scan
                                                      +  'along-track: %s\n' % track
                                        + '</description>\n')
                            kml.write('<TimeStamp><when>%s</when></TimeStamp>\n' % timestamp)
                            if prod == 'AF':
                                if conf < 30:
                                    kml.write('<styleUrl> modis_conf_low </styleUrl>\n')
                                elif conf < 80:
                                    kml.write('<styleUrl> modis_conf_med </styleUrl>\n')
                                else:
                                    kml.write('<styleUrl> modis_conf_high </styleUrl>\n')
                            elif prod == 'TF':
                                kml.write('<Style>\n'+'<PolyStyle>\n'
                                            +'<color>%s</color>\n' % cols[t]
                                            +'<outline>0</outline>\n'+'</PolyStyle>\n'
                                            +'</Style>\n')
                            elif prod == 'FRP':
                                frpx = min(40,np.ceil(frp/10.)-1)
                                kml.write('<styleUrl> %s </styleUrl>\n' % frp_style[frpx] )
                            elif prod == 'NF':
                                kml.write('<styleUrl> no_fire </styleUrl>\n')

                            kml.write('<Polygon>\n<outerBoundaryIs>\n<LinearRing>\n<coordinates>\n')

                            km_lon=km_lat/np.cos(lat*np.pi/180)  # 1 km in longitude

                            sq_track_size_km=track
                            sq2_lat=km_lat * sq_track_size_km/2
                            sq_scan_size_km=scan
                            sq2_lon=km_lon * sq_scan_size_km/2

                            kml.write('%s,%s,0\n' % (lon - sq2_lon, lat - sq2_lat))
                            kml.write('%s,%s,0\n' % (lon - sq2_lon, lat + sq2_lat))
                            kml.write('%s,%s,0\n' % (lon + sq2_lon, lat + sq2_lat))
                            kml.write('%s,%s,0\n' % (lon + sq2_lon, lat - sq2_lat))
                            kml.write('%s,%s,0\n' % (lon - sq2_lon, lat - sq2_lat))

                            kml.write('</coordinates>\n</LinearRing>\n</outerBoundaryIs>\n</Polygon>\n</Placemark>\n')
                        kml.write('</Folder>\n')

                kml.write('</Folder>\n')

            kml.write('</Document>\n</kml>\n')

        print 'Created file %s' % kml_path
    else:
        print 'Any detections to be saved as %s' % kml_path

def burned_algorithm(data):
    """
    Computes mask of burned scar pixels

    :param data: data dictionary with all the necessary bands M7, M8, M10 and M11
    :return C: Mask of burned scar pixels

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com) 2019-01-03
    """
    # Parameters
    RthSub=0.05
    Rth=0.81
    M07UB=0.19
    M08LB=0.00
    M08UB=0.28
    M10LB=0.07
    M10UB=1.00
    M11LB=0.05
    eps=1e-30
    # Bands
    M7=data.M7
    M8=data.M8
    M10=data.M10
    M11=data.M11
    # Eq 1a
    M=(M8.astype(float)-RthSub)/(M11.astype(float)+eps)
    C1=np.logical_and(M>0,M<Rth)
    # Eq 1b
    C2=np.logical_and(M8>M08LB,M8<M08UB)
    # Eq 1c
    C3=M7<M07UB
    # Eq 1d
    C4=M11>M11LB
    # Eq 1e
    C5=np.logical_and(M10>M10LB,M10<M10UB)
    # All the conditions at the same time
    C=np.logical_and(np.logical_and(np.logical_and(np.logical_and(C1,C2),C3),C4),C5)
    return C

if __name__ == "__main__":
    bbox=[-132.86966,-102.0868788,44.002495,66.281204]
    time = ("2012-09-11T00:00:00Z", "2012-09-12T00:00:00Z")
    data=retrieve_af_data(bbox,time)
    # Save the data dictionary into a matlab structure file out.mat
    sio.savemat('out.mat', mdict=data)

