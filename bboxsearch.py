## Search/Download - sandbox
from cmr import CollectionQuery, GranuleQuery
import json
import requests
import urlparse
import os
import sys

api = GranuleQuery()
fire = GranuleQuery()

#MOD14: bounding box = Colorado (gps coord sorted lat,lon; counterclockwise)
MOD14granules = api.parameters(
                        short_name="MOD14",
                        platform="Terra",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2017-01-01T00:00:00Z", "2017-01-07T00:00:00Z") #time start,end
                        )
print "MOD14 gets %s hits in this range" % MOD14granules.hits()
MOD14granules = api.get(10)

#MOD03: geoloc data for MOD14
MOD03granules = api.parameters(
                        short_name="MOD03",
                        platform="Terra",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2017-01-01T00:00:00Z", "2017-01-07T00:00:00Z") #time start,end
                        )
print "MOD03 gets %s hits in this range" % MOD03granules.hits()
MOD03granules = api.get(10)

#VNP14: fire data, res 750m
VNP14granules = fire.parameters(
                        short_name="VNP14",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2017-01-01T00:00:00Z", "2017-01-07T00:00:00Z") #time start,end
                        )
print "VNP14 gets %s hits in this range" % VNP14granules.hits()
VNP14granules = fire.get(10)

#VNP14IMGTDL_NRT: granules with resolution 375m
VNP14hiresgranules = fire.parameters(
                        short_name="VNP14IMGTDL_NRT",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2017-01-01T00:00:00Z", "2017-01-07T00:00:00Z") #time start,end
                        )
print "VNP14(hi-res) gets %s hits in this range" % VNP14hiresgranules.hits()
VNP14hiresgranules = fire.get(10)

#VNP03MODLL: geoloc data for VNP14
VNP03granules = fire.parameters(
                        short_name="VNP03MODLL",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2017-01-01T00:00:00Z", "2017-01-07T00:00:00Z") #time start,end
                        )
print "VNP03 gets %s hits in this range" % VNP03granules.hits()
VNP03granules = fire.get(10)

data = []
def download(granules):
    for granule in granules:
        url = granule['links'][0]['href']
        filename=os.path.basename(urlparse.urlsplit(url).path)

        # to store as object in memory (maybe not completely downloaded until accessed?)
        # with requests.Session() as s:
        #    data.append(s.get(url))

        # download -  a minimal code without various error checking and corrective actions
        # see wrfxpy/src/ingest/downloader.py
        try:
            chunk_size = 1024*1024
            s = 0
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

'''
BE CAREFUL!! - the script below triggers automatic download of VERY
LARGE .hdf files!
'''            
            
#MOD14 = download(MOD14granules)
MOD03 = download(MOD03granules)
VNP14 = download(VNP14granules)
VNP14hires = download(VNP14hiresgranules)
VNP03 = download(VNP03granules)