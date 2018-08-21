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
                        temporal=("2016-10-10T01:02:00Z", "2016-10-12T00:00:30Z") #time start,end
                        )
print MOD14granules.hits()
MOD14granules = api.get(10)

#VNP03MODLL: bounding box = Colorado (gps coord sorted lat,lon; counterclockwise)
VIIRSgranules = fire.parameters(
                        short_name="VNP03MODLL",
                        downloadable=True,
                        polygon=[(-109.0507527,40.99898), (-109.0698568,37.0124375), (-102.0868788,36.9799819),(-102.0560592,40.999126), (-109.0507527,40.99898)],
                        temporal=("2016-10-10T01:02:00Z", "2016-10-12T00:00:30Z") #time start,end
                        )
print VIIRSgranules.hits()
VIIRSgranules = fire.get(10)

data = []
for granule in MOD14granules:
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