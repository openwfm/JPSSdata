import glob, re, sys, os, pytz, requests
from JPSSD import time_iso2datetime, time_datetime2iso, get_url
from utils import clean_dir
import os.path as osp
from datetime import datetime,timedelta
from tzwhere import tzwhere

if len(sys.argv) != 3:
    print 'Error: python %s firename year' % sys.argv[0]
    sys.exit(1)

firename = sys.argv[1]
year = sys.argv[2]
if year == str(datetime.today().year):
    year = 'current_year'
dst_in = 'perim_orig'
dst_out = 'perim'

clean_dir(dst_in)
clean_dir(dst_out)
baseurl = 'https://rmgsc.cr.usgs.gov/outgoing/GeoMAC/'
url = osp.join(baseurl,year+'_fire_data/KMLS/')
r = requests.get(url, stream=True)
content = r.content
plist = re.findall('([a-z\d\s-]+%s[\d\s-]+.kml)' % firename,content,re.IGNORECASE)
for p in plist:
    get_url(osp.join(url,p),osp.join(dst_in,p))

files = glob.glob(osp.join(dst_in, '*.kml'))
print 'Transforming KML files to UTC from %s to %s' % (dst_in, dst_out)
for k,file in enumerate(files):
    f = open(file,"r")
    f_str = ''.join(f.readlines())
    f.close()
    name = re.findall(r'<name>(.*?)</name>',f_str,re.DOTALL)[0]
    match = re.match(r'(.*) ([0-9]+)-([0-9]+)-([0-9]+) ([0-9]{2})([0-9]{2})',name).groups()
    case = match[0]
    date = (match[3],match[1],match[2],match[4],match[5])
    time_iso = '%04d-%02d-%02dT%02d:%02d:00Z' % tuple([ int(d) for d in date ])
    time_datetime = time_iso2datetime(time_iso)
    # Calculate time zone from lon/lat in the first file
    if not k:
        print 'Computing GMT from coordinates in the first file...'
        coord = re.findall(r'<coordinates>(.*?)</coordinates>',f_str,re.DOTALL)[0]
        lon,lat,_ = map(float,re.findall(r'([-]?[0-9.]+)',coord,re.DOTALL))
        tz = tzwhere.tzwhere(forceTZ=True)
        timezone_str = tz.tzNameAt(lat,lon,forceTZ = True)
        timezone = pytz.timezone(timezone_str)
        gmt = timezone.utcoffset(time_datetime).total_seconds()/3600.
        print 'GMT%d' % gmt
    new_time_datetime = time_datetime + timedelta(hours=-gmt)
    new_time_iso = time_datetime2iso(new_time_datetime)
    new_name = case + ' %02d-%02d-%04d %02d%02d' % (new_time_datetime.month,
                                                    new_time_datetime.day,
                                                    new_time_datetime.year,
                                                    new_time_datetime.hour,
                                                    new_time_datetime.minute)
    new_file = osp.join(dst_out,new_name+'.kml')
    new_f = open(new_file,"w")
    new_f_str = re.sub(r'(?is)<name>(.*?)</name>(?is)', '<name>'+new_name+'</name>', f_str)
    new_f_str = re.sub(r'(?is)</Placemark>(?is)','<TimeStamp><when>'+new_time_iso+'</when></TimeStamp>\n</Placemark>',new_f_str)
    # not always same structure: be careful to loose information
    new_f_str = re.sub(r'(?is)<Placemark>(.*?)</Placemark>(?is)','',new_f_str,1)
    new_f.write(new_f_str)
    new_f.close()
    print '> new file %s created.' % new_file
