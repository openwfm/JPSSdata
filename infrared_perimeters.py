import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from JPSSD import time_iso2num
import saveload as sl
import re, glob, sys, os

def read_infrared_perimeters(dst,plot=False):
    files = glob.glob(osp.join(dst, '*.kml'))
    perimeters = dict({})
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
                iso_time = date[6:10]+'-'+date[0:2]+'-'+date[3:5]+'T'+date[11:13]+':'+date[13:15]+':00'
                polygons = re.findall(r'<Polygon>(.*?)</Polygon>',f_str,re.DOTALL)
                buckets = [re.split('\r\n\s+',re.findall(r'<coordinates>(.*?)</coordinates>',p,re.DOTALL)[0])[1:] for p in polygons]
                coordinates = [np.array(re.split(',',b)[0:2]).astype(float) for bucket in buckets for b in bucket]
            except Exception as e:
                print 'Error: file %s has not proper structure' % file
                sys.exit(1)
            lons = np.array([coord[0] for coord in coordinates])
            lats = np.array([coord[1] for coord in coordinates])
            if plot:
                plt.plot(lons,lats,'*')
            perimeters.update({iso_time: dict({'file': file, 'lons': lons, 'lats': lats, 'iso_time': iso_time})})
        if plot:
            plt.show()
        sl.save(perimeters,'perimeters')
    else:
        print 'Warning: No KML files in the path specified'
        perimeters = []
    return perimeters

def process_infrared_perimeters(dst,plot=False):
    perim_file = 'perimeters'
    if (osp.isfile(perim_file) and os.access(perim_file,os.R_OK)):
        perimeters = sl.load(perim_file)
    else:
        perimeters = read_infrared_perimeters(dst,plot)
    if perimeters:
        perim = sorted(perimeters.values(), key = lambda x: x['iso_time'])
        X = []
        for p in perim:
            time_num = time_iso2num(p['iso_time'])
            X.append(np.c_[(p['lons'],p['lats'],time_num*np.ones(p['lons'].shape))])
        return np.concatenate(X)
    else:
        print 'Warning: input argument empty'
        return []

if __name__ == "__main__":
    plot = False
    dst = './pioneer_perim'

    X = process_infrared_perimeters(dst,plot)
    print X
