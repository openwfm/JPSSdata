#GOES16 via AWS
Lauren Hearn, 10.2018
Requirements:
- must have rclone installed
- configure new remote via rclone titled 'goes16aws'
- must have subdirectory with rclone in path
'''
from subprocess import Popen, PIPE, STDOUT
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

buckets = Popen('rclone lsd goes16aws:noaa-goes16', cwd='./rclone', shell=True) 
date = raw_input("What date/time would you like to see(please use format <Year>/<Day of Year>/<Hour>)? ")
type(str)
# todo - change date format to be consistent with JPSS/datetime

path = 'goes16aws:noaa-goes16/ABI-L2-MCMIPC/' + date
print path

# get all files for given date/time, (i.e. 2018/262/02)
cmd = 'rclone ls ' + path
print cmd
files = Popen('cmd', cwd='./rclone', shell=True)
process = Popen(cmd, cwd='./rclone', shell=True, stdout=PIPE, stderr=STDOUT)
output = process.communicate()[0]
print 'available files for ' + date + ' are:', output # list available files for the given hour

# copy all files to given local directory
download = Popen('rclone copyto ' + path + ' ./', cwd='./rclone', shell=True)

#now we open files to look around
C_file = './rclone/' + raw_input("What file would you like to see? ") # allow a moment for download to complete
C = Dataset(C_file, 'r')

#will add mapping from https://github.com/blaylockbk/pyBKB_v2/blob/master/BB_GOES16/mapping_GOES16_FireTemperature.ipynb 
