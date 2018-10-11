#GOES16 via AWS
<<<<<<< HEAD
'''
=======
>>>>>>> 6865887e9899dda98cbc28b01115267dc3a5c30a
Lauren Hearn, 10.2018
Requirements:
- must have rclone installed
- configure new remote via rclone titled 'goes16aws'
- must have subdirectory with rclone in path
'''
<<<<<<< HEAD

from subprocess import call, PIPE, STDOUT
=======
from subprocess import Popen, PIPE, STDOUT
>>>>>>> 6865887e9899dda98cbc28b01115267dc3a5c30a
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
args = ['rclone', 'lsd', 'ls', 'goes16aws:noaa-goes16', './rclone']

buckets = call([args[0], args[1], args[3]], cwd=args[4], shell=True) 
=======
buckets = Popen('rclone lsd goes16aws:noaa-goes16', cwd='./rclone', shell=True) 
>>>>>>> 6865887e9899dda98cbc28b01115267dc3a5c30a
date = raw_input("What date/time would you like to see(please use format <Year>/<Day of Year>/<Hour>)? ")
type(str)
# todo - change date format to be consistent with JPSS/datetime

path = 'goes16aws:noaa-goes16/ABI-L2-MCMIPC/' + date
print path

<<<<<<< HEAD
# get all files for given date/time, i.e. 2018/262/02
files = call([args[0],args[2], path], cwd=args[4])
process = call([args[0],args[2], path], cwd=args[4], stdout=PIPE, stderr=STDOUT)
=======
# get all files for given date/time, (i.e. 2018/262/02)
cmd = 'rclone ls ' + path
print cmd
files = Popen('cmd', cwd='./rclone', shell=True)
process = Popen(cmd, cwd='./rclone', shell=True, stdout=PIPE, stderr=STDOUT)
>>>>>>> 6865887e9899dda98cbc28b01115267dc3a5c30a
output = process.communicate()[0]
print 'available files for ' + date + ' are:', output # list available files for the given hour

# copy all files to given local directory
download = call([args[0], 'copyto', path, './'], cwd=args[4])

#now we open files to look around
C_file = './' + args[0] + "/" + raw_input("What file would you like to see? ") # allow a moment for download to complete
C = Dataset(C_file, 'r')

#will add mapping from https://github.com/blaylockbk/pyBKB_v2/blob/master/BB_GOES16/mapping_GOES16_FireTemperature.ipynb 
<<<<<<< HEAD

'''
Notes: 
-may need "shell=True" only for Windows environment.
-call still not working, properly. Will adjust in the next couple days
'''
=======
>>>>>>> 6865887e9899dda98cbc28b01115267dc3a5c30a
