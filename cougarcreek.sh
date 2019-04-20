#!/bin/bash

wrfout=/uufs/chpc.utah.edu/common/home/kochanski-group3/farguella/JPSSdata/cougarcreek/wrfout_d04_2015-08-11_00:00:00
python -u process.py $wrfout 20150811000000 14 &> cougarcreek.log

