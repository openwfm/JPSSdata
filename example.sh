#!/bin/bash

wrfout=/uufs/chpc.utah.edu/common/home/kochanski-group4/farguella/JPSSdata/example/wrfout
python -u process.py $wrfout 20181108195500 0.066 &> example.log
