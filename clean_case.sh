#!/bin/bash

rm $(find . -maxdepth 1 -type l | grep .hdf) 2> /dev/null
rm $(find . -maxdepth 1 -type l | grep .h5) 2> /dev/null
rm $(find . -maxdepth 1 -type l | grep .nc) 2> /dev/null
