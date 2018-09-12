# JPPSdata
## Edit - Utilize JPPSD.py, followed by utilizing matlab functions for visualization

*A tool for searching the cmr API [found here](https://cmr.earthdata.nasa.gov)*

### Authors:
Lauren Hearn,
Jan Mandel,
Angel Caus,
James Haley

*As a joint project supporting the [OpenWFM project](https://github.com/openwfm) with support from UC Denver.*

### To do:
- add download ability (completed)
- download only certain leaves of the overall tree structure
- reconcile data from M?D14 with geolocation data from M?D03
- extend MODIS search to VIIRS (completed)
- continue translating ncvarinfo to proper python grammar

### Tools:
- JPPSD.py gathers data from MODIS/VIIRS
- out.mat reads data downloaded from JPPSD.py, with ability to plot granules
- sugarloaf.py plots sample data from Sugarloaf fire into mesh and plot
- ncvarinfo.py is a python version of code originally found [here](https://github.com/openwfm/wrf-fire/blob/master/other/Matlab/netcdf/private/ncvarinfo.m). 
