# JPPSdata
## Edit - Utilize JPPSD.py, followed by utilizing matlab functions for visualization

https://github.com/openwfm/JPSSdata
mirror: http://repo.or.cz/git-browser/by-commit.html?r=JPSSData.git


### Authors:
Lauren Hearn,
Jan Mandel,
Angel Caus,
James Haley

University of Colorado Denver

*A subproject in the [OpenWFM project](https://github.com/openwfm) with support from NASA NNX13AH59G and NSF 1664175
### To do:
- add download ability (completed)
- download only certain leaves of the overall tree structure
- reconcile data from M?D14 with geolocation data from M?D03 (done)
- extend MODIS search to VIIRS (completed)
- continue translating ncvarinfo to proper python grammar
- interpolate to a fire mesh and set up upper and lower bounds for fire arrival time
- integrate in WRFXPY  https://github.com/openwfm/wrfxpy

### Contains:
- JPPSD.py gather data from MODIS/VIIRS for a given time window and bounding box 
- get_af_data.py a test driver for JPSSD.py to download all data for a given window
- interpolation.py not just interpolation but various utilities of more general nature
- saveload.py a convenience utility to save and load Python objects
- setup.py reads data and makes into input for statistical interpolation to estimate fire arrival time
- out.mat reads data downloaded by JPPSD.py, with ability to plot granules
- utils.py is a part of utils.py from https://github.com/openwfm/wrfxpy
- sugarloaf.py test driver to download sample data for Sugarloaf fire into mesh and plot
- ncvarinfo.py is a python version of code originally found [here](https://github.com/openwfm/wrf-fire/blob/master/other/Matlab/netcdf/private/ncvarinfo.m). 

### External links
- Tools for searching the CMR API https://cmr.earthdata.nasa.gov
- VIIRS Active Fires users guide https://lpdaac.usgs.gov/sites/default/files/public/product_documentation/vnp14_user_guide_v1.3.pdf
