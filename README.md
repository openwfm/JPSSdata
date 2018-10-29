# JPPSdata
### Usage:
1) Run case.py as:

	$ python case.py wrfout start_time days

	Generates the fire mesh, downloads all the granules in between the dates and intersecting with the fire mesh, reads all the important information inside them and saves everything in a text file called 'data'. It creates as well a csv file and a KML file with all the detections called 'fire_detections.csv' and 'fire_detections.kml'. It is also created a KML file with the ground detections called 'nofire.kml'. The input variales are:

	- wrfout:  string, link to the wrfout file of WRF-SFIRE simulation.	
	- start_time - string, YYYYMMDDHHMMSS where:
		- YYYY - year
		- MM - month
		- DD - day
		- HH - hour
		- MM - minute
		- SS - second
	- days: integer, number of days of simulation.

2) Run setup.py as:

	$ python setup.py

	Processes all the granules and creates the upper and lower bounds for the fire arrival time. It saves everything in a text file called 'result' and in a Matlab file called 'result.mat'.

5) The Matlab file result.mat can be used to run the Multigrid method which are going to define a fire arrival time curve in between the upper and lower bounds as a rigid plate deformed by forces. In order to do that, the next steps are necessary.

6) Link the Matlab file 'result.mat' into the private fire_interpolation repository in:

	ssh://repo.openwfm.org/home/git/fire_interpolation

7) Run in Matlab the script jpss_mg.m as:

	\>\> jpss_mg

	Generates the fire arrival time in a 2D array called 'a' using the Multigrid technique and saves everything in a Matlab file called 'mgout.mat'. It shows in different figures the different levels and how they are changing all the time.

8) Link back the Matlab file 'mgout.mat' into the JPSSData repository and run contline.py as:

	$ python contline.py

	Generates a contour line representation of the results in a KML file called 'perimeters.kml'. It can be opened in Google Earth application as well with step 2) and it generates a movie of the interpolation with the fire detections.

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
- reconcile data from M?D14 with geolocation data from M?D03 (completed)
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

### External links:
- Tools for searching the CMR API https://cmr.earthdata.nasa.gov
- VIIRS Active Fires users guide https://lpdaac.usgs.gov/sites/default/files/public/product_documentation/vnp14_user_guide_v1.3.pdf
- GOES16.py filters and downloads GOES satellite files from AWS.
