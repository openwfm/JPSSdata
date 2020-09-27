# JPPSdata

### Requirements

	1) Install Anaconda 3:
		wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
		chmod +x Anaconda3-2019.10-Linux-x86_64.sh
		./Anaconda3-2019.10-Linux-x86_64.sh

	2) Create anaconda environment named jpssdata:
		conda create -n jpssdata python=2.7 basemap netcdf4 scikit-learn scikit-image h5py pandas requests

	3) Install other necessary packages:
                conda activate jpssdata
		conda install -c conda-forge pyhdf
		pip install python-cmr

### Usage Support Vector Machine (SVM):
Run process.py as:

	python process.py wrfout start_time days

OR

	python process.py lon1,lon2,lat1,lat2 start_time days

Generates the fire mesh, downloads all the granules in between the dates and intersecting with the fire mesh, reads all the important information inside them and saves everything in a binary file called 'data'. It creates as well a KML file with all the fire detections called 'fire_detections.kml'. After that, it runs a postprocessing of the data creating an intermediate binary file called 'result'. Finally, it estimates the fire arrival time using SVM machine learning technique, creating an output file 'svm.mat' with the results.

	The input variales from 'python process.py coord start_time days' are:

	- coord: string:
				1) link to the wrfout file of WRF-SFIRE simulation or
				2) bounding box coordinates separated by commas
					lon1,lon2,lat1,lat2
	- start_time - string, YYYYMMDDHHMMSS where:
		- YYYY - year
		- MM - month
		- DD - day
		- HH - hour
		- MM - minute
		- SS - second
	- days: number, number of days of simulation (can be decimal).

For different configurations of the SVM run, create file called 'conf.json' using similar structure than in 'conf_example.json' file. In order to find out what are the flags, look into 'utils.py' file.

For running SVM using different weights depending on the confidence levels, run inside JPSSdata repository:

	git clone https://github.com/Fergui/libsvm_weights

and set 'dyn_pen' flag in 'conf.json' to true.

There is also an example.sh bash script which needs to be run in Kinspeak or change the wrfout path to some existent wrfout file.

### Usage L2 minimization:
1) Run case.py as:

	$ python case.py wrfout start_time days

	Generates the fire mesh, downloads all the granules in between the dates and intersecting with the fire mesh, reads all the important information inside them and saves everything in a binary file called 'data'. It creates as well a csv file and a KML file with all the detections called 'fire_detections.csv' and 'fire_detections.kml'. It is also created a KML file with the ground detections called 'nofire.kml'. The input variales are:

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
Angel Farguell,
James Haley

University of Colorado Denver

*A subproject in the [OpenWFM project](https://github.com/openwfm) with support from NASA NNX13AH59G and NSF 1664175
### To do:
- add download ability (completed)
- download only certain leaves of the overall tree structure (completed)
- reconcile data from M?D14 with geolocation data from M?D03 (completed)
- extend MODIS search to VIIRS (completed)
- continue translating ncvarinfo to proper python grammar
- interpolate to a fire mesh and set up upper and lower bounds for fire arrival time (completed)
- integrate in WRFXPY  https://github.com/openwfm/wrfxpy

### Contains:
- process.py drives all the simulation from acquiring the data, preprocessing the data, and running SVM.
- svm.py runs SVM in a 3D mesh and estimates fire arrival time as the minimum zero of the 3D boundary.
- JPPSD.py gather data from MODIS/VIIRS for a given time window and bounding box
- get_af_data.py a test driver for JPSSD.py to download all data for a given window
- GOES16.py filters and downloads GOES satellite files from AWS
- read_goes.py parses netCDF files for relevant fire info in given bounding box/time
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
