from svm import SVM3
import numpy as np
from scipy.io import loadmat,savemat
from JPSSD import read_fire_mesh
import sys
from contour2kml import contour2kml

m = loadmat('fire.mat')
search = True
Xg = m['Xg'];
Xf = m['Xg'];

for exp in (1,):

    def exp1():
        C = 10.
        kgam = 1.
        return Xg, Xf, C, kgam
    def exp2():
        C = np.concatenate((50*np.ones(len(Xg)), 100.*np.ones(len(Xf))))
        kgam = 5.
        return Xg, Xf, C, kgam

    # Creating the options
    options = {1 : exp1, 2 : exp2}

    # Defining the option depending on the experiment
    Xg, Xf, C, kgam = options[exp]()

    # Creating the data necessary to run SVM3 function
    X = np.concatenate((Xg, Xf))
    y = np.concatenate((-np.ones(len(Xg)), np.ones(len(Xf))))

    # Running SVM classification
    F = SVM3(X,y,C=C,kgam=kgam,search=search,plot_result=True,plot_data=True)

# fxlon,fxlat,bbox,time_esmf=read_fire_mesh(wrfout)
bbox = m['bbox']
fxlon,fxlat=np.meshgrid(np.linspace(bbox[0],bbox[1],num=100),
                        np.linspace(bbox[2],bbox[3],num=111))

sys.stdout.flush()
# Fire arrival time in seconds from the begining of the simulation
tign_g = np.array(F[2])*float(tscale)+scale[0]-time_num_interval[0]
# Creating the dictionary with the results
svm = { 'X': np.array(X), 'y': np.array(y), 'c': np.array(c),
		'fxlon': np.array(F[0]), 'fxlat': np.array(F[1]), 
		'Z': np.array(F[2]), 'tign_g': np.array(tign_g), 
		'C': C, 'kgam': kgam}

svm_file = 'svm.mat'
savemat(svm_file, mdict=svm)
print 'The results are saved in ' + svm_file
svm = loadmat(svm_file)
print 'Loading' + smf_file

# Interpolation of tign_g
if fire_interp:
	try:
		print '>> Interpolating the results in the fire mesh'
		t_interp_1 = time()
		points = np.c_[np.ravel(F[0]),np.ravel(F[1])]
		values = np.ravel(tign_g)
		tign_g_interp = interpolate.griddata(points,values,(fxlon,fxlat))
		t_interp_2 = time()
		if notnan:
			with np.errstate(invalid='ignore'):
				tign_g_interp[np.isnan(tign_g_interp)] = np.nanmax(tign_g_interp)
		print 'elapsed time: %ss.' % str(abs(t_interp_2-t_interp_1))
		svm.update({'fxlon_interp': np.array(fxlon), 
			'fxlat_interp': np.array(fxlat),
			'tign_g_interp': np.array(tign_g_interp)})
	except:
		print 'Warning: longitudes and latitudes from the original grid are not defined...'
		print '%s file is not compatible with fire_interp=True! Run again the experiment from the begining.' % bounds_file

# Save resulting file
savemat(svm_file, mdict=svm)
print 'The results are saved in ' + svm_file

print ''
print '>> Computing contour lines of the fire arrival time <<'
print 'Computing the contours...'
try:
	# Granules numeric times
	Z = np.array(F[2])*tscale+scale[0]
	# Creating contour lines
	contour_data = get_contour_verts(F[0], F[1], Z, time_num_granules, contour_dt_hours=6, contour_dt_init=6, contour_dt_final=6)
	print 'Creating the KML file...'
	# Creating the KML file
	contour2kml(contour_data,contour_file)
	print 'The resulting contour lines are saved in perimeters_svm.kml file'
except:
	print 'Warning: contour creation problem'
	print 'Run: python contlinesvm.py'

print ''
print '>> DONE <<'
t_final = time()
print 'Elapsed time for all the process: %ss.' % str(abs(t_final-t_init))
sys.exit()
