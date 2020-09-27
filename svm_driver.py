from svm import SVM3
import numpy as np
from scipy.io import loadmat,savemat
from JPSSD import read_fire_mesh
import sys
import datetime as dt
from contour2kml import contour2kml

m = loadmat('fire.mat')
search = True
Xg = m['Xg'];
Xf = m['Xg'];
tscale = m['tscale']
epoch = m['epoch'].flatten()
# time = col 3 of Xg Xf is in tscale seconds since epoch
epoch_dt = dt.datetime(year=epoch[0],month=epoch[1],day=epoch[2],
                       hour=epoch[3],minute=epoch[4],second=epoch[5])
svm_file = 'svm.mat'

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

    sys.stdout.flush()
    tign_g = np.array(F[2])
    # Creating the dictionary with the results
    svm = { 'X': np.array(X), 'y': np.array(y), 'C': np.array(C),
    		'fxlon': np.array(F[0]), 'fxlat': np.array(F[1]), 
    		'Z': np.array(F[2]), 'tign_g': np.array(tign_g), 
    		'C': C, 'kgam': kgam, 
                'tscale': tscale, 'epoch':epoch,'bbox':bbox}
    # time is epoch + tscale * tign_g

    savemat(svm_file, mdict=svm)
    print ('The results are saved in %s' % svm_file)
    
svm = loadmat(svm_file)
print ('Loading from %s' % smf_file)

# still have bbox and epoch_dt from above
# fxlon,fxlat,bbox,time_esmf=read_fire_mesh('wrfout')
# assuming time_esmf is the beginning of the simulation
# esmf format is YYYY-MM-DD_hh:mm:ss

# wrfout not available - making it up, testing only
time_esmf = '2019-07-24_22:00:00'
fxlon,fxlat=np.meshgrid(np.linspace(bbox[0],bbox[1],num=100),
                        np.linspace(bbox[2],bbox[3],num=111))
#end making it up

time_dt = dt.datetime.strptime(time_esmf,'%Y-%m-%d_%H:%M:%S')
# Interpolation of tign_g
if fire_interp:
	try:
		print '>> Interpolating the results in the fire mesh'
		t_interp_1 = time()
		points = np.c_[np.ravel(F[0]),np.ravel(F[1])]
		values = np.ravel(tign_g)*tscale + (epoch_dt - time_dt).total_seconds()
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
print '>> DONE <<'
t_final = time()
print 'Elapsed time for all the process: %ss.' % str(abs(t_final-t_init))
sys.exit()
