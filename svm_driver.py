from svm import SVM3
import numpy as np
from scipy.io import loadmat,savemat
from scipy import interpolate
from JPSSD import read_fire_mesh
import sys
import datetime as dt
from contour2kml import contour2kml
from time import time

t_init = time()

m = loadmat('fire.mat')
search = True
Xg = m['Xg'];
Xf = m['Xf'];
tscale = m['tscale']
epoch = m['epoch'].ravel()
bbox = m['bbox'].ravel()
# time = col 3 of Xg Xf is in tscale seconds since epoch
epoch_dt = dt.datetime(year=epoch[0],month=epoch[1],day=epoch[2],
                       hour=epoch[3],minute=epoch[4],second=epoch[5])
svm_file = 'svm.mat'
svm_file2 = 'svm_out.mat'

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
    Fx, Fy, Fz = SVM3(X,y,C=C,kgam=kgam,search=search,plot_result=True,plot_data=True)
 
    sys.stdout.flush()
    tign_g = np.array(Fz)
    # Creating the dictionary with the results
    svm = { 'X': np.array(X), 'y': np.array(y), 'C': np.array(C),
    		'fxlon': np.array(Fx), 'fxlat': np.array(Fy), 
    		'Z': np.array(Fz), 'tign_g': np.array(tign_g), 
    		'C': C, 'kgam': kgam, 
                'tscale': tscale, 'epoch':epoch,'bbox':bbox}
    # time is epoch + tscale * tign_g

    savemat(svm_file, mdict=svm)
    print ('The results are saved in %s' % svm_file)
    
 
print ('Loading from %s' % svm_file)
svm = loadmat(svm_file)
#print(svm)

print('data bbox %s %s %s %s' % (
    np.amin(svm['fxlon']), np.amax(svm['fxlon']),
    np.amin(svm['fxlat']), np.amax(svm['fxlat'])))

tscale=svm['tscale']
epoch=svm['epoch'].ravel()
epoch_dt = dt.datetime(year=epoch[0],month=epoch[1],day=epoch[2],
                       hour=epoch[3],minute=epoch[4],second=epoch[5])
print('data epoch %s' % epoch_dt)

# still have bbox and epoch_dt from above
try:
    fxlon,fxlat,bbox,time_esmf=read_fire_mesh('wrfout')
    time_dt = dt.datetime.strptime(time_esmf,'%Y-%m-%d_%H:%M:%S')
    print('wrfout time %s' % time_dt)
    print('wrfout bbox %s %s %s %s' % (
       np.amin(fxlon), np.amax(fxlon),
       np.amin(fxlat), np.amax(fxlat)))
except:
    print('cannot read wrfout, using existing values')
    bbox = svm['bbox']
    fxlon,fxlat=np.meshgrid(np.linspace(bbox[0],bbox[1],num=100),
                        np.linspace(bbox[2],bbox[3],num=111))
    time_dt = epoch_dt

# assuming time_esmf is the beginning of the simulation
# esmf format is YYYY-MM-DD_hh:mm:ss


# Interpolation of tign_g
fire_interp = True
notnan = True
if fire_interp:
		print '>> Interpolating the results in the fire mesh'
		t_interp_1 = time()
		points = (svm['fxlon'].ravel(),svm['fxlat'].ravel())
		values = (svm['Z']*tscale + (epoch_dt - time_dt).total_seconds()).ravel()
		tign_g_interp = interpolate.griddata(points,values,(fxlon,fxlat))
		t_interp_2 = time()
                nans = np.sum(np.isnan(tign_g_interp));
                nums = np.sum(~np.isnan(tign_g_interp));
                frac = float(nans)/(nans+nums)
                print('interpolated array is %s%s nan' % (100*frac,'%'))
		if notnan:
			with np.errstate(invalid='ignore'):
				tign_g_interp[np.isnan(tign_g_interp)] = np.nanmax(tign_g_interp)
		print 'elapsed time: %ss.' % str(abs(t_interp_2-t_interp_1))
                svm2 = svm
		svm2.update({'fxlon_interp': np.array(fxlon), 
			'fxlat_interp': np.array(fxlat),
			'tign_g_interp': np.array(tign_g_interp)})

# Save resulting file
savemat(svm_file2, mdict=svm)
print 'The results are saved in file %s' % svm_file2

print ''
print '>> DONE <<'
t_final = time()
print 'Elapsed time for all the process: %ss.' % str(abs(t_final-t_init))
sys.exit()
