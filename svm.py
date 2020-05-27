#
# Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
# Angel Farguell (angel.farguell@gmail.com)
#
# to install:
#       conda install scikit-learn
#       conda install scikit-image

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from scipy import interpolate, spatial
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from time import time
from infrared_perimeters import process_infrared_perimeters
import sys
import saveload as sl

def verify_inputs(params):
    required_args = [("search", False), ("norm", True),
        ("notnan", True), ("artil", False), ("hartil", 0.2),
        ("artiu", True), ("hartiu", 0.1), ("downarti", True),
        ("dminz", 0.1), ("confal", 100), ("toparti", False),
        ("dmaxz", 0.1), ("confau", 100), ("plot_data", False),
        ("plot_scaled", False), ("plot_decision", False),
        ("plot_poly", False), ("plot_supports", False),
        ("plot_result", False)]
    # check each argument that should exist
    for key, default in required_args:
        if key not in params:
            params.update({key: default})
    return params

def preprocess_data_svm(data, scale, minconf=80.):
    """
    Preprocess satellite data from JPSSD to use in Support Vector Machine directly
    (without any interpolation as space-time 3D points)

    :param data: dictionary of satellite data from JPSSD
    :param scale: time scales
    :param minconf: optional, minim fire confidence level to take into account
    :return X: matrix of features for SVM
    :return y: vector of labels for SVM
    :return c: vector of confidence level for SVM

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-09-24
    """

    # confidence of ground detections
    gconf = 95.

    # scale from seconds to days
    tscale = 24.*3600.

    detlon = np.concatenate([d['lon_fire'] for d in data.itervalues()])
    detlat = np.concatenate([d['lat_fire'] for d in data.itervalues()])
    confs = np.concatenate([d['conf_fire'] for d in data.itervalues()])
    bb = (detlon[confs > minconf].min(),detlon[confs > minconf].max(),detlat[confs > minconf].min(),detlat[confs > minconf].max())
    dc = (bb[1]-bb[0],bb[3]-bb[2])
    bf = (bb[0]-dc[0]*.3,bb[1]+dc[0]*.3,bb[2]-dc[1]*.3,bb[3]+dc[1]*.3)
    print bf

    # process all the points as space-time 3D nodes
    XX = [[],[]]
    cf = []
    for gran in data.items():
        print '> processing granule %s' % gran[0]
        tt = (gran[1]['time_num']-scale[0])/tscale
        conf = gran[1]['conf_fire']>=minconf
        xf = np.c_[(gran[1]['lon_fire'][conf],gran[1]['lat_fire'][conf],np.repeat(tt,conf.sum()))]
        print 'fire detections: %g' % len(xf)
        XX[0].append(xf)
        mask = np.logical_and(gran[1]['lon_nofire'] >= bf[0],
                np.logical_and(gran[1]['lon_nofire'] <= bf[1],
                 np.logical_and(gran[1]['lat_nofire'] >= bf[2],
                                 gran[1]['lat_nofire'] <= bf[3])))
        xg = np.c_[(gran[1]['lon_nofire'][mask],gran[1]['lat_nofire'][mask],np.repeat(tt,mask.sum()))]
        print 'no fire detections: %g' % len(xg)
        coarsening = 1 + max((len(xg)-len(xf))//50,0)
        print 'no fire coarsening: %d' % coarsening
        print 'no fire detections reduction: %g' % len(xg[::coarsening])
        XX[1].append(xg[::coarsening])
        cf.append(gran[1]['conf_fire'][conf])

    Xf = np.concatenate(tuple(XX[0]))
    Xg = np.concatenate(tuple(XX[1]))
    X = np.concatenate((Xg,Xf))
    y = np.concatenate((-np.ones(len(Xg)),np.ones(len(Xf))))
    c = np.concatenate((gconf*np.ones(len(Xg)),np.concatenate(tuple(cf))))
    print 'shape X: ', X.shape
    print 'shape y: ', y.shape
    print 'shape c: ', c.shape
    print 'len fire: ', len(X[y==1])
    print 'len ground: ', len(X[y==-1])

    return X,y,c

def preprocess_result_svm(lons, lats, U, L, T, scale, time_num_granules, C=None):
    """
    Preprocess satellite data from JPSSD and setup to use in Support Vector Machine

    :param lons: longitud grid
    :param lats: latitde grid
    :param U: upper bound grid
    :param L: lower bound grid
    :param T: mask grid
    :param scale: time scales
    :param time_num_granules: times of the granules
    :return X: matrix of features for SVM
    :return y: vector of labels for SVM
    :return c: vector of confidence level for SVM

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-04-01
    """

    # Flatten coordinates
    lon = np.ravel(lons).astype(float)
    lat = np.ravel(lats).astype(float)

    # Temporal scale to days
    tscale = 24*3600
    U = U/tscale
    L = L/tscale

    # Ensuring U>=L always
    print 'U>L: ',(U>L).sum()
    print 'U<L: ',(U<L).sum()
    print 'U==L: ',(U==L).sum()

    # Reshape to 1D
    uu = np.ravel(U)
    ll = np.ravel(L)
    tt = np.ravel(T)

    # Maximum and minimums to NaN data
    uu[uu==uu.max()] = np.nan
    ll[ll==ll.min()] = np.nan

    # Mask created during computation of lower and upper bounds
    mk = tt==scale[1]-scale[0]
    # Masking upper bounds outside the mask
    uu[mk] = np.nan

    # Creating minimum value for the upper bounds
    muu = uu[~np.isnan(uu)].min()
    # Creating maximum value for the lower bounds
    mll = ll[~np.isnan(ll)].max()

    ### Reduction of the density of lower bounds
    # Creation of low bounds mask (True values are going to turn Nan's in lower bound data)
    lmk = np.copy(mk)
    ## Reason: We do not care about lower bounds below the upper bounds which are far from the upper bounds
    # temporary lower mask, all False (values of the mask where the mask is False, inside the fire mask)
    tlmk = lmk[~mk]
    # set to True all the bounds less than floor of minimum of upper bounds in fire mask
    tlmk[~np.isnan(ll[~mk])] = (ll[~mk][~np.isnan(ll[~mk])] < np.floor(muu))
    # set lower mask from temporary mask
    lmk[~mk] = tlmk
    ## Reason: Coarsening of the lower bounds below the upper bounds to create balance
    # create coarsening of the lower bound data below the upper bounds to be similar amount that upper bounds
    kk = (~np.isnan(ll[~lmk])).sum()/(~np.isnan(uu)).sum()
    if kk:
        # temporary lower mask, all True (values of the lower mask where the lower mask is False, set to True)
        tlmk = ~lmk[~lmk]
        # only set a subset of the lower mask values to False (coarsening)
        tlmk[::kk] = False
        # set lower mask form temporary mask
        lmk[~lmk] = tlmk
    ## Reason: We care about the maximum lower bounds which are not below upper bounds
    # temporary lower mask, all True (values of the mask where the mask is True, outside the fire mask)
    tlmk = lmk[mk]
    # temporary lower mask 2, all True (subset of the previous mask where the lower bounds is not Nan)
    t2lmk = tlmk[~np.isnan(ll[mk])]
    # set to False in the temporary lower mask 2 where lower bounds have maximum value
    t2lmk[ll[mk][~np.isnan(ll[mk])] == mll] = False
    # set temporary lower mask from temporary lower mask 2
    tlmk[~np.isnan(ll[mk])] = t2lmk
    # set lower mask from temporary lower mask
    lmk[mk] = tlmk
    ## Reason: Coarsening of the not maximum lower bounds not below the upper bounds to create balance
    # set subset outside of the fire mask for the rest
    # create coarsening of the not maximum lower bounds not below the upper bounds to be similar amount that the current number of lower bounds
    kk = (ll[mk][~np.isnan(ll[mk])] < mll).sum()/(~np.isnan(ll[~lmk])).sum()
    if kk:
        # temporary lower mask, values of the current lower mask outside of the original fire mask
        tlmk = lmk[mk]
        # temporary lower mask 2, subset of the previous mask where the lower bound is not Nan
        t2lmk = tlmk[~np.isnan(ll[mk])]
        # temporary lower mask 3, subset of the previous mask where the lower bounds are not maximum
        t3lmk = t2lmk[ll[mk][~np.isnan(ll[mk])] < mll]
        # coarsening of the temporary lower mask 3
        t3lmk[::kk] = False
        # set the temporary lower mask 2 from the temporary lower mask 3
        t2lmk[ll[mk][~np.isnan(ll[mk])] < mll] = t3lmk
        # set the temporary lower mask from the temporary lower mask 2
        tlmk[~np.isnan(ll[mk])] = t2lmk
        # set the lower mask from the temporary lower mask
        lmk[mk] = tlmk

    # Masking lower bounds from previous lower mask
    ll[lmk] = np.nan

    # Values different than NaN in the upper and lower bounds
    um = np.array(~np.isnan(uu))
    lm = np.array(~np.isnan(ll))
    # Define all the x, y, and z components of upper and lower bounds
    ux = lon[um]
    uy = lat[um]
    uz = uu[um]
    lx = lon[lm]
    ly = lat[lm]
    lz = ll[lm]

    # Create the data to call SVM3 function from svm.py
    X = np.c_[np.concatenate((lx,ux)),np.concatenate((ly,uy)),np.concatenate((lz,uz))]
    y = np.concatenate((-np.ones(len(lx)),np.ones(len(ux))))
    # Print the shape of the data
    print 'shape X: ', X.shape
    print 'shape y: ', y.shape

    if C is None:
        c = 80*np.ones(y.shape)
    else:
        c = np.concatenate((np.ravel(C[0])[lm],np.ravel(C[1])[um]))

    # Clean data if not in bounding box
    bbox = (lon.min(),lon.max(),lat.min(),lat.max(),time_num_granules)
    geo_mask = np.logical_and(np.logical_and(np.logical_and(X[:,0] >= bbox[0],X[:,0] <= bbox[1]), X[:,1] >= bbox[2]), X[:,1] <= bbox[3])
    btime = (0,(scale[1]-scale[0])/tscale)
    time_mask = np.logical_and(X[:,2] >= btime[0], X[:,2] <= btime[1])
    whole_mask = np.logical_and(geo_mask, time_mask)
    X = X[whole_mask,:]
    y = y[whole_mask]
    c = c[whole_mask]

    return X,y,c

def make_fire_mesh(fxlon, fxlat, it, nt):
    """
    Create a mesh of points to evaluate the decision function

    :param fxlon: data to base x-axis meshgrid on
    :param fxlat: data to base y-axis meshgrid on
    :param it: data to base z-axis meshgrid on
    :param nt: tuple of number of nodes at each direction, optional
    :param coarse: coarsening of the fire mesh
    :return xx, yy, zz: ndarray

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-04-01
    """

    xx = np.repeat(fxlon[:, :, np.newaxis], nt, axis=2)
    yy = np.repeat(fxlat[:, :, np.newaxis], nt, axis=2)
    tt = np.linspace(it[0],it[1],nt)
    zz = np.swapaxes(np.swapaxes(np.array([np.ones(fxlon.shape)*t for t in tt]),0,1),1,2)

    return xx, yy, zz

def make_meshgrid(x, y, z, s=(50,50,50), exp=.1):
    """
    Create a mesh of points to evaluate the decision function

    :param x: data to base x-axis meshgrid on
    :param y: data to base y-axis meshgrid on
    :param z: data to base z-axis meshgrid on
    :param s: tuple of number of nodes at each direction, optional
    :param exp: extra percentage of time steps in each direction (between 0 and 1), optional
    :return xx, yy, zz: ndarray

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-02-20
    Modified version of:
    https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py
    """

    if not isinstance(s, tuple):
        print '>> FAILED <<'
        print 'The number of nodes at each direction is not a tuple: ', s
        sys.exit(1)
    # number of nodes in each direction
    sx, sy, sz = np.array(s).astype(int)
    # extra step sizes in each direction
    brx = int(sx * exp)
    bry = int(sy * exp)
    brz = int(sz * exp)
    # grid lengths in each directon
    lx = x.max() - x.min()
    ly = y.max() - y.min()
    lz = z.max() - z.min()
    # grid resolutions in each direction
    hx = lx / (sx - 2*brx - 1)
    hy = ly / (sy - 2*bry - 1)
    hz = lz / (sz - 2*brz - 1)
    # extrem values for each dimension
    x_min, x_max = x.min() - brx * hx, x.max() + brx * hx
    y_min, y_max = y.min() - bry * hy, y.max() + bry * hy
    z_min, z_max = z.min() - brz * hz, z.max() + brz * hz
    # generating the mesh grid
    xx, yy, zz = np.meshgrid(np.linspace(y_min, y_max, sy),
                             np.linspace(x_min, x_max, sx),
                             np.linspace(z_min, z_max, sz))
    return xx, yy, zz

def frontier(clf, xx, yy, zz, bal=.5, plot_decision = False, plot_poly=False, using_weights=False):
    """
    Compute the surface decision frontier for a classifier.

    :param clf: a classifier
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param zz: meshgrid ndarray
    :param bal: number between 0 and 1, balance between lower and upper bounds in decision function (in case not level 0)
    :param plot_decision: boolean of plotting decision volume
    :param plot_poly: boolean of plotting polynomial approximation
    :return F: 2D meshes with xx, yy coordinates and the hyperplane z which gives decision functon 0

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-02-20
    Modified version of:
    https://www.semipol.de/2015/10/29/SVM-separating-hyperplane-3d-matplotlib.html
    """

    # Creating the 3D grid
    XX = np.c_[np.ravel(xx), np.ravel(yy), np.ravel(zz)]

    # Evaluating the decision function
    print '>> Evaluating the decision function...'
    sys.stdout.flush()
    t_1 = time()
    if using_weights:
        from libsvm_weights.python.svmutil import svm_predict
        _, _, p_vals = svm_predict([], XX, clf, '-q')
        ZZ = np.array([p[0] for p in p_vals])
    else:
        ZZ = clf.decision_function(XX)
    t_2 = time()
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))
    hist = np.histogram(ZZ)
    print 'counts: ', hist[0]
    print 'values: ', hist[1]
    print 'decision function range: ', ZZ.min(), '~', ZZ.max()

    # Reshaping decision function volume
    Z = ZZ.reshape(xx.shape)
    print 'decision function shape: ', Z.shape
    sl.save((xx,yy,zz,Z),'decision')

    if plot_decision:
        try:
            from skimage import measure
            from shiftcmap import shiftedColorMap
            verts, faces, normals, values = measure.marching_cubes_lewiner(Z, level=0, allow_degenerate=False)
            # Scale and transform to actual size of the interesting volume
            h = np.divide([xx.max()-xx.min(), yy.max() - yy.min(), zz.max() - zz.min()],np.array(xx.shape)-1)
            verts = verts * h
            verts = verts + [xx.min(), yy.min(), zz.min()]
            mesh = Poly3DCollection(verts[faces], facecolor='orange', alpha=.9)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle("Decision volume")
            col = [(0, .5, 0), (.5, .5, .5), (.5, 0, 0)]
            cm = colors.LinearSegmentedColormap.from_list('GrRdD',col,N=50)
            midpoint = 1 - ZZ.max() / (ZZ.max() + abs(ZZ.min()))
            shiftedcmap = shiftedColorMap(cm, midpoint=midpoint, name='shifted')
            kk = 1+np.divide(xx.shape,50)
            X = np.ravel(xx[::kk[0],::kk[1],::kk[2]])
            Y = np.ravel(yy[::kk[0],::kk[1],::kk[2]])
            T = np.ravel(zz[::kk[0],::kk[1],::kk[2]])
            CC = np.ravel(Z[::kk[0],::kk[1],::kk[2]])
            p = ax.scatter(X, Y, T, c=CC, s=.1, alpha=.5, cmap=shiftedcmap)
            cbar = fig.colorbar(p)
            cbar.set_label('decision function value', rotation=270, labelpad=20)
            ax.add_collection3d(mesh)
            ax.set_zlim([xx.min(),xx.max()])
            ax.set_ylim([yy.min(),yy.max()])
            ax.set_zlim([zz.min(),zz.max()])
            ax.set_xlabel("Longitude normalized")
            ax.set_ylabel("Latitude normalized")
            ax.set_zlabel("Time normalized")
            plt.savefig('decision.png')
        except Exception as e:
            print 'Warning: something went wrong when plotting...'
            print e

    if plot_poly:
        fig = plt.figure()
    # Computing fire arrival time from previous decision function
    print '>> Computing fire arrival time...'
    sys.stdout.flush()
    t_1 = time()
    # xx 2-dimensional array
    Fx = xx[:, :, 0]
    # yy 2-dimensional array
    Fy = yy[:, :, 0]
    # zz 1-dimensional array
    zr = zz[0, 0]
    # Initializing fire arrival time
    Fz = np.zeros(Fx.shape)
    # For each x and y
    for k1 in range(Fx.shape[0]):
        for k2 in range(Fx.shape[1]):
            # Approximate the vertical decision function by a piecewise polynomial (cubic spline interpolation)
            pz = interpolate.CubicSpline(zr, Z[k1,k2])
            # Compute the real roots of the the piecewise polynomial
            rr = pz.roots()
            # Just take the real roots between min(zz) and max(zz)
            realr = rr.real[np.logical_and(abs(rr.imag) < 1e-5, np.logical_and(rr.real > zr.min(), rr.real < zr.max()))]
            if len(realr) > 0:
                # Take the minimum root
                Fz[k1,k2] = realr.min()
                # Plotting the approximated polynomial with the decision function
                if plot_poly:
                    try:
                        plt.ion()
                        plt.plot(pz(zr),zr)
                        plt.plot(Z[k1,k2],zr,'+')
                        plt.plot(np.zeros(len(realr)),realr,'o',c='g')
                        plt.plot(0,Fz[k1,k2],'o',markersize=3,c='r')
                        plt.title('Polynomial approximation of decision_function(%f,%f,z)' % (Fx[k1,k2],Fy[k1,k2]))
                        plt.xlabel('decision function value')
                        plt.ylabel('Z')
                        plt.legend(['polynomial','decision values','roots','fire arrival time'])
                        plt.xlim([Z.min(),Z.max()])
                        plt.ylim([zz.min(),zz.max()])
                        plt.show()
                        plt.pause(.001)
                        plt.cla()
                    except Exception as e:
                        print 'Warning: something went wrong when plotting...'
                        print e
            else:
                # If there is not a real root of the polynomial between zz.min() and zz.max(), just define as a Nan
                Fz[k1,k2] = np.nan
    t_2 = time()
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))
    F = [Fx,Fy,Fz]

    return F

def SVM3(X, y, C=1., kgam=1., fire_grid=None, it=None, **params):
    """
    3D SuperVector Machine analysis and plot

    :param X: Training vectors, where n_samples is the number of samples and n_features is the number of features.
    :param y: Target values
    :param C: Penalization (argument of svm.SVC class), optional
    :param kgam: Scalar multiplier for gamma (capture more details increasing it), optional
    :param fire_grid: The longitud and latitude grid where to have the fire arrival time, optional
    :return F: tuple with (longitude grid, latitude grid, fire arrival time grid)

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-02-20
    Modified version of:
    https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py
    """
    # add default values for parameters not specified
    params = verify_inputs(params)
    print 'svm params: ',params

    t_init = time()

    col = [(0, .5, 0), (.5, 0, 0)]
    cm_GR = colors.LinearSegmentedColormap.from_list('GrRd',col,N=2)
    col = [(1, 0, 0), (.25, 0, 0)]
    cm_Rds = colors.LinearSegmentedColormap.from_list('Rds',col,N=100)

    # if fire_grid==None: creation of the grid options
    # number of vertical nodes per observation
    vN = 1
    # number of horizontal nodes per observation
    hN = 5

    # using different weights for the data
    if isinstance(C,(list,tuple,np.ndarray)):
        using_weights = True
        from libsvm_weights.python.svm import svm_problem, svm_parameter
        from libsvm_weights.python.svmutil import svm_train
        from sklearn.utils import compute_class_weight
    else:
        using_weights = False

    # Data inputs
    X = np.array(X).astype(float)
    y = np.array(y)

    # Original data
    oX = np.array(X).astype(float)
    oy = np.array(y)

    # Visualization of the data
    X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]
    if params['plot_data']:
        try:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle("Plotting the original data to fit")
            ax.scatter(X0, X1, X2, c=y, cmap=cm_GR, s=1, alpha=.5, vmin=y.min(), vmax=y.max())
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_zlabel("Time (days)")
            plt.savefig('original_data.png')
        except Exception as e:
            print 'Warning: something went wrong when plotting...'
            print e

    # Normalization of the data into [0,1]^3
    if params['norm']:
        xmin = X0.min()
        xlen = X0.max() - X0.min()
        x0 = np.divide(X0 - xmin, xlen)
        ymin = X1.min()
        ylen = X1.max() - X1.min()
        x1 = np.divide(X1 - ymin, ylen)
        zmin = X2.min()
        zlen = X2.max() - X2.min()
        x2 = np.divide(X2 - zmin, zlen)
        X0, X1, X2 = x0, x1, x2
        X[:, 0] = X0
        X[:, 1] = X1
        X[:, 2] = X2

    # Creation of fire and ground artificial detections
    if params['artil'] or params['artiu'] or params['toparti'] or params['downarti']:
        # Extreme values at z direction
        minz = X[:, 2].min()
        maxz = X[:, 2].max()
        # Division of lower and upper bounds for data and confidence level
        fl = X[y==np.unique(y)[0]]
        fu = X[y==np.unique(y)[1]]

        # Artifitial extensions of the lower bounds
        if params['artil']:
            # Create artificial lower bounds
            flz = np.array([ np.unique(np.append(np.arange(f[2],minz,-params['hartil']),f[2])) for f in fl ])
            # Definition of new ground detections after artificial detections added
            Xg = np.concatenate([ np.c_[(np.repeat(fl[k][0],len(flz[k])),np.repeat(fl[k][1],len(flz[k])),flz[k])] for k in range(len(flz)) ])
            if using_weights:
                cl = C[y==np.unique(y)[0]]
                Cg = np.concatenate([ np.repeat(cl[k],len(flz[k])) for k in range(len(flz)) ])
        else:
            Xg = fl
            if using_weights:
                cl = C[y==np.unique(y)[0]]
                Cg = cl

        # Artifitial extensions of the upper bounds
        if params['artiu']:
            # Create artificial upper bounds
            fuz = np.array([ np.unique(np.append(np.arange(f[2],maxz,params['hartiu']),f[2])) for f in fu ])
            # Definition of new fire detections after artificial detections added
            Xf = np.concatenate([ np.c_[(np.repeat(fu[k][0],len(fuz[k])),np.repeat(fu[k][1],len(fuz[k])),fuz[k])] for k in range(len(fuz)) ])
            # Define new confidence levels
            if using_weights:
                cu = C[y==np.unique(y)[1]]
                Cf = np.concatenate([ np.repeat(cu[k],len(fuz[k])) for k in range(len(fuz)) ])
        else:
            Xf = fu
            if using_weights:
                cu = C[y==np.unique(y)[1]]
                Cf = cu

        # Bottom artificial lower bounds
        if params['downarti']:
            # Creation of the x,y new mesh of artificial lower bounds
            xn, yn = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
                np.linspace(X[:, 1].min(), X[:, 1].max(), 20))
            # All the artificial new mesh are going to be below the data
            zng = np.repeat(minz-params['dminz'],len(np.ravel(xn)))
            # Artifitial lower bounds
            Xga = np.c_[np.ravel(xn),np.ravel(yn),np.ravel(zng)]
            # Definition of new ground detections after down artificial lower detections
            Xgn = np.concatenate((Xg,Xga))
            # Definition of new confidence level
            if using_weights:
                Cga = np.ones(len(Xga))*params['confal']
                Cgn = np.concatenate((Cg,Cga))
        else:
            Xgn = Xg
            if using_weights:
                Cgn = Cg

        # Top artificial upper bounds
        if params['toparti']:
            # Creation of the x,y new mesh of artificial upper bounds
            xn, yn = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
                                np.linspace(X[:, 1].min(), X[:, 1].max(), 20))
            # All the artificial new mesh are going to be over the data
            znf = np.repeat(maxz+params['dmaxz'],len(np.ravel(xn)))
            # Artifitial upper bounds
            Xfa = np.c_[np.ravel(xn),np.ravel(yn),np.ravel(znf)]
            # Definition of new fire detections after top artificial upper detections
            Xfn = np.concatenate((Xf,Xfa))
            # Definition of new confidence level
            if using_weights:
                Cfa = np.ones(len(Xfa))*params['confau']
                Cfn = np.concatenate((Cf,Cfa))
        else:
            Xfn = Xf
            if using_weights:
                Cfn = Cf

        # New definition of the training vectors
        X = np.concatenate((Xgn, Xfn))
        # New definition of the target values
        y = np.concatenate((np.repeat(np.unique(y)[0],len(Xgn)),np.repeat(np.unique(y)[1],len(Xfn))))
    	# New definition of the confidence level
        if using_weights:
            C = np.concatenate((Cgn, Cfn))
        # New definition of each feature vector
    	X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]

    # Printing number of samples and features
    n0 = (y==np.unique(y)[0]).sum().astype(float)
    n1 = (y==np.unique(y)[1]).sum().astype(float)
    n_samples, n_features = X.shape
    print 'n_samples =', n_samples
    print 'n_samples_{-1} =', int(n0)
    print 'n_samples_{+1} =', int(n1)
    print 'n_features =', n_features

    # Visualization of scaled data
    if params['plot_scaled']:
        try:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle("Plotting the data scaled to fit")
            ax.scatter(X0, X1, X2, c=y, cmap=cm_GR, s=1, alpha=.5, vmin=y.min(), vmax=y.max())
            ax.set_xlabel("Longitude normalized")
            ax.set_ylabel("Latitude normalized")
            ax.set_zlabel("Time normalized")
            plt.savefig('scaled_data.png')
        except Exception as e:
            print 'Warning: something went wrong when plotting...'
            print e

    # Reescaling gamma to include more detailed results
    gamma = 1. / (n_features * X.std())
    print 'gamma =', gamma

    # Creating the SVM model and fitting the data using Super Vector Machine technique
    print '>> Creating the SVM model...'
    sys.stdout.flush()
    if using_weights:
        gamma = kgam*gamma
        # Compute class balanced weights
        cls, _ = np.unique(y, return_inverse=True)
        class_weight = compute_class_weight("balanced", cls, y)
        prob = svm_problem(C,y,X)
        arg = '-g %.15g -w%01d %.15g -w%01d %.15g -m 1000 -h 0' % (gamma, cls[0], class_weight[0],
                                            cls[1], class_weight[1])
        param = svm_parameter(arg)
        print '>> Fitting the SVM model...'
        t_1 = time()
        clf = svm_train(prob,param)
        t_2 = time()
    else:
        t_1 = time()
        if params['search']:
            print '>> Searching for best value of C and gamma...'
            # Grid Search
            # Parameter Grid
            param_grid = {'C': np.logspace(0,5,6), 'gamma': gamma*np.logspace(0,5,6)}
            # Make grid search classifier
            grid_search = GridSearchCV(svm.SVC(cache_size=2000,class_weight="balanced",probability=True), param_grid, n_jobs=-1, verbose=1, cv=5, iid=False)
            print '>> Fitting the SVM model...'
            # Train the classifier
            grid_search.fit(X, y)
            print "Best Parameters:\n", grid_search.best_params_
            clf = grid_search.best_estimator_
            print "Best Estimators:\n", clf
        else:
            gamma = kgam*gamma
            clf = svm.SVC(C=C, kernel="rbf", gamma=gamma, cache_size=2000, class_weight="balanced") # default kernel: exp(-gamma||x-x'||^2)
            print clf
            print '>> Fitting the SVM model...'
            # Fitting the data using Super Vector Machine technique
            clf.fit(X, y)
        t_2 = time()
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))

    if not using_weights:
        # Check if the classification failed
        if clf.fit_status_:
            print '>> FAILED <<'
            print 'Failed fitting the data'
            sys.exit(1)
        print 'number of support vectors: ', clf.n_support_
        print 'score of trained data: ', clf.score(X,y)

    # Creating the mesh grid to evaluate the classification
    print '>> Creating mesh grid to evaluate the classification...'
    nnodes = np.ceil(np.power(n_samples,1./n_features))
    if fire_grid is None:
        # Number of necessary nodes
        hnodes = hN*nnodes
        vnodes = vN*nnodes
        print 'number of horizontal nodes (%d meshgrid nodes for each observation): %d' % (hN,hnodes)
        print 'number of vertical nodes (%d meshgrid nodes for each observation): %d' % (vN,vnodes)
        # Computing resolution of the mesh to evaluate
        sdim = (hnodes,hnodes,vnodes)
        print 'grid_size = %dx%dx%d = %d' % (sdim[0],sdim[1],sdim[2],np.prod(sdim))
        t_1 = time()
        xx, yy, zz = make_meshgrid(X0, X1, X2, s=sdim)
        t_2 = time()
    else:
        fxlon = np.divide(fire_grid[0] - xmin, xlen)
        fxlat = np.divide(fire_grid[1] - ymin, ylen)
        if it is None:
            it = (X2.min(),X2.max())
        else:
            it = np.divide(np.array(it) - zmin, zlen)
        vnodes = vN*nnodes
        sdim = (fxlon.shape[0],fxlon.shape[1],vnodes)
        print 'fire_grid_size = %dx%dx%d = %d' % (sdim + (np.prod(sdim),))
        t_1 = time()
        xx, yy, zz = make_fire_mesh(fxlon, fxlat, it, sdim[2])
        t_2 = time()
        print 'grid_created = %dx%dx%d = %d' % (zz.shape + (np.prod(zz.shape),))
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))

    # Computing the 2D fire arrival time, F
    print '>> Computing the 2D fire arrival time, F...'
    sys.stdout.flush()
    F = frontier(clf, xx, yy, zz, plot_decision=params['plot_decision'], plot_poly=params['plot_poly'], using_weights=using_weights)

    print '>> Creating final results...'
    sys.stdout.flush()
    # Plotting the Separating Hyperplane of the SVM classification with the support vectors
    if params['plot_supports']:
        try:
            if using_weights:
                supp_ind = np.sort(clf.get_sv_indices())-1
                supp_vec = X[supp_ind]
            else:
                supp_ind = clf.support_
                supp_vec = clf.support_vectors_
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle("Plotting the 3D Separating Hyperplane of an SVM")
            # plotting the separating hyperplane
            ax.plot_surface(F[0], F[1], F[2], color='orange', alpha=.3)
            # computing the indeces where no support vectors
            rr = np.array(range(len(y)))
            ms = np.isin(rr,supp_ind)
            nsupp = rr[~ms]
            # plotting no-support vectors (smaller)
            ax.scatter(X0[nsupp], X1[nsupp], X2[nsupp], c=y[nsupp], cmap=cm_GR, s=.5, vmin=y.min(), vmax=y.max(), alpha=.1)
            # plotting support vectors (bigger)
            ax.scatter(supp_vec[:, 0], supp_vec[:, 1], supp_vec[:, 2], c=y[supp_ind], cmap=cm_GR, s=10, edgecolors='k', linewidth=.5, alpha=.5);
            ax.set_xlim(xx.min(),xx.max())
            ax.set_ylim(yy.min(),yy.max())
            ax.set_zlim(zz.min(),zz.max())
            ax.set_xlabel("Longitude normalized")
            ax.set_ylabel("Latitude normalized")
            ax.set_zlabel("Time normalized")
            plt.savefig('support.png')
        except Exception as e:
            print 'Warning: something went wrong when plotting...'
            print e

    # Plot the fire arrival time resulting from the SVM classification normalized
    if params['plot_result']:
        try:
            Fx, Fy, Fz = np.array(F[0]), np.array(F[1]), np.array(F[2])
            with np.errstate(invalid='ignore'):
                Fz[Fz > X2.max()] = np.nan
            if params['notnan']:
                Fz[np.isnan(Fz)] = X2.max()
                Fz = np.minimum(Fz, X2.max())
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle("Fire arrival time normalized")
            # plotting fire arrival time
            p = ax.plot_surface(Fx, Fy, Fz, cmap=cm_Rds,
                           linewidth=0, antialiased=False)
            ax.set_xlim(xx.min(),xx.max())
            ax.set_ylim(yy.min(),yy.max())
            ax.set_zlim(zz.min(),zz.max())
            cbar = fig.colorbar(p)
            cbar.set_label('Fire arrival time normalized', labelpad=20, rotation=270)
            ax.set_xlabel("Longitude normalized")
            ax.set_ylabel("Latitude normalized")
            ax.set_zlabel("Time normalized")
            plt.savefig('tign_g.png')
        except Exception as e:
            print 'Warning: something went wrong when plotting...'
            print e

    # Translate the result again into initial data scale
    if params['norm']:
        f0 = F[0] * xlen + xmin
        f1 = F[1] * ylen + ymin
        f2 = F[2] * zlen + zmin
        FF = [f0,f1,f2]

    # Set all the larger values at the end to be the same maximum value
    oX0, oX1, oX2 = oX[:, 0], oX[:, 1], oX[:, 2]
    FFx, FFy, FFz = FF[0], FF[1], FF[2]

    if params['notnan']:
        with np.errstate(invalid='ignore'):
            FFz[np.isnan(FFz)] = np.nanmax(FFz)
    FF = [FFx,FFy,FFz]

    # Plot the fire arrival time resulting from the SVM classification
    if params['plot_result']:
        try:
            # Plotting the result
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle("Plotting the 3D graph function of a SVM")
            FFx, FFy, FFz = np.array(FF[0]), np.array(FF[1]), np.array(FF[2])
            # plotting original data
            ax.scatter(oX0, oX1, oX2, c=oy, cmap=cm_GR, s=1, alpha=.5, vmin=y.min(), vmax=y.max())
            # plotting fire arrival time
            ax.plot_wireframe(FFx, FFy, FFz, color='orange', alpha=.5)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_zlabel("Time (days)")
            plt.savefig('result.png')
        except Exception as e:
            print 'Warning: something went wrong when plotting...'
            print e

    print '>> SUCCESS <<'
    t_final = time()
    print 'TOTAL elapsed time: %ss.' % str(abs(t_final-t_init))
    plt.close()

    return FF


if __name__ == "__main__":
    # Experiment to do
    exp = 1
    search = True

    # Defining ground and fire detections
    def exp1():
        Xg = [[0, 0, 0], [2, 2, 0], [2, 0, 0], [0, 2, 0]]
        Xf = [[0, 0, 1], [1, 1, 0], [2, 2, 1], [2, 0, 1], [0, 2, 1]]
        C = 10.
        kgam = 1.
        return Xg, Xf, C, kgam
    def exp2():
        Xg = [[0, 0, 0], [2, 2, 0], [2, 0, 0], [0, 2, 0],
            [4, 2, 0], [4, 0, 0], [2, 1, .5], [0, 1, .5],
            [4, 1, .5], [2, 0, .5], [2, 2, .5]]
        Xf = [[0, 0, 1], [1, 1, 0.25], [2, 2, 1], [2, 0, 1], [0, 2, 1], [3, 1, 0.25], [4, 2, 1], [4, 0, 1]]
        C = np.concatenate((np.array([50.,50.,50.,50.,50.,50.,
                        1000.,100.,100.,100.,100.]), 100.*np.ones(len(Xf))))
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
    SVM3(X,y,C=C,kgam=kgam,search=search,plot_result=True)
