#
# Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
# Angel Farguell (angel.farguell@gmail.com)
#
# to install:
#       conda install scikit-learn
#       conda install scikit-image

from sklearn import svm
from scipy import interpolate, spatial
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
from time import time

def preprocess_data_svm(lons, lats, U, L, T, scale, time_num_granules):
    """
    Preprocess satellite data from JPSSD and setup to use in Support Vector Machine

    :param lon:
    :param lat:
    :param U:
    :param L:
    :param T:
    :param scale:
    :param time_num_granules:
    :return X: matrix of features for SVM
    :return y: vector of labels for SVM

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-04-01
    """

    # Flatten coordinates
    lon = np.reshape(lons,np.prod(lons.shape)).astype(float)
    lat = np.reshape(lats,np.prod(lats.shape)).astype(float)

    # Temporal scale to days
    tscale = 24*3600
    U = U/tscale
    L = L/tscale

    # Ensuring U>=L always
    print 'U>L: ',(U>L).sum()
    print 'U<L: ',(U<L).sum()
    print 'U==L: ',(U==L).sum()

    # Reshape to 1D
    uu = np.reshape(U,np.prod(U.shape))
    ll = np.reshape(L,np.prod(L.shape))
    tt = np.reshape(T,np.prod(T.shape))

    # Maximum and minimums to NaN data
    uu[uu==uu.max()] = np.nan
    ll[ll==ll.min()] = np.nan

    # Mask created during computation of lower and upper bounds
    mk = tt==scale[1]-scale[0]
    # Masking upper bounds outside the mask
    uu[mk] = np.nan
    # Creating maximum value considered of the upper bounds
    nuu = uu[~np.isnan(uu)]
    muu = nuu.max() # could be a different value like a mean value
    # Create a mask with lower bound less than the previous maximum upper bound value
    with np.errstate(invalid='ignore'):
        low = (ll <= muu)
    if low.sum() > 10000:
        # Create a mask with all False of low size
        mask = np.repeat(False,len(low[low == True]))
        # Take just a subset of the nodes
        clear_level = 50
        mask[0::clear_level] = True
        # Mask the subset
        low[low == True] = mask
    # Eliminate all the previous elements from the mask
    mk[low] = False
    # Masking lower bounds outside the mask
    ll[mk] = np.nan

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

    # Create the data to call SVM3 function from svm3test.py
    X = np.c_[np.concatenate((lx,ux)),np.concatenate((ly,uy)),np.concatenate((lz,uz))]
    y = np.concatenate((-np.ones(len(lx)),np.ones(len(ux))))
    # Print the shape of the data
    print 'shape X: ', X.shape
    print 'shape y: ', y.shape

    return X,y

def make_fire_mesh(fxlon, fxlat, it, nt, coarse=10):
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

    coarse = int(coarse)
    x = fxlon[0][0::coarse]
    y = fxlat[:,0][0::coarse]
    xx, yy, zz = np.meshgrid(x,y,np.linspace(it[0],it[1],nt))

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
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, sx),
                             np.linspace(y_min, y_max, sy),
                             np.linspace(z_min, z_max, sz))
    return xx, yy, zz

def frontier(clf, xx, yy, zz, bal=.5, plot_poly=False):
    """
    Compute the surface decision frontier for a classifier.

    :param clf: a classifier
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param zz: meshgrid ndarray
    :param bal: number between 0 and 1, balance between lower and upper bounds in decision function (in case not level 0)
    :param plot_poly: boolean of plotting polynomial approximation (if tech=='poly')
    :return F: 2D meshes with xx, yy coordinates and the hyperplane z which gives decision functon 0

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-02-20
    Modified version of:
    https://www.semipol.de/2015/10/29/SVM-separating-hyperplane-3d-matplotlib.html
    """

    # Creating the 3D grid
    XX = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Evaluating the decision function
    print '>> Evaluating the decision function...'
    t_1 = time()
    Z = clf.decision_function(XX)
    t_2 = time()
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))
    hist = np.histogram(Z)
    print 'counts: ', hist[0]
    print 'values: ', hist[1]
    print 'decision function range: ', Z.min(), '~', Z.max()

    # Reshaping decision function volume
    Z = Z.reshape(xx.shape)
    print 'decision function shape: ', Z.shape

    if plot_poly:
        fig = plt.figure()
    # Computing fire arrival time from previous decision function
    print '>> Computing fire arrival time...'
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
            else:
                # If there is not a real root of the polynomial between zz.min() and zz.max(), just define as a Nan
                Fz[k1,k2] = np.nan
    t_2 = time()
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))
    F = np.array([Fx,Fy,Fz.T])

    return F

def SVM3(X, y, C=1., kgam=1., norm=True, fire_grid=None):
    """
    3D SuperVector Machine analysis and plot

    :param X: Training vectors, where n_samples is the number of samples and n_features is the number of features.
    :param y: Target values
    :param C: Weight to not having outliers (argument of svm.SVC class), optional
    :param kgam: Scalar multiplier for gamma (capture more details increasing it)
    :param norm: Normalize the data in the interval (0,1) in all the directions, optional
    :param fire_grid: The longitud and latitude grid where to have the fire arrival time
    :return F: tuple with (longitude grid, latitude grid, fire arrival time grid)

    Developed in Python 2.7.15 :: Anaconda 4.5.10, on MACINTOSH.
    Angel Farguell (angel.farguell@gmail.com), 2019-02-20
    Modified version of:
    https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py
    """

    t_init = time()

    # Plot options
    # plot original data
    plot_data = False
    # plot scaled data with artificial data
    plot_scaled = False
    # plot polynomial approximation (if postech=='poly')
    plot_poly = False
    # plot full hyperplane vs detections with support vectors
    plot_supports = False
    # plot resulting fire arrival time vs detections
    plot_result = False

    # Other options
    # number of horizontal nodes per observation
    hN = 5
    # number of vertical nodes per observation
    vN = 1
    # resolution of artificial upper bounds vertical to the fire detections
    hartil = .15
    # resolution of artificial lower bounds vertical to the ground detections
    hartiu = .05
    # creation of over and under artificial upper and lower bounds in the pre-processing
    arti = True
    # creation of an artifitial mesh of top upper bounds
    toparti = True
    # creation of an artifitial mesh of down lower bounds
    downarti = True
    # proportion over max of z direction for upper bound artifitial creation
    dmaxz = 0.1
    # below min of z direction for lower bound artifitial creation
    dminz = 0.1
    # if not Nans in the data are wanted (all Nans are going to be replaced by the maximum value)
    notnan = True
    # coarsening of the fire mesh
    coarse = 10
    # interpolate in to the original fire mesh
    interp = True

    # Data inputs
    X = np.array(X).astype(float)
    y = np.array(y)

    # Original data
    oX = np.array(X).astype(float)
    oy = np.array(y)

    # Visualization of the data
    X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]
    if plot_data:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.suptitle("Plotting the original data to fit")
        ax.scatter(X0, X1, X2, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k', vmin=y.min(), vmax=y.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Normalization of the data into [0,1]^3
    if norm:
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
    if arti:
        # Extreme values at z direction
        minz = X[:, 2].min()
        maxz = X[:, 2].max()
        # Division of lower and upper bounds
        fl = X[y==np.unique(y)[0]]
        fu = X[y==np.unique(y)[1]]
        # Create artificial lower bounds
        flz = np.array([ np.unique(np.append(np.arange(f[2],minz,-hartil),f[2])) for f in fl ])
        # Create artificial upper bounds
        fuz = np.array([ np.unique(np.append(np.arange(f[2],maxz,hartiu),f[2])) for f in fu ])
        # Definition of new ground detections after artificial detections added
        Xg = np.concatenate([ np.c_[(np.repeat(fl[k][0],len(flz[k])),np.repeat(fl[k][1],len(flz[k])),flz[k])] for k in range(len(flz)) ])
        # Definition of new fire detections after artificial detections added
        Xf = np.concatenate([ np.c_[(np.repeat(fu[k][0],len(fuz[k])),np.repeat(fu[k][1],len(fuz[k])),fuz[k])] for k in range(len(fuz)) ])
        # Top artificial upper bounds
        if toparti:
            # Creation of the x,y new mesh of artificial upper bounds
            xn, yn = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
                                np.linspace(X[:, 1].min(), X[:, 1].max(), 20))
            # All the artificial new mesh are going to be over the data
            znf = np.repeat(maxz+dmaxz,len(xn.ravel()))
            # Artifitial upper bounds
            Xfa = np.c_[(xn.ravel(),yn.ravel(),znf.ravel())]
            # Definition of new fire detections after top artificial upper detections
            Xfn = np.concatenate((Xf,Xfa))
            if downarti:
                # All the artificial new mesh are going to be below the data
                zng = np.repeat(minz-dminz,len(xn.ravel()))
                # Artifitial lower bounds
                Xga = np.c_[(xn.ravel(),yn.ravel(),zng.ravel())]
                # Definition of new ground detections after down artificial lower detections
                Xgn = np.concatenate((Xg,Xga))
                # New definition of the training vectors
                X = np.concatenate((Xgn, Xfn))
                # New definition of the target values
                y = np.concatenate((np.repeat(np.unique(y)[0],len(Xgn)),np.repeat(np.unique(y)[1],len(Xfn))))
            else:
                # New definition of the training vectors
                X = np.concatenate((Xg, Xfn))
                # New definition of the target values
                y = np.concatenate((np.repeat(np.unique(y)[0],len(Xg)),np.repeat(np.unique(y)[1],len(Xfn))))
        else:
            # New definition of the training vectors
            X = np.concatenate((Xg, Xf))
            # New definition of the target values
            y = np.concatenate((np.repeat(np.unique(y)[0],len(Xg)),np.repeat(np.unique(y)[1],len(Xf))))
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
    if plot_scaled:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.suptitle("Plotting the data scaled to fit")
        ax.scatter(X0, X1, X2, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k', vmin=y.min(), vmax=y.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Reescaling gamma to include more detailed results
    gamma = kgam / (n_features * X.std())
    print 'gamma =', gamma

    # Creating the SVM model
    print '>> Creating the SVM model...'
    clf = svm.SVC(C=C, kernel="rbf", gamma=gamma, cache_size=1000, class_weight="balanced") # default kernel: exp(-gamma||x-x'||^2)
    print clf

    # Fitting the data using Super Vector Machine technique
    print '>> Fitting the SVM model...'
    t_1 = time()
    clf.fit(X, y)
    t_2 = time()
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))

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
        it = (X2.min(),X2.max())
        vnodes = vN*nnodes
        sdim = (len(fxlon),len(fxlat),vnodes)
        print 'fire_grid_size = %dx%dx%d = %d' % (sdim + (np.prod(sdim),))
        t_1 = time()
        xx, yy, zz = make_fire_mesh(fxlon, fxlat, it, sdim[2], coarse=coarse)
        t_2 = time()
        print 'grid_created = %dx%dx%d = %d' % (zz.shape + (np.prod(zz.shape),))
    print 'elapsed time: %ss.' % str(abs(t_2-t_1))

    # Computing the 2D fire arrival time, F
    print '>> Computing the 2D fire arrival time, F...'
    F = frontier(clf, xx, yy, zz, plot_poly=plot_poly)

    print '>> Creating final results...'
    # Plotting the Separating Hyperplane of the SVM classification with the support vectors
    if plot_supports:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.suptitle("Plotting the 3D Separating Hyperplane of an SVM")
        # plotting the separating hyperplane
        ax.plot_wireframe(F[0], F[1], F[2], color='orange')
        # computing the indeces where no support vectors
        rr = np.array(range(len(y)))
        ms = np.isin(rr,clf.support_)
        nsupp = rr[~ms]
        # plotting no-support vectors (smaller)
        ax.scatter(X0[nsupp], X1[nsupp], X2[nsupp], c=y[nsupp], cmap=plt.cm.coolwarm, s=.5, vmin=y.min(), vmax=y.max())
        # plotting support vectors (bigger)
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], clf.support_vectors_[:, 2], c=y[clf.support_], cmap=plt.cm.coolwarm, s=20, edgecolors='k');
        ax.set_xlim(xx.min(),xx.max())
        ax.set_ylim(yy.min(),yy.max())
        ax.set_zlim(zz.min(),zz.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Plot the fire arrival time resulting from the SVM classification normalized
    if plot_result and (F.size != 0):
        Fx, Fy, Fz = np.array(F[0]), np.array(F[1]), np.array(F[2])
        with np.errstate(invalid='ignore'):
            Fz[Fz > X2.max()] = np.nan
        if notnan:
            Fz[np.isnan(Fz)] = X2.max()
            Fz = np.minimum(Fz, X2.max())
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # plotting fire arrival time
        p = ax.plot_surface(Fx, Fy, Fz, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax.set_xlim(xx.min(),xx.max())
        ax.set_ylim(yy.min(),yy.max())
        ax.set_zlim(zz.min(),zz.max())
        fig.colorbar(p)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Translate the result again into initial data scale
    if norm:
        f0 = F[0] * xlen + xmin
        f1 = F[1] * ylen + ymin
        f2 = F[2] * zlen + zmin
        F = np.array([f0,f1,f2])

    # Set all the larger values at the end to be the same maximum value
    oX0, oX1, oX2 = oX[:, 0], oX[:, 1], oX[:, 2]
    Fx, Fy, Fz = F[0], F[1], F[2]
    with np.errstate(invalid='ignore'):
        Fz[Fz > oX2.max()] = np.nan
    if notnan:
        Fz[np.isnan(Fz)] = oX2.max()
        Fz = np.minimum(Fz, oX2.max())

    if (not fire_grid is None) and (interp):
        Flon = fire_grid[0]
        Flat = fire_grid[1]
        points = np.c_[Fx.ravel(),Fy.ravel()]
        values = Fz.ravel()
        Ffire = interpolate.griddata(points,values,(Flon,Flat))
        F = np.array([Flon,Flat,Ffire])

    # Plot the fire arrival time resulting from the SVM classification
    if plot_result and (F.size != 0):
        # Plotting the result
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.suptitle("Plotting the 3D graph function of a SVM")
        Fx, Fy, Fz = F[0], F[1], F[2]
        # plotting original data
        ax.scatter(oX0, oX1, oX2, c=oy, cmap=plt.cm.coolwarm, s=2, vmin=y.min(), vmax=y.max())
        # plotting fire arrival time
        ax.plot_wireframe(Fx, Fy, Fz, color='orange')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    print '>> SUCCESS <<'
    t_final = time()
    print 'TOTAL elapsed time: %ss.' % str(abs(t_final-t_init))

    plt.show()

    return F


if __name__ == "__main__":
    # Experiment to do
    exp = 1

    # Defining ground and fire detections
    def exp1():
        Xg = [[0, 0, 0], [2, 2, 0], [2, 0, 0], [0, 2, 0]]
        Xf = [[0, 0, 1], [1, 1, 0], [2, 2, 1], [2, 0, 1], [0, 2, 1]]
        return Xg, Xf
    def exp2():
        Xg = [[0, 0, 0], [2, 2, 0], [2, 0, 0], [0, 2, 0], [4, 2, 0], [4, 0, 0], [2, 1, 0.5]]
        Xf = [[0, 0, 1], [1, 1, 0.25], [2, 2, 1], [2, 0, 1], [0, 2, 1], [3, 1, 0.25], [4, 2, 1], [4, 0, 1]]
        return Xg, Xf

    # Creating the options
    options = {1 : exp1, 2 : exp2}

    # Defining the option depending on the experiment
    Xg, Xf = options[exp]()

    # Creating the data necessary to run SVM3 function
    X = np.concatenate((Xg, Xf))
    y = np.concatenate((-np.ones(len(Xg)), np.ones(len(Xf))))

    # Running SVM classification
    SVM3(X,y,C=10.)
