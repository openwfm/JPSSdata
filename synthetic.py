import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as colors
from svm import SVM3
from scipy.io import savemat
from scipy import interpolate

def plot_case(xx,yy,tign_g,X_satellite=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.contour(xx,yy,tign_g,30)
    if X_satellite is not None:
        ax.scatter(X_satellite[:,0],X_satellite[:,1],
                X_satellite[:,2],s=5,color='r')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("T")
    plt.savefig('syn_case.png')

def plot_data(X,y):
    col = [(0, .5, 0), (.5, 0, 0)]
    cm_GR = colors.LinearSegmentedColormap.from_list('GrRd',col,N=2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                c=y, cmap=cm_GR, s=1, alpha=.5, 
                vmin=y.min(), vmax=y.max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("T")
    plt.savefig('syn_data.png')

def cone_point(xx,yy,nx,ny):
    cx = nx*.5
    cy = ny*.5
    tign_g = np.minimum(1e3,10+(2e3/cx)*np.sqrt(((xx-cx)**2+(yy-cy)**2)/2))
    tsat = (tign_g.max()-tign_g.min())*.5
    tt1d = np.ravel(tign_g)
    mask = tt1d < tt1d.max()
    xx1d = np.ravel(xx)[mask]
    yy1d = np.ravel(yy)[mask]
    tt1d = tt1d[mask]
    X_satellite = np.array([[cx*.7,cy*.7,tsat]])
    return tign_g,xx1d,yy1d,tt1d,X_satellite

def cone_points(xx,yy,nx,ny):
    cx = nx*.5
    cy = ny*.5
    tign_g = np.minimum(1e3,10+(2e3/cx)*np.sqrt(((xx-cx)**2+(yy-cy)**2)/2))
    tsat = (tign_g.max()-tign_g.min())*.5
    tt1d = np.ravel(tign_g)
    mask = tt1d < tt1d.max()
    xx1d = np.ravel(xx)[mask]
    yy1d = np.ravel(yy)[mask]
    tt1d = tt1d[mask]
    N = 10
    X_satellite = np.c_[np.linspace(cx*.7,cx,N+1),
                        np.linspace(cy*.7,cy,N+1),
                        np.linspace(tsat,tign_g.min(),N+1)][:-1]
    return tign_g,xx1d,yy1d,tt1d,X_satellite

def slope(xx,yy,nx,ny):
    ros = (10,30) # rate of spread
    cx = round(nx*.5)
    s1 = 10+np.arange(0,cx*ros[0],ros[0])
    s2 = ros[1]+np.arange(cx*ros[0],cx*ros[0]+(nx-cx)*ros[1],ros[1])
    s = np.concatenate((s1,s2))
    tign_g = np.reshape(np.repeat(s,ny),(nx,ny)).T
    xx1d = np.ravel(xx)
    yy1d = np.ravel(yy)
    tt1d = np.ravel(tign_g)
    X_satellite = None
    return tign_g,xx1d,yy1d,tt1d,X_satellite

def preprocess_svm(xx,yy,tt,epsilon,weights,X_satellite=None):
    wforecastg,wforecastf,wsatellite = weights
    for_fire = np.c_[xx.ravel(),yy.ravel(),tt.ravel() + epsilon]
    for_nofire = np.c_[xx.ravel(),yy.ravel(),tt.ravel() - epsilon]
    X_forecast = np.concatenate((for_nofire,for_fire))
    y_forecast = np.concatenate((-np.ones(len(for_nofire)),np.ones(len(for_fire))))
    c_forecast = np.concatenate((wforecastg*np.ones(len(for_nofire)),wforecastf*np.ones(len(for_fire))))
    if X_satellite is not None:
        X = np.concatenate((X_forecast,X_satellite)) 
        y = np.concatenate((y_forecast,np.ones(len(X_satellite))))
        c = np.concatenate((c_forecast,wsatellite*np.ones(len(X_satellite))))
    else:
        X = X_forecast
        y = y_forecast
        c = c_forecast
    return X,y,c

if __name__ == "__main__":
    ## SETTINGS
    # Experiments: 1) Cone with point, 2) Slope, 3) Cone with points
    exp = 2
    # hyperparameter settings
    wforecastg = 50
    wforecastf = 50
    wsatellite = 50
    kgam = 1
    # epsilon for artificial forecast in seconds
    epsilon = 1 
    # dimensions
    nx, ny = 50, 50
    # plotting data before svm?
    plot = True

    ## CASE
    xx,yy = np.meshgrid(np.arange(0,nx,1),
                        np.arange(0,ny,1))
    # select experiment
    experiments = {1: cone_point, 2: slope, 3: cone_points}
    tign_g,xx1d,yy1d,tt1d,X_satellite = experiments[exp](xx,yy,nx,ny)
    if plot:
        plot_case(xx,yy,tign_g,X_satellite)

    ## PREPROCESS
    if X_satellite is None:
        wsatellite = 0
    X,y,c = preprocess_svm(xx1d,yy1d,tt1d,epsilon,
                    (wforecastg,wforecastf,wsatellite), 
                    X_satellite)
    if plot:
        plot_data(X,y)

    ## SVM
    # options for SVM
    options = {'downarti': False, 'plot_data': True, 
                'plot_scaled': True, 'plot_supports': True, 
                'plot_result': True, 'plot_decision': True,
                'artiu': False, 'hartiu': .2, 
                'artil': False, 'hartil': .2,
                'notnan': True}
    if (wforecastg == wforecastf and
        (wsatellite == 0 or wsatellite == wforecastg)):
        c = wforecastg
    # running SVM
    F = SVM3(X, y, C=c, kgam=kgam, **options)

    ## POSTPROCESS
    # interpolation to validate
    points = np.c_[np.ravel(F[0]),np.ravel(F[1])]
    values = np.ravel(F[2])
    zz_svm = interpolate.griddata(points,values,(xx,yy))
    # output dictionary
    svm = {'xx': xx, 'yy': yy, 'zz': tign_g, 
            'zz_svm': zz_svm, 'X': X, 'y': y, 'c': c, 
            'fxlon': F[0], 'fxlat': F[1], 'Z': F[2], 
            'epsilon': epsilon, 'options': options}
    # output file
    if wsatellite:
        filename = 'syn_fg%d_ff%d_s%d_k%d_e%d.mat' % (wforecastg,wforecastf,
                                                        wsatellite,kgam,epsilon)
    else:
        filename = 'syn_fg%d_ff%d_k%d_e%d.mat' % (wforecastg,wforecastf,
                                                        kgam,epsilon)
    savemat(filename, mdict=svm)
    print 'plot_svm %s' % filename
