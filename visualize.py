import numpy as np
import saveload as sl
import JPSSD as J

data,fxlon,fxlat=sl.load('data')
U,L,T=sl.load('result')
xx=fxlon
yy=fxlat
zz=U 
zz[zz==np.inf]=99999999999
J.plot_3D(xx,yy,zz)

zz=L
zz[zz==-np.inf]=0
J.plot_3D(xx,yy,zz)

zz=T
zz[zz==np.inf]=99999999999
J.plot_3D(xx,yy,zz)

