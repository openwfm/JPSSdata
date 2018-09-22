import numpy as np
import saveload as sl
import JPSSD as J

data,fxlon,fxlat=sl.load('data')
U,L,T=sl.load('result')
xx=fxlon
yy=fxlat
zz=U 
J.plot_3D(xx,yy,zz)

zz=L
J.plot_3D(xx,yy,zz)

zz=T
J.plot_3D(xx,yy,zz)
