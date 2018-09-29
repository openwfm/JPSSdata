# following https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
import numpy as np

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours

from pprint import pprint 
import matplotlib.pyplot 
z = [
     [1.4, 1, 1.4], 
     [1, 0, 1], 
     [1.4, 1, 1.4]
     ] 
      
cn = matplotlib.pyplot.contour(z) 

contours = get_contour_verts(cn)

print 'contours are collections of line, each line consisting of poins with x and y coordinates'

for c in contours:
    print c
