from svm import SVM3
import numpy as np
from scipy.io import loadmat

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
    SVM3(X,y,C=C,kgam=kgam,search=search,plot_result=True,plot_data=True)
