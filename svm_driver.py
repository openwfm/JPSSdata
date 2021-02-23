from svm import SVM3
import numpy as np

 
for exp in (1,2):

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
    SVM3(X,y,C=C,kgam=kgam,search=search,plot_result=True,plot_data=True)
