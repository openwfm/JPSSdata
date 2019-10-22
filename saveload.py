# a simple utility to save to file and load back
import pickle

def save(obj,file):
    """
    :param obj: object to be saved
    :param file: file name
    """
    with open(file,'wb') as f:
       pickle.dump(obj,f,protocol=-1)

def load(file):
    """
    :param file: file name
    :return: the object read
    """
    with open(file,'rb') as f:
        try:
            # python 2
            return pickle.load(f)
        except ImportError:
            # python 3
            return pickle.load(f,encoding='latin1')
