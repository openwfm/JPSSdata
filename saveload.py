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
    try:
        # python 2
        with open(file,'rb') as f:
            return pickle.load(f)
    except:
        # python 3
        with open(file,'rb') as f:
            return pickle.load(f,encoding='latin1')
