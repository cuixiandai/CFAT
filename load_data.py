import numpy as np
from spectral.io import envi
import os.path
from pathlib import Path
import scipy.io as sio

def load_data(name):
    if name == 'FL_T':
        path = 'Datasets/Flevoland/T_Flevoland_14cls.mat'
        first_read =sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)
        T[: ,:, 0]=first_read
        del first_read
        T[: ,:, 1]=sio.loadmat(path)['T12']
        T[: ,:, 2]=sio.loadmat(path)['T13']
        T[: ,:, 3]=sio.loadmat(path)['T22']
        T[: ,:, 4]=sio.loadmat(path)['T23']
        T[: ,:, 5]=sio.loadmat(path)['T33']

        labels = sio.loadmat('Datasets/Flevoland/Flevoland_gt.mat')['gt'] 
##############################################################################

    elif name == 'SF':
        T = sio.loadmat('Datasets/san_francisco/SanFrancisco_Coh.mat')['T']
        T=T.astype(np.complex64)
        
        labels = sio.loadmat('Datasets/san_francisco/SanFrancisco_gt.mat')['gt'] 

##############################################################################
    elif name == 'ober':
        path = 'Datasets/Oberpfaffenhofen/T_Germany.mat'
        first_read =sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)
        T[: ,:, 0]=first_read
        del first_read
        T[: ,:, 1]=sio.loadmat(path)['T12']
        T[: ,:, 2]=sio.loadmat(path)['T13']
        T[: ,:, 3]=sio.loadmat(path)['T22']
        T[: ,:, 4]=sio.loadmat(path)['T23']
        T[: ,:, 5]=sio.loadmat(path)['T33']

        labels = sio.loadmat('Datasets/Oberpfaffenhofen/Label_Germany.mat')['label']

##############################################################################
    else:
        print("Incorrect data name")
        
    return T, labels

