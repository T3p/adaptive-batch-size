import numpy as np
import math

"""Helper functions"""


def apply_along_axis2(func1d,axis,X,Y):
    """Extension of numpy.apply_along_axis to functions of two parameters"""
    if len(X.shape)<=axis:
        X = np.expand_dims(X,axis=axis)
 
    if len(Y.shape)<=axis:
        Y = np.expand_dims(Y,axis=axis)   

    split = X.shape[axis]
    Z = np.concatenate((X,Y),axis)    

    def aux(z):
        return func1d(z[:split],z[split:])

    return np.apply_along_axis(aux,axis,Z)


def inv(x):
    """Generalization of np.linalg.inv including scalars"""
    return 1./x if np.isscalar(x) else np.linalg.inv(x)

def det(x):
    """Generalization of np.linalg.det including scalars"""
    return x if np.isscalar(x) else np.linalg.det(x)

def cholesky(x):
    """Generalization of np.linalg.cholesky including scalars"""
    return math.sqrt(x) if np.isscalar(x) else np.linalg.cholesky(x)
