import numpy as np
from numpy.linalg import inv

__version__ = 0.1

def gj(b,y,X,Z):
    """Observations of g_j(b).

    This defines the deviations from the predictions of our model; i.e.,
    e_j = Z_ju_j, where EZ_ju_j=0.

    Can replace this function to testimate a different model.
    """
    return Z*(y - X*b)

def gN(b,data):
    """Averages of g_j(b).

    This is generic for data, to be passed to gj.
    """
    e = gj(b,*data)

    # Check to see more obs. than moments.
    assert e.shape[0] > e.shape[1]
    
    return e.mean(axis=0)

def Omegahat(b,data):
    e = gj(b,*data)

    # Recenter! We have Eu=0 under null.
    # Important to use this information.
    e = e - e.mean(axis=0) 
    
    return e.T@e/e.shape[0]

def J(b,W,data):

    m = gN(b,data) # Sample moments @ b
    N = data[0].shape[0]

    return N*m.T@W@m # Scale by sample size

from scipy.optimize import minimize_scalar

def two_step_gmm(data):

    # First step uses identity weighting matrix
    W1 = np.eye(gj(1,*data).shape[1])

    b1 = minimize_scalar(lambda b: J(b,W1,data)).x 

    # Construct 2nd step weighting matrix using
    # first step estimate of beta
    W2 = inv(Omegahat(b1,data))

    return minimize_scalar(lambda b: J(b,W2,data))


def print_version():
    print(__version__)
