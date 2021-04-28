import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize_scalar, minimize
from scipy.optimize import minimize as scipy_min

__version__ = "0.3.1"

######################################################
# Beginning of procedural version of gmm routines
def gj(b,data):
    """Observations of g_j(b).

    This defines the deviations from the predictions of our model; i.e.,
    e_j = Z_ju_j, where EZ_ju_j=0 for *each observation*.

    Returns an Nxl matrix (observaions x moment conditions).

    Can replace this function to testimate a different model.

    If passed value of b is a scalar expand to make a k-vector.
    """
    y,X,Z = data
    
    # Construct vector of identical parameter if b a scalar.
    if np.isscalar(b): b = np.array([b]*X.shape[1]).reshape((-1,1))
        
    return Z*(y - X*b)

def gN(b,data):
    """Averages of g_j(b).

    This is generic for data, to be passed to gj.
    """
    e = gj(b,data)

    # Check to see more obs. than moments.
    assert e.shape[0] > e.shape[1]
    
    return e.mean(axis=0).reshape((-1,1))

def Omegahat(b,data):
    e = gj(b,data)

    # Recenter! We have Eu=0 under null.
    # Important to use this information.
    e = e - e.mean(axis=0) 
    
    return e.T@e/e.shape[0]

def J(b,W,data):

    m = gN(b,data) # Sample moments @ b
    N = data[0].shape[0]

    return (N*m.T@W@m).squeeze() # Scale by sample size

def minimize(f,b_init=None):
    if b_init is None:
        return minimize_scalar(f).x
    else:
        return scipy_min(f,b_init).x

def one_step_gmm(data,W=None,b_init=None):

    if b_init is None:
        b_init = 0

    if W is None:
        W = np.eye(gj(b_init,data).shape[1])

    b = minimize(lambda b: J(b,W,data),b_init=b_init)

    return b, J(b,W,data)

def two_step_gmm(data,b_init=None):

    # First step uses identity weighting matrix
    b1 = one_step_gmm(data,b_init=b_init)[0]

    # Construct 2nd step weighting matrix using
    # first step estimate of beta
    W2 = inv(Omegahat(b1,data))

    return one_step_gmm(data,W=W2,b_init=b1)

def continuously_updated_gmm(data,b_init=None):

    # First step uses identity weighting matrix
    W = lambda b: np.inv(Omegahat(b,data))

    bhat = minimize(lambda b: J(b,W(b),data))

    return bhat

def print_version():
    print(__version__)

# End of procedural version of gmm routines
######################################################

