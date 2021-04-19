import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize_scalar, minimize

__version__ = 0.2

def gj(b,data):
    """Observations of g_j(b).

    This defines the deviations from the predictions of our model; i.e.,
    e_j = Z_ju_j, where EZ_ju_j=0.

    Can replace this function to testimate a different model.
    """
    y,X,Z = data
    return Z*(y - X*b)

class GMM(object):

    def __init__(self,gj,data,B,W=None):
        """GMM problem for restrictions E(gj(b0))=0, estimated using data with b0 in R^k.

           - If supplied B is a positive integer k, then 
             space taken to be R^k.  
           - If supplied B is a k-vector, then
             parameter space taken to be R^k with B a possible
             starting value for optimization.
        """
        self.gj = gj
        self.data = data

        self.W = W

        try:
            self.k = len(B)
            self.b_init = np.array(B)
        except TypeError:
            self.k = B
            self.b_init = np.zeros(self.k)

        self.ell = gj(self.b_init,self.data).shape[1]

        if type(data) is tuple:
            self.N = data[0].shape[0]
        else:
            self.N = data.shape[0]

        if self.k == 1:
            self.minimize = lambda f: minimize_scalar(f).x
        else:
            self.minimize = lambda f,b_init=self.b_init: minimize(f,b_init).x
            
            
    def gN(self,b):
        """Averages of g_j(b).

        This is generic for data, to be passed to gj.
        """
        e = self.gj(b,self.data)

        # Check to see more obs. than moments.
        assert e.shape[0] > e.shape[1]

        return e.mean(axis=0)

    def Omegahat(self,b):

        e = self.gj(b,self.data)

        # Recenter! We have Eu=0 under null.
        # Important to use this information.
        e = e - e.mean(axis=0) 

        return e.T@e/e.shape[0]

    def J(self,b,W):

        m = self.gN(b) # Sample moments @ b
        N = self.N

        return N*m.T@W@m # Scale by sample size

    def two_step_gmm(self):

        # First step uses identity weighting matrix
        W1 = np.eye(self.ell)

        b1 = self.minimize(lambda b: self.J(b,W1))

        # Construct 2nd step weighting matrix using
        # first step estimate of beta
        W2 = inv(self.Omegahat(b1))

        return self.minimize(lambda b: self.J(b,W2),b_init=b1)

    def continuously_updated_gmm():

        # First step uses identity weighting matrix
        W = lambda b: np.inv(self.Omegahat(b))

        bhat = self.minimize(lambda b: self.J(b,W(b)))

        return bhat

def print_version():
    print(__version__)
