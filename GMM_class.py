import gmm
import numpy as np

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
        gmm.gj = gj  # Overwrite member of gmm module
        self.data = data

        self.W = W

        self.b = None

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

        self.minimize = gmm.minimize
            
    def gN(self,b):
        """Averages of g_j(b).

        This is generic for data, to be passed to gj.
        """
        return gmm.gN(b,self.data)

    def Omegahat(self,b):

        return gmm.Omegahat(b,self.data)
    
    def J(self,b,W):

        return gmm.J(b,W,self.data)

    def one_step_gmm(self,W=None,b_init=None):

        self.b = gmm.one_step_gmm(self.data,W,b_init=self.b_init)

        return self.b
    
    def two_step_gmm(self):

        self.b = gmm.two_step_gmm(self.data,b_init=self.b_init)[0]

        return self.b

    def continuously_updated_gmm(self):

        self.b = gmm.continuously_updated_gmm(self.data,b_init=self.b_init)[0]
        
        return self.b



if __name__=='__main__':
    #foo = GMM(gmm.gj,
    pass
