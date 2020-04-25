from scipy.stats import distributions as iid

# Code to convolve two random variables having pmfs.
# Exploits =scipy.stats= base rv_discrete class.
#
# Credit to ARE212 Spring 2020 for ConvolvedDiscrete Class.

class ConvolvedDiscrete(iid.rv_discrete):
    """Class to convolve two discrete random variables.
    """
    def __init__(self, r, s):
        self.discrete_rv1 = r
        self.discrete_rv2 = s
        super(ConvolvedDiscrete, self).__init__(name="ConvolvedDiscrete")
        
    def _pmf(self, z):
        f = 0
        r = self.discrete_rv1
        s = self.discrete_rv2
        
        for k in range(len(s.xk)):
            f = f + r.pmf(z - s.xk[k])*s.pk[k]
        return f
        
    def _cdf(self, z):
        F = 0
        r = self.discrete_rv1
        s = self.discrete_rv2
        
        for k in range(len(s.xk)):
            F = F + r.cdf(z - s.xk[k])*s.pk[k]
        return F


from scipy.stats import distributions as iid

# Code to convolve a random variable with a pmf and another having a cdf
# Exploits =scipy.stats= base rv_continuous class.

class ConvolvedContinuousAndDiscrete(iid.rv_continuous):

    """Convolve (add) a continuous rv x and a discrete rv s,
       returning the resulting cdf."""

    def __init__(self,f,s):
        self.continuous_rv = f
        self.discrete_rv = s
        super(ConvolvedContinuousAndDiscrete, self).__init__(name="ConvolvedContinuousAndDiscrete")
        
    def _cdf(self,z):
        F=0
        s = self.discrete_rv
        x = self.continuous_rv
        
        for k in range(len(s.xk)):
            F = F + x.cdf(z-s.xk[k])*s.pk[k]
        return F

    def _pdf(self,z):
        f=0
        s = self.discrete_rv
        x = self.continuous_rv
        
        for k in range(len(s.xk)):
            f = f + x.pdf(z-s.xk[k])*s.pk[k]
        return f


if __name__ == "__main__":  # If running this code instead of importing...
    x = iid.norm()    # Create continuous rv

    Omega = (-1, 0, 1)  # Sample space for discrete rvs

    # Create two discrete rvs
    r = iid.rv_discrete(values=(Omega, (1/3., 1/2., 1/6.)))
    s = iid.rv_discrete(values=(Omega, (5/6., 1/12., 1/12.)))

    # Create new convolved rv:
    y = ConvolvedContinuousAndDiscrete(x, s)

    t = ConvolvedDiscrete(r, s)
