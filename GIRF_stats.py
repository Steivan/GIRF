# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""
import numpy as np
from scipy import optimize, stats
from scipy.stats import distributions, rv_discrete, rv_continuous, uniform
import scipy.special

# **********************************************************

def get_E_t(T, E_0, E_T):
    b   = np.log(E_T / E_0) / (T-1)
    t_  = np.arange(T)
    E_t = E_0 * np.exp(b*t_)
    E1  = E_0 * np.exp(b*(T+1))   
    return E_t, E1                       # array(0...T-1), scalar  T+1   
    
# *********************************

def logN_E_V(mu, s):
    s2 = s**2
    E  = np.exp(mu+s2/2)
    V  = E**2 * (np.exp(s2) - 1.)
    return E, V
    
def logN_mu_s(E, V):
    s2 = np.log(1. + V / (E**2))
    mu = np.log(E) - s2/2 
    return mu, s2**0.5

def normalize(p):
    p_ = np.array(p)                      # convert into numpy array 
    is_neg = p_ < 0.                      # identify negative values
    p_ = p_ * (1-is_neg)                  # set negative values to zero
    if p_.max() == 0.: p_[-1] = 1.        # ensure that sum is positive
    return p_ / p_.sum()                  # return normaplized probabilities

def get_counts(x, y_lo=None, y_hi=None):
    # Get support y and counts c from array x with (simulated) discrete random variables 
    # - x   : array[K] with discrete random variables
    # - [ymin, ..., ymax]: default range of support
    # - [y_lo, ..., y_hi]: optional extended bounds for support
    # - y   : array[n] with support
    # - c   : array[n] with counts
    ymin = x.min()
    ymax = x.max()
    if y_lo is not None: ymin = min(y_lo, ymin)
    if y_hi is not None: ymax = max(y_hi, ymax)
    n = int(ymax+1-ymin)
    y = np.linspace(ymin, ymax, n)
    c = np.zeros(n)
    for i, y_i in enumerate(y):
        c[i] = (x == y_i).sum()    # count instances with x[k] == y_i
    return y, c    

def get_emp_pmf(x):
    # Get support emp_x and emp_pmf from array[n] x with simulated discrete random variables 
    x_       = np.array(x)
    emp_x, y = get_counts(x_, y_lo=0, y_hi=x_.max()+1)
    return emp_x, normalize(y)

def get_np_B(mean, f_Panjer):
    n = int(np.ceil(mean / (1-f_Panjer)))
    p = mean / n
    return n, p
    
def get_rp_NB(mean, f_Panjer):
    r = mean / (f_Panjer - 1)
    p = 1 / f_Panjer
    return r, p
    
def get_Panjer_par(lda, f_Panjer):
# E[N] = lda >= 0.0 and Var[N] = lda * f_Panjer > 0.0
#   Remark: Distribution with lda = integer and f_Panjer = 0.0 is not member of the Panjer class !!
# Degenerate : lda = 0.0 or f_Panjer = infinite
# Binomial   : 0.0 < f_Panjer < 1.0 -> f_Panjer = 1 - lda / n   =>  f_Panjer discrete
# Poisson    : f_Panjer = 1.0
# NegBinomial: 1.0 < f_Panjer < infinite 
    eps = 1.0E-6
    if (lda < eps) or (f_Panjer > 1.0/eps):                 # special case p0 = 1.0
        a    = 0.0
        b    = 0.0
        plda = 0.0
        nr   = 0.0
        p0   = 1.0
    elif f_Panjer > 1.0 + eps:                              # case negative Binomial
        r, p = get_rp_NB(lda, f_Panjer)
        if abs(r - 1.0) < eps: r = 1.0
        if r == 1.0: p = 1.0 / (lda + 1.0)                 # geometric distribution
        a    = 1.0 - p
        b    = a * (r - 1.0)
        plda = p
        nr   = r
        p0   = p**r
    elif (f_Panjer > 1.0 - eps):                            # case Poisson
        a    = 0.0
        b    = lda
        plda = lda
        nr   = np.infty
        p0   = np.exp(-lda)
    else:                                                   # case Binomial
        if f_Panjer < eps: f_Panjer = eps
        n, p = get_np_B(lda, f_Panjer)
        a    = p / (p - 1.0)
        b    = -a * (n + 1)
        plda = p
        nr   = n
        p0   = (1.0 - p)**n
    return a, b, plda, nr, p0

def get_Panjer_distr(lda, f_Panjer):
    a, b, plda, nr, p0 = get_Panjer_par(lda, f_Panjer)
    if a == 0.0:
        if b == 0.0: kind = "Degenerate"
        else: kind = "Poisson"
    elif b == 0.:
        kind = 'Geometric'
    else:
        if f_Panjer < 1.0: kind = "Binomial"
        else: kind = "negBinomial"
    if kind == "Degenerate": N_max = 2
    elif kind == "Binomial": N_max = int(nr) + 2
    else: N_max = int(lda * (1 + 5 * f_Panjer**0.5)) + 2
    P_ = np.zeros(N_max)
    P_[0] = p0
    p_k_1 = p0
    for k in range(1, N_max):
        p_k = (a + b / k) * p_k_1
        if p_k > 0.0: P_[k] = P_[k - 1] + p_k
        else: P_[k] = P_[k-1]
        if P_[k] > 1.0: P_[k] = 1.0
        p_k_1 = p_k
    P_[N_max - 1] = 1.0
    return kind, a, b, plda, nr, P_

def get_rvs_cdf(rvs):
    try:
        L = len(rvs)
    except:
        L = 0
    if L > 1:
        x = np.append(rvs, rvs)
        x.sort()
        y = np.linspace(0, (L-1)/L, len(rvs))
        y = np.append(y, y)
        y.sort()
        y = np.roll(y, -1)
        y[-1] = 1
    else:
        if L == 0:
            try:
                r = rvs * 1
            except:
                r = 0
        else:
           r = rvs[0]     
        x = np.array([r, r])
        y = np.array([0, 1])
    return x, y
        
# **********************************************************

class scipy_Distribution(object):
# Generic wrapper for scipy distributions    
    def __init__(self, distr, args):
        self.eps   = 1.E-6
        self.kind  = 'scipy distr'
        self._set_None
        if isinstance(distr, str):
            self.kind = distr
            self._init_scipy_distr(distr, args)
    def _set_None(self):        
        self.is_discrete   = False
        self.is_continuous = False
        self.dist          = None
        self.args          = None
        self.rv            = None
    def _init_scipy_distr(self, distr, args):
        self.args  = args
        try:
            self.dist          = getattr(distributions, distr)
            self.is_discrete   = isinstance(self.dist, rv_discrete)
            self.is_continuous = isinstance(self.dist, rv_continuous)
            self.rv            = self.dist(*args)
        except:
            self._set_None
    def rvs(self, size, sort=False):
        rvs = self.rv.rvs(size=size)
        if sort: rvs.sort()
        return rvs
    def pdf(self, q):
        if self.is_discrete:
            return self.rv.pmf(q)
        elif self.is_continuous:
            return self.rv.pdf(q)
        else: 
            return None
    def pmf(self, q):
        return self.pdf(q)
    def cdf(self, q):
        return self.rv.cdf(q)
    def ppf(self, P):
        return self.rv.ppf(P)
    def isf(self, S):
        return self.rv.isf(S)
    def stats(self, moments='mvsk'):
        return self.rv.stats(moments=moments)
    def test(self):
        print(f'testing: {self.kind} ({self.args}) distribution')
        print(self.is_continuous, self.is_discrete)
        q0= self.rvs(10, True)
        p = self.pdf(q0) 
        P = self.cdf(q0)
        q1= self.ppf(P)
        q2= self.isf(1-P)
        s = self.stats()
        print(q0)
        print(q2)
        print(q1)
        print(s)
        print()
        
class scipy_discrete_custom(scipy_Distribution):
# See also 'class empdiscrete(rv_discrete)' in 'generic_distr.py'    
    def __init__(self, pk=[1.]):
        self.kind = 'scipy discrete custom'
        self.eps  = 1.E-8
        self.init_p(pk)
    def init_r(self, r):
        xk, pk = get_emp_pmf(r)
        self.init_x_p(xk, pk)
    def init_lda_f(self, lda, f):
        kind, a, b, plda, nr, pk = get_Panjer_distr(lda, f)
        self.init_p(pk)
    def init_p(self, pk):    
        self.init_x_p(np.arange(len(pk)), pk)
    def init_x_p(self, xk, pk):
        self.is_discrete   = True
        self.is_continuous = False
        self.dist  = None
        self.args  = None
        self.xk    = np.array(xk)
        self.rv    = stats.rv_discrete(name='custm', values=(xk, normalize(pk)))
        self.Mu    = self.rv.mean()
        self.X_max = self.xk.max()
        self.p     = self.rv.pmf(self.X_max)
    def dH_dX(self, X):
        return 0.
    def dF_dX(self, X):
        return 0.
    def F_lo_hi(self, X):
        F_lo = self.F(X - self.eps)
        F_hi = self.F(X + self.eps)
        return F_lo, F_hi
    def pmf(self, X):
        return self.rv.pmf(X)
    def test_init(self):
        xk = np.arange(5)
        pk = (0.1, 0.2, 0., 0.4, 0.3)       
        self.init_x_p(xk, pk)
        
class scipy_Binom(scipy_Distribution):
    def __init__(self, n=5, p=0.5):
        self.init_n_p(n, p)
    def init_lda_f(self, lda, f):
        p = min(max(self.eps, 1-f), 1.-self.eps)
        n = int(np.ceil(lda/p))
        p = lda / n
        self.init_n_p(n, p)
    def init_n_p(self, n, p):
        self._init_scipy_distr('binom', (n, p, 0))
        self.eps  = 1E-6
        self.kind = 'scipy Binomial'
        
class scipy_Poisson(scipy_Distribution):
    def __init__(self, lda=5):
        self.init_lda(lda)
    def init_lda(self, lda):
        self._init_scipy_distr('poisson', (lda, 0))
        self.eps  = 1E-6
        self.kind = 'scipy Poisson'

class scipy_negBinom(scipy_Distribution):
    def __init__(self, r=5, p=0.5):
        self.init_r_p(r, p)
    def init_lda_f(self, lda, f):
        f = min(max(1+self.eps, f), 1/self.eps)
        r = lda / (f-1)
        p = 1 / f
        self.init_r_p(r, p)
    def init_r_p(self, r, p):
        self._init_scipy_distr('nbinom', (r, p, 0))
        self.eps  = 1E-6
        self.kind = 'scipy negBinomial'

class scipy_Panjer(scipy_Distribution):
# select Panjer class depending on f_Panjer = Var[N] / E[N] 
# and return discrete probability distribution P(N) with E[N] = mean and Var[N] = mean * f_Panjer
    def __init__(self, mean=5, f_Panjer=1, eps = 0.01):
        a, b, plda, nr, p0 = get_Panjer_par(mean, f_Panjer)
        if abs(f_Panjer-1) < eps:
            P = scipy_Poisson(mean)
        elif f_Panjer > 1:
            r, p = get_rp_NB(mean, f_Panjer)
            P = scipy_negBinom(r, p)
        else:
            n, p = get_np_B(mean, f_Panjer)
            P = scipy_Binom(n, p)
        self.kind          = P.kind
        self.eps           = P.eps
        self.is_discrete   = P.is_discrete
        self.is_continuous = P.is_continuous
        self.dist          = P.dist
        self.args          = P.args
        self.rv            = P.rv
    def init_lda_f(self, lda, f_Panjer, L=0):
        kind, a, b, plda, nr, P_ = get_Panjer_distr(lda, f_Panjer)
        self.Panjer_ab =(a, b)
        self.kind = 'scipy Panjer (' + kind +')'
        if kind == 'Poisson':
            self._init_scipy_distr('poisson', (plda, L))
        elif kind == 'Binomial':
            self._init_scipy_distr('binom', (nr, plda, L))
            self.X_max = L + nr
            self.p     = self.rv.pmf(self.X_max)
        elif kind == 'Geometric':
            self._init_scipy_distr('geom', (plda, L-1))
        elif kind == 'negBinomial':
            self._init_scipy_distr('nbinom', (nr, plda, L))
        else: 
            self._init_scipy_distr('bernoulli', (plda, L))
    def init_E_V(self, E, V):
        self.init_lda_f(E, V/E)
        
class scipy_Norm(scipy_Distribution):
    def __init__(self, mu=0, sigma=1):
        self.init_mu_s(mu, sigma)
    def init_E_V(self, E, V):
        self.init_mu_s(E, V**0.5)
    def init_mu_s(self, mu, s):
        self._init_scipy_distr('norm', (mu, s))  # Parameters: mu, sigma
        self.eps  = 1E-6
        self.kind = 'scipy Normal'
        
class scipy_logNorm(scipy_Distribution):
    def __init__(self, mu=0, sigma=1):
        self.init_mu_s(mu, sigma)
    def init_E_V(self, E, V):
        mu, s = logN_mu_s(E, V)
        self.init_mu_s(mu, s)
    def init_mu_s(self, mu, s):
        self._init_scipy_distr('lognorm', (s, 0, np.exp(mu)))  # Parameters: sigma, location, scale
        self.eps  = 1E-6
        self.kind = 'scipy logNormal'
        
class scipy_Gamma(scipy_Distribution):
    def __init__(self, alpha=1., beta=1.):
        self.kind = 'scipy Gamma'
        self.eps  = 1.0E-8
        self.init_alpha_beta(alpha, beta)
    def init_Mu_CoV(self, Mu=1., CoV=1.):
        k = 1. / (CoV**2)
        self.init_k_theta(k, Mu/k)
    def init_E_V(self, E=1., V=1.):
        beta  = E/V 
        alpha = E*beta
        self.init_alpha_beta(alpha, beta)
    def init_alpha_beta(self, alpha=1., beta=1.):
        self.init_scipy_Gamma(alpha, 0., 1./beta)
    def init_k_theta(self, k=1., theta=1.):
        self.init_scipy_Gamma(k, 0., theta)
    def init_scipy_Gamma(self, a, loc=0., scale=1.):
        self.alpha = a
        self.beta  = 1. / scale
        self.k     = a
        self.theta = scale
        self.loc   = loc
        self._init_scipy_distr('gamma', (a, loc, scale))
    def test_init(self):
        self.init_alpha_beta(1.8, 1.2)

if __name__ == "__main__":

# generic    
    P = scipy_Distribution('norm', (0,1))
    P.test()

#discrete    
    lda = 5
    df  = 0.05
    
    B = scipy_Binom()
    B.init_lda_f(lda, 1-df)
    B.test()   
    Pa = scipy_Panjer(lda, 1-df)
    Pa.test()
    
    P = scipy_Poisson(lda)
    P.test()
    Pa = scipy_Panjer(lda, 1)
    Pa.test()

    NB = scipy_negBinom()
    NB.init_lda_f(lda, 1+df)
    NB.test()
    Pa = scipy_Panjer(lda, 1+df)
    r = Pa.rvs(100)
    Pa.test()
    
    D =  scipy_discrete_custom()
    D.init_r(r)
    D.test()

# continuous
    E = 10
    V = 5
    
    Pa = scipy_Panjer()
    Pa.init_E_V(E, V)
    Pa.test()

    N = scipy_Norm(0,1)
    N.init_E_V(E, V)
    N.test()
    
    LN = scipy_logNorm()
    LN.init_E_V(E, V)
    LN.test()
    
    G = scipy_Gamma()
    G.init_E_V(E, V)
    G.test()
    
