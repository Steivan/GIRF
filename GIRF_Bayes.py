# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 13:02:37 2023

@author: steiv
"""
from scipy import special
from scipy import linalg
from scipy.stats import linregress, gamma, poisson, nbinom, binom, norm, lognorm

from GIRF_stats import *

# ******************************************************
# def get_xys(N_sim, t):
# def get_N_sim(T, lda_0, lda_T, beta, f_Panjer, K):
# def get_N_stats(N_sim, T, K):
# ******************************************************
# def get_Panjer_dist(x, pmf, f_lo=0.95, f_hi=1.05):
# def Poisson_lgN_fit(years, N_sim, T, K, iterations=5):   
# def mu_Phi_Y(E, V, X=None, logNorm=True, smooth_Phi_t=True):
# def get_mu_Phi_Y(X, EX, VX, Xtot, Etot, Vtot, logNorm=True, smooth_Phi_t=True):
# def get_a_b_c(Phi, Y_mu, t_, Phi_a, Phi_b, Phi_c):
# def get_c(Phi, Y_mu, Phi_c):
# def get_f_calibration(t, t0, X, E_X, V_X, X_tot, E_tot, V_tot, Phi_a, Phi_b, Phi_c, logNorm=True):
# def lnLikelihood(Y, distr, *args):
# ******************************************************
# def NegBin_p_fit(Y_, p0_t, r_t, Phi_c, max_it=10, eps= 0.0001, check_ln_P=False):# def Poisson_fit(Y_, E_X, Phi_c, max_it=10, eps= 0.0001, check_ln_P=False):
# def Bin_p_fit(Y_, p0_t, n_t, Phi_c, max_it=10, eps= 0.0001, check_ln_P=False):# def negBin_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
# ******************************************************
# def Poisson_fit(Y_, E_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
# def NegBin_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
# def Bin_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
# def Panjer_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False, f_lo=0.9, f_hi=1.1):
# def Gamma_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
# def Gamma_hf_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):    
# ******************************************************
# def N_logN_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, Is_logN=True, smooth_Phi_t=True, Trend=True, check_ln_P=False):
# def logN_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, smooth_Phi_t=True, check_ln_P=False):
# def logN_fit(Y_, E_X, V_X, Phi_c, smooth_Phi_t=True, check_ln_P=False):
# def N_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, smooth_Phi_t=True, check_ln_P=False):
# def N_fit(Y_, E_X, V_X, Phi_c, smooth_Phi_t=True, check_ln_P=False):
# ******************************************************

# ***************************************************************************************************

def get_xys(N_sim, t):
    # Get support and counts for year t from array N_sim with simulated discrete random variables 
    # - N_sim: array[K,T] with K simulations during T years
    # - t    : evaluation year
    # - Nt = N_sim.transpose()[t]: array[K] with simulations for year t
    # - x    : array[n] with time of counts -> x-location of scatter plot
    # - y    : array[n] with support        -> y-location of scatter plot
    # - s    : array[n] with counts         -> size of markers in scatter plot
    y, s = get_counts(N_sim.transpose()[t])    
    x    = np.ones(len(y)) * t
    return x, y, s    

def get_N_sim(T, lda_0, lda_T, beta, f_Panjer, K):
    # Generate a set of K simulations for T years (0 ... T-1): 
    # - lda_0, lda_T: given means at t=0 and t=T-1    
    # - years  : array[T] with years[t] = t
    # - lda_gen: array[T] with expected mean per year (geometric series) -> reference for generative model
    # - alpha(lda_gen), beta: Parameters of Gamma ditribution with mean=lda_gen and variance=lda_gen/beta
    # - lda_sim: array[T] with random mean drawn from Gamma(alpha, beta) -> basis for simulation
    # - N_sim  : array[K,T] with discrete random variables N_sim[k,t] = Xkt >= 0 with: 
    #     mean     E[Xkt] = lda_sim[t]
    #     variance V[Xkt] = lda_sim[t] * f_Panjer
    
    # geometric series:
    a = np.log(lda_0)
    b = (np.log(lda_T) - a) / (T-1)
    years   = np.arange(T)
    lda_gen = np.exp(a + b*years)
    
    # random lda_i=lda_sim[t=i] ´´] ~ Gamma(alpha_i, 1/beta) with 
    # - E[lda_i] = lda_gen[t=i]
    # - V[lda_i] = lda_gen[t=i] / beta
    # random Xki's~(P, NB or B)(lda_i, f_Panjer) with:
    # - E[Xki]  = lda_i
    # - V[Xki] >= lda_i * f_Panjer
    lda_sim = np.zeros(T)
    N_sim = np.zeros((T,K))            # N_sim[T,K] to be transposed into N_sim[K,T]
    for i, lda_i in enumerate(lda_gen):
        # random Gamma:  
        alpha_i = lda_i * beta
        lda_sim[i]  = gamma.rvs(alpha_i, scale=1/beta, size=1)
        if f_Panjer == 1:
        # Poisson case:    
            N_sim[i] = poisson.rvs(lda_sim[i], size=K)
        elif f_Panjer > 1:
        # NegBinomial case:    
            p_ = 1/f_Panjer
            r_ = lda_sim[i]/(f_Panjer-1)
            N_sim[i] = nbinom.rvs(r_, p_, size=K)
        else:
        # Binomial case: 
        # -> Variance = lda*f_P_ with f_P_ = 1-p_ >= f_Panjer !      
            n_ = int(np.ceil(lda_sim[i]/(1-f_Panjer)))
            p_ = lda_sim[i]/n_
            N_sim[i] = binom.rvs(n_, p_, size=K)
    return years, lda_gen, lda_sim, N_sim.transpose()

def get_N_stats(N_sim, T, K):
    # N_sim   : array[K,T] with discrete random variables Xkt
    # N_sim_T : array[T,K] with discrete random variables X_tk
    # - lda_sim[t] = mean of X_tk
    # - var_sim[t] = sample variance of X_tk
    # N_tot   : array[K] with Xk = N_tot[k] = Sum_t Xkt
    # lda_tot : mean of Xk
    # var_tot : sample variance of Xk
    
    def get_E_V(x, K):
        E1 = x.sum()
        E2 = (x*x).sum()
        E  = E1 / K
        V  = (E2 - E1**2 / K) / (K-1)
        return E, V
    
    N_tot = N_sim.sum(axis=1)
    N_sim_T = N_sim.transpose().copy()
    lda_sim = np.zeros(T)
    var_sim = np.zeros(T)
    for t in range(T):
        lda_sim[t], var_sim[t] = get_E_V(N_sim_T[t], K)
    lda_tot, var_tot = get_E_V(N_tot, K)
    # print(lda_sim.sum(), lda_tot)
    # print(var_sim.sum(), var_tot)
    return lda_sim, var_sim, N_tot, lda_tot, var_tot

def get_Panjer_dist(x, pmf, f_lo=0.95, f_hi=1.05):
    xy  = x * pmf
    xxy = x * xy
    E1  = xy.sum()
    V   = xxy.sum() - E1**2 
    f_Panjer   = V / E1
    if f_Panjer > f_hi:
    # NegBinomial case:    
        p_   = 1 / f_Panjer
        r_   = E1 / (f_Panjer-1)
        dist = nbinom(r_, p_)
    elif f_Panjer < f_lo:
    # Binomial case: 
    # -> Variance = lda*f_P_ with f_P_ = 1-p_ >= f_Panjer !      
        n_       = int(np.ceil(E1 / (1 - f_Panjer)))
        p_       = E1 / n_
        dist     = binom(n_, p_)
        f_Panjer = 1 - p_
        V        = E1 * f_Panjer
    else:
    # Poisson case:    
        dist     = poisson(E1)
        f_Panjer = 1
        V        = E1
    
    return dist, E1, V, f_Panjer

def Poisson_lgN_fit(years, N_sim, T, K, iterations=5):   
    # years : array[T] with years[t] = t
    # N_sim : array[K,T] with discrete random variables Xkt
    
    def get_lda_regr(t, x):
        # linear regression of (t_i, ln(x_i))
        slope, intercept, rvalue, pvalue, stderr = linregress(t, np.log(x))
        return intercept, slope, np.exp(intercept + slope * t)
    
    # statistics    
    lda_sim, var_sim, N_tot, lda_tot, var_tot = get_N_stats(N_sim, T, K)
    
    # Regression
    # - lda0 : fit geometric series to simulated means (years, lda_sim)
    # - f_   : uncertainty-ratio for lda_sim 
    # - g_   : variation-ratio for (lda_sim - lda0)
    # - cred : credibility factor g_ / (g_ + f_/K)
    # - mu   : credibility weighted mixture of lda_sim and lda0

    a_regr, b_regr, lda0  = get_lda_regr(years, lda_sim)
    d_lda = lda_sim - lda0
    
    f_ = var_sim.sum() / lda_sim.sum()
    g_ = (d_lda * d_lda).sum() / lda0.sum() 
    
    cred = g_ / (g_ + f_ / K)
    mu = cred * lda_sim + (1-cred) * lda0
    
    return lda_sim, var_sim, N_tot, lda_tot, var_tot, a_regr, b_regr, lda0, f_, g_, cred, mu    

def mu_Phi_Y(E, V, X=None, logNorm=True, smooth_Phi_t=True):
    q0 = 0.25
    if logNorm:
        # E = exp(mu+Phi/2)
        # V = E^2*(exp(Phi)-1)
        Phi = np.log(1+V/(E**2))
        if smooth_Phi_t:
            Phi_0 = Phi.mean()
            Phi   = Phi*0 + Phi_0
        mu = np.log(E) - Phi/2 
        try:
            is_0 = X <= 0.
            X = is_0 * q0 + (1-is_0) * X
            Y = np.log(X)  - mu
            return mu, Phi, Y
        except:
            return mu, Phi
    else:
        # E = mu
        # V = Phi
        if smooth_Phi_t:
            V_0 = V.mean()
            V   = V*0 + V_0
        try: 
            X *= 1
            return E, V, X-E
        except:
            return E, V
        
def get_mu_Phi_Y(X, EX, VX, Xtot, Etot, Vtot, logNorm=True, smooth_Phi_t=True):
    # Convert X ~ logN() or N() with given (EX, VX) into Y ~ N(0, Phi)
    #  - X   : array[T] with random variables
    #  - EX  : array[T] with means     E[X(t)]   = EX[t]
    #  - VX  : array[T] with variances Var[X(t)] = VX[t]
    #  - Xtot: scalar random variable
    #  - Etot: mean     E[Xtot]   = Etot
    #  - Vtot: variance Var[Xtot] = Vtot
    
    mu    , Phi    , Y     = mu_Phi_Y(EX  , VX  , X=X   , logNorm=logNorm, smooth_Phi_t=smooth_Phi_t)
    mu_tot, Phi_tot, Y_tot = mu_Phi_Y(Etot, Vtot, X=Xtot, logNorm=logNorm, smooth_Phi_t=smooth_Phi_t)
        
    return mu, Phi, Y, mu_tot, Phi_tot, Y_tot   

def get_a_b_c(Phi, Y_mu, t_, Phi_a, Phi_b, Phi_c):
    # - Solve linear equation V = M*U for MAP estimates [a,b] = U^T 
    M = np.zeros((2,2))
    V = np.zeros(2)
    
    Phi_inv = (1/Phi).sum()
    
    M[0][0] = 1/Phi_a + Phi_inv
    M[0][1] = (t_/Phi).sum()
    M[1][0] = M[0][1]
    M[1][1] = 1/Phi_b + (t_*t_/Phi).sum()
    
    V[0]    = (Y_mu/Phi).sum()
    V[1]    = (Y_mu*t_/Phi).sum()
    
    U       = linalg.inv(M).dot(V)
    a, b    = U[0], U[1]
    c       = V[0] / (1/Phi_c + Phi_inv)        # c = limes solution for a when Phi_b -> 0
    return a, b, c 

def get_c(Phi, Y_mu, Phi_c):
    M00 = 1/Phi_c + (1/Phi).sum()
    V0  = (Y_mu/Phi).sum()
    U0  = V0 / M00
    return U0
    
def get_f_calibration(t, t0, X, E_X, V_X, X_tot, E_tot, V_tot, Phi_a, Phi_b, Phi_c, logNorm=True):
    # Bayesiann evaluation of trended and overall calibration factors
    # A) Empirical data during observation period (T years)):
    #  - t     : array[T] with observation period
    #  - X     : array[T] with empirical data X[t]
    #  - X_tot : scalar with accumulated empirical data
    # B) Generative model:
    #  - E_X, V_X    : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    #  - E_tot, V_tot: overall mean and variance as derived from the generative model
    # C) Calibration model:
    #  - logNorm=False: X_i ~ N(E_i, V_i)
    #  - logNorm=True : X_i ~ logN(mu_i, Phi_i) with (mu_i, Phi_i) derived from (E_i, V_i) with get_mu_Phi_Y()
    #  Trendrd MAP calibration:
    #  - t0: reference time, used as time of intersection in the regression analysis
    #  - f_trend[t] = exp(a + b(t-t0)) 
    #  -          a ~ N(0, Phi_a) 
    #  -          b ~ N(0, Phi_b) 
    #  Overall MAP calibration:
    #  - f_tot = exp(c) 
    #  -     c ~ N(0, Phi_c) 
    
    _, Phi, Y_mu, _, Phi_tot, Y_tot_mu_tot = get_mu_Phi_Y(X, E_X, V_X, X_tot, E_tot, V_tot)

    # Scaling and trend based on annual data
    t_ = t - t0
    a, b, c = get_a_b_c(Phi, Y_mu, t_, Phi_a, Phi_b, Phi_c)
    f_trend = np.exp(a+b*t_)
    
    # Scaling based on overall data
    c_tot = get_c(Phi_tot, Y_tot_mu_tot, Phi_c)
    f_tot = np.exp(c_tot)
    
    return a, b, c_tot, f_trend, f_tot

def lnLikelihood(Y, distr, *args):
    try:
        if is_discrete(distr):
            P = distr.pmf(Y, *args)
        else:
            P = distr.pdf(Y, *args)
    except:
        P = np.ones(len(Y))
    return np.log(P).sum()    

# ***************************************************************************************************

def get_c_lo_bound(a):
    return (-np.log(a)).max()

def c_confine(c, c_min):
    dc = c.min() - c_min
    if dc < 0:
        c = c - dc
    return c

def R_c_Poisson(c, Y_, E, f_P, Phi_c):
    #  - R(c) = Phi_c * Sum_t lda_0[t]*(Y[t]/lda_t[c]-1) 
    lda_0 = E
    n       = len(c)
    f_c     = np.zeros(n)
    exp_c   = np.exp(c)
    for i in range(n):
        lda_c   = lda_0 * exp_c[i] 
        f_c[i] = (lda_0 * (Y_ / lda_c - 1)).sum()
    return Phi_c * f_c

def R_c_negBin(c, Y_, E, f_P, Phi_c):
    #  R(c) = Phi_c * Sum_t r_0[t]*(ln(1 + Y[t]/(r_t[c]-1)) - 1/2*Y[t]/(Y[t]+r_t[c]-1)/(r_t[c]-1) - ln(f_P))
    r_0     = E / (f_P - 1)
    try:
        n   = len(c)
    except:
        n   = 1
        c   = np.ones(n) * c
    f_c     = np.zeros(n)
    exp_c   = np.exp(c)
    ln_f_P  = np.log(f_P)
    for i in range(n):
        rc_1   = r_0 * exp_c[i] - 1
        z      = Y_ / rc_1
        z1     = np.log(1 + z)
        z2     = z / (Y_ + rc_1) / 2
        f_c[i] = (r_0 * (z1 - z2 - ln_f_P)).sum()
    return Phi_c * f_c

def R_c_Bin(c, Y_, E, f_P, Phi_c):
    #  - R(c) = Phi_c * Sum_t n_0[t]*(ln(1 + Y[t]/(n_t[c]-Y[t])) - 1/2*Y[t]/n_t[c]/(n_t[c]-Y[t]) + ln(f_P))   
    n_0     = E / (1-f_P)
    n       = len(c)
    f_c     = np.zeros(n)
    exp_c   = np.exp(c)
    ln_f_P  = np.log(f_P)
    for i in range(n):
        nc     = n_0 * exp_c[i]
        nc_Y   = nc - Y_
        z      = Y_ / nc_Y
        z1     = np.log(1 + z)
        z2     = z / nc / 2
        f_c[i] = (n_0 * (z1 - z2 + ln_f_P)).sum()
    return Phi_c * f_c

def R_c_Gamma(c, Y_, E, f_P, Phi_c):
    #  - R(c) = Phi_c * Sum_t r_0[t]*(ln(Y[t]/(alpha_t[c]-1)) - 1/2/(alpha_t[c]-1) - ln(f_P))    
    a_0     = E / f_P
    n       = len(c)
    f_c     = np.zeros(n)
    exp_c   = np.exp(c)
    ln_f_P  = np.log(f_P)
    for i in range(n):
        ac_1   = a_0 * exp_c[i] - 1
        z1     = np.log(Y_ / ac_1)
        z2     = 1 / 2 / ac_1 
        f_c[i] = (a_0 * (z1 - z2 - ln_f_P)).sum()
    return Phi_c * f_c

def Poisson_fit(Y_, E_X, Phi_c, max_it=10, eps= 0.0001, check_ln_P=False):
    # Bayesiann evaluation of the overall calibration factors
    # A) Empirical data during observation period (T years)):
    #  - Y_ : array[T] with empirical data Y[t]
    # B) Generative model:
    #  - E_X : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    #  - Poisson assumption: f_P = V_X / E_X = 1
    # C) Calibration model:
    #  - X_t ~ P(lda_t(c))
    #  - with:
    #    - lda_t(c) = lda_0[t] * e^c = E_X[t] * e^c
    #  Overall MAP calibration:
    #  - f_c = e^c 
    #  -   c ~ N(0, Phi_c) 
    #  Solvig the Bayes equation: Sum_t (Y[t] - lda_0[t] * e^c ) - c/Phi_c = 0
    
    def get_ln_P(c0):
        n    = 11
        D_c  = Phi_c**0.5 / 10
        c_P  = np.linspace(c0-D_c, c0+D_c, n)
        f_c  = np.exp(c_P) 
        ln_P = np.ones(n)
        for i, c_i in enumerate(c_P):
            for j, Y_j in enumerate(Y_):
                 ln_P[i] *= poisson.pmf(Y_j, E_X[j] * f_c[i])
            ln_P[i] = np.log(ln_P[i]) - c_i**2 / Phi_c / 2
        return c_P, ln_P    
    
    S_Y = Y_.sum()
    S_E = E_X.sum() 
    c   = 0                                            # initialize c(0)
    D_c = 1
    it  = 0 
    while (abs(D_c) > eps) and (it < max_it):
        c_  = np.log((S_Y - c / Phi_c) / S_E)          # iteration: c(i) -> c(i+1)
        D_c = c_ - c
        c   = c_
        it  += 1

    if check_ln_P:
        print(c)
        c_P, ln_P = get_ln_P(c)
        print(np.round(np.array([c_P, ln_P]).transpose(), 3))

    c_T = c    
    return c, c_T    

def negBin_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
    # Bayesiann evaluation of the overall calibration factors
    # A) Empirical data during observation period (T years)):
    #  - Y_: array[T] with empirical data Y[t]
    # B) Generative model:
    #  - E_X, V_X : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    # C) Calibration model:
    #  - X_t ~ NB(p_t, r_t(c))
    #  - with:
    #    - f_P    = Sum_t(V_X[t]) / Sum_t(E_X[t]) > 1
    #    - p_t    = 1 / f_P     
    #    - r_t(c) = r_0[t] * e^c = E_X[t] / (f_P - 1) * e^c
    #  Overall MAP calibration:
    #  - f_c = e^c 
    #  -   c ~ N(0, Phi_c) 
    #  Solvig the Bayes equation L(c) = R(c) with:
    #  - L(c) = c / e^c
    #  - R(c) = Phi_c * Sum_t r_0[t]*(ln(1 + Y[t]/(r_t[c]-1)) - 1/2*Y[t]/(Y[t]+r_t[c]-1)/(r_t[c]-1) - ln(f_P))   
    dc_lo = 0.01
    
    def get_ln_P(c0, p, r_0):
        n    = 11
        D_c  = Phi_c**0.5 / 10
        c_P  = np.linspace(c0-D_c, c0+D_c, n)
        f_c  = np.exp(c_P) 
        ln_P = np.ones(n)
        for i, c_i in enumerate(c_P):
            for j, Y_j in enumerate(Y_):
                 ln_P[i] *= nbinom.pmf(Y_j, r_0[j] * f_c[i], p)
            ln_P[i] = np.log(ln_P[i]) - c_i**2 / Phi_c / 2
        return c_P, ln_P    
 
    def L_c(c):
        return c * np.exp(-c)

    c, c_T = Poisson_fit(Y_, E_X, Phi_c, max_it=max_it, eps=eps)     # Poisson case and initial for NB case
    f_P    = V_X.sum() / E_X.sum()
    if f_P > 1:                                                    # negBin case
        
        dc   = 0.01
        dc_2 = 2 * dc  
        D_c  = 1
        it   = 0 
        c_min = get_c_lo_bound(E_X / (f_P - 1)) + dc_lo
        while (abs(D_c) > eps) and (it < max_it):
        # calibration based on annual data    
            c_   = np.array([c-dc, c, c+dc])
            c_   = c_confine(c_, c_min)
            f_L  = L_c(c_)
            f_R  = R_c_negBin(c_, Y_, E_X, f_P, Phi_c)
            D_c  = - dc_2 * (f_R[1] - f_L[1]) / ((f_R[2] - f_R[0]) - (f_L[2] - f_L[0]))
            c   += D_c
            it  += 1

        c_T = c
        D_c = 1
        it  = 0 
        c_min = get_c_lo_bound(E_X.sum() / (f_P - 1)) + dc_lo
        while (abs(D_c) > eps) and (it < max_it):
        # calibration based on agregate data    
            c_   = np.array([c_T-dc, c_T, c_T+dc])
            c_   = c_confine(c_, c_min)
            f_L  = L_c(c_)
            f_R  = R_c_negBin(c_, Y_.sum(), E_X.sum(), f_P, Phi_c)
            D_c  = - dc_2 * (f_R[1] - f_L[1]) / ((f_R[2] - f_R[0]) - (f_L[2] - f_L[0]))
            c_T += D_c
            it  += 1

    if check_ln_P:
        print(c)
        p         = 1 / f_P
        r_0       = E_X / (f_P - 1)
        c_P, ln_P = get_ln_P(c, p, r_0)
        print(np.round(np.array([c_P, ln_P]).transpose(), 3))    

    return c, c_T   
    
def Bin_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
    # Bayesiann evaluation of the overall calibration factors
    # A) Empirical data during observation period (T years)):
    #  - Y_: array[T] with empirical data Y[t]
    # B) Generative model:
    #  - E_X, V_X : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    # C) Calibration model:
    #  - X_t ~ NB(p_t, r_t(c))
    #  - with:
    #    - f_P    = Sum_t(V_X[t]) / Sum_t(E_X[t]) < 1
    #    - p_t    = 1 - f_P     
    #    - n_t(c) = n_0[t] * e^c = E_X[t] / p_t * e^c
    #  Overall MAP calibration:
    #  - f_c = e^c 
    #  -   c ~ N(0, Phi_c) 
    #  Solvig the Bayes equation L(c) = R(c) with:
    #  - L(c) = c / e^c
    #  - R(c) = Phi_c * Sum_t n_0[t]*(ln(1 + Y[t]/(n_t[c]-Y[t])) - 1/2*Y[t]/n_t[c]/(n_t[c]-Y[t]) - ln(1/f_P))    
    dc_lo = 0.01
    
    def get_ln_P(c0, p, n_0):
        n    = 11
        D_c  = Phi_c**0.5 / 10
        c_P  = np.linspace(c0-D_c, c0+D_c, n)
        f_c  = np.exp(c_P) 
        ln_P = np.ones(n)
        for i, c_i in enumerate(c_P):
            for j, Y_j in enumerate(Y_):
                 ln_P[i] *= binom.pmf(Y_j, n_0[j] * f_c[i], p)
            ln_P[i] = np.log(ln_P[i]) - c_i**2 / Phi_c / 2
        return c_P, ln_P    
 
    def L_c(c):
        return c * np.exp(-c)

    c, c_T = Poisson_fit(Y_, E_X, Phi_c, max_it=max_it, eps=eps)     # Poisson case and initial for NB case
    f_P  = V_X.sum() / E_X.sum()
    if f_P < 1:                                                   # Binomial case
        
        dc   = 0.01
        dc_2 = 2 * dc  
        D_c  = 1
        it   = 0 
        c_min = get_c_lo_bound(E_X / (1-f_P) / Y_) + dc_lo
        while (abs(D_c) > eps) and (it < max_it):
        # calibration based on annual data    
            c_  = np.array([c-dc, c, c+dc])
            c_  = c_confine(c_, c_min)
            f_L = L_c(c_)
            f_R = R_c_Bin(c_, Y_, E_X, f_P, Phi_c)
            D_c = - dc_2 * (f_R[1] - f_L[1]) / ((f_R[2] - f_R[0]) - (f_L[2] - f_L[0]))
            c   += D_c
            it  += 1

        c_T = c
        D_c = 1
        it  = 0 
        c_min = get_c_lo_bound(E_X.sum() / (1-f_P) / Y_.sum()) + dc_lo
        while (abs(D_c) > eps) and (it < max_it):
        # calibration based on agregate data    
            c_  = np.array([c_T-dc, c_T, c_T+dc])
            c_  = c_confine(c_, c_min)
            f_L = L_c(c_)
            # f_R = R_T(c_)
            f_R = R_c_Bin(c_, Y_.sum(), E_X.sum(), f_P, Phi_c)
            D_c = - dc_2 * (f_R[1] - f_L[1]) / ((f_R[2] - f_R[0]) - (f_L[2] - f_L[0]))
            c_T += D_c
            it  += 1
 
    if check_ln_P:
        print(c)
        p         = 1 - f_P
        n_0       = E_X / (1-f_P)
        c_P, ln_P = get_ln_P(c, p, n_0)
        print(np.round(np.array([c_P, ln_P]).transpose(), 3))    

    return c, c_T   
    
def Panjer_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False, f_lo=0.9, f_hi=1.1):
    # f_P in [f_lo, f_hi] : Poisson case 
    f_lo = min(1., max(f_lo, 0.5))
    f_hi = max(1., min(f_hi, 2.0))
    
    f_P  = V_X.sum() / E_X.sum()
    if f_P < f_lo:                                 # Binomial case
        return Bin_fit(Y_, E_X, V_X, Phi_c, max_it=max_it, eps=eps, check_ln_P=check_ln_P)
    elif f_P > f_hi:                              # negBinomial case
        return negBin_fit(Y_, E_X, V_X, Phi_c, max_it=max_it, eps=eps, check_ln_P=check_ln_P)
    else:                                 # negBinomial case
        return Poisson_fit(Y_, E_X, Phi_c, max_it=max_it, eps=eps, check_ln_P=check_ln_P)
                  
def Gamma_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
    # Bayesiann evaluation of the overall calibration factors
    # A) Empirical data during observation period (T years)):
    #  - Y_: array[T] with empirical data Y[t]
    # B) Generative model:
    #  - E_X, V_X : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    # C) Calibration model:
    #  - X_t ~ G(alpha_t(c), beta_t)
    #  - with:
    #    - f_P    = Sum_t(V_X[t]) / Sum_t(E_X[t]) > 0
    #    - beta_t    = 1 / f_P     
    #    - alpha_t(c) = alpha_0[t] * e^c = E_X[t] / f_P * e^c
    #  Overall MAP calibration:
    #  - f_c = e^c 
    #  -   c ~ N(0, Phi_c) 
    #  Solvig the Bayes equation L(c) = R(c) with:
    #  - L(c) = c / e^c
    #  - R(c) = Phi_c * Sum_t r_0[t]*(ln(Y[t]/(alpha_t[c]-1)) - 1/2/(alpha_t[c]-1) - ln(f_P))    
    
    # def get_ln_P(c0):
    #     n    = 11
    #     D_c  = Phi_c**0.5 / 10
    #     c_P  = np.linspace(c0-D_c, c0+D_c, n)
    #     f_c  = np.exp(c_P) 
    #     ln_P = np.ones(n)
    #     for i, c_i in enumerate(c_P):
    #         for j, Y_j in enumerate(Y_):
    #              ln_P[i] *= nbinom.pmf(Y_j, a_0[j] * f_c[i], p)
    #         ln_P[i] = np.log(ln_P[i]) - c_i**2 / Phi_c / 2
    #     return c_P, ln_P    
    dc_lo = 0.01
 
    def L_c(c):
        return c * np.exp(-c)

    f_P  = V_X.sum() / E_X.sum()
    c, c_T = Poisson_fit(Y_, E_X, Phi_c, max_it=max_it, eps=eps)     # Poisson case and initial for NB case
    
    dc   = 0.01
    dc_2 = 2 * dc  
    D_c  = 1
    it   = 0 
    c_min = get_c_lo_bound(E_X / f_P) + dc_lo
    while (abs(D_c) > eps) and (it < max_it):
    # calibration based on annual data  
        c_  = np.array([c-dc, c, c+dc])
        c_  = c_confine(c_, c_min)
        f_L = L_c(c_)
        f_R = R_c_Gamma(c_, Y_, E_X, f_P, Phi_c)
        D_c = - dc_2 * (f_R[1] - f_L[1]) / ((f_R[2] - f_R[0]) - (f_L[2] - f_L[0]))
        c  += D_c
        it += 1

    c_T  = c
    D_c  = 1
    it   = 0 
    c_min = get_c_lo_bound(E_X.sum() / f_P) + dc_lo
    while (abs(D_c) > eps) and (it < max_it):
    # calibration based on agregate data    
        c_   = np.array([c_T-dc, c_T, c_T+dc])
        c_   = c_confine(c_, c_min)
        f_L  = L_c(c_)
        f_R  = R_c_Gamma(c_, Y_.sum(), E_X.sum(), f_P, Phi_c)
        D_c  = - dc_2 * (f_R[1] - f_L[1]) / ((f_R[2] - f_R[0]) - (f_L[2] - f_L[0]))
        c_T += D_c
        it  += 1

    # if check_ln_P:
    #     print(c)
    #     c_P, ln_P = get_ln_P(c)
    #     print(np.round(np.array([c_P, ln_P]).transpose(), 3))    

    return c, c_T   
    
def Gamma_hf_fit(Y_, E_X, V_X, Phi_c, max_it=10, eps=0.0001, check_ln_P=False):
    # high-frequency approximation: alpha*e^c-1 -> alpha*e^c-1

    f_P  = V_X.sum() / E_X.sum()
    a_0     = E_X / f_P 
    t_Y     = (a_0 * np.log(Y_ / a_0)).sum()           # sufficient statistics
    a_tot   = a_0.sum()
    t_Y_T   = a_tot * np.log(Y_.sum() / a_tot)         # aggr. stats.            
    T_2     = len(Y_) / 2
    ln_f_P  = np.log(f_P)
    
    c   = 0                                            # initialize c(0)
    D_c = 1
    it  = 0 
    while (abs(D_c) > eps) and (it < max_it):
        c_  = (t_Y - (T_2 + c/Phi_c)/np.exp(c)) / a_tot -ln_f_P
        D_c = c_ - c
        c   = c_
        it  += 1

    c_T = c                                            # initialize c(0)
    D_c = 1
    it  = 0 
    while (abs(D_c) > eps) and (it < max_it):
        c_  = (t_Y_T - (1/2 + c_T/Phi_c)/np.exp(c_T)) / a_tot -ln_f_P
        D_c = c_ - c_T
        c_T = c_
        it  += 1
    
    return c, c_T

# ***************************************************************************************************

def NegBin_p_fit(Y_, p0_t, r_t, Phi_c, max_it=10, eps= 0.0001, check_ln_P=False):
    # Bayesiann calibration of the p-parameters
    # A) Empirical data during observation period (T years)):
    #  - Y_ : array[T] with BINARY empirical data Y[t] in {0, 1}
    # B) Calibration model:
    #  - N_t ~ B(p_t(c), n_t)
    #  - with logistic linking:
    #    - p_t(c) = p0_t * e^c / (p0_t * e^c + (1-p0_t) * e^-c)
    #  Overall MAP calibration:
    #  -   c ~ N(0, Phi_c) 
    #  Solvig the Bayes equation

    def get_ceil(a, b):
        a_ = np.ceil(a)
        b_ = np.ceil(b)
        is_a = a_ > b_
        return a_*is_a + b_*(1-is_a)

    t_nom_1 = 2 * Phi_c * r_t * (1-p0_t) 
    t_nom_2 = 2 * Phi_c * Y_*p0_t
    
    def get_R_c(c):
        R_c = np.zeros(len(c))
        e_c = np.exp(c)
        for i, e_c_i in enumerate(e_c):
            t_nom   = t_nom_1 / e_c_i - t_nom_2 * e_c_i
            t_denom = p0_t * e_c_i + (1-p0_t) / e_c_i
            R_c[i]  = (t_nom / t_denom).sum()
        return R_c    

    c   = 0
    dc  = 0.01
    D_c = 1
    it  = 0 
    while (abs(D_c) > eps) and (it < max_it):
        c_   = np.array([c-dc, c, c+dc])
        L_c  = c_
        R_c  = get_R_c(c_)
        d_dc = ((R_c[2] - L_c[2]) - (R_c[0] - L_c[0])) / dc / 2
        D_c  = (R_c[1] - L_c[1]) / d_dc
        c   -= D_c
        it  += 1
    
    p_t = p0_t/(p0_t+(1-p0_t)*np.exp(-2*c))
    return c, p_t, r_t

def Bin_p_fit(Y_, p0_t, n_t, Phi_c, max_it=10, eps= 0.0001, check_ln_P=False):
    # Bayesiann calibration of the p-parameters
    # A) Empirical data during observation period (T years)):
    #  - Y_ : array[T] with BINARY empirical data Y[t] in {0, 1}
    # B) Calibration model:
    #  - N_t ~ B(p_t(c), n_t)
    #  - with logistic linking:
    #    - p_t(c) = p0_t * e^c / (p0_t * e^c + (1-p0_t) * e^-c)
    #  Overall MAP calibration:
    #  -   c ~ N(0, Phi_c) 
    #  Solvig the Bayes equation

    def get_ceil(a, b):
        a_ = np.ceil(a)
        b_ = np.ceil(b)
        is_a = a_ > b_
        return a_*is_a + b_*(1-is_a)

    n_t   = get_ceil(n_t, Y_)                         # ensure that: n_t >= Y_
    t_nom_1 = 2 * Phi_c * Y_ * (1-p0_t) 
    t_nom_2 = 2 * Phi_c * (n_t-Y_)*p0_t
    
    def get_R_c(c):
        R_c = np.zeros(len(c))
        e_c = np.exp(c)
        for i, e_c_i in enumerate(e_c):
            t_nom   = t_nom_1 / e_c_i - t_nom_2 * e_c_i
            t_denom = p0_t * e_c_i + (1-p0_t) / e_c_i
            R_c[i]  = (t_nom / t_denom).sum()
        return R_c    

    c   = 0
    dc  = 0.01
    D_c = 1
    it  = 0   
    while (abs(D_c) > eps) and (it < max_it):
        c_   = np.array([c-dc, c, c+dc])
        L_c  = c_
        R_c  = get_R_c(c_)
        d_dc = ((R_c[2] - L_c[2]) - (R_c[0] - L_c[0])) / dc / 2
        D_c  = (R_c[1] - L_c[1]) / d_dc
        c   -= D_c
        it  += 1
    
    p_t = p0_t/(p0_t+(1-p0_t)*np.exp(-2*c))
    return c, p_t, n_t

# ***************************************************************************************************

def N_logN_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, Is_logN=True, smooth_Phi_t=True, Trend=True, check_ln_P=False):
    # Bayesiann evaluation of the calibration parameters (a, b) and (c, c_tot)
    # A) Empirical data during observation period (T years)):
    #  - t_ : array[T] with time t    
    #  - Y_ : array[T] with empirical data Y[t]
    # B) Generative model:
    #  - E_X, V_X : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    #  - logN assumption:
    #    - smooth_Phi_t=True : 
    #      - Is_logN=True : Phi_t = mean(ln(1+V_X / (E_X)^2))
    #      - Is_logN=False: Phi_t = mean(V_X)
    #    - smooth_Phi_t=False: 
    #      - Is_logN=True : Phi_t = ln(1+V_X / (E_X)^2)
    #      - Is_logN=False: Phi_t = V_X
    #    - Is_logN=True : mu_t = ln(E_X) - Phi_t/2)
    #    - Is_logN=False: mu_t = E_X
    # C) Calibration model:   
    #  -Is_logN=True : X_t ~ logN(mu_t(*), Phi_t) and X_tot ~ logN(mu_tot+c_tot, Phi_tot)
    #  -Is_logN=False: X_t ~    N(mu_t(*), Phi_t) and X_tot ~    N(mu_tot+c_tot, Phi_tot)
    #  - with:
    #    - mu_t(a, b) = mu_t_0 + a + b * (t-t0)
    #    - t0 = T/2
    #    - mu_t(c)    = mu_t_0 + c
    # D) check_ln_P=True: 
    #  - Evaluate the likelihood function in the neighbourhood of the solution (a, b)

    mu_t    = None
    sigma_t = None
    
    def get_n_D(x0, n0, Phi_x, N=50):
        n_x = 2*n0+1
        D_x = Phi_x**0.5 / N
        x_P = np.linspace(x0-n0*D_x, x0+n0*D_x, n_x)
        return n_x, x_P
        
    def get_ln_P(Y_, mu_t, t_, a, b, Is_logN):
        P_0 = 1
        for k, Y_k in enumerate(Y_):
            mu_k = mu_t[k] + a + b * t_[k]
            if Is_logN:
                P_0 *= lognorm.pdf(Y_k, scale=np.exp(mu_k), s=sigma_t[k])
            else:
                P_0 *= norm.pdf(Y_k, loc=mu_k, scale=sigma_t[k])
        return np.log(P_0)

    def get_ln_P_ab(a0, b0, n0=4):
        # Evaluate the loglikelihood function in the surroundings of (a0,b0)
        na, a_P = get_n_D(a0, n0, Phi_a)
        nb, b_P = get_n_D(b0, n0, Phi_b)
        ln_P    = np.zeros((na, nb))
        for i, a_i in enumerate(a_P):
            for j, b_j in enumerate(b_P):
                ln_P[i][j] = get_ln_P(Y_, mu_t, t_, a_i, b_j, Is_logN) - a_i**2 / Phi_a / 2 - b_j**2 / Phi_b / 2
        return ln_P[n0][0], ln_P - ln_P[n0][n0]   
 
    def get_ln_P_c(c0, n0=4):
        # Evaluate the loglikelihood function in the surroundings of (c0)
        n, c_P = get_n_D(c0, n0, Phi_c)
        ln_P   = np.zeros(n)
        for i, c_i in enumerate(c_P):
            ln_P[i] = get_ln_P(Y_, mu_t, t_, c_i, 0, Is_logN) - c_i**2 / Phi_c / 2
        return ln_P[n0], ln_P - ln_P[n0] 
    
    t0 = (t_.min() + t_.max()) / 2
    t_ = t_ - t0
    
    mu_t, Phi_t, Y_mu, mu_tot, Phi_tot, Y_mu_tot = get_mu_Phi_Y(Y_, E_X, V_X, Y_.sum(), E_X.sum(), V_X.sum(), 
                                                                logNorm=Is_logN, smooth_Phi_t=smooth_Phi_t)
    
    a, b, c = get_a_b_c(Phi_t, Y_mu, t_, Phi_a, Phi_b, Phi_c)  # Y_mu = ln(Y_) - mu_t   or  Y_ - mu_t
    c_tot   = get_c(Phi_tot, Y_mu_tot, Phi_c)
    
    if check_ln_P:
        sigma_t = Phi_t**0.5

        if Trend:
            ln_P_00, ln_P_ab = get_ln_P_ab(a, b)
            print('a, b, ln_P_00:', a, b, ln_P_00)
            print(np.round(np.array(ln_P_ab), 3))    

        ln_P_0, ln_P_c = get_ln_P_c(c)
        print('c, ln_P_0:', c, ln_P_0)
        print(np.round(np.array(ln_P_c), 3))    
        print()
        
    if Trend:
        return a, b, t0, c, c_tot   
    else:
        return c, c_tot

def logN_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, smooth_Phi_t=True, check_ln_P=False):
    # logNormal fit: scale and trend
    a, b, t0, c, c_tot = N_logN_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, 
                                          Is_logN=True, smooth_Phi_t=smooth_Phi_t, check_ln_P=check_ln_P)
    return a, b, t0, c, c_tot 

def logN_fit(Y_, E_X, V_X, Phi_c, smooth_Phi_t=True, check_ln_P=False):
    # logNormal-fit: scale only
    try:
        t_ = np.arange(len(Y_))
    except:
        t_ = np.array([1])
    c, c_tot  = N_logN_fit_trend(t_, Y_, E_X, V_X, Phi_c, Phi_c, Phi_c, Is_logN=True, smooth_Phi_t=smooth_Phi_t, Trend=False, check_ln_P=check_ln_P)
    return c, c_tot    

def N_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, smooth_Phi_t=True, check_ln_P=False):
    # Normal fit: location and trend
    a, b, t0, c, c_tot = N_logN_fit_trend(t_, Y_, E_X, V_X, Phi_a, Phi_b, Phi_c, Is_logN=False, smooth_Phi_t=smooth_Phi_t, check_ln_P=check_ln_P)
    return a, b, t0, c, c_tot 

def N_fit(Y_, E_X, V_X, Phi_c, smooth_Phi_t=True, check_ln_P=False):
    # lNormal-fit: location only
    try:
        t_ = np.arange(len(Y_))
    except:
        t_ = np.array([1])
    c, c_tot  = N_logN_fit_trend(t_, Y_, E_X, V_X, Phi_c, Phi_c, Phi_c, Is_logN=False, smooth_Phi_t=smooth_Phi_t, Trend=False, check_ln_P=check_ln_P)
    return c, c_tot    

# ***************************************************************************************************

def Panjer_sim_test_data(T, par_emp, par_sim):
    
    def N_sim(param):
        lda_0, lda_T, beta, f_Panjer, K = param
        years, lda_ref, lda_gen, N_sim  = get_N_sim(T, lda_0, lda_T, beta, f_Panjer, K)
        a_mod = np.log(lda_0)
        b_mod = (np.log(lda_T) - a_mod) / (T-1)
        return years, lda_ref, lda_gen, N_sim, a_mod, b_mod 
        
    # Proxy for empirical data:    
    # -> get_N_sim() for details
    data_emp = N_sim(par_emp) 
    
    
    # Proxy for generative model:    
    # -> get_N_sim() for details
    data_sim = N_sim(par_sim) 
    
    return data_emp, data_sim
    
   
# ***************************************************************************************************

def get_f_calibration(t, t0, X, E_X, V_X, X_tot, E_tot, V_tot, Phi_a, Phi_b, Phi_c, logNorm=True):
    # Bayesiann evaluation of trended and overall calibration factors
    # A) Empirical data during observation period (T years)):
    #  - t     : array[T] with observation period
    #  - X     : array[T] with empirical data X[t]
    #  - X_tot : scalar with accumulated empirical data
    # B) Generative model:
    #  - E_X, V_X    : arrays[T] with (simulated) means E_X[t] and variance V_X[t] as derived from a generative model
    #  - E_tot, V_tot: overall mean and variance as derived from the generative model
    # C) Calibration model:
    #  - logNorm=False: X_i ~ N(E_i, V_i)
    #  - logNorm=True : X_i ~ logN(mu_i, Phi_i) with (mu_i, Phi_i) derived from (E_i, V_i) with get_mu_Phi_Y()
    #  Trendrd MAP calibration:
    #  - t0: reference time, used as time of intersection in the regression analysis
    #  - f_trend[t] = exp(a + b(t-t0)) 
    #  -          a ~ N(0, Phi_a) 
    #  -          b ~ N(0, Phi_b) 
    #  Overall MAP calibration:
    #  - f_tot = exp(c) 
    #  -     c ~ N(0, Phi_c) 
    
    _, Phi, Y_mu, _, Phi_tot, Y_tot_mu_tot = get_mu_Phi_Y(X, E_X, V_X, X_tot, E_tot, V_tot)

    # Scaling and trend based on annual data
    t_ = t - t0
    a, b, c = get_a_b_c(Phi, Y_mu, t_, Phi_a, Phi_b, Phi_c)
    f_trend = np.exp(a+b*t_)
    
    # Scaling based on overall data
    c_tot = get_c(Phi_tot, Y_tot_mu_tot, Phi_c)
    f_tot = np.exp(c_tot)
    
    return a, b, c_tot, f_trend, f_tot

# ***************************************************************************************************

# if __name__ == "__main__":

