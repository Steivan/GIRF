# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""
import numpy as np
from scipy.stats import poisson
from scipy import signal
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']  = 600
plt.rcParams['savefig.dpi'] = 600

from Transscript import ts_start, ts_stop
from GIRF_models import Red_Fields, D_Bino, D_Pois, D_negB, D_Panj, D_Gamm, D_logN, D_Norm, get_freq_model
from GIRF_claim import  my_round, Red_Claim, Red_Ann, Red_Overall
from GIRF_stats import get_E_t, logN_mu_s, scipy_discrete_custom, scipy_Panjer, scipy_logNorm
from GIRF_Bayes import Poisson_fit, negBin_fit, Bin_fit, Gamma_fit, \
                       N_logN_fit_trend, logN_fit_trend, N_fit_trend, \
                       Panjer_sim_test_data, Poisson_lgN_fit, logN_fit, get_f_calibration  
#                       R_c_Poisson, R_c_negBin, R_c_Bin, R_c_Gamma, Panjer_fit, NegBin_p_fit, Bin_p_fit, Gamma_hf_fit, 
from GIRF_plot import print_file_name, plot_claims_cont, print_claims_count, plot_claims_count, plot_R_c, \
                      plot_evol_all, print_all_param

# *********************************

def get_trended_model(param_0, cal_param):
    (T, E_0, E_T, f_Panjer) = param_0
    (a, b, t0) = cal_param
    E_0_r = E_0 * np.exp(a+b*(  0-t0)) 
    E_T_r = E_T * np.exp(a+b*(T-1-t0)) 
    run_param = (T, E_0_r, E_T_r, f_Panjer)
    return run_param

def get_X_max(T, E_0, E_T, f_Panjer):
    b = np.log(E_T / E_0) / (T-1)                    # derive annual growth from period 0...T-1
    E1 = E_0 * np.exp(b*(T+1))                       # mean and variance at t=T+1
    X_max = E1 + 3 * (E1*f_Panjer)**0.5              # evaluate X_max at T+1 for the projected distr.
    return X_max

def get_Panjer_series(T, E_0, E_T, f_Panjer):
    # Create list with a series of Panjer-distributions P_annual(o)
    # - P(t): Panjer distribution with mean E(t) and variance V(t)
    # - P1  : projected distribution at T+1 
    # - The mean E(t) and the variance V(t) are geometric series with:
    #   - E(t=  0) = E_0 
    #   - E(t=T-1) = E_T 
    #   - V(t)     = E(t)*f_Panjer

    E_annual, E_proj = get_E_t(T, E_0, E_T)
    P_annual = []
    for t, E_t in enumerate(E_annual):
        P_annual.append(scipy_Panjer(E_t, f_Panjer))        # annual distributions for o=-T,...,-1  
    P_proj = scipy_Panjer(E_proj, f_Panjer)                 # projected distribution for o=+1
    return P_annual, P_proj   

def get_p_f(series, q):
    T = len(series) 
    try:
        p_f  = np.zeros((T, len(q)))
        p_f1 = np.zeros(len(q))
    except:
        p_f  = np.zeros(T)
        p_f1 = 0
    return T, p_f, p_f1    
    
def get_series_pmf(my_series, q, D1=None):
    T, pmf, pmf1 = get_p_f(my_series, q)
    for t, D_t in enumerate(my_series):
        pmf[t] = D_t.pmf(q)
    if (D1 is None):
        return pmf
    else:
        pmf1 = D1.pmf(q)
        return pmf, pmf1
    
def get_series_pdf(my_series, q, D1=None):
    T, pdf, pdf1 = get_p_f(my_series, q)
    for t, D_t in enumerate(my_series):
        pdf[t] = D_t.pdf(q)
    if (D1 is None):
        return pdf
    else:
        pdf1 = D1.pdf(q)
        return pdf, pdf1
    
def get_series_rvs(my_series, D1=None, size=1):
    T = len(my_series) 
    if size <= 1:
        size = 1
        rvs  = np.zeros(T) 
        p_r  = np.zeros(T) 
        rvs1 = 0
        p_r1 = 0 
    else:
        rvs  = np.zeros((T, size)) 
        p_r  = np.zeros((T, size)) 
        rvs1 = np.zeros(size)
        p_r1 = np.zeros(size)
    for t, D_t in enumerate(my_series):
        rvs[t] = D_t.rvs(size, False)
        p_r[t] = D_t.pmf(rvs[t])
    if (D1 is None):
        return rvs, p_r
    else:
        rvs1 = D1.rvs(size, False)
        p_r1 = D1.pmf(rvs1)
        return rvs, p_r, rvs1, p_r1
    
def get_Y_t(Y_t, Panjer_series):    
    if Y_t is None:
        Y_t, p_Y = get_series_rvs(Panjer_series, size=1)                # random variable and pmf per year
    else:
        p_Y = np.zeros(len(Y_t))
        for i, Pi in enumerate(Panjer_series): p_Y[i] = Pi.pmf(Y_t[i])  # probabilieties for 'observations'
    return Y_t, p_Y    

def get_series_EV(my_series, D1=None):
    T = len(my_series) 
    E = np.zeros(T) 
    V = np.zeros(T) 
    for t, D_t in enumerate(my_series):
        E[t], V[t] = D_t.stats('mv')
    if (D1 is None):
        return E, V
    else:
        E1, V1 = D1.stats('mv')
        return E, V, E1, V1

def get_logN_EV_series(E, V, E1=None, V1=None):
    # Create list with a series of logNormal-distributions lnN(t)
    # - The mean E(t) and the variance V(t) are given with:
    #   len(E) = len(V)
    # - E1, V1: projections for T+1
    
    T = min(len(E), len(V))
    lnN_series = []
    for t in range(T):
        mu, s = logN_mu_s(E[t], V[t])
        lnN_series.append(scipy_logNorm(mu=mu, sigma=s))      
    if (E1 is None) or (V1 is None):
        return lnN_series
    else:
        mu, s = logN_mu_s(E1, V1)    
        lnN1 = scipy_logNorm(mu=mu, sigma=s)
        return lnN_series, lnN1    

def get_conv(T, x, y_t, is_pmf):
    # convolute the set of T distributions captured in the array y_t(T, N) on the support x(N) 
    # - is_pmf = True : discrete distribution defined on integer support    i = 0, ..., N-1
    #          = False: discrete distribution defined on discrete support x_i = 0, dx, 2*dx, ..., (N-1)*dx 
    # - Adjust the densities to the support compressed by a factor T
    def get_norm_pmf(f, dx):
        y = f * dx
        dy = 1 - y.sum()
        y[-1] += dy 
        return y
    
    N_steps = len(x) - 1
    x_max   = x.max()
    if is_pmf:
        d_x = 1
    else:
        d_x = x_max / N_steps                              # interval length on initial support
    y_conv  = get_norm_pmf(y_t[0], d_x)                    # normalized probabilities per interval
    for t in range(1, T):
        y2 = get_norm_pmf(y_t[t], d_x) 
        y_conv = signal.convolve(y_conv, y2)

    N_conv   = len(y_conv)
    x_conv   = np.linspace(0, x_max, N_conv)
    d_x_conv = d_x / T
    y_conv   = y_conv / d_x_conv                           # densities on new compressed support
    return x_conv, y_conv
    
# *********************************

def get_stats(T, pdf_supp, pmf_supp, data_t, data1, is_sim=True):
# data for period t=0...T-1
    (pdf_t, q0,  p0,  E_t,  V_t,  S_t,  pmf_t) = data_t            # series for t=0...T-1
# Overall distributions for the average
    x1_conv, y1_conv = get_conv(T, pmf_supp, pmf_t, True )         # evaluate aggregate pmf via convolution
    x2_conv, y2_conv = get_conv(T, pdf_supp, pdf_t, False)         # evaluate aggregate pdf via convolution
# Average of aggregates
    E_tot = E_t.sum() / T
    V_tot = V_t.sum() / (T**2)   
# Overall logNorm fit and pdf's at mean and mean +/- one std   
    mu, s = logN_mu_s(E_tot, V_tot)
# Common scaling factor: mode/f_max <= 1. for t<0 and <=2. for t=1
    D1 = scipy_discrete_custom()
    D1.init_x_p(x1_conv, y1_conv)
    D2 = scipy_discrete_custom()
    D2.init_x_p(x1_conv, y1_conv)
    E1, V1 = D1.stats('mv')
    E2, V2 = D2.stats('mv')
# data projected for T=1
    (pdf1,  q10, p10, E1_m, V1_m, S1_m, pmf1 ) = data1             # data for projectionT=+1  
    if is_sim:
        E1_1 = (pmf_supp*pmf1).sum()    
        dx = pdf_supp[1] - pdf_supp[0]
        E1_2 = (pdf_supp*pdf1).sum() * dx 
    else: 
        E1_1 = float(E1_m)
        E1_2 = float(E1_m)
# relevant statistics
    range_stats = (q0[0][1], q0[-1][1], V_t.sum() / E_t.sum())    # mu_0, mu_T, f_P
    tot_stats_1 = (E1, V1**0.5, E1_1)                             # aggregate and projected pmf
    tot_stats_2 = (E2, V2**0.5, E1_2)                             # lognoemal fit
    aggr_dist = ((x1_conv, y1_conv), (x2_conv, y2_conv))
    stats     = (range_stats, tot_stats_1, tot_stats_2)
    return aggr_dist, stats

def get_moments(data, N_years, N_param, std_min=1e-4):
    data_moments = np.zeros((N_years+1, N_param, 2))       # 0: ovearll data, [1..N_years]: annual data
    for j in range(N_years+1):
        for k in range(N_param):
            d_j_k = data[:, j:j+1, k:k+1] 
            data_moments[j][k][0] = d_j_k.mean() / Red_Fields[k][2]
            data_moments[j][k][1] = d_j_k.std()  / Red_Fields[k][2] + std_min
    return data_moments        

def lin_regr(data_moments, Yr_min, Yr_max, k):
    j2 = Yr_max + 1 - Yr_min
    x = np.linspace(Yr_min, Yr_max, j2)
    y_tot_mean = data_moments[0:1,    k:k+1, 0:1].transpose()[0][0][0]    # Overall moments as scalars
    y_tot_std  = data_moments[0:1,    k:k+1, 1:2].transpose()[0][0][0]
    y_mean     = data_moments[1:j2+1, k:k+1, 0:1].transpose()[0][0]       # Annual moments as vectors
    y_std      = data_moments[1:j2+1, k:k+1, 1:2].transpose()[0][0]
    slope_m, y0_m = np.polyfit(x, y_mean, 1)                              # Lin regression as scalars
    slope_s, y0_s = np.polyfit(x, y_std , 1)   
    t_ref = (Yr_min + Yr_max) / 2                                         # Anchor point
    y_ref_m = y0_m + slope_m * t_ref
    y_ref_s = y0_s + slope_s * t_ref
    return y_tot_mean, y_tot_std, x, y_mean, y_std, t_ref, y_ref_m, slope_m, y_ref_s, slope_s

def get_ab(A, B, C, D, E):    
#  I) A * a + C * b = D    
# II) C * a + B * b = E    
    det = A * B - np.square(C)
    if abs(det) > 1e-6:
        a = (B*D - C*E) / det
        b = (A*E - C*D) / det
    else:
        a = None
        b = None
    return a, b

def m_s(E, V):
    try:
        E += 1E-6
    except:
        E += 0
    s2 = np.log(1. + V / np.square(E))
    m = np.log(E) - s2/2
    return m, np.sqrt(s2)

def get_ab_MAP_N(t0, t_, Y_t, mt, st, sigma_a, sigma_b):
    t_ = t_ - t0
    st_2 = np.square(st)
    A = 1/np.square(sigma_a) + (1/st_2).sum()
    B = 1/np.square(sigma_b) + (np.square(t_)/st_2).sum()
    C =  (t_/st_2).sum()
    D = ((Y_t - mt) / st_2).sum()
    E = ((Y_t - mt) * t_ / st_2).sum()
    return get_ab(A, B, C, D, E)

def get_ab_MAP_logN(t0, t_, Y_t, mean_t, std_t, sigma_a, sigma_b):
    mt, st = m_s(mean_t, np.square(std_t))
    return get_ab_MAP_N(t0, t_, np.log(Y_t), mt, st, sigma_a, sigma_b)

def get_ab_MAP(t0, t_, Y_t, mean_t, std_t, sigma_a, sigma_b, param_type, distr_fit=False):
    if distr_fit:
        E_X = mean_t
        V_X = std_t**2
        Phi_a = sigma_a**2 
        Phi_b = sigma_b**2 
        Phi_c = sigma_a   # *len(t_) 
        t_ = t_ - t0
        if   param_type == D_Norm:
            a, b, _, _, _ = N_fit_trend(t_, Y_t, E_X, V_X, Phi_a, Phi_b, Phi_c)
        elif  param_type in [D_Bino, D_Pois, D_negB, D_Panj, D_Gamm, D_logN]:
            a, b, _, _, _ = logN_fit_trend(t_, Y_t, E_X, V_X, Phi_a, Phi_b, Phi_c)
        else:
            a = b = None 
    else:
        if param_type == D_Norm:
            a, b = get_ab_MAP_N(t0, t_, Y_t, mean_t, std_t, sigma_a, sigma_b)
        elif param_type in [D_Bino, D_Pois, D_negB, D_Panj, D_Gamm, D_logN]:
            a, b = get_ab_MAP_logN(t0, t_, Y_t, mean_t, std_t, sigma_a, sigma_b)
        else:
            a = b = None
    return a, b 

def claims_count_gen_process(process_param, Y_t=None, plt_true=True, N=100, fn_ID='dummy_plot'):
    # Claims count: probability distribtions for the 'generative process'
    run_label = 'Generative process '
    caption   =  ['negative binomial distributions', 'fitted log normal distributions', ]
    cal_param = (0, 0, 0)    
    
    run_param = get_trended_model(process_param, cal_param)
    (T, E_0, E_T, f_Panjer) = run_param

    # get upper limit for the support (plotted in y-direction)
    X_max = int(get_X_max(T, E_0, E_T, f_Panjer))
    
    # discrete and continuous support
    q = np.arange(X_max)
    x = np.linspace(0., X_max, N)

    # series with Panjer distributions used to emulate the process    
    Panjer_series, P1 = get_Panjer_series(T, E_0, E_T, f_Panjer)
    pmf_t, pmf1 = get_series_pmf(Panjer_series, q, D1=P1)               # array with pmf's and projection
    
    Y_t, p_Y = get_Y_t(Y_t, Panjer_series)                              # generate Y_t (if None) and evaluate prob.
    E_p, V_p, E1_p, V1_p = get_series_EV(Panjer_series, D1=P1)          # vectors with mean and variance
    
    # series with fitted logNormal distributions
    logN_series, logN1 = get_logN_EV_series(E_p, V_p, E1=E1_p, V1=V1_p)
    pdf_t, pdf1 = get_series_pdf(logN_series, x, D1=logN1)              # array with pdf's
    S_p   = V_p**0.5   
    S1_p  = V1_p**0.5

    q0    = np.array([E_p-S_p, E_p, E_p+S_p]).transpose()               # array with mean and mean +/- std
    q10   = np.array([E1_p-S1_p, E1_p, E1_p+S1_p])
    p0    = np.zeros((T, 3))                                            # and pdf's
    for t in range(T):
        p0[t] = logN_series[t].pdf(q0[t])
    p10 = logN1.pdf(q10)

    data_t = (pdf_t, q0,  p0,  E_p,  V_p,  S_p,  pmf_t)                # series for t=0...T-1
    data1  = (pdf1,  q10, p10, E1_p, V1_p, S1_p, pmf1 )                # data for projectionT=+1
    aggr_dist, stats = get_stats(T, x, q, data_t, data1, is_sim=False)   
    if plt_true: 
        plot_claims_cont(T, x, q, data_t, data1, Y_t, p_Y, aggr_dist, stats, caption, fn_ID=fn_ID)    
    else:
        fn_ID = 'no plot'
     
    M_EV_p = (run_label, cal_param, run_param, E_p, V_p, E1_p, V1_p, stats, fn_ID)
    
    return Y_t, X_max, M_EV_p
    
def get_discrete_series(X_t, X1=None):
    # X_t = array(T, N)
    # X1  = array(N)
    D_series = []
    for t, rvs_t in enumerate(X_t):
        D = scipy_discrete_custom()
        D.init_r(rvs_t)
        D_series.append(D)        
    if (X1 is None):
        return D_series
    else:
        D1 = scipy_discrete_custom()
        D1.init_r(X1)
        return D_series, D1    

def claims_count_Bayes(process_param, model_param, Y_t, X_max, Phi_a, Phi_b, Phi_c, K=50,
                       plt_true=True, N=100, fn_ID_list=['dummy_plot', 'dummy_plot', 'dummy_plot']):
    # Calibration of the claims-count model
    # Evaluate MAP calibration parameters and conditional probability distribtions for the 'generative model'
    # - (default) prior model   : M_EV_m
    # - scale calibration       : M_EV_c
    # - linear-trend calibration: M_EV_ab
    run_label = ['Prior model        ', 'Level calibration  ', 'Linear-trend cal.', ]
    caption   = ['simulated Poisson distributions', 'fitted log normal distributions', ]

    def run_model(model_param, cal_param, Y_t, run_label, caption, fn_ID):
        
        run_param = get_trended_model(model_param, cal_param)
        (T, E_0, E_T, f_Panjer) = run_param
   
        q = np.arange(int(X_max))                 
        x = np.linspace(0., X_max, N)
    
        # Generative model
        Panjer_series, P1    = get_Panjer_series(T, E_0, E_T, f_Panjer)
        X_s, p_s, X1_s, p1_s = get_series_rvs(Panjer_series, D1=P1, size=K)  
        E0, V0, E10, V10     = get_series_EV(Panjer_series, D1=P1)
        
        # Discrete distribution derived from simulations
        sim_series, sim1  = get_discrete_series(X_s, X1_s)
        pmf_t, pmf1       = get_series_pmf(sim_series, q, D1=sim1)        
        E_t, V_t, E1, V1  = get_series_EV(sim_series, D1=sim1)      
        
        # Fitted logNormal distributions
        logN_series, logN1 = get_logN_EV_series(E_t, V_t, E1, V1)
        pdf_t, pdf1 = get_series_pdf(logN_series, x, D1=logN1)            
        S_t = V_t**0.5   
        S1  = V1**0.5
        
       # ticks foe mean and +/- std
        q0  = np.array([E_t-S_t, E_t, E_t+S_t]).transpose()    
        q10 = np.array([E1-S1, E1, E1+S1])
        p_Y = np.zeros(len(Y_t))
        p0  = np.zeros((T, 3))                                
        for t in range(T):
            p_Y[t] = logN_series[t].pmf(Y_t[t])
            p0[t]  = logN_series[t].pdf(q0[t])
        p10 = logN1.pdf(q10)
        
        data_t = (pdf_t, q0,  p0,  E_t, V_t, S_t, pmf_t)                # series for t=0...T-1
        data1  = (pdf1,  q10, p10, E1,  V1,  S1,  pmf1 )                # data for projectionT=+1
        aggr_dist, stats = get_stats(T, x, q, data_t, data1)
        if plt_true: 
            plot_claims_cont(T, x, q, data_t, data1, Y_t, p_Y, aggr_dist, stats, caption, fn_ID=fn_ID)    
        else:
            fn_ID = 'no plot'
             
        return (run_label, cal_param, run_param, E_t, V_t, E1, V1, stats, fn_ID)

#   Check generative process:
#   - generate observ. Y_t (if None) and evaluate prob.
    (T, E_0, E_T, f_Panjer) = process_param
    Panjer_series, P1 = get_Panjer_series(T, E_0, E_T, f_Panjer)
    Y_t, p_Y = get_Y_t(Y_t, Panjer_series)                           # generate 
    
#   Run the generative model for three parameter sets:
#   - prior        : -> M_EV_m   
#   - level        : -> M_EV_c
#   - linear trend : -> M_EV_ab

    t_ = np.arange(len(Y_t))        
    k = 0
    M_EV_m = run_model(model_param, (0, 0, 0), Y_t, run_label[k], caption, fn_ID=fn_ID_list[k])

    (_, _, _, E, V, E1, V1, _, _) = M_EV_m
    a, b, t0, c, c_tot = N_logN_fit_trend(t_, Y_t, E, V, Phi_a, Phi_b, Phi_c)   
   
    k = 1
    M_EV_c  = run_model(model_param, (c_tot, 0, 0), Y_t, run_label[k], caption, fn_ID=fn_ID_list[k])

    k = 2
    M_EV_ab = run_model(model_param, (a, b, t0) , Y_t, run_label[k], caption, fn_ID=fn_ID_list[k])
    
    return Y_t, M_EV_m, M_EV_c, M_EV_ab

# ********************************************************

def claims_count_calibration(T, process_param, model_param, Y_t, Phi_a, Phi_b, Phi_c, K_sim=50,
                             fn_ID_plt=['dummy_plot', 'dummy_plot', 'dummy_plot', 'dummy_plot'], 
                             fn_ID_txt=['dummy_print', 'dummy_print']):
# Define overall and control variables
    plt_True = True
    N        = 100                                 # length of continuous support
# Define and run the generative process
    Y_t, Y_max, M_EV_p = claims_count_gen_process(process_param, Y_t=Y_t, 
                         plt_true=plt_True, N=N, fn_ID=fn_ID_plt[0]) 
# Define, run and calibrate the generative model
    Y_t, M_EV_m, M_EV_c, M_EV_ab = claims_count_Bayes(process_param, model_param, Y_t, Y_max, 
                                   Phi_a, Phi_b, Phi_c, K=K_sim, plt_true=plt_True, N=N,
                                   fn_ID_list=fn_ID_plt[1:])
# Document the process    
    print_claims_count(Y_t, K_sim, M_EV_p, M_EV_m, M_EV_c, M_EV_ab, fn_ID_list=fn_ID_txt)
  
# ********************************************************

def get_claims_count_stats(T, process_param, model_param, Y_t_0, Phi_a, Phi_b, Phi_c, K_sim=50, N_run=20,
                           fn_ID_uncond='dummy_plot', fn_ID_cond='dummy_plot'):   
    Y_t = None         # initialize and update with Y_t_0                              
    
    def get_stats_E(Y_t):
    # model parameters
        (T, E_0_p, E_T_p, f_P_p) = process_param         # Specification of gen. process
        E_p_t, E1_p = get_E_t(T, E_0_p, E_T_p)           # means: array for period 0...T-1 and scalar for T+1
        (_, E_0_m, E_T_m, f_P_m) = model_param           # Specification of gen. model
        E_m_t, E1_m = get_E_t(T, E_0_m, E_T_m)           # means: array for period 0...T-1 and scalar for T+1
    # calculated calibration parameters
        ln_f_0 = np.log(E_0_p / E_0_m)
        ln_f_T =  np.log(E_T_p / E_T_m)
        a0 = (ln_f_T + ln_f_0)/2                         # a0, b0: calibration of means at t=0 and t=T-1
        b0 = (ln_f_T - ln_f_0)/T
        c0 = np.log(E_p_t.sum()) - np.log(E_m_t.sum())   # c0: calibration of overall mean
    # calculated aggregate statistics for gen. process
        N0 = Y_t.sum() / T                               # 'observed' mean
        # E0 = E_p_t.sum() / T                             # model mean
        # S0 = (E_p_t.sum()*f_P_p)**0.5 / T                # model standard deviation
        n_N0 = (4, 10)                                   # indices of fields carrying N0
        return np.array([c0,  0, E_0_p, E_T_p, N0, E1_p, 
                         a0, b0, E_0_p, E_T_p, N0, E1_p]), n_N0

    def get_M_stats(M):
    # model parameters and statistics
        (label_1, cal_ab, param, E_p, V_p, E1, V1, stats, plot_FN) = M
        (a, b, t0)         = cal_ab
        (T, E_0, E_T, f_P) = param
        (range_stats, tot_stats_1, tot_stats_2) = stats
        (mu_0, mu_T, f_P_2) = range_stats
        (mu, s, mu1)        = tot_stats_1 
        return (a, b, mu_0, mu_T, f_P_2, mu, E1)

    l2, l3, l4, l5 = '$\Delta \mu_{-T}$', '$\Delta \mu_{-1}$', '$\Delta \hat{\mu}$', '$\Delta \mu_{1}$'   
    labels = ('$\Delta c$', ''         ,   l2, l3, l4, l5,
              '$\Delta a$', '$\Delta b^* $', l2, l3, l4, l5)  

    # Y_t, X_max, M_EV_p = process_evolution(process_param, Y_t=Y_t_0, plt_true=False) 
    # get upper limit for the support (plotted in y-direction)
    Y_t = Y_t_0
    (T, E_0, E_T, f_Panjer) = process_param
    Panjer_series, P1 = get_Panjer_series(T, E_0, E_T, f_Panjer)
    Y_t, p_X = get_Y_t(Y_t, Panjer_series)               # generate 
    X_max = int(get_X_max(T, E_0, E_T, f_Panjer))

    stats_E, n_N0 = get_stats_E(Y_t)    # model-based means  
    
    N0_distr    = []
    stats_distr = []
    for r in range(N_run):
        Y_t = Y_t_0
        Y_t, _, M_EV_c, M_EV_ab = claims_count_Bayes(process_param, model_param, Y_t, X_max, 
                                  Phi_a, Phi_b, Phi_c, K=K_sim, plt_true=False)
        N0 = Y_t.sum() / T
        N0_distr.append(N0)
        stats_E[n_N0[0]] = N0                             # adjust stats_E with run-specific expectation   
        stats_E[n_N0[0]] = N0
        c, _, mu_0_c,  mu_T_c,  f_P_c,  mu_c,  E1_c  = get_M_stats(M_EV_c)
        a, b, mu_0_ab, mu_T_ab, f_P_ab, mu_ab, E1_ab = get_M_stats(M_EV_ab)
        stats_record = np.array([c, 0, mu_0_c,  mu_T_c,  mu_c,  E1_c, 
                                 a, b, mu_0_ab, mu_T_ab, mu_ab, E1_ab])
        new_rec = stats_record - stats_E
        stats_distr.append(new_rec)
    stats_distr = np.array(stats_distr).transpose()
    stats_distr[7] *= (T-1)                           # replace delta_b by delta_b*
    
    N0 = np.array(N0_distr).mean()
    stats_E[n_N0[0]] = N0                             # adjust stats_E with mean
    stats_E[n_N0[0]] = N0

    if Y_t_0 is None:
        fn_ID = fn_ID_uncond
    else:
        fn_ID = fn_ID_cond
    plot_claims_count(K_sim, N_run, stats_distr, labels, fn_ID=fn_ID)
    
# ********************************************************

def run_calibration_comparison(param_lists, E, V, Y_t, Phi_c, fn_ID='dummy_print'):   
# E, V : mean and variance of (default) prior generative model
# Y_t  : simulated 'observations'
# Phi_c: variance of prior pi(c) 
# -> Evaluate calibration parameter c_tot with logN approximation
# -> Plot curves L(c) and R(c) 
     
    t_ = np.arange(len(Y_t))        
    Phi_a, Phi_b = Phi_c, Phi_c
    a, b, t0, c, c_tot = N_logN_fit_trend(t_, Y_t, E, V, Phi_a, Phi_b, Phi_c)       
    
    plot_R_c(param_lists, Y_t, E, V, Phi_c, c0=c_tot, fn_ID=fn_ID)
    
# ********************************************************

def model_comparison_LaTeX(model_param, f_P_list, i_f_P, K_sim=50, nr_runs=1, first=False, last=False,
                           fn_ID_mod='dummy_print',  fn_ID_par='dummy_print'):
    
    def print_results(results):
        def get_label(n, f_P, c_mod, Is_mean):
            if f_P > 1: 
                s_l = '$\\fnegBin$  '
            elif f_P < 1: 
                s_l = '$\\fBinomial$   '
            else:
                s_l = '$\\fPoisson$   '
            if Is_mean: 
                return '    ' + s_l.strip() + ' mean    '
            else:
                if n==2:
                    return s_l
                elif n==3:
                    return f'   $f_P$: {f_P:.1f}    '
                elif n==4:
                    return f'$c^{{mod}}$: {c_mod:.3f}'
                else:
                    return '                '
            
        def get_s_c(c, c_T, n=3, Is_OK=True, Is_bold=False, Is_P=False):
            if Is_OK:
                if Is_bold:
                    s1, s2 = f'& \\textbf{{{c:.{n}f}}} ', f'& \\textbf{{{c_T:.{n}f}}} '
                else:
                    s1, s2 = f'&      {c:.{n}f}     ',  f'&      {c_T:.{n}f}     '
            else:
                s1 = s2 = '&        -       ' 
            if Is_P:                  # Poisson case: c = c_T
                return s1
            else:
                return s1 + s2
                
        def get_s(n0, data, Is_mean=False):
            [n, f_P, c_mod, f_all, c_NB, c_NB_T, c_P, c_P_T, c_B, c_B_T, c_G, c_G_T, c_lN, c_lN_T] = data  
            s1 = get_label(n, f_P, c_mod, Is_mean)
            
            NB_OK = (f_all>1 and not Is_mean) or (f_P>1 and Is_mean)
            B_OK  = (f_all<1 and not Is_mean) or (f_P<1 and Is_mean)
            s_NB = get_s_c(c_NB, c_NB_T, Is_OK=NB_OK, Is_bold=f_P> 1) 
            s_P  = get_s_c(c_P,  c_P_T,  Is_OK=True,  Is_bold=f_P==1, Is_P=True) 
            s_B  = get_s_c(c_B,  c_B_T,  Is_OK=B_OK,  Is_bold=f_P< 1) 
            s_G  = get_s_c(c_G,  c_G_T,  Is_OK=True,  Is_bold=False ) 
            s_lN = get_s_c(c_lN, c_lN_T, Is_OK=True,  Is_bold=False ) 
           
            return f'{s1} & {f_all:.3f}  &' + s_NB + s_P + s_B + s_G + s_lN + '\\\\'
        
        n0 = (len(results)+1) // 2
        if first:
            print()
            file_name = print_file_name(fn_ID_par)
            print(f'Calibration parameters for LaTeX table: {file_name}')
            print()
            ts_start(file_name, wa='w')       # start transscript to file (erase prior content)
            print('\\begin{tabular}{lccccccccccc}')
            print('\\toprule')
            print('\\multicolumn{2}{c}{Model} && \\multicolumn{9}{c}{Calibration parameters $c^a_{MAP}$ and $c^p_{MAP}$} \\\\ ')
            print('\\cmidrule(rl){1-2}\\cmidrule(ll){4-12}')
            print('Case & $f_P^{sim}$ && $\\fnegBin^a$ & $\\fnegBin^p$ & $\\fPoisson^a = \\fPoisson^p$ & $\\fBinomial^a$ & $\\fBinomial^p$ & $\\fGamma^a$ & $\\fGamma^p$ & $\\flogNorm^a$ & $\\flogNorm^p$ \\\\') 
            print('\\bottomrule')
            print('\\toprule')
        for i, data in enumerate(results):
            print(get_s(n0, data))
            
        print('\\midrule')
        print(get_s(n0, np.array(results).mean(axis=0), Is_mean=True))
        if last:
            print('\\bottomrule')
            print('\\end{tabular}')
            ts_stop()                                   # stop transscript to file
            print()
        else:
            print('\\midrule')
    
    f_Panjer = f_P_list[i_f_P]
    # Unpack parameters defined in GIRF_models.py / param_model_comparison()
    (T, par_emp_0, par_sim_0, Phi) = model_param        
    # Parameters for generative process:    
    (gp_lda_0, gp_lda_T, gp_beta) = par_emp_0
    gp_ln_lda = (np.log(gp_lda_0)+np.log(gp_lda_T)) / 2
    K_emp   = 1
    par_emp = (gp_lda_0, gp_lda_T, gp_beta, f_Panjer, K_emp)
    # Parameters for generative model:    
    (gm_lda_0, gm_lda_T, gm_beta) = par_sim_0
    gm_ln_lda = (np.log(gm_lda_0)+np.log(gm_lda_T)) / 2
    K       = K_sim
    par_sim = (gm_lda_0, gm_lda_T, gm_beta, f_Panjer, K)
    # Calibrate
    (Phi_a, Phi_b, Phi_c) = Phi
    
    str_f_P = f'{f_P_list[0]:.1f} & {f_P_list[1]:.1f} & {f_P_list[2]:.1f} \\\\'
    if first:
        file_name = print_file_name(fn_ID_mod)
        print(f'Model parameters for LaTeX table: -> {file_name}')
        print()
        ts_start(file_name, wa='w')       # start transscript to file (erase prior content)
        print('\\begin{tabular}{lrSSrSrScS}')
        print('\\toprule')
        print('Model & $T$ & $\\lambda_{-T}^{ref}$ & $\\lambda_{-1}^{ref}$ & $\\widehat{\ln{\\lambda}^{ref}_{o}}$ & $\\beta$ & $K$ & $f_P^{\\fnegBin}$ & $f_P^{\\fPoisson}$ & $f_P^{\\fBinomial}$ \\\\') 
        print('\\bottomrule')
        print('\\toprule')
        print(f'Generative process & {T} & {gp_lda_0:.1f} & {gp_lda_T:.1f} & ${gp_ln_lda:.3f}$ & {gp_beta:.1f} & {K_emp} & {str_f_P}')
        print(f'Generative model   & {T} & {gm_lda_0:.1f} & {gm_lda_T:.1f} & ${gm_ln_lda:.3f}$ & {gm_beta:.1f} & {K_sim} & {str_f_P}')
        print('\\bottomrule')
        print('\\end{tabular}')
        ts_stop()                                     # stop transscript to file
        
    # Comparison of calibration:
    # - logN    : a, b, c      -> trend and level of smmothed data
    # - Poisson : c_P0,  c_P1  -> level of simulated and smoothed    
    # - negBin  : c_NB0, c_NB1 -> level of simulated and smoothed    
    # - Bin     : c_B0,  c_B1  -> level of simulated and smoothed    
    
    t0 = T / 2
    t_ = np.arange(T) - t0
    results = []
    for n in range(nr_runs):
        # Get empirical data and data for the generative model
        # -> get_N_sim() for details
        data_emp, data_sim = Panjer_sim_test_data(T, par_emp, par_sim)
        years, lda_emp, _      , N_emp, a_mod, b_mod = data_emp
        _    , lda_ref, lda_gen, N_sim, a_sim, b_sim = data_sim
    
        # Get statistics and regression fit
        # - statistics: lda_sim_, var_sim_, N_tot, lda_tot
        # - prior     : lda0, a0, b0
        # - posterior : a, b, mu
        lda_sim_, var_sim_, N_tot, lda_tot, var_tot, a_regr, b_regr, lda0, f_, g_, gamma, mu = Poisson_lgN_fit(years, N_sim, T, K) 
        
        # Derive simulated Panjer factors
        # - f_sim : annual ratios V[Xt]/E[Xt]
        # - f_tot : overall ratio V[X]/E[X]
        f_sim = var_sim_ / lda_sim_
        f_all = var_sim_.sum() / lda_sim_.sum()

        # fitting to initial variables
        c_P, _  = Poisson_fit(N_emp[0], lda_sim_, Phi_c)
        c_NB, _ = negBin_fit(N_emp[0], lda_sim_, var_sim_, Phi_c)
        c_B, _  = Bin_fit(N_emp[0], lda_sim_, var_sim_, Phi_c)
        c_lN, _ = logN_fit(N_emp[0], lda_sim_, var_sim_, Phi_c)
        c_G, _  = Gamma_fit(N_emp[0], lda_sim_, var_sim_, Phi_c)

        # fitting to smoothed variables
        a, b, c, f_trend, f_tot = get_f_calibration(years, 0, N_emp, mu, mu*f_all, N_emp.sum(), lda_tot, var_tot, Phi_a, Phi_b, Phi_c, logNorm=True)
        c_mod   = (a_mod-a_sim)+(b_mod-b_sim)*(T-1)/2
        # c_P, _  = Poisson_fit(N_emp[0], mu, Phi_c)
        # c_NB, _ = negBin_fit(N_emp[0], mu, mu*f_all, Phi_c)
        # c_B, _  = Bin_fit(N_emp[0], mu, mu*f_all, Phi_c)
        # c_lN, _ = logN_fit(N_emp[0], mu, mu*f_all, Phi_c)
        # c_G, _  = Gamma_fit(N_emp[0], mu, mu*f_all, Phi_c)

        N_T    = np.array([N_emp[0].sum()])
        E_T    = np.array([lda_tot])
        V_T    = np.array([var_tot])
        c_P_T, _  = Poisson_fit(N_T, E_T, Phi_c)
        c_NB_T, _ = negBin_fit(N_T, E_T, V_T, Phi_c)
        c_B_T, _  = Bin_fit(N_T, E_T, V_T, Phi_c)
        c_lN_T, _ = logN_fit(N_T, E_T, V_T, Phi_c)
        c_G_T, _  = Gamma_fit(N_T, E_T, V_T, Phi_c)
        
    #     print(f'Calibration, run {n+1}')
    #     print('f_P      :', f_Panjer, f_all)
    #     print('Mod  a, b:', a_mod, b_mod)
    #     print('Sim  a, b:', a_sim, b_sim)
    #     print('Regr a, b:', a_regr, b_regr)
    #     print()
    #     print('mod a - c:', a_mod-a_sim, b_mod-b_sim, (a_mod-a_sim)+(b_mod-b_sim)*(T-1)/2)
    #     # print('lnN a - c:', a0, b0, a0+b0*(T-1)/2)
    #     print('lnN a - c:', a, b, a+b*(T-1)/2)
    #     print()
    #     print('f_P      :', np.round([f_Panjer, f_all],3))
    #     print('mod      :', np.round([c_mod],3))
    #     print('negBin   :', np.round([c_NB, c_NB_T],3))
    #     print('Poisson  :', np.round([c_P , c_P_T ],3))
    #     print('Bin      :', np.round([c_B , c_B_T ],3))
    #     print('Gamma    :', np.round([c_G , c_G_T ],3))
    #     print('lnN      :', np.round([c_lN, c_lN_T, c],3))
        
        results.append([n+1, f_Panjer, c_mod, f_all, c_NB, c_NB_T, c_P, c_P_T, c_B, c_B_T, 
                        c_G, c_G_T, c_lN, c_lN_T])
    
    # print(np.round(results, 3))
    # print()
    
    print_results(results)

        
# ********************************************************

def get_data_arr(L_list, N_years, N_data, Year_min):
    N_sim = len(L_list)
    data = np.zeros((N_sim, N_years+1, N_data))          # 0: ovearll data, [1..N_years]: annual data
    for i in range(N_sim):
        L_obj = L_list[i]
        for j in range(N_years+1):
            Yr = Year_min-1+j
            if j == 0:
                L = L_obj                    # Overall statistics
            else:
                L = L_obj.Red_Ann[Yr]        # Annual statistics 
            data[i][j] = L.get_red_stats()    
            # data[i, j, 0] = L.Claim_Freq
            # data[i, j, 1] = L.Clos_Freq
            # data[i, j, 2] = L.Inc
            # data[i, j, 3] = L.Paid
            # data[i, j, 4] = L.Lag_R
            # data[i, j, 5] = L.Lag_I - L.Lag_R
            # data[i, j, 6] = L.Lag_P - L.Lag_R
            # data[i, j, 7] = L.Nr_I
            # data[i, j, 8] = L.Nr_P
    return data 

def get_param_ab(T, param, log_True):
    param_ab = np.zeros((2,len(log_True)))
    for i, log_T_i in enumerate(log_True):
        y1, y2 = param[0][i], param[1][i]
        if log_T_i: y1, y2 = np.log(y1), np.log(y2)
        param_ab[0][i] = (y2+y1)/2 
        param_ab[1][i] = (y2-y1)/(T-1)
    return param_ab 

def get_param(t, t0, param_ab, log_True):
    param = param_ab[0] + param_ab[1] * (t - t0)
    for i, log_T_i in enumerate(log_True):
        if log_T_i: param[i] = np.exp(param[i])
    return param

def run_full_calibration(iterations, N_sim, param_labels, process_param, model_param, Date_Sub, Year_min, Year_max, 
                         cal_True, Phi_cal, plt_dist=False, fn_ID_plt='dummy_plot', fn_ID_txt='dummy_print'):
    
    def get_stats(p_ab, t0, log_True, Date_Sub, Year_min, Year_max, N_sim):
        L = Red_Claim()
        L_list = []
        for i in range(N_sim):
            L_tot = Red_Overall()
            for Year in range(Year_min, Year_max+1):      
                [freq, f_clos, mean_I, mean_P, t_rep, t_inc, t_paid, n_inc, n_paid] = get_param(Year, t0, p_ab, log_True)
                L_yr = Red_Ann()
                N = 1 + poisson.rvs(max(0., freq-1))
                for k in range(N):
                    L._init_random(mean_I, t_rep, t_inc, t_paid, n_inc, n_paid, Date_Sub, Year=Year)
                    L_yr.add_Claim(L)
                L_tot.add_Claim(L_yr)
            L_list.append(L_tot)         
        return L_list

    N_param = len(process_param[0])
    param_list = []
    T        = Year_max - Year_min + 1
    t_ref    = (Year_min + Year_max) / 2
    for k, T_k in enumerate(cal_True):
        if T_k: param_list.append(k)
    # data_list                   = [   0,      1,      2,      3,     4,     5,      6,     7,      8]   
    # process_param, model_param : [[freq, f_clos, mean_I, mean_P, t_rep, t_inc, t_paid, n_inc, n_paid],  # @ t=-T
    #                               [freq, f_clos, mean_I, mean_P, t_rep, t_inc, t_paid, n_inc, n_paid]]  # @ t=-1
    
    # LaTeX table with parameters   
    print_all_param(param_labels, process_param, model_param, T, cal_True, Phi_cal, fn_ID=fn_ID_txt)

    log_True = np.ones(N_param)
    for k in range(N_param): log_True[k] = Red_Fields[k][4] == D_logN

    proc_ab0  = get_param_ab(T, process_param, log_True)
    model_ab0 = get_param_ab(T, model_param, log_True)
    ab_calc = proc_ab0 - model_ab0

    O_list = get_stats(proc_ab0, t_ref, log_True, Date_Sub, Year_min, Year_max, N_sim)
    O_list[0].Print()
    proc_data = get_data_arr(O_list, T, N_param, Year_min)      # -> distributions of gen process
    Y = proc_data[0]                                            # use 1st simulation as 'observation'
    
    [Phi_a, Phi_b, Phi_c] = Phi_cal
    a  = np.zeros((iterations+1, N_param))
    b  = np.zeros((iterations+1, N_param))
    sigma_a = np.zeros((iterations+1, N_param))
    sigma_b = np.zeros((iterations+1, N_param))
    
    sigma_a[0] = np.sqrt(Phi_a)
    sigma_b[0] = np.sqrt(Phi_b)
    
    for i in range(iterations):
        p_ab = model_ab0 + np.array([a[i], b[i]])
        param0 = get_param(t_ref, t_ref, p_ab, log_True)
        print('Iteration', i)
        print('  p Mod:', my_round(param0, 3))

        L_list = get_stats(p_ab, t_ref, log_True, Date_Sub, Year_min, Year_max, N_sim)
        mod_data = get_data_arr(L_list, T, N_param, Year_min)    
        
        if i == 0: 
            q_min_max = plot_evol_all(T, N_param, i, Y, proc_data, mod_data, log_True,
                                      fn_ID=fn_ID_plt)
            
        data_moments = get_moments(mod_data, T, N_param)
        
        Y0  = np.zeros(N_param)
        Y_t = np.zeros((N_param, T))
        t_ = Year_min + np.arange(T)
        for k in param_list:                         # range(N_param):
            Y0[k]  = Y[0][k] / Red_Fields[k][2]
            Y_t[k] = Y[1:T+1,k:k+1].transpose()[0] / Red_Fields[k][2]
            y_tot_mean, y_tot_std, x, y_mean, y_std, t_ref, y_ref_m, slope_m, y_ref_s, slope_s = lin_regr(data_moments, Year_min, Year_max, k)
            a_MAP, b_MAP = get_ab_MAP(t_ref, t_, Y_t[k], y_mean, y_std, sigma_a[i][k], sigma_b[i][k], Red_Fields[k][4], distr_fit=True)

            print(Red_Fields[k][5], 'da, db:', my_round([a_MAP, b_MAP], 3))
            if a_MAP is None: a_MAP = 0
            if b_MAP is None: b_MAP = 0
            a[i+1][k] = a[i][k] + a_MAP
            b[i+1][k] = b[i][k] + b_MAP
            a0, b0 = ab_calc[0][k], ab_calc[1][k]
            print('   new a,  b', my_round([a[i+1][k], b[i+1][k], a0, b0], 3))
            
        sigma_a[i+1] = sigma_a[i] / 2
        sigma_b[i+1] = sigma_b[i] / 2
        
    p_ab = model_ab0 + np.array([a[i], b[i]])
    param0 = get_param(t_ref, t_ref, p_ab, log_True)

    L_list = get_stats(p_ab, t_ref, log_True, Date_Sub, Year_min, Year_max, N_sim)
    mod_data = get_data_arr(L_list, T, N_param, Year_min)    
    
    plot_evol_all(T, N_param, iterations, Y, proc_data, mod_data, log_True, q_min_max=q_min_max,
                  fn_ID=fn_ID_plt)
        
    # c  = np.zeros((iterations+1, N_param))
    # b0 = np.zeros(N_param)
    # sigma_c = np.zeros((iterations+1, N_param))
    
    # plot_calibration(proc_ab0, model_ab0, a, b, c, sigma_a, sigma_b, sigma_c, Y, T, iterations, N_param, log_True)

def get_freq_model_new_default(K=100):
    process_param, model_param, _, _ = get_freq_model(use_default_obs=False)
    
    # series with Panjer distributions used to emulate the process    
    (T, E_0, E_T, f_Panjer) =process_param
    Panjer_series, P1 = get_Panjer_series(T, E_0, E_T, f_Panjer)    
    Y_t, p_Y = get_Y_t(None, Panjer_series)                              # generate Y_t (if None) and evaluate prob.
    
    # series with Panjer distributions used to emulate the process    
    (T, E_0, E_T, f_Panjer) = model_param
    Panjer_series, P1 = get_Panjer_series(T, E_0, E_T, f_Panjer)  
    X_KT = []
    for k in range(K):
        X_t, p_X = get_Y_t(None, Panjer_series)                              # generate Y_t (if None) and evaluate prob.
        X_KT.append(X_t)
    X_KT = np.array(X_KT) 
    E_X = X_KT.mean(axis=0)
    V_X = X_KT.var(axis=0)
    
    # copy / paste console output to: GIRF_models.py
    print('# random variables drawn from:')
    print(f'#  - process_param = {process_param}')
    print('Y_t_default = np.array(', list(Y_t.astype(int)),')')
    print(f'#  - model_param, K = {model_param}, {K}')
    print('E_default = np.array(', list(np.round(E_X, 3)),')')
    print('V_default = np.array(', list(np.round(V_X, 3)),')')
    print('s_default = np.array(', list(np.round(V_X**0.5, 3)),')')

if __name__ == "__main__":
    
    # Create default parameters: copy / paste from console to GIRF_models.py
    K = 100
    get_freq_model_new_default(K=K)
    