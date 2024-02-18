# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']  = 600
plt.rcParams['savefig.dpi'] = 600

from Transscript import ts_start, ts_stop
from GIRF_models import D_Bino, D_Pois, D_negB, D_Panj, D_Gamm, D_Norm, D_logN, Red_Fields, GIRF_fn_dict
from GIRF_stats import get_E_t, logN_mu_s, scipy_logNorm, scipy_Norm, scipy_Panjer, scipy_Gamma, get_rvs_cdf
from GIRF_Bayes import logN_fit, get_c_lo_bound, R_c_Poisson, R_c_negBin, R_c_Bin, R_c_Gamma

# *********************************

def save_plot(plt, fn_ID, fn_append=''):
    # Save plot to 'file_name' defined in GIRF_fn_dict wit all extensions (graphic formats)
    # in the ext_list
    (file_name, ext_list, _) = GIRF_fn_dict[fn_ID]
    file_name += fn_append
    for ext in ext_list:
        plt.savefig(file_name + ext, bbox_inches='tight', pad_inches=0)
    return file_name, ext_list  

def print_file_name(fn_ID, fn_append=''):
    # Get file name and extension from dictionary GIRF_fn_dict
    # use the first entry in the ext_list as type for saviing text
    (file_name, ext_list, _) = GIRF_fn_dict[fn_ID]
    file_name += fn_append
    return file_name + ext_list[0]

# *********************************

def plot_claims_cont(T, pdf_supp, pmf_supp, data_t, data1, Y_t, p_Y, aggr_dist, stats, caption, fn_ID='dummy_plot'):    
# Modified version of: 
#    Plot_Distr(k, data, moments, Y, Year_min, Year_max, Is_Int, N_sim, i_iter, n_iter)
# in module: ..\Distributions\ExEx_calibrate    

    def get_max_indices(y0_t, y0_1, y0_tot, y1_t, y1_1, y1_tot):
        def max_int(y, f):
            y_ = (y >= y.max()*f) 
            w_ = np.linspace(0, 1, len(y))
            n_lo = np.argmax(y_ * (1-w_))
            n_hi = np.argmax(y_ * w_)
            return (n_lo, n_hi)
        n_t = [[], []]       
        f = 0.005
        for i, y_i in enumerate(y0_t): n_t[0].append(max_int(y_i, f))
        for i, y_i in enumerate(y1_t): n_t[1].append(max_int(y_i, f))
        n1 = [max_int(y0_1, f), max_int(y1_1, f)]
        
        f = 0.0005
        n_tot = [max_int(y0_tot, f), max_int(y1_tot, f)]
        return n_t, n1, n_tot
        
    def plt_distr(ax, t, t_y, x, n12, is_pmf, c='k', ls='-', lw=0.5):
        # plot hor lines representing the pmf's if 'is_pmf' is True,
        # otherwise plot 'continuous' pdf's
        # - X  : support of pmf / pdf plotted in y-direction
        # - t_y: t + normalized pmf / pdf plotted in x-direction
        # the routine is also used to draw horizontal lines at [mu-s, mu, mu+s] with different linestyles
        def get_ls(ls, i):
            try:
                return ls[i] 
            except: 
                return ls
                    
        (n1, n2) = n12  
        if is_pmf:
            for i in range(n1, n2):
                x_hi = x[i]
                ls_i = get_ls(ls, i)
                ax.plot([t, t_y[i]], [x_hi, x_hi], c, ls=ls_i , lw=lw)     # pmf for year t 
        else:
            ax.plot(t_y[n1:n2], x[n1:n2], c, ls=ls , lw=lw)                # pdf for year t 
    
    def get_x_min(x, n): return x[n[0]]                                    # lower bound for plot
    def get_x_max(x, n): return x[n[1]]                                    # upper bound for plot
    
    def get_x_range(x, n, X_max, w=0.75):                                  # plot range
        x1 = get_x_min(x, n)
        x2 = get_x_max(x, n)
        y_min    =                 w * x1
        y_max    = (1-w) * X_max + w * x2
        return y_min, y_max

    def plt_data_t(ax, t, x, y, X_max, f_max, n_t, is_pmf, c1, lw0, lw1):
        if t == 1:
            y_min, y_max = 0, X_max
        else:
            y_min, y_max = get_x_range(x, n_t, X_max)
        if t == -T: y_min = 0.    
        ax.plot([t, t], [y_min, y_max], c1, lw=lw0)                        # vertical axis for year t
        plt_distr(ax, t, t+y/f_max, x, n_t, is_pmf, lw=lw1)                # pdf / pmf for year t 
        
    def my_f(x, n=1): return f'{x:.{n}f}'                                  # format function
    
    def s_range(range_stats):                                              # string with range parameters
        (mu_0, mu_T, f_P) = range_stats
        s1, s2, s_f = my_f(mu_0), my_f(mu_T), my_f(f_P, 2)
        return '\n$\mu_t = [' + s1 + ', \cdots , ' + s2 + ']$ ; ' + '$f_P = ' + s_f + '$'
    
    def s_tot(tot_stats, draw_N):                                          # overall statistics
        (m, s, _) = tot_stats
        if draw_N:
            s_N = my_f(Y_t.mean(), 1)
            return '$\hat{{N}} = ' + s_N+ '$'                              # 'observed' 
        else:
            s_m = my_f(m, 1)
            s_s = my_f(s, 2)
            return '$\hat{{\mu}} = ' + s_m + '$' + '\n$\hat{{\sigma}} = ' + s_s + '$'      # model
    
    def s_proj(mu1):                                                        # overall statistics
        s_m = my_f(mu1, 1)
        return '$\mu_1 = ' + s_m + '$'                                      # projected mean

    (pdf_t, q0,  p0,  E_t, V_t, S_t, pmf_t)  = data_t
    (pdf1,  q10, p10, E1,  V1,  S1,  pmf1)   = data1
    (range_stats, tot_stats_1, tot_stats_2)  = stats
    ((x1_conv, y1_conv), (x2_conv, y2_conv)) = aggr_dist
    caption = (caption[0]+s_range(range_stats), caption[1])    
# Time stamps
    t = np.arange(0, T) - T
    t_tot = 0
    t1    = 1

# Average of aggregates
    E_tot = E_t.sum() / T
    V_tot = V_t.sum() / (T**2)   
    Y_tot = Y_t.sum() / T
    S_tot = V_tot**0.5
    
# Overall logNorm fit and pdf's at mean and mean +/- one std   
    mu, s = logN_mu_s(E_tot, V_tot)
    logN  = scipy_logNorm(mu=mu, sigma=s)
    x_E   = [E_tot-S_tot, E_tot, E_tot+S_tot]
    pdf_E = logN.pdf(x_E)
    p_tot = logN.pdf(Y_tot)

# Common scaling factor: mode/f_max <= 1. for t<0 and <=2. for t=1
    f_max = max(max(pmf_t.max(), pdf_t.max()), max(y1_conv.max(), y1_conv.max())/2 )
        
    fig, axs = plt.subplots(2)

    c0, c1 = 'k', 'gray'
    lw0, lw1 = 0.25, 0.5
    ls0, ls1, ls2 = '-', '--', ':'
    mk0, mk1, mk2, mk3 = '.', 0, 'd', 'D'
    ms0, ms1, ms2 = 4, 12, 16
    
    n_t, n1, n_tot = get_max_indices(pmf_t, pmf1, y1_conv, pdf_t, pdf1, y2_conv)
    X_max = max(pdf_supp.max(), pmf_supp.max(), get_x_max(x1_conv, n_tot[0]), get_x_max(x2_conv, n_tot[1]))
    
    for a in range(2):
        ax = axs[a]
        
        ax.plot([t[0], 2], [0, 0], c1, lw=lw0)                         # draw x-axis 
        
        if a == 0:                                                     # discrete pmf (Panjer case)
            is_pmf = True
            x, y = pmf_supp, pmf_t
            y1   = pmf1
            x_conv, y_conv = x1_conv, y1_conv
            tot_stats = tot_stats_1
            lw_1 = lw0
            
        else:                                                          # continuous pdf (logNormal case)     
            is_pmf = False
            x, y = pdf_supp, pdf_t
            y1   = pdf1
            x_conv, y_conv = x2_conv, y2_conv
            tot_stats = tot_stats_2
            lw_1 =lw1
        
        for i in range(T):                                                   # annual distributions
            plt_data_t(ax, t[i], x, y[i], X_max, f_max, n_t[a][i], is_pmf, c1, lw0, lw1)
        plt_data_t(ax, t1, x, y1, X_max, f_max, n1[a], is_pmf, c1, lw0, lw1) # projected distribution
   
        if not is_pmf:                                                       # draw ticks for mean and mean +/- std 
            ls = [ls2, ls0, ls2]
            for i in range(T):                                               # annual distributions
                plt_distr(ax, t[i], t[i]+p0[i]/f_max, q0[i], (0, len(q0[i])),  
                          True, ls=ls, lw=lw1)
            plt_distr(ax, t1, t1+p10/f_max, q10, (0, len(q10)),              # projected distribution
                      True, ls=ls, lw=lw1)
            plt_distr(ax, t_tot, t_tot+pdf_E/f_max, x_E, (0, len(x_E)),      # aggregate distribution
                      True, ls=ls, lw=lw1)
          
        ax.scatter(t, E_t,     c=c0, marker=mk1, s=ms0)                      # evolution of mean
        ax.scatter(t, E_t+S_t, c=c0, marker=mk1, s=ms0)                      #              mean + 1 stdev
        ax.scatter(t, E_t-S_t, c=c0, marker=mk1, s=ms0)                      #              mean - 1 stdev

        ax.scatter(t1, E1,    c=c0, marker=mk1, s=ms0)                       # projection of mean
        ax.scatter(t1, E1+S1, c=c0, marker=mk1, s=ms0)                       #              mean + 1 stdev
        ax.scatter(t1, E1-S1, c=c0, marker=mk1, s=ms0)                       #              mean - 1 stdev

        y_min, y_max = get_x_range(x_conv, n_tot[a], X_max)
        ax.plot([t_tot, t_tot], [y_min, y_max], c1, lw=lw0)                  # vertical axis for aggregate distribution
        plt_distr(ax, t_tot, t_tot+y_conv/f_max, x_conv, n_tot[a], 
                  is_pmf, lw=lw_1)  # aggregate distribution
        ax.text(t_tot, 0, s_tot(tot_stats, False), color='k', fontsize=8, ha='center', va='bottom') 
        if a==0: 
            ax.text(t_tot+0.1, y_max, s_tot(tot_stats_1,  True), color='k', fontsize=8, ha='center', va='bottom') 
        
        ax.scatter(t+p_Y/f_max, Y_t, c=c0, marker=mk2, s=ms1)                # random observations by year
        ax.scatter([t_tot+p_tot/f_max], [Y_tot], c=c0, marker=mk3, s=ms2)    # average of random observations
                
        ax.scatter(t_tot, E_tot,       c=c0, marker=mk1, s=ms0)              # overall average
        ax.scatter(t_tot, E_tot+S_tot, c=c0, marker=mk1, s=ms0)              #         average + 1 stdev
        ax.scatter(t_tot, E_tot-S_tot, c=c0, marker=mk1, s=ms0)              #         average  - 1 stdev

        ax.text(-T , X_max-0.5, caption[a], color='k', fontsize=8, ha='left', va='top') 

    axs[0].text(1, X_max-0.5, s_proj(tot_stats_1[2]), color='k', fontsize=8, ha='center', va='top') 
    axs[0].get_xaxis().set_visible(False)
    
    file_name, ext_list = save_plot(plt, fn_ID)
    plt.show()
        
# *********************************

def print_claims_count(Y_t, K, M_EV_p, M_EV_m, M_EV_c, M_EV_ab, fn_ID_list=['dummy_print', 'dummy_print']):
# Y_t     : List with 'simulated' observations
# K       : count of simulation runs    
# M_EV_p  : generative process
# M_EV_m  : prior model
# M_EV_c  : scale calibration
# M_EV_ab : linear-trend calibration
    n_dec = 3
    
    def get_s_x(x, n): 
        # if x == int(x): return f' & {int(x)}'
        # else:           return f' & {x:.{n}f}'
        return f' & {x:.{n}f}'

    def get_str(i, label, a, b, t0, mu_0, mu_T, f_P, N_, mu, mu1, n1=1, n2=2, is_line_2=False):
        s_d  = ' & '
        s1   = s_d + s_d + s_d
        s3_1 = s_d
        if is_line_2:
            if i == 0: 
                label = 'Actual pmf'
                s3_1 = get_s_x(N_, 1)
            else:      
                label = 'Simulated pmf'
        else:
            if i == 2: s1 = get_s_x(a, 2) + s_d + s_d
            if i == 3: s1 = get_s_x(a, 2) + get_s_x(b, 3) + get_s_x(t0, 1)
        s0 = f'{label} '
        s2 = get_s_x(mu_0, 1) + get_s_x(mu_T, 1) + get_s_x(f_P, 2)
        s3 = s3_1 + get_s_x(mu, 1) + get_s_x(mu1, 1)
        return s0 + s1 + s2 + s3 + '\\\\'

    def get_M_stats(M):
        (_, cal_ab, param, E_p, _, E1, _, _, _) = M
        (T, E_0, E_T, _) = param
        E_1 = E_T * np.exp((np.log(E_T)-np.log(E_0))/(T-1)*2)
        E_t,_ = get_E_t(T, E_0, E_T)
        return T, E_0, E_T, E_t, E_p, E_1, E1, cal_ab

    def s_x(x, n_dec=n_dec): return f'{x:.{n_dec}f}'
    
    def print_stats(s_label, X, ln_X, o_0, slope, intersection, n_dec=n_dec):
        print(f'- {s_label}:')
        print('    ', list(np.round(X, n_dec-1)))
        print(f'      mean = {s_x(X.mean())}')
        print(' ln:', list(np.round(ln_X, n_dec-1)))
        print(f'      mean = {s_x(ln_X.mean())}')
        print(f'      LR (o_0 = {o_0}): slope = {s_x(slope)} / intersection = {s_x(intersection)}')

    def analyze(t, x, ln_x, y, ln_y):
        mean_x = x.mean()
        slo_ln_x, int_ln_x, r, p, se = stats.linregress(t, ln_x)
        
        mean_y = y.mean()
        slo_ln_y, int_ln_y, r, p, se = stats.linregress(t, ln_y)
        
        c = np.log(mean_y / mean_x)
        a = int_ln_y - int_ln_x 
        b = slo_ln_y - slo_ln_x
        return c, (a,b)
    
# Parameters and statistics from gen process and gen model
    T, Ep_0, Ep_T, Ep_t, Ep_p, Ep_1, Ep1,      _ = get_M_stats(M_EV_p)
    T, Em_0, Em_T, Em_t, Em_p, Em_1, Em1,      _ = get_M_stats(M_EV_m)
    _,    _,    _,    _,    _,    _,   _, cal_c  = get_M_stats(M_EV_c)
    _,    _,    _,    _,    _,    _,   _, cal_ab = get_M_stats(M_EV_ab)
    
    (c_MAP, _, _)  = cal_c
    (a_MAP, b_MAP, t0) = cal_ab
    
    mu_t = Em_p

# Create LaTeX table with parameters and results
    filename = print_file_name(fn_ID_list[0])
    print()
    print(f'Model parameters for LaTeX table: -> {filename}')
    print('Simulation runs     :', K)
    print()
    
    ts_start(filename, wa='w')                       # start transscript to file (erase prior content)
    print('\\begin{tabular}{lrrrSSSrSS}')
    print('\\toprule')
    print(' & \multicolumn{3}{c}{$\mbox{Calibration}$} & \multicolumn{3}{c}{$\mbox{Annual}$} & \multicolumn{2}{c}{$\mbox{Average}$} & $\mbox{Projected}$ \\\\')
    print('\cmidrule(rl){2-4}\cmidrule(rl){5-7}\cmidrule(rl){8-9}')
    print(' Model & $a$ or $c$ & $b$ & $o_0$ & $\mu_{-12}$ & $\mu_{-1}$ & $f_{P}$ & $\hat{N}$ & $\hat{\mu}$ & $\mu_{1}$ \\\\')
    print('\\bottomrule')
    print('\\toprule')
    
    for i, M in enumerate([M_EV_p, M_EV_m, M_EV_c, M_EV_ab]): 
# model parameters and statistics
        (label_1, cal_ab, param, E_p, V_p, E1, V1, m_stats, plot_FN) = M
        (a, b, t0)         = cal_ab
        (T, E_0, E_T, f_P) = param
        E_t, E1 = get_E_t(T, E_0, E_T)
        mu  = E_t.sum() / T
        print(get_str(i, label_1, a, b, t0-T, E_0, E_T, f_P, 0, mu, E1))
# simulated statistics
        N_ = Y_t.sum() / T
        (range_stats, tot_stats_1, tot_stats_2) = m_stats
        (mu_0, mu_T, f_P_2) = range_stats
        (mu_1, s_1, E1_1) = tot_stats_1                            # pmf  statistics
        (mu_2, s_2, E1_2) = tot_stats_2                            # logN statistics
        print(get_str(i, '', 0, 0, 0, mu_0, mu_T, f_P_2, N_, mu_1, E1_1, is_line_2=True))
        if i<3: 
            print('    \\midrule')
        else:
            print('    \\bottomrule')

    # simulated annual 'observations' drawn from generative process
    print('\\toprule')
    s1 = 'Random variables used & $N_o$ & \multicolumn{8}{l}{$('
    n = len(Y_t)
    for i in range(n): 
        Yi = Y_t[i]
        if Yi < 10: 
            si = f' ~~{Yi}~~'
        else:
            si = f' {Yi}~~'
        if i < n-1:
            si += ','
        else:
            si += ')$} \\\\'
        s1 += si
    print(s1)
    
    # simulated annual means drawn from (prior) generative model
    # (_, _, _, mu_t, _, _, _, _, _) = M_EV_m
    s2 ="for the 'toy model'& $\mu^{0}_{o}$ & \multicolumn{8}{l}{$("
    n = len(mu_t)
    for i in range(n): 
        mui = mu_t[i]
        si = f' {mui:.1f}'
        if i < n-1:
            si += ','
        else:
            si += ')$} \\\\'
        s2 += si
    print(s2)
    print('\\bottomrule')
    print('\\end{tabular}')
    ts_stop()                               # stop transscript to file
    print()
    
# Create file with stats

    # t_0 = (T-1) / 2
    o_0 = t0 - T
    t = np.arange(T) - t0
    
    ln_mu_p = np.linspace(np.log(Ep_0), np.log(Ep_T), T)
    mu_p = np.exp(ln_mu_p)
    slo_mu_p, int_mu_p, r, p, se = stats.linregress(t, ln_mu_p)
        
    ln_Y_t =np.log(Y_t)
    slo_Y_t, int_Y_t, r, p, se = stats.linregress(t, ln_Y_t)
    
    ln_mu_m = np.linspace(np.log(Em_0), np.log(Em_T), T)
    mu_m = np.exp(ln_mu_m)
    slo_mu_m, int_mu_m, r, p, se = stats.linregress(t, ln_mu_m)

    ln_mu_t = np.log(mu_t)
    slo_mu_t, int_mu_t, r, p, se = stats.linregress(t, ln_mu_t)
    
    c_calc, (a_calc, b_calc) = analyze(t, mu_m, ln_mu_m, mu_p, ln_mu_p) 
    c_sim, (a_sim, b_sim) = analyze(t, mu_t, ln_mu_t, Y_t, ln_Y_t) 
    
    filename = print_file_name(fn_ID_list[1])
    print()
    print(f'Statistics: -> {filename}')
    print('Simulation runs     :', K)
    print()

    ts_start(filename, wa='w')                       # start transscript to file (erase prior content)
    print('\\begin{verbatim}')

    print('o:   ', list(np.arange(-T,0)))
    
    print('Generative process:')
    print_stats('Specified mu_o', mu_p, ln_mu_p, o_0, slo_mu_p, int_mu_p)    
    print_stats('Observed Y_o (K=1)',   Y_t,  ln_Y_t,  o_0, slo_Y_t,  int_Y_t)    
    # print('')

    print('Generative model:')
    print_stats('Specified mu_o', mu_m, ln_mu_m, o_0, slo_mu_m, int_mu_m)    
    print_stats(f'Simulated mu_o (K={K})', mu_t, ln_mu_t, o_0, slo_mu_t, int_mu_t)    
    # print('')

    print('Generative process and prior model:')
    print(f'- gen process: mu_1 = {s_x(Ep_1)}')
    print(f'- gen model  : mu_1 = {s_x(Em_1)}')
    
    print('Scale calibration and projection:')    
    mu_1_m = Em_1 * np.exp(c_calc)    # Em_1: adjustment applied to modelled  projection
    mu_1_f = Em_1 * np.exp(c_sim )    
    mu_1_B = Em_1 * np.exp(c_MAP )
    print(f'- model: c = ln({s_x(mu_p.mean())} / {s_x(mu_m.mean())}) = {s_x(c_calc)} / \
mu_1 = {s_x(mu_1_m)}')
    print(f'- freq : c = ln({s_x(Y_t .mean())} / {s_x(mu_t.mean())}) = {s_x(c_sim )} / \
mu_1 = {s_x(mu_1_f)}')
    print(f'- Bayes: c =                      {s_x(c_MAP )} / \
mu_1 = {s_x(mu_1_B)}')
    # print('')

    print(f'Linear-trend calibration and projection (o_0 = {o_0}):')  
    mu_1_m = Em_1 * np.exp(a_calc + b_calc * (1-o_0))    # Em_1: adjustment applied to modelled  projection
    mu_1_f = Em_1 * np.exp(a_sim  + b_sim  * (1-o_0))
    mu_1_B = Em_1 * np.exp(a_MAP  + b_MAP  * (1-o_0))
    print(f'- model: a = {s_x(int_mu_p)} - {s_x(int_mu_m)} = {s_x(a_calc)} / \
b = {s_x(slo_mu_p, n_dec=n_dec+1)} - {s_x(slo_mu_m, n_dec=n_dec+1)} = {s_x(b_calc, n_dec=n_dec+1)} / \
mu_1 = {s_x(mu_1_m)}')
    print(f'- freq : a = {s_x(int_Y_t )} - {s_x(int_mu_t)} = {s_x(a_sim )} / \
b = {s_x(slo_Y_t, n_dec=n_dec+1 )} - {s_x(slo_mu_t, n_dec=n_dec+1)} = {s_x(b_sim , n_dec=n_dec+1)} / \
mu_1 = {s_x(mu_1_f)}')
    print(f'- Bayes: a =                 {s_x(a_MAP)} / \
b =                   {s_x(b_MAP, n_dec=n_dec+1)} / \
mu_1 = {s_x(mu_1_B)}')

    print('\\end{verbatim}')
    ts_stop()                                                    # stop transscript to file

    print()
    
# *********************************

def print_all_param(param_labels, process_param, model_param, T, cal_True, Phi_cal, fn_ID='dummy_print'):
    def print_line(s0, s1, s_list, n_dec=None, is_bool=False):
        s_bool = ['F', 'T']
        s = s0 + ' & ' +s1
        for k, s_k in enumerate(s_list):
            if is_bool:
                s_k = s_bool[s_k]
            else: 
                if not (n_dec==None):
                    if s_k==0:
                        s_k = ''
                    else:    
                        if abs(s_k) >= 1000:
                            f_code = 'e'
                        else:
                            f_code='f'
                        s_k = f'{s_k:.{n_dec}{f_code}}'
            s += ' & ' + s_k
        print(s + ' \\\\')  

    filename = print_file_name(fn_ID)
    Phi_cal = np.array(Phi_cal)
    is_true = np.array(cal_True)
    print()
    print(f'Model parameters for LaTeX table: -> {filename}')
    print()
    ts_start(filename, wa='w')                       # start transscript to file (erase prior content)
    print('\\begin{tabular}{lcSSSSSSSSS}')
    print('\\toprule')
    print_line('', 'Period', param_labels, n_dec=None)
    print('\\bottomrule')
    print('\\toprule')
    print_line('Generative process', f'o=-{T}', process_param[0], n_dec=1)
    print_line('', 'o=-1', process_param[1], n_dec=1)
    print('\\midrule')
    print_line('Generative model', f'o=-{T}', model_param[0], n_dec=1)
    print_line('', 'o=-1', model_param[1], n_dec=1)
    print('\\midrule')
    print_line('Calibration item', '', cal_True, n_dec=0, is_bool=True)
    print('\\midrule')
    print_line('Priors', '$\phi_{\\ell,a}$', Phi_cal[0]*is_true, n_dec=2)
    print_line('', '$\phi_{\\ell,b}$', Phi_cal[1]*is_true, n_dec=2)
    print_line('', '$\phi_{\\ell,c}$', Phi_cal[2]*is_true, n_dec=2)
    print('\\bottomrule')
    print('\\end{tabular}')
    ts_stop()                               # stop transscript to file
    print()
    
def plot_claims_count(M_sim, N_run, stats_distr, labels, fn_ID='dummy_plot'):
    N_plot = 100
    sub_pl = (2, 2)                                     # switch between (2,2) and (2,3) sublots
    plot_prior = not True

    if sub_pl[1] == 2:
        w_ratios = [1, 3]
    else:
        w_ratios = None

    c0, c1 = 'k', 'gray'
    lw0, lw1, lw2, lw3 = 0.25, 0.5, 1.0, 2.0
    ls0, ls1, ls2 = '-', '--', ':'
    # mk0, mk1, mk2, mk3 = '.', 0, 'd', 'D'
    # ms0, ms1, ms2 = 4, 12, 16
    
    #          c     -    mu_-T, mu_0,   N, mu_1,
    #          a     b    mu_-T, mu_0,   N, mu_1,
    c_list  = [c0,   '',     c0,   c0,  c0,   c0,
               c0,   c0,     c0,   c0,  c0,   c0,]
    lw_list = [lw2,   0,    lw2,  lw2, lw2,  lw3,
               lw2, lw2,    lw2,  lw2, lw2,  lw3,]
    ls_list = [ls0,  '',    ls2,  ls1, ls0,  ls0,
               ls0, ls1,    ls2,  ls1, ls0,  ls0,]

    def get_xy(i, j):
        def append_E_V(k, xy):
            (rvs_, E_, V_, L_, c_, lw_, ls_) = xy
            (rvs_k, E_k, V_k) = moments_list[k]
            
            rvs_.append(rvs_k)
            E_.append(E_k)
            V_.append(V_k)
            L_.append(labels[k])
            c_.append(c_list[k])
            lw_.append(lw_list[k])
            ls_.append(ls_list[k])
            return (rvs_, E_, V_, L_, c_, lw_, ls_)
        k = 6*i
        xy =([], [], [], [], [], [], [])                 # rvs_, E_, V_, L_, c_, lw_, ls_
        if j==0: 
            xy = append_E_V(k, xy)                       # p(c) or p(a)
            if i==1: xy = append_E_V(k+1, xy)            # p(b)
        if j==1: 
            xy = append_E_V(k+2, xy)                     # p(mu_-T)
            xy = append_E_V(k+3, xy)                     # p(mu_-1)
            if sub_pl[1] == 2:                       # case sub_pl 2, 2
                xy = append_E_V(k+4, xy)                 # p(N)
                xy = append_E_V(k+5, xy)                 # p(mu_1)
        if j==2:                                     # case sub_pl 2, 3
            xy = append_E_V(k+4, xy)                     # p(N)
            xy = append_E_V(k+5, xy)                     # p(mu_1)
        return xy     

    def min_max_EV(E, V, x_min, x_max, n_sigma=3):
        dx = (V**0.5) * n_sigma
        lo, hi = E-dx, E+dx 
        if x_min is None: 
            return lo, hi
        else: 
            return min(x_min, lo), max(x_max, hi)

    def get_min_max(j):

        def min_max_k(k_list):
            x_min, x_max = None, None
            for k in k_list:
                (rvs, E, V) = moments_list[k]
                x_min, x_max = min_max_EV(E, V, x_min, x_max)
                x_min = min(x_min, rvs.min())
                x_max = max(x_max, rvs.max())
            return x_min, x_max        
        
        list_0 = [0,     6,  7]                                # c, a, b
        list_1 = [2, 3,  8,  9]                                # mu_-T(c), mu_-1(c), mu_-T(a,b), mu_-1(a,b)
        list_2 = [4, 5, 10, 11]                                # N(c),     mu_1(c),  N(a,b),     mu_1(a,b)
        if j == 0: 
            x_min, x_max = min_max_k(list_0)  
            if plot_prior:
                x_min, x_max = min_max_EV(0, 1, x_min, x_max)  # N(0,1) prior for a, b, and c 
            return x_min, x_max
        if sub_pl[1] == 2:                       # case sub_pl 2, 2
            if j == 1: return min_max_k(list_1+list_2)    
        else:                                    # case sub_pl 2, 3
            if j == 1: return min_max_k(list_1)            
            if j == 2: return min_max_k(list_2)            
            
    def plot_Norm(ax, rvs_k, E, V, L, y_max, c='k', ls='-', lw=0.5):
        D = scipy_Norm()
        D.init_E_V(E, V)
        x_lo, x_hi = min_max_EV(E, V, None, None)
        x = np.linspace(x_lo, x_hi, N_plot)
        p = D.cdf(x)
        y_max = max(y_max, p.max())
        # ax.plot(x, p, c, lw=lw, ls=ls)
     # empirical cdf and empirical mean  
        x_k, y_k = get_rvs_cdf(rvs_k)
        ax.plot(x_k, y_k, c, lw=lw, ls=ls, label=L)
        x_ = x_k.mean()
        ax.plot([x_, x_], [0, 1], c, lw=lw0, ls=ls)
        return y_max
 
# Structure of stats_distr:
#     stats_distr.append([c, 0, mu_0_c,  mu_T_c,  mu_c,  s_c, 
#                         a, b, mu_0_ab, mu_T_ab, mu_ab, s_ab])
# stats_distr = np.array(stats_distr).transpose()

    moments_list = []
    for i, rvs_i in enumerate(stats_distr):
        E = rvs_i.mean()
        V = rvs_i.var(ddof=1)
        moments_list.append([rvs_i, E, V])
        
    fig, axs = plt.subplots(sub_pl[0], sub_pl[1], 
                            gridspec_kw={'width_ratios': w_ratios}, 
                            figsize=(10,4))

    for j in range(sub_pl[1]):
        x_min, x_max = get_min_max(j)
        for i in range(sub_pl[0]):
            ax = axs[i][j]
            if i==0: ax.get_xaxis().set_visible(False)
            ax.plot([min(-0.05, x_min), x_max], [0, 0],        c1, lw=lw1)
            ax.plot([min(    0, x_min), x_max], [1, 1],        c1, lw=lw1)
            ax.plot([0, 0],                     [-0.05, 0.05], c1, lw=lw2)
            
            (rvs_, E_, V_, L_, c_, lw_, ls_) = get_xy(i, j)
            y_max = 0
            if j==0:
                if plot_prior:
                    y_max = plot_Norm(ax, None, 0, 1, '$\pi()$', y_max)         # prior distributions foa a, b, and c 
            for k in range(len(E_)):
                y_max = plot_Norm(ax, rvs_[k], E_[k], V_[k], L_[k], y_max, c=c_[k], ls=ls_[k], lw=lw_[k])
            ax.get_yaxis().set_visible(False)
            ax.legend(loc='best', frameon=False)

    file_name, ext_list = save_plot(plt, fn_ID)
    plt.show()
    
# *********************************

def plot_R_c(param_lists, Y_t, E, V, Phi_c, c0=0, dc_lo=0.015, N_plot = 200, fn_ID='dummy_plot'):
    # Plot the functions R(c¦Y_t, E, V) for P, B, NB, and G distributions
    # - Y_t   : array with observations drawn from generic process
    # - E, V  : arrays with means and variances of generic model 
    # - Phi_c : variance of prior pi(c)
    # - dc_lo : minimum distance of c_lo and c0_lo from singularity of R(c)
    eps = 0.001
    
    def get_c(c_lo, c_hi):
        return np.linspace(c_lo+eps, c_hi-eps, N_plot)
    
    def get_L_c(c):
        return c * np.exp(-c)
    def plt_ax12(ax1, ax2, x1, y1, x2, y2, c='k', ls=None, lw=1.0, labels=('','')):
        ax1.plot(x1, y1, c=c, ls=ls, lw=lw, label=labels[0])           
        ax2.plot(x2, y2, c=c, ls=ls, lw=lw, label=labels[1])           
        
    def L_c_plt(ax1, ax2, c_lo, c_hi, c0_lo, c0_hi, c='k', ls=None, lw=1, labels=('','')): 
        c1 = get_c(c_lo, c_hi)
        c2 = get_c(c0_lo, c0_hi)
        L1 = get_L_c(c1)
        L2 =get_L_c(c2)
        plt_ax12(ax1, ax2, c1, L1, c2, L2, c=c, ls=ls, lw=lw, labels=labels)
        
    def R_P_plt(ax1, ax2, c_lo, c_hi, c0_lo, c0_hi, E, f_P, c='k', ls=None, lw=1, labels=('','')):
        c1 = get_c(c_lo, c_hi)
        c2 = get_c(c0_lo, c0_hi)
        R1 = R_c_Poisson(c1, Y_t, E, f_P, Phi_c)
        R2 = R_c_Poisson(c2, Y_t, E, f_P, Phi_c)
        plt_ax12(ax1, ax2, c1, R1, c2, R2, c=c, ls=ls, lw=lw, labels=labels)
    
    def R_nB_plt(ax1, ax2, c_lo, c_hi, c0_lo, c0_hi, E, f_P, c='k', ls=None, lw=1, labels=('','')):
        c_min = get_c_lo_bound(E / (f_P - 1)) + dc_lo
        c_lo  = max(c_lo, c_min)
        c0_lo = max(c0_lo, c_min)
        c1 = get_c(c_lo, c_hi)
        c2 = get_c(c0_lo, c0_hi)
        R1 = R_c_negBin(c1, Y_t, E, f_P, Phi_c)
        R2 = R_c_negBin(c2, Y_t, E, f_P, Phi_c)
        plt_ax12(ax1, ax2, c1, R1, c2, R2, c=c, ls=ls, lw=lw, labels=labels)
    
    def R_B_plt(ax1, ax2, c_lo, c_hi, c0_lo, c0_hi, E, f_P, c='k', ls=None, lw=1, labels=('','')):
        c_min = get_c_lo_bound(E / (1-f_P) / Y_t) + dc_lo
        c_lo  = max(c_lo, c_min)
        c0_lo = max(c0_lo, c_min)
        c1 = get_c(c_lo, c_hi)
        c2 = get_c(c0_lo, c0_hi)
        R1 = R_c_Bin(c1, Y_t, E, f_P, Phi_c)
        R2 = R_c_Bin(c2, Y_t, E, f_P, Phi_c)
        plt_ax12(ax1, ax2, c1, R1, c2, R2, c=c, ls=ls, lw=lw, labels=labels)
    
    def R_G_plt(ax1, ax2, c_lo, c_hi, c0_lo, c0_hi, E, f_P, c='k', ls=None, lw=1, labels=('','')):
        c_min = get_c_lo_bound(E / f_P) + dc_lo
        c_lo  = max(c_lo, c_min)
        c0_lo = max(c0_lo, c_min)
        c1 = get_c(c_lo, c_hi)
        c2 = get_c(c0_lo, c0_hi)
        R1 = R_c_Gamma(c1, Y_t, E, f_P, Phi_c)
        R2 = R_c_Gamma(c2, Y_t, E, f_P, Phi_c)
        plt_ax12(ax1, ax2, c1, R1, c2, R2, c=c, ls=ls, lw=lw, labels=labels)
        
    def get_L(L_ist, i, j):
        return (L_ist[0][i][j], L_ist[1][i][j])

    # Unpack model and plot parameters defined in: GIRF_models.py / param_R_c_models()
    (f_P_lists, c_ranges, (col_list, colors), d_labels, LR_labels) = param_lists
    
    [P_range, B_range, nB_range, G_range, c0_range] = c_ranges
    (P_lo, P_hi)     = P_range
    (B_lo, B_hi)     = B_range
    (nB_lo, nB_hi)   = nB_range
    (G_lo, G_hi)     = G_range
    (dc0_lo, dc0_hi) = c0_range
    
    [col0, col1, col2] = col_list
    
    [L_, P_, B_, nB_, G_] = d_labels
    [L_L, L_P, L_B, L_nB, L_G] = LR_labels

    c, c0 = logN_fit(Y_t, E, V, Phi_c)
    c0_lo = c0 + dc0_lo 
    c0_hi = c0 + dc0_hi
    
    fig, axs = plt.subplots(2, 3)
    
    for i, L in enumerate(f_P_lists): 
        ax1 = axs[0,i]
        ax2 = axs[1,i]
        for j, f_P in enumerate(L):
            
            c_P_l,  c_P_h  =  P_lo[i][j],  P_hi[i][j]
            c_B_l,  c_B_h  =  B_lo[i][j],  B_hi[i][j]
            c_nB_l, c_nB_h = nB_lo[i][j], nB_hi[i][j]
            c_G_l,  c_G_h  =  G_lo[i][j],  G_hi[i][j]
            
            col = colors[i][j]

            L_c_plt(ax1, ax2, -4, 5, c0_lo, c0_hi, 
                    c=col0, ls='-', labels=get_L(L_L, i, j))
            
            R_G_plt(ax1, ax2, c_G_l, c_G_h, c0_lo, c0_hi, E, f_P, 
                    c=col0, ls=':', lw=0.5, labels=get_L(L_G, i, j))

            if i==0:
                R_B_plt(ax1, ax2, c_B_l, c_B_h, c0_lo, c0_hi, E, f_P, 
                        c=col, ls='-.', lw=1.5, labels=get_L(L_B, i, j))
            if i==1:
                R_B_plt(ax1, ax2, c_B_l, c_B_h, c0_lo, c0_hi, E, f_P-0.2, 
                        c=col2, ls='-.', lw=1.5, labels=get_L(L_B, i, j))

                R_P_plt(ax1, ax2, c_P_l, c_P_h, c0_lo, c0_hi, E, f_P, 
                        c=col, ls='-', labels=get_L(L_P, i, j))

                R_nB_plt(ax1, ax2, c_nB_l, c_nB_h, c0_lo, c0_hi, E, f_P+0.2, 
                         c=col2, ls='--', lw=1.5, labels=get_L(L_nB, i, j))
            if i==2:
                R_nB_plt(ax1, ax2, c_nB_l, c_nB_h, c0_lo, c0_hi, E, f_P, 
                         c=col, ls='--', lw=1.5, labels=get_L(L_nB, i, j))
    L_0 = get_L_c(c0)             
    for i in range(2):        
        axs[i][0].sharey(axs[i][1])
        axs[i][1].sharey(axs[i][2])
        axs[i][1].get_yaxis().set_visible(False)
        axs[i][2].get_yaxis().set_visible(False)
        for j in range(3):
            axs[i][j].scatter(c0, L_0, c=col0, marker='D', s=20)        
            axs[i][j].legend(fontsize=6, loc='upper right')

    x0_lo, x0_hi =  axs[0][0].get_xlim()
    y0_lo, y0_hi =  axs[0][0].get_ylim()
    x1_lo, x1_hi =  axs[1][0].get_xlim()
    y1_lo, y1_hi =  axs[1][0].get_ylim()
    for j in range(3):
        axs[0][j].plot([x1_lo, x1_hi],[y0_lo, y0_lo], 'k', ls='-', lw=2)
        axs[0][j].plot([x0_lo, x0_lo],[y1_lo, y1_hi], 'k', ls='-', lw=2)

    
    file_name, ext_list = save_plot(plt, fn_ID)
    plt.show()
    
# *********************************

def row_col(k, n):
    r = k // n
    c = k % n 
    return r, c

def plot_evol_all(T, N_param, iteration, Y0, proc_data, mod_data, log_True, q_min_max=[[],[]],
                  N_points=100, fn_ID='dummy_plot'):

    distr_list  = [D_Panj, D_Panj, D_logN, D_logN, D_logN, D_Norm, D_logN, D_logN, D_logN]
    Is_discrete = [  True,   True,  False,  False,  False,  False,  False,  False,  False]

    lw1, lw2, lw3 = 0.1, 0.2, 0.4
    ms1 = 4
    
    def get_pmf_pdf(E_Y, V_Y, E_Z, V_Z, k, t):
        
        if t == 0 and Is_discrete[k]:
            param_type = D_Gamm
        else:
            param_type = distr_list[k]
        
        if param_type in [D_Panj, D_Bino, D_negB, D_Pois]:
            distr_Y = scipy_Panjer()
            distr_Y.init_lda_f(E_Y, V_Y/E_Y)

            distr_Z = scipy_logNorm()            
            distr_Z.init_E_V(E_Z, V_Z)
        else:
            if param_type in [D_Gamm]:
                distr_Y = scipy_Gamma()
                distr_Z = scipy_Gamma()
            elif param_type in [D_logN]:
                distr_Y = scipy_logNorm()
                distr_Z = scipy_logNorm()
            else:
                distr_Y = scipy_Norm()
                distr_Z = scipy_Norm()

            distr_Y.init_E_V(E_Y, V_Y)
            distr_Z.init_E_V(E_Z, V_Z)
            
        return distr_Y, distr_Z
    
    def plot_Y_Z(ax, x_t, Y0, pdmf_Y0, q_k, pdmf_Y, pdmf_Z, pdmf_max, Is_discr=False, c_Y='k', c_Z='k', lw_Y=lw1, lw_Z=lw2):
        p_Y  = pdmf_Y  / pdmf_max
        p_Z  = pdmf_Z  / pdmf_max
        p_Y0 = pdmf_Y0 / pdmf_max
        
        if Is_discr:
            for i, q_i in enumerate(q_k):
                ax.plot([x_t, x_t + p_Y[i]], [q_i, q_i], c_Y, lw=lw_Y)   
        else:
            ax.plot(x_t + p_Y, q_k, c_Y, lw=lw_Y)
            
        ax.plot(x_t + p_Z, q_k, c_Z, lw=lw_Z)
        ax.scatter(x_t + p_Y0, Y0, c='k', marker='d', s=ms1)
        
    
    fig, axs = plt.subplots(3, 3, sharex = True)    
    
    q_min = q_min_max[0] 
    q_max = q_min_max[1] 
    
    min_max_ok = (len(q_min) == N_param) and (len(q_min) == N_param)
    if not min_max_ok:
        q_min = []
        q_max = []
    
    for k in range(N_param):
        i, j = row_col(k, 3)
        ax = axs[i][j]
    
        Y0_k = Y0[:, k:k+1]
        P_k = proc_data[:, :, k:k+1]
        M_k = mod_data[:, :, k:k+1]
        
        if min_max_ok:
            q_lo = q_min[k]
            q_hi = q_max[k]
        else:
            Y0_min = Y0_k.min()
            Y0_max = Y0_k.max()
            q_lo = min(P_k.min(), M_k.min())
            q_hi = max(P_k.max(), M_k.max())
            if log_True[k]: 
                q_lo = 0
                q_hi = min(q_hi, 2*Y0_max)
            else:
                d_Y0 = Y0_max - Y0_min
                q_lo = max(q_lo, Y0_min-d_Y0/2)
                q_hi = min(q_hi, Y0_max+d_Y0/2)
            if Is_discrete[k]:
                q_hi = max(q_hi, 5)
            q_min.append(q_lo)
            q_max.append(q_hi)
        
        q_k_0 = np.linspace(q_lo, q_hi, N_points)
        if Is_discrete[k]:
            q_k_t = np.arange(int(np.ceil(q_hi)))
        else:
            q_k_t = q_k_0
        
        Y0_list     = []
        p_Y0_list   = []
        pdmf_Y_list = []
        pdmf_Z_list = []
        pdmf_max  = 0.
        for t in range(T+1):
            Y0_k_t = Y0_k[t][0]
            # moments of gen Proces and gen Model
            P_k_t = P_k[:, t:t+1].transpose()[0,0]
            E_Y = P_k_t.mean()
            V_Y = P_k_t.var(ddof=1)
            M_k_t = M_k[:, t:t+1].transpose()[0,0]
            E_Z = M_k_t.mean()
            V_Z = M_k_t.var(ddof=1)

            dist_Y, dist_Z = get_pmf_pdf(E_Y, V_Y, E_Z, V_Z, k, t)
            
            if t == 0:
                q_k = q_k_0
            else:
                q_k = q_k_t
                
            pdmf_Y  = dist_Y.pdf(q_k)
            pdmf_Z  = dist_Z.pdf(q_k)
            pdmf_Y0 = dist_Z.pdf(Y0_k_t)      # get likelihood from Calibrated distribution !
            
            if t == 0:
                Y0_k_0   = Y0_k_t
                pdmf_Y0_0 = pdmf_Y0
                pdmf_Y_0  = pdmf_Y.copy()
                pdmf_Z_0  = pdmf_Z.copy()
                pdmf_max  = max(pdmf_max, pdmf_Y_0.max()/2, pdmf_Z.max()/2)
            else:
                Y0_list.append(Y0_k_t)
                p_Y0_list.append(pdmf_Y0)
                pdmf_Y_list.append(pdmf_Y.copy())
                pdmf_Z_list.append(pdmf_Z.copy())
                pdmf_max = max(pdmf_max, pdmf_Y.max(), pdmf_Z.max())

        ax.plot([-T, 2], [0, 0], 'gray', lw=lw1)
        
        ax.plot([0,0], [q_lo, q_hi], 'gray', lw=lw1)

        x_t = 0
        n = -20   # last 20 points of overall distribution are not plotted 
        plot_Y_Z(ax, x_t, Y0_k_0, pdmf_Y0_0, q_k_0[:n], pdmf_Y_0[:n], pdmf_Z_0[:n], pdmf_max, lw_Y=lw1, lw_Z=lw3)
        for t in range(T):
            x_t   = t - T
            
            Y0_k_t = Y0_list[t]
            pdmf_Y0 = p_Y0_list[t]
            pdmf_Y  = pdmf_Y_list[t]
            pdmf_Z  = pdmf_Z_list[t]
            
            ax.plot([x_t ,x_t ], [q_lo, q_hi], 'gray', lw=lw1)
            
            # ax.plot(x_t  + pdmf_Y / pdmf_max, q_k, 'k', lw=lw1)
            # ax.plot(x_t  + pdmf_Z / pdmf_max, q_k, 'k', lw=lw2)
            # ax.scatter(x_t  + pdmf_Y0 / pdmf_max, Y0_k_t, c='k', marker='d', s=ms1) 
            
            plot_Y_Z(ax, x_t, Y0_k_t, pdmf_Y0, q_k_t, pdmf_Y, pdmf_Z, pdmf_max, Is_discr=Is_discrete[k], lw_Y=lw1, lw_Z=lw2)
            
        ax.text(1, q_hi, Red_Fields[k][5], ha='center', va='top')     

    
    fig.suptitle(f'Generative process and model calibration: iteration {iteration}')
    save_plot(plt, fn_ID, fn_append=f'{iteration}')
    # plt.savefig(f'evolution_all_ab_it_{iteration}.png', bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'evolution_all_ab_it_{iteration}.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()    
    
    return [q_min, q_max]

# if __name__ == "__main__":

