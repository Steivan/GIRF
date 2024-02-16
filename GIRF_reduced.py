# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']  = 600
plt.rcParams['savefig.dpi'] = 600

from GIRF_models import get_patterns
from GIRF_plot import save_plot

# ******************************************************

def get_pattern(t, Y, t_max=None):
    # Convert the series Y_i = Y(t_i+) into a step function 
    # Pts = [[0,0], [t_0,0], [t_0,Y_0], [t_1,Y_0], ... , [t_n,Y_n]] where t_n <= T
    
    def Pts_append(Pts, ti, Yi):
        Pts[0].append(ti)
        Pts[1].append(Yi)
        return Pts

    if t_max == None: t_max = max(t)
    Pts = [[],[]]
    t_i = 0
    Y_i = 0
    Pts = Pts_append(Pts, t_i, Y_i)            # append start point [0,0]
    for i, t_i in enumerate(t):
        if t_i <= t_max:
            Pts = Pts_append(Pts, t_i, Y_i)    # append lower right point of step
            Y_i = Y[i]
            Pts = Pts_append(Pts, t_i, Y_i)    # append upper left point of step
    Pts = Pts_append(Pts, t_max, Y_i)          # append end point [t_max, P(t_max)]
    return np.array(Pts)

def get_count(Pt):
    # Count the number of steps in the pattern Pt
    count = 0 
    Pt_ = Pt[0]
    for i, Pt_i in enumerate(Pt):
        if Pt_i != Pt_:
            count += 1 
            Pt_ = Pt_i 
    return count        

def get_IP_tau(Pt, t):
    # Evaluate the incurred amaount, the incurred lag, the paid ratio and the paid lag at time t,  
    # given the pattern Pt
    
    t_0 = Pt[0]                      # list time stamps t_i
    I_0 = Pt[1]                      # list icurred amounts I_i = I(t_i+)
    P_0 = Pt[2]                      # list cumulative paid amounts P_i = P(t_i+)

    t_lo  = t_0 <= t                 # extract conditional pattern for the period <=T
    # I_zer = I_0 <= 0.                # identify incurred values <= 0
    t_max = max(t_0)                 # get last data point (t_max, I_max, P_max) in P
    I_max = I_0[len(I_0)-1]
    P_max = P_0[len(P_0)-1]
    
    t_1    = np.roll(t_0, 1)         # get list D_t with length of time intervals
    t_1[0] = 0
    D_t    = (t_0 - t_1) * t_lo
    
    I_1    = np.roll(I_0, 1)         # get values of I and P for the time intervals
    I_1[0] = 0
    P_1    = np.roll(P_0, 1)
    P_1[0] = 0
    
    if t >= t_max:                   # get values at time T
        I1 = I_max 
        P1 = P_max 
    else:    
        i0 = min(t_lo.sum(), len(I_1)-1)
        I1 = I_1[i0]
        P1 = P_1[i0]
        
    I_int = (D_t * I_1).sum() + (t-D_t.sum()) * I1   # Integrate I(t) and P(s)from 0 to t
    P_int = (D_t * P_1).sum() + (t-D_t.sum()) * P1
    
    if I1 > 0:                                       # Normalize  
        P_rel   = P1 / I1                            # Paid/Incurred ratio  
        tau_I_t = t - I_int / I1                     # Incurred lag
        tau_P_t = t - P_int / I1                     # Paid lag
    else:
        P_rel = 0
        tau_I_t = 0
        tau_P_t = 0
        
    return  I1, tau_I_t, P_rel, tau_P_t

def get_IP_t(t_IP_tau, Pt, t):
    # Evaluate the statistics at time t and append data to t_IP_tau
    
    I1, tau_I_t, P_rel, tau_P_t = get_IP_tau(Pt, t)  
    
    t_IP_tau[0].append(t)           # time
    t_IP_tau[1].append(I1)          # incurred
    t_IP_tau[2].append(tau_I_t)     # incurred lag
    t_IP_tau[3].append(P_rel)       # Paid / incurred ratio
    t_IP_tau[4].append(tau_P_t)     # paid lag
    
    return t_IP_tau

def get_polygon(X, Y, Y_max):
    x = [0] + list(X)                    # insert origin
    y = [0] + list(Y)
    x_last = x[-1]
    y_last = y[-1]
    if y_last < Y_max:                   # case I* > P*: elevate polygon up to I* = Y_max
        x = x + [x_last,    0]
        y = y + [Y_max, Y_max]
    else:
        if y[-2] == y_last:              # case y[-2] == y[-1] == y_last
            x[-1] = 0
        else:
            x = x + [0]                  # draw upper bound back to y-axis
            y = y + [y_last]
    x = x + [0]                          # close the loop
    y = y + [0]
    return [x, y]        
        
def plot_IP(tau_grid, I, P, tau_eval, col_list, fn_ID='dummy_plot'):    
# Create a chart depicting the reduced variables for:
#  a) incurred claims    
#  b) paid claims  
 
    # eps = 0.001
    ax_titles = ['a) incurred', 'b) paid']

    def m_str(s1, s_lo, s_hi, s2=''):
        b, k = '{', '}'
        def bsk(c, s): 
            if s.strip() == '':
                return ''
            else:
                return f'{c}{b}{s}{k}'
        
        if s1[0] == ' ':
            s_m = f"${s1[1:]}"
        else:
            s_m = f"$\{s1}"
        s_lh = f"{bsk('_', s_lo)}{bsk('^', s_hi)}{s2}$"    
        return s_m + s_lh
    
    def plt_x_tick(axs, x, c='k', ls='-', lw=0.5, d_x=0.05):
        axs.plot([x, x], [-d_x, d_x], c=c, ls=ls, lw=lw)

    def plt_y_tick(axs, y, c='k', ls='-', lw=0.5, d_y=0.05):
        axs.plot([-d_y, d_y], [y, y], c=c, ls=ls, lw=lw)

    def plt_xy(axs, x, y, c='k', ls='-', lw=0.5, d_x=0.025, d_y=0.05):
        axs.plot([x, x], [-d_y, y+d_y], c=c, ls=ls, lw=lw)
        axs.plot([-d_x, x+d_x], [y, y], c=c, ls=ls, lw=lw)
    
    def plt_arrow(axs, x2, y2, x1, y1, arr_st='<->', c='k', ls='-', w=0.5):
        axs.annotate('', xy=(x1, y1), xytext=(x2, y2), 
                     arrowprops=dict(arrowstyle=arr_st, color=c, ls=ls, lw=w))
    
    def disp_arrow(axs, label, x2, x1, y, arr_st='<->', c='k', ls='-', w=0.5):  
        plt_arrow(axs, x2, y, x1, y, arr_st=arr_st, c=c, ls=ls, w=w)
        axs.text((x1+x2)/2, y+0.05, label, fontsize=fs, color=c, va='bottom', ha='center')
    
    s_0 = '$0$'
    s_t = '$time$'
    kappa = '\\kappa'
    
    s_t_a = m_str(' t', kappa, 'occ')
    s_t_r = m_str(' t', kappa, 'rep')
    s_t_s = m_str(' t', ''  , 'sub')
    s_t_c = m_str(' t', kappa, 'clo')
    s_arsc = [s_t_a, s_t_r, s_t_s, s_t_c, ]

    s_tau_r = m_str('tau', kappa, 'rep')
    s_tau_I = m_str('tau', kappa, 'I'  )
    s_tau_P = m_str('tau', kappa, 'P'  )
    
    s_I_t = m_str(' I', kappa, '', '(t)')
    s_P_t = m_str(' P', kappa, '', '(t)')
    
    s_U = m_str(' U', kappa, '')
    s_I = m_str(' I', kappa, '*')
    s_P = m_str(' P', kappa, '*')

    s_N_I = m_str(' N', kappa, 'I'   )
    s_N_P = m_str(' N', kappa, 'P'   )
     
    # time lags: accident date, reported date, submission date, closure date
    tau_arsc = [0.0, 0.0 , 0.0, 1.0]      # placeholder
    # Paid amount, Incurred amount, Ultimate amount
    PIU = [0.0 , 0.0 , 1.0]               # placeholder

    IP_pol = [[], []]
    
    # Long-tail pattern 
    # tau_grid: list with lags tau_i (>=0) for which I_i and P_i are given 
    tau_r, tau_c = min(tau_grid), max(tau_grid)
    t_r = tau_r / tau_c                                    # t_rep relative to t_closure
    t_s = tau_eval / tau_c                                 # t_sub relative to t_closure
    t_IP = np.array([tau_grid, I, P])                      # array with time_grid, incurred and paid
    I_max = max(I)
    I1, tau_I_T, P_rel, tau_P_T = get_IP_tau(t_IP, tau_eval)  
    Pt_I = get_pattern(tau_grid, I, tau_eval)
    Pt_P = get_pattern(tau_grid, P, tau_eval)
    t_T = Pt_I[0] / tau_c
    I_T = Pt_I[1] / I_max
    P_T = Pt_P[1] / I_max
    N_I = get_count(I_T)
    N_P = get_count(P_T)
    IP_pol[0].append(get_polygon(t_T, I_T, I_T[-1]))       # Incurred polygon (integrated area)
    IP_pol[1].append(get_polygon(t_T, P_T, I_T[-1]))       # Paid polygon (integrated area)

    tau_arsc[1] = t_r
    tau_arsc[2] = t_s
    PIU[0] = P_T[-1]
    PIU[1] = I_T[-1]

    fs = 6
    fig, ax = plt.subplots(1,2) 
    ax[0].set_box_aspect(0.5)
    ax[1].set_box_aspect(0.5)
    
    c0 = col_list[0][2]
    c1 = col_list[0][0]
    c2 = col_list[0][3]
    c3 = 'gray'
    c4 = 'lightgray'
    j = 0
    for i in range(2):
        axs = ax[i]
        axs.set_title(ax_titles[i],fontsize=8)
        axs.set_axis_off()   
        
        axs.fill(IP_pol[i][j][0], IP_pol[i][j][1], color=c4)
        
        plt_arrow(axs, -0.05, 0, 1.05, 0, arr_st='->', c=c0, ls='-', w=0.5)
        axs.text(-0.075, 0.0, s_0, fontsize=fs, color=c0, va='center', ha='right')
        axs.text( 1.075, 0.0, s_t, fontsize=fs, color=c0, va='center', ha='left')
        plt_x_tick(axs, tau_arsc[1])
        
        plt_arrow(axs, 0, -0.075, 0, 1.1, arr_st='->', c=c0, ls='-', w=0.5)
        axs.text(-0.075, PIU[1], s_I, fontsize=fs, color=c2, va='center', ha='right')
        plt_xy(axs, tau_arsc[2], PIU[1], c=c2, ls='-', lw=0.5)

        axs.text(-0.025, PIU[2], s_U, fontsize=fs, color=c3, va='center', ha='right')
        plt_xy(axs, tau_arsc[3], PIU[2], c=c3, ls='--', lw=0.5)
        for k, x_k in enumerate(tau_arsc):
            if k==3:
                c_k = c3
            else:
                c_k = c0
            axs.text(x_k, -0.075, s_arsc[k], fontsize=fs, color=c_k, va='top', ha='center')
            if i==1:
                axs.text(-0.075, PIU[0], s_P, fontsize=fs, color=c2, va='center', ha='right')

    # Long-tail case 
    x1 = tau_arsc[1]
    x2 = tau_arsc[2]
    x3 = tau_arsc[3]
    y0 = PIU[0]
    y1 = PIU[1]
    y2 = PIU[2]

    # Long-tail incurred
    axs = ax[0] 
    axs.plot(t_T, I_T, color=c1)
    axs.plot([x2, x3], [y1, y2], color=c3, ls='--')
    x = tau_I_T / tau_c 
    y = y1*0.6
    axs.plot([x,x], (0, y1), color=c2, ls='--', lw=0.5)
    axs.text(x+0.05, y, s_I_t, fontsize=fs, color=c1, va='center', ha='left')   
    y = y1*3/4
    disp_arrow(axs, s_tau_I, 0, x, y, arr_st='<->', c=c2, ls='--', w=0.5)
    axs.text(x+0.05, 0.05, s_N_I+f'$={N_I}$', fontsize=fs, color=c2, va='bottom', ha='left') 
    
    # Long-tail paid
    axs = ax[1] 
    axs.plot(t_T, P_T, color=c1)
    axs.plot([x2, x3], [y0, y2], color=c3, ls='--')
    axs.plot([-0.05, x2], [y0, y0], color=c0, ls=':', lw=0.5)
    x = tau_P_T / tau_c
    y = y1/2
    axs.plot([x,x], (0, y1), color=c2, ls='--', lw=0.5)
    axs.text(x+0.02, y-0.02, s_P_t, fontsize=fs, color=c1, va='center', ha='left')
    y = y1/2
    disp_arrow(axs, s_tau_P, 0, x, y, arr_st='<->', c=c2, ls='--', w=0.5)
    disp_arrow(axs, s_tau_r, 0, x1, 0.1, arr_st='<->', c=c2, ls='--', w=0.5)
    axs.text(x+0.02, 0.05, s_N_P+f'$={N_P}$', fontsize=fs, color=c2, va='bottom', ha='left') 

    file_name, ext_list = save_plot(plt, fn_ID)
    plt.show()    

def plot_t_IP(t, L_I, P, index_flat, col_list, lw_list, ls_list, label_list, fn_ID='dummy_plot'):
# Create a chart depicting the temporal evolution of patterns and lags:
#  a) Input: paid and incurred patterns (5 cases with favorable, unbiased and adverse IBNER):     
#  b) resulting temporal evolution of incurred lags    
#  c) resulting temporal evolution of paid ratios    
#  d) resulting temporal evolution of paid lags   
 
    eps = 0.001
    sub_kappa = '_{\\kappa}'

    ax_titles = [
        ['a) paid and incurred patterns', 'c) paid/incurred ratios', ],
        ['b) incurred lags',              'd) paid lags',            ],
        ]
    x_label = [
        [None, None ],
        ['$t$ [mos.]', '$t$ [mos.]'],
        ]
    y_label = [
        [f'$I{sub_kappa}$, $P{sub_kappa}$ [m CHF]', f'$P{sub_kappa}$ / $I{sub_kappa}$' ],
        [f'$\\tau{sub_kappa}^I$ [mos.]',       f'$\\tau{sub_kappa}^P$ [mos.]' ],
        ]

    def plot_IP(ax, t_IP_tau, c, lw, ls, label):
        t_list = t_IP_tau[0]
        t_max  = max(t_list)
        for k in range(4):
            i = k % 2 
            j = k // 2 
            y_list = t_IP_tau[k+1]
            y_max = max(y_list)
            if k == 0: 
                ax_label = label
            else:
                ax_label = None
                
            axs = ax[i,j]    
            axs.set_title(ax_titles[i][j]) 
            axs.plot([0, t_max], [0, 0], 'k', lw=0.25)    
            axs.plot([0, 0], [0, y_max], 'k', lw=0.25)    
            if k == 2:
                axs.plot([0, t_max], [1, 1], 'k', lw=0.25)        
            axs.plot(t_list, y_list, c, lw=lw, ls=ls, label=ax_label)   
            axs.set_xlabel(x_label[i][j])
            axs.set_ylabel(y_label[i][j])
        
    fig, ax = plt.subplots(2,2, sharex = True, figsize=(10,4))

    #Incurred patterns
    for i, I in enumerate(L_I):
        T_IP_tau = [[], [], [], [], []] 
        t_IP = np.array([t, I, P])
        
        T_IP_tau = get_IP_t(T_IP_tau, t_IP, 0)
        for T in t:
            T_IP_tau = get_IP_t(T_IP_tau, t_IP, T-eps)
            T_IP_tau = get_IP_t(T_IP_tau, t_IP, T+eps)
        T_IP_tau = get_IP_t(T_IP_tau, t_IP, max(t)+30)
        if i == index_flat: T_IP_flat =  T_IP_tau
        plot_IP(ax, T_IP_tau, col_list[0][i], lw_list[0][i], ls_list[0][i], label_list[0][i])

    #Incurred flat
    plot_IP(ax, T_IP_flat, col_list[0][index_flat], lw_list[0][index_flat], ls_list[0][index_flat], None)

    # Paid pattern
    pP = get_pattern(t, P)    
    ax[0,0].plot(pP[0], pP[1], color=col_list[1][0], lw=lw_list[1][0], ls=ls_list[1][0], label=label_list[1][0])

    ax[0,0].legend(loc='best', frameon=not False, fontsize=4)
    
    file_name, ext_list = save_plot(plt, fn_ID)
    plt.show()    

# ************************************************************************

if __name__ == "__main__":
    
    gray_scale = True
    time_grid, Inc_list, Paid, index_flat, label_list, color_list, lw_list, ls_list = get_patterns(gray_scale = gray_scale)
    
    plot_t_IP(time_grid, Inc_list, Paid, index_flat, color_list, lw_list, ls_list, label_list, fn_ID='patt_lags')
    
    t_eval = 75                                                   # lag [month] for the evaluation of reduced variables   
    i_eval = 3                                                    # index of incurred pattern to be evaluated                                        
    plot_IP(time_grid, Inc_list[i_eval], Paid, t_eval, color_list, fn_ID='red_var')

