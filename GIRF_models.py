# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""

import numpy as np
import datetime as dt

# *********************************

# current date
today = dt.datetime.now().date()

# submission date
year_sub = today.year -1 * (today.month < 9)            # submission year = previous year if month < 9
date_sub = dt.date(year_sub, 7, 1)                      # set submission date to mid-year
sub_year = date_sub.year
days_per_year = 365.25

Units_I = 1.e6

# List with names of probability distributions
D_Bino = 'Bin'
D_Pois = 'Poiss'
D_negB = 'negB'
D_Panj = 'Panj'
D_Gamm = 'Ga'
D_Norm = 'N'
D_logN = 'logN'

# Dictionnary containing the names of the reduced variables and respective modelling parameters
#                 Field Name,    Is_Int, Units,  Mod_1,  Mod_2,  Label 
#                                                  c      a,b
Red_Fields = {
              0: ('Rep. Count' , True,  1,       D_negB, D_logN, r'$N^{loss}$'  ),
              1: ('Clos. Count', True,  1,       D_negB, '__',   r'$N^{clo}$'   ),
              2: ('Incurred'   , False, Units_I, D_logN, D_logN, r'$\hat{I}^*$' ),
              3: ('Paid'       , False, Units_I, D_logN, '__',   r'$\hat{P}^*$' ),
              4: ('Rep. Lag'   , False, 1,       D_logN, D_logN, r'$\tau^{rep}$'),
              5: ('Inc. Lag'   , False, 1,       D_Norm, D_Norm, r'$\tau^I$'    ),
              6: ('Paid Lag'   , False, 1,       D_logN, D_logN, r'$\tau^P$'    ),
              7: ('Inc. Count' , True,  1,       D_negB, D_logN, r'$\hat{n}^I$' ),
              8: ('Paym. Count', True,  1,       D_negB, D_logN, r'$\hat{n}^P$' ),
             }

# Dictionnary with Figure file names (plot) and Table file names (print)
plt_list = ['.pdf', '.png']
txt_list = ['.txt']
GIRF_fn_dict = {
#    fn_ID             file_name                ext       LaTeX 
    'dummy_plot'   : ('dummy_plot_file',        plt_list, None),
    'dummy_print'  : ('dummy_print_file',       txt_list, None),

    'red_var'      : ('Reduced_variables',      plt_list, 'Figure 4.1.'),
    'patt_lags'    : ('Patterns_lags',          plt_list, 'Figure 4.2.'),
    
    'pN_GP'        : ('pdf_evolution',          plt_list, 'Figure 6.1.a)'),
    'pN_GM_0'      : ('pdf_simulation',         plt_list, 'Figure 6.1.b)'),
    'pN_GM_c'      : ('pdf_fit_c',              plt_list, 'Figure 6.1.c)'),
    'pN_GM_ab'     : ('pdf_fit_ab',             plt_list, 'Figure 6.1.d)'),
    'pN_par'       : ('T_pdf_fit_param',        txt_list, 'Table 6.1. and C.1'),
    
    'cond_stats'   : ('cond_calibration_stats', plt_list, 'Figure 6.2.'),
    'uncond_stats' : ('calibration_stats',      plt_list, 'Figure 6.3.'),
    
    'fit_comp'     : ('fit_comparison',         plt_list, 'Figure 6.4.'),
    'fit_comp_mod' : ('T_comp_model_param',     txt_list, 'Table 6.2. and C.2'),
    'fit_comp_par' : ('T_comp_calibr_param',    txt_list, 'Table 6.3 and C.4.'),
    
    'all_it'       : ('evolution_all_ab_it_',   plt_list, 'Figure 6.5. and 6.6'),  # iteration_nr is appended to file_name
    'all_it_par'   : ('T_all_fit_param',        txt_list, 'Table C.4'),
    
    }
    
# *********************************

def get_patterns(gray_scale = True):
# Samples of incurred patterns for given paid pattern shown in Figure 4.2
# Basis for evaluating reduced variables shown in Figure 4.1
# gray_scale = True/False is used for figures in printed/online version
    
    time_grid =  [15, 20, 30, 40, 50, 65, 90,120]        # time stamps in [months] since occurrence

    Inc_list  = [                                        # list with examples of incurred patterns I_i = I(t_i+) 
         [44, 44, 40, 40, 40, 34, 34, 32],               # favorable, decrease
         [32, 37, 37, 44, 44, 37, 35, 32],               # favorable, bump

         [32, 32, 32, 32, 32, 32, 32, 32],               # unbiased, flat

         [20, 20, 24, 24, 24, 30, 30, 32],               # adverse , increase
         [32, 27, 27, 20, 20, 27, 29, 32],               # adverse , dip
        ]   
    index_flat = 2                                       # index of flat incurred pattern
    
    Paid       = [ 0,  3,  7, 12, 18, 23, 28, 32]        # cumulative paid pattern P_i = P(t_i+) 
    
    # labels and plot-parameters for the incurred patterns and the paid pattern:
    label_list = [
        ['favorable: decrease', 'favorable: bump', 'unbiased: flat', 'adverse  : increase', 'adverse  : dip'],
        ['cumululative paid']
        ]
    if gray_scale:
        color_list = [[ 'dimgrey', 'gainsboro', 'black', 'grey', 'darkgrey', 'lightgray', ], ['gray']]
    else:
        color_list = [[ 'b',       'c',         'k',     'r',    'm',        'g',         ], ['gray']]
    lw_list = [[   1,   1,   1,   1,   1], [1.5]]
    ls_list = [['--','--', '-','-.','-.'], ['-']]

    return time_grid, Inc_list, Paid, index_flat, label_list, color_list, lw_list, ls_list
    
# *********************************

# Defaults created with 'get_freq_model_new_default()' in module or GIRF_main.py (or GIRF_calibrate.py) 

# random variables drawn from:
#  - process_param = (12, 5.0, 15.0, 1.25)
Y_t_default = np.array( [ 3, 4, 6, 7, 10, 10, 7, 8, 12, 8, 18, 13] )
#  - model_param, K = (12, 10.0, 20.0, 1.0), 100
E_default   = np.array( [10.630, 10.365,11.315, 11.855, 12.365, 13.580, 14.620, 15.645, 16.120, 18.025, 18.780, 19.955] )
V_default   = np.array( [11.183, 11.742, 9.916, 12.484, 11.442, 12.004, 16.506, 16.909, 15.856, 17.534, 21.572, 24.063] )

def get_freq_model(use_default_obs=False):
# Frequency model used in Figurs 6.1, 6.2 and 6.3
# -> Sample calibration and calibration statistics

    T = 12                                                 # length of observation period
   
    E_0_p, E_T_p, f_P_p = 5.0, 15.0, 1.25                  # Gen process: negative Binomial model
    
    E_0_m, E_T_m, f_P_m = 10.0, 20.0, 1.0                  # Gen process: Poisson model  
    
    Phi_a, Phi_b, Phi_c = 1., 1./(T-1), 1.                 # Variance of the priors for a, b, c
    
    if use_default_obs:
        Y_t = Y_t_default                                  # use default sample of simulated observations
    else:
        Y_t = None                                         # use generative model to draw a random set

    process_param = (T, E_0_p, E_T_p, f_P_p)
    model_param   = (T, E_0_m, E_T_m, f_P_m)
    Phi =(Phi_a, Phi_b, Phi_c)
    
    return process_param, model_param, Phi, Y_t

# *********************************

def param_R_c_models():
    # List of lists with Panjer factors: [[B-list: f_P < 1], [P-list: f_P = 1], [NB-list: f_P > 1]]
    f_P_lists = [[0.5, 0.80], [1.0], [1.2, 2.0]]
    
    # Lists with lower & higer plot ranges for c:
    P_lo  = [[0, 0], [-1.4], [0, 0]]                 # Posson, lower ranges
    P_hi  = [[0, 0], [ 5.0], [0, 0]]                 #         upper ranges
    P_range = (P_lo, P_hi)
  
    B_lo  = [[-0.7, -1.2], [-1.2], [0, 0]]           # Binomial, lower ranges
    B_hi  = [[ 5.0,  5.0], [ 5.0], [0, 0]]           #           upper ranges
    B_range = (B_lo, B_hi)
  
    nB_lo = [[0, 0], [-1.6], [-1.6, -2.4]]           # negative Binomial, lower ranges
    nB_hi = [[0, 0], [ 5.0], [ 5.0,  5.0]]           #                    upper ranges
    nB_range = (nB_lo, nB_hi)
  
    G_lo  = [[-1.2, -1.5], [-1.5], [-2.1, -2.0]]     # Gamma, lower ranges
    G_hi  = [[ 0.2,  0.7], [ 0.8], [ 1.0,  1.7]]     #        upper ranges
    G_range = (G_lo, G_hi)
    
    dc0_lo, dc0_hi = -0.12, +0.08
    c0_range = (dc0_lo, dc0_hi)
    c_ranges = [P_range, B_range, nB_range, G_range, c0_range]

    # color coding
    col0, col1, col2 = 'k', 'darkgrey', 'gray'   
    col_list = [col0, col1, col2]
    colors   = [[col1, col2], [col0], [col2, col1]]

    # Distribution-name labels L(c) and R(c) for P, B. NB, and G
    L_, P_, B_, nB_, G_ = 'L(c)', 'Poisson', 'binomial', 'neg bin', 'gamma'
    d_labels = [L_, P_, B_, nB_, G_]
    
    # Parameter labels (if applicable, else '_')
    L_L  =  [[['_', '_'], ['_'], ['_', '_']],  
             [['_', L_], [L_], ['_', L_]]]
    L_P  =  [[['_', '_'], [f'f_P={f_P_lists[1][0]:.1f}'], ['_', '_']],  
             [['_', '_'], [P_], ['_', '_']]]
    L_B  =  [[[f'f_P={f_P_lists[0][0]:.1f}', f'f_P={f_P_lists[0][1]:.1f}'], ['f_P=0.8'], ['_', '_']],  
             [['_', B_], [B_], ['_', '_']]]
    L_nB =  [[['_', '_'], ['f_P=1.2'], [f'f_P={f_P_lists[2][0]:.1f}', f'f_P={f_P_lists[2][1]:.1f}']],  
             [['_', '_'], [nB_], ['_', nB_]]]
    L_G  =  [[['_', '_'], ['_'], ['_', '_']],  
             [['_', G_], [G_], ['_', G_]]]
    LR_labels = [L_L, L_P, L_B, L_nB, L_G]
    
    param_lists = (f_P_lists, c_ranges, (col_list, colors), d_labels, LR_labels)
    
    E, V, Y_t = E_default, V_default, Y_t_default
    Phi_c = 1
    
    return param_lists, E, V, Y_t, Phi_c
    

# *********************************

def param_model_comparison():
    # Observation period
    T        = 10
    
    # Parameters for generative process:    
    lda_0    = 10
    lda_T    = 12.5
    beta     = 5 
    par_emp = (lda_0, lda_T, beta)
    
    # Parameters for generative model:    
    lda_0    = 5
    lda_T    = 10
    par_sim = (lda_0, lda_T, beta)
    
    # Calibrate
    Phi_a = 1
    Phi_b = 1
    Phi_c = 1
    Phi = (Phi_a, Phi_b, Phi_c)
    
    param = (T, par_emp, par_sim, Phi)
    
    return param

# *********************************

def get_full_model():
# period    
    T = 12                                              # length of observation period

    obs_period = (T, date_sub, year_sub - T, year_sub - 1)
    
# parameters for generative process and generative model
    # param :        [freq, f_clos, mean_I, mean_P, tau_rep, tau_inc, tau_paid,  N_I,  N_P]
    param_labels  = ['$\\E[N_o^{inc}]$', '$\\E[N_o^{clo}]$', '$\\E[I_o]$', '$\\E[P_o]$', '$\\E[\\tau_o^{rep}]$', 
                     '$\\E[\\tau_o^I]$', '$\\E[\\tau_o^P]$', '$\\E[N_o^I]$', '$\\E[N_o^P]$']
    process_param = [[  10,      0,  1.0E8,      0,     4.5,     2.5,     15.0,    5,   10],       # o = -T
                     [  30,      0,  1.5E8,      0,     3.5,    -1.5,     10.0,    8,    5],       # o = -1
                     ]
    model_param   = [[  20,      0,  2.0E8,      0,     3.0,     0.0,     18.0,   10,   12],       # o = -T
                     [  40,      0,  2.5E8,      0,     3.0,     0.0,     18.0,   10,   12],       # o = -1
                     ]
# features to be calibrated    
    cal_True       = [True,  False,   True,  False,    True,    True,     True, True, True]
    n_param = len (process_param[0])
    
# normal priors pi(c; mu, phi=sigma^2) with mu = 0
    Phi_a = np.ones(n_param)
    Phi_b = np.ones(n_param) / 10
    Phi_c = np.ones(n_param)
    Phi_cal = [Phi_a, Phi_b, Phi_c]
    
    return param_labels, obs_period, process_param, model_param, cal_True, Phi_cal

# *********************************

if __name__ == "__main__":
    
    params1 = get_patterns(gray_scale = True)
    time_grid, Inc_list, Paid, index_flat, label_list, color_list, lw_list, ls_list = params1
    
    params2 = get_freq_model(use_default_obs=False)
    process_param, model_param, Phi, Y_t = params2
    
    params3 = param_R_c_models()
    param_lists, E, V, Y_t, Phi_c = params3
    
    params4 = param_model_comparison()
    (T, par_emp, par_sim, Phi) = params4
    
    params5 = get_full_model()
    param_labels, obs_period, process_param, model_param, cal_True, Phi_cal = params5