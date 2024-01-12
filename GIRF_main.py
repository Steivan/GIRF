# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""

from GIRF_models import get_patterns, get_freq_model, param_R_c_models, param_model_comparison, get_full_model
from GIRF_reduced import plot_IP, plot_t_IP
from GIRF_calibrate import claims_count_calibration, get_freq_model_new_default, get_claims_count_stats, \
                           run_calibration_comparison, model_comparison_LaTeX, run_full_calibration

# **********************************************************

def patterns_and_red_var(plot_reduced=True, plot_patterns=True, gray_scale=True):
    # Patterns shown in Figure 4.2 
    # Depiction of reduced variables in Figure 4.1
    
    time_grid, Inc_list, Paid, index_flat, label_list, color_list, lw_list, ls_list = get_patterns(gray_scale = gray_scale)
    
    plot_t_IP(time_grid, Inc_list, Paid, index_flat, color_list, lw_list, ls_list, label_list, fn_ID='patt_lags')

    t_eval = 75                                                   # lag [month] for the evaluation of reduced variables   
    i_eval = 3                                                    # index of incurred pattern to be evaluated                                        
    plot_IP(time_grid, Inc_list[i_eval], Paid, t_eval, color_list, fn_ID='red_var')

# **********************************************************
    
def calibrate_claims_count():
    # Calibration of the claims count shown in Figure 6.1
    # - generative process defined by 'process_param'
    # - Y_t: simulated 'observations'
    # - generative model defined by 'model_param' 
    # - priors defined by 'Phi'
    
    use_default = True                      # True : use default observations, i.e., Y_t = Y_t_default
                                            # False: draw observations Y_t from generative process parameters
    
    process_param, model_param, Phi, Y_t = get_freq_model(use_default_obs=use_default)
    T = process_param [0]
    (Phi_a, Phi_b, Phi_c) = Phi
    
    K = 200                                                       # count of simulations
    claims_count_calibration(T, process_param, model_param, Y_t, Phi_a, Phi_b, Phi_c, K_sim=K,
                             fn_ID_plt=['pN_GP', 'pN_GM_0', 'pN_GM_c', 'pN_GM_ab'], 
                             fn_ID_txt= 'pN_par')

# **********************************************************
    
def claims_count_stats():
    # claims count calibration statistics shown in Figure2 6.2 and 6.3
    
    use_default = True                      # True : use default sample of simulated observations Y_t
                                            # False: draw observations Y_t from generative process 
    
    process_param, model_param, Phi, Y_t = get_freq_model(use_default_obs=use_default)
    T = process_param [0]
    (Phi_a, Phi_b, Phi_c) = Phi
    
    K = 200                                     # count of simulations per run
    N = 100                                     # count of calibration runs
    # Conditional statistics: Y_t is given
    Y_t_0 = Y_t
    get_claims_count_stats(T, process_param, model_param, Y_t_0, Phi_a, Phi_b, Phi_c, K_sim=K, N_run=N)
    
    # Unconditional statistics: Y_t is redrawn in each run
    Y_t_0 = None
    get_claims_count_stats(T, process_param, model_param, Y_t_0, Phi_a, Phi_b, Phi_c, K_sim=K, N_run=N,
                           fn_ID_uncond='uncond_stats', fn_ID_cond='cond_stats')
    
# **********************************************************
    
def calibration_comparison():
    # Fitting comparison shown in Figure 6.4
    # (same underlying model as in Figures 6.1, 6.2, and 6.3)
    # calibration comparison: plot L(c) and R(c) for P, B, NB, and G
    # compare roots with logN solution
    
    param_lists, E, V, Y_t, Phi_c = param_R_c_models()
    run_calibration_comparison(param_lists, E, V, Y_t, Phi_c, fn_ID='fit_comp')
    
# **********************************************************
    
def model_comparison():
    # Comparison of calibration parameters derived with different methods:
    # - Generative models: NB, P, and B
    # - Parametric models fitted to simulated distributions: NB (if f_P > 1), P, NB (if f_P < 1), G, and logN 
    # - Calibration parameters derived on annual basis and period basis
    # => Print LaTeX content for table with model parameters (Table 6.2)  
    # => Print LaTeX content for table with calibration parameters (Table 6.3)  
    
    model_param = param_model_comparison()
    
    nr_runs =  5                    # count calibration runs per generative model
    K_sim   = 50                    # count of simulation runs for the generative models 
    f_P_list = [1.5, 1.0, 0.6]      # list with three values for f_P: [NB, P, B] -> input for table contents
    n = len(f_P_list)
    for i_f_P in range(n):
        first = i_f_P == 0
        last  = i_f_P == n-1
        model_comparison_LaTeX(model_param, f_P_list, i_f_P, K_sim=K_sim, nr_runs=nr_runs, first=first, last=last,
                               fn_ID_mod='fit_comp_mod', fn_ID_par='fit_comp_par')
    
# **********************************************************

def full_calibration():
    
    param_labels, obs_period, process_param, model_param, cal_True, Phi_cal = get_full_model()
    (T, Date_Sub, Year_min, Year_max) = obs_period

    iterations = 5
    N_sim = 200

    run_full_calibration(iterations, N_sim,  param_labels, process_param, model_param, Date_Sub, Year_min, Year_max, 
                         cal_True, Phi_cal, plt_dist=False, fn_ID_plt='all_it', 
                         fn_ID_txt='all_it_par')

if __name__ == "__main__":
    
# Generic Integrated Rating Framework (GIRF):
# *******************************************
# - Input : the parameters for the various models are defined in: 
#           - GIRF_models.py     
# - Output: the output file names (figures and tables) are defined in: 
#           - GIRF_models.py / GIRF_fn_dict     

    # Plots 'claims representation and reduced variables'
    # and 'patterns and lags' (Figures 4.1 and 4.2) 
    patterns_and_red_var()
    
    # Create default parameters: copy / paste from console to 
    # module GIRF_models.py
    get_freq_model_new_default(K=100)

    # Four plots 'calibration of the annual observations' an print
    # parameters (Figure 6.1 (a)-(d) and Table 6.1)
    calibrate_claims_count()
    
    # Plots 'conditional' and 'unconditional calibration statistics
    # (Figures 6.2 and 6.3)
    claims_count_stats()
    
    # Plot 'fitting comparison' (Figure 6.4)
    calibration_comparison()

    # Print model and calibration parameters (Tables 6.2 and 6.3)
    model_comparison()
    
    # Run full calibration model (Figures 6.5 and 6.6 and Table C4)
    full_calibration()

