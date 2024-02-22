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

# module nr 1
def patterns_and_red_var():
    r"""
    
    Claim representation and reduced variables
    ------------------------------------------

    Parameters
    ----------
    None.

    Input
    -----
    - Model parameters are imported from 'GIRF_models.py' via 'get_patterns()'

    Execution
    ---------
    The routines 'plot_IP()' and 'plot_t_IP()' imported from 'GIRF_reduced.py' generates
    a plot with the claim representation and a plot with derived temporal developments.
    
    Output
    ------
    - Plot with claim representation and reduced variables 'reduced_variables.pdf' (-> Figure 4.1)
    - Plot with claim patterns and lags                    'Patterns_lags.pdf'     (-> Figure 4.2)

    Returns
    -------
    None.

    """
    
    time_grid, Inc_list, Paid, index_flat, label_list, color_list, lw_list, ls_list = get_patterns()
    
    plot_t_IP(time_grid, Inc_list, Paid, index_flat, color_list, lw_list, ls_list, label_list, fn_ID='patt_lags')

    t_eval = 75                                                   # lag [month] for the evaluation of reduced variables   
    i_eval = 3                                                    # index of incurred pattern to be evaluated                                        
    plot_IP(time_grid, Inc_list[i_eval], Paid, t_eval, color_list, fn_ID='red_var')

# **********************************************************
    
# module nr 2
def get_new_default_param(K_sim=100):
    r"""
    
    Generation of defaullt random variables
    ---------------------------------------

    Parameters
    ----------
    K_sim : Integer, optional
        Count of simulations per run. The default is 100.
        
    Execution
    ---------
    The routine 'get_freq_model_new_default()' imported from 'GIRF_calibrate.py' 
    generates K=1 set of random variables for the generative process and K=K_sim
    sets of random variables for the generative model.
    
    Output
    ------
    Statistics are written to the console and need to copy/pasted into the 
    'GIRF_models.py' module (lines 122 ff.)

    Returns
    -------
    None.

    """
    
    # Create default parameters: copy / paste from console to 
    # module GIRF_models.py
    
    get_freq_model_new_default(K=K_sim)
    
# **********************************************************
    
# module nr 3
def calibrate_claims_count(K_sim=100, use_default = True):
    r"""

    Calibration of the claims-count process
    ---------------------------------------

    Parameters
    ----------
    K_sim : Integer, optional
        Count of simulations per run. The default is 100.
    use_default: Boolean, optional
        if True : Use the default sample of simulated observations Y_t.
        if False: Draw  aset of observations Y_t from generative process.

    Input
    -----
    - Model parameters are imported from 'GIRF_models.py' via 'get_freq_model()'

    Execution
    ---------
    The routine 'claims_count_calibration()' imported from 'GIRF_calibrate.py' generates
    a table and plots with probability distributions for the generative process, 
    the prior model, and the models calibrated on an annual level and a period level.
    
    Output
    ------
    - LaTeX table with model parameters and statistics  'T_pdf_fit_param.txt' (-> Table 6.1 and C.1)
    - LaTeX summary file with detaile statistics        'T_pdf_fit_stats.txt' (-> Section C.1.2)
    - Pmf's and fitted pdf's for the generative model   'pdf_evolution.pdf'   (-> Figure 6.1 a)
    - Pmf's and pdf's for the prior  generative process 'pdf_simulation.pdf'  (-> Figure 6.1 b)
    - Pmf's and pdf's for the period-level calibration  'pdf_fit_c.pdf'       (-> Figure 6.1 c)
    -                         annual-level calibration  'pdf_fit_ab.pdf'      (-> Figure 6.1 d)

    Returns
    -------
    None.

    """
    
    # Four plots 'calibration of the annual observations' and print
    # parameters (Figure 6.1 (a)-(d), Table 6.1 = C.1, and stats file for App. C.1.2)

    # Calibration of the claims count shown in Figure 6.1
    # - generative process defined by 'process_param'
    # - Y_t: simulated 'observations'
    # - generative model defined by 'model_param' 
    # - K: count of simulations
    # - use_default = True : use default observations, i.e., Y_t = Y_t_default
    #               = False: draw observations Y_t from generative process parameters
    # - priors defined by 'Phi'
    
    process_param, model_param, Phi, Y_t = get_freq_model(use_default_obs=use_default)
    T = process_param [0]
    (Phi_a, Phi_b, Phi_c) = Phi
    
    claims_count_calibration(T, process_param, model_param, Y_t, Phi_a, Phi_b, Phi_c, K_sim=K_sim,
                             fn_ID_plt=['pN_GP', 'pN_GM_0', 'pN_GM_c', 'pN_GM_ab'], 
                             fn_ID_txt=['pN_par', 'pN_stats'])

# **********************************************************
    
# module nr 4
def claims_count_stats(N_run=100, K_sim=200, use_default=True):
    r"""
    
    Claims count calibration statistics
    -----------------------------------

    Parameters
    ----------
    N_run : Integer, optional
        Count of calibration runs. The default is 100.
    K_sim : Integer, optional
        Count of simulations per run. The default is 200.
    use_default: Boolean, optional
        if True : Use the default sample of simulated observations Y_t.
        if False: Draw  aset of observations Y_t from generative process.

    Input
    -----
    - Model parameters are imported from 'GIRF_models.py' via 'get_freq_model()'

    Execution
    ---------
    Two sets of charts are generated with 'get_claims_count_stats()' 
    imported from 'GIRF_calibrate.py':
        - 1st set:   conditional, i.e., same Y_t is used in each calibration run.
        - 2nd set: unconditional, i.e., a new set of observations Y_t is generated in 
                   each calibration run.
    
    Output
    ------
    - Charts with   conditional statistics 'cond_calibration_stats.pdf' (-> Figure 6.2)
    - Charts with unconditional statistics      'calibration_stats.pdf' (-> Figure 6.3)

    Returns
    -------
    None.

    """
    
    # get model parameters
    process_param, model_param, Phi, Y_t = get_freq_model(use_default_obs=use_default)
    T = process_param [0]
    (Phi_a, Phi_b, Phi_c) = Phi
    
    # Conditional statistics: Y_t is given
    Y_t_0 = Y_t
    get_claims_count_stats(T, process_param, model_param, Y_t_0, Phi_a, Phi_b, Phi_c, K_sim=K_sim, N_run=N_run,
                           fn_ID_uncond='uncond_stats', fn_ID_cond='cond_stats')
    
    # Unconditional statistics: Y_t is redrawn in each run
    Y_t_0 = None
    get_claims_count_stats(T, process_param, model_param, Y_t_0, Phi_a, Phi_b, Phi_c, K_sim=K_sim, N_run=N_run,
                           fn_ID_uncond='uncond_stats', fn_ID_cond='cond_stats')
    
# **********************************************************
    
# module nr 5
def calibration_comparison():
    r"""
    
    Evaluation of the MAP estimates for the Panjer class
    ----------------------------------------------------
    Plot L(c) and R(c) for P, B, NB, and G and compare roots with logN solution
    - Curves are shown for a selection of Panjer factors: NB, P, and B
    
    Parameters
    ----------
    None.
    
    Input
    -----
    - Model parameters are imported from 'GIRF_models.py' via 'param_R_c_models()'
      (same underlying model as in Figures 6.1, 6.2, and 6.3)

    Execution
    ---------
    The charts are generated with 'run_calibration_comparison()' 
    imported from 'GIRF_calibrate.py'
    
    Output
    ------
    - Charts with curves L(c) and R(c) 'fit_comparison.pdf' (-> Figure 6.4)
    
    Returns
    -------
    None.

    """
    
    param_lists, E, V, Y_t, Phi_c = param_R_c_models()
    run_calibration_comparison(param_lists, E, V, Y_t, Phi_c, fn_ID='fit_comp')
    
# **********************************************************
    
# module nr 6
def model_comparison(N_run=5, K_sim=50, f_P_list=[1.5, 1.0, 0.6]):
    r"""
    
    Comparison of calibration parameters derived with different methods
    -------------------------------------------------------------------
    - Generative models: NB (f_P>1), P (f_P=1), and B (f_P<1)
    - Parametric models fitted to simulated distributions: 
        - Discrete models  : NB (only if f_P > 1), P, NB (only if f_P < 1)
        - Continuous models: G and logN 
    - Calibration parameters evaluated on annual level and period level
    
    Parameters
    ----------
    N_run : Integer, optional
        Count calibration runs per generative model. The default is 5.
    K_sim : Integer, optional
        Count of simulation runs for the generative models. The default is 50.
    f_P_list : List with floats, optional
        List with f_P values for [NB, P, B] cases. The default is [1.5, 1.0, 0.6].

    Input
    -----
    - Model parameters are imported from 'GIRF_models.py' via 'param_model_comparison()'

    Execution
    ---------
    The iterative calibration procedure is performed with 'model_comparison_LaTeX()' 
    imported from 'GIRF_calibrate.py'
    
    Output
    ------
    - Parameters: Create LaTeX table 'T_comp_model_param.txt.' (-> Table 6.2 = C.2)
    - Comparison: Create LaTeX table 'T_comp_calibr_param.txt' (-> Table 6.3 = C.3)
    
    Returns
    -------
    None.

    """
    
    model_param = param_model_comparison()
    
    n = len(f_P_list)
    for i_f_P in range(n):
        first = i_f_P == 0
        last  = i_f_P == n-1
        model_comparison_LaTeX(model_param, f_P_list, i_f_P, K_sim=K_sim, nr_runs=N_run, first=first, last=last,
                                fn_ID_mod='fit_comp_mod', fn_ID_par='fit_comp_par')
    
# **********************************************************

# module #7
def full_calibration(Nr_iter=5, K_sim=200):
    r"""
    
    Calibrate 7/9 model features and monitor the remaining 2/9
    ==========================================================
    
    Parameters
    ----------
    Nr_iter : Integer, optional
        Count of iterations. The default is 5.
    K_sim : Integer, optional
        Count of simulations runs. The default is 200.

    Input
    -----
    - Model parameters are imported from 'GIRF_models.py' via 'get_full_model()'

    Execution
    ---------
    The iterative calibration procedure is performed with 'run_full_calibration()' 
    imported from 'GIRF_calibrate.py'
    
    Output
    ------
    - Parameters       : Create LaTeX table 'T_all_fit_param.txt (-> Table  C.4)
    - Initial model    : Create figure 'evolution_all_it_0.pdf'  (-> Figure 6.5)
    - Calibrated model : Create figure 'evolution_all_it_5.pdf'  (-> Figure 6.6)
    
    Returns
    -------
    None.

    """
    
    param_labels, obs_period, process_param, model_param, cal_True, Phi_cal = get_full_model()
    (T, Date_Sub, Year_min, Year_max) = obs_period

    run_full_calibration(Nr_iter, K_sim,  param_labels, process_param, model_param, Date_Sub, Year_min, Year_max, 
                          cal_True, Phi_cal, plt_dist=False, fn_ID_plt='all_it', 
                          fn_ID_txt='all_it_par')

if __name__ == "__main__":
    
# Generic Integrated Rating Framework (GIRF):
# *******************************************
# - Input : the parameters for the various models are defined in: 
#           - GIRF_models.py     
# - Output: the output file names (figures and tables) are defined in: 
#           - GIRF_models.py / GIRF_fn_dict    
# 

    selection = [1, 2, 3, 4, 5, 6, 7]
    
    # Plots 'claims representation and reduced variables'
    # and 'patterns and lags' (Figures 4.1 and 4.2) 
    if 1 in selection: patterns_and_red_var()
    
    # Create default parameters: copy / paste from console to 
    # module GIRF_models.py
    if 2 in selection: get_new_default_param(K_sim=100)

    # Four plots 'calibration of the annual observations' and print
    # parameters (Figure 6.1 (a)-(d), Table 6.1 = C.1, and stats file for App. C.1)
    if 3 in selection: calibrate_claims_count()
    
    # Plots 'conditional' and 'unconditional calibration statistics
    # (Figures 6.2 and 6.3)
    if 4 in selection: claims_count_stats(N_run=100, K_sim=200, use_default=True)
    
    # Plot 'fitting comparison' (Figure 6.4)
    if 5 in selection: calibration_comparison()

    # Print model and calibration parameters (Tables 6.2 = C.2 and 6.3 = C.3)
    if 6 in selection: model_comparison(N_run=5, K_sim=50, f_P_list=[1.5, 1.0, 0.6])
    
    # Run full calibration model (Figures 6.5 and 6.6 and Table C.4)
    if 7 in selection: full_calibration(Nr_iter=5, K_sim=200)

