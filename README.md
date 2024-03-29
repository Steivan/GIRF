# GIRF
Generic Integrated Rating Framework

## Python library GIRF
The python library GIRF comprises the modules listed in Table D.1 (dependencies are indicated by a ’+’ symbol). The library is used to generate most figures and tables in the printed document and in the online supplementary.

![image](https://github.com/Steivan/GIRF/assets/87634614/d6b37781-4ca4-4715-84a8-451840c29cec)

## Modules
The most relevant modules for a user are the main module ``GIRF_main.py`` used to run the various top-level routines and the module ``GIRF_models.py`` containing the parameters for the various example models.

#### GIRF_main
Following code extract from the ``GIRF_main.py`` module provides an overview of the routines used to generate the figures and tables:

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


#### GIRF_reduced
The module ``GIRF_reduced.py`` is used to generate the chart depicting the reduced variables and the chart depicting the claims development patters and the temporal evolution of the lags.

#### GIRF_calibrate
The module ``GIRF_calibrate.py`` contains the routines used to calibrate the example models.

#### GIRF_plot
The module ``GIRF_plot.py``  contains the routines used to generate the figures with the results of the simulations and the figures containing analytical results. The module is also used to generate the tables containing the respective parameters.

#### GIRF_claim
The module ``GIRF_claim.py`` contains the classes used to represent the reduced variables on a claims, an annualand a periodlevel. It also contains a routine used to generate reduced claims.

#### GIRF_Bayes
The module ``GIRF_Bayes.py`` contains the routines used to evaluate the calibration parameters with the help of various parametric distributions fitted to the simulated distributions.

#### GIRF_models
The module ``GIRF_models.py`` contains the some global parameters and the specific parameters used in the sample models. The dictionary ``Red_Fields`` is used to assign a parametric distribution family to each reduced variable. The dictionary ``GIRF_fn_dict`` is used to specify the names and the formats of the output files (Figures and LaTeX tables).

#### GIRF_stats
The module ``GIRF_stats.py`` contains some classes which are used as wrappers to the scipy library.

#### Transscript
The module ``Transscript.py`` used to redirect the output from the console to a text file (and the console).
