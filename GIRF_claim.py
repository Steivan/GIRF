# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:21:29 2024

@author: Stefan Bernegger
"""

import numpy as np 
from scipy.stats import lognorm, poisson, beta
import datetime as dt

from GIRF_models import today, date_sub, sub_year, days_per_year, Units_I

# ********************************************************

def get_beta_rvs(x_min, x_max, mean, sigma, size=1, sort_true=True):
    scale = x_max - x_min 
    if scale == 0.:
        x = np.ones(size) * x_max
    else:
        m     = (mean - x_min) / scale
        v     = np.square(sigma / scale)
        v_max = m*(1-m)*0.99
        if v > v_max: v = v_max
        c = m*(1-m)/v - 1.
        a = c * m
        b = c - a
        x = beta.rvs(a, b, loc=x_min, scale=scale, size=size)
        if sort_true: x = np.sort(x)
    return x    

# ********************************************************

def my_round(x, n):
    y = x.copy()
    f = 10**n
    for i in range(len(y)):
        try:
            y[i] = int(y[i]*f)/f
        except:
            y[i] = y[i]
    return y

# ********************************************************

def get_reduced_claim(mean, t_rep, t_inc, t_paid, t_sub, n_inc, n_paid):
# Input:
# - mean: Mean of loss amount    
# - t_rep, t_inc, t_paid: Mean Reported Lag, Incurred Lag and Paid Lag  
# - t_sub: Submission Lag
# - n_inc, n_paid: Mean count of Incurred Adjustments and count of Payments
# Output:
# - Reported: True if reporting-date <= submission-date, i.e., Reporting Lag <= Submission Lag 
# - Closed: True is closure-date <= submission-date
# - Inc: Latest incurred estimate at submission date
# - Paid: Paid amount till submission date    
# - t_r, t_i, t_p: Simulated Reported Lag, Incurred Lag and Paid Lag  
# - r_p: Paid / Incurred ratio
# - N_I, N_P: Simulated count of Incurred Adjustments and count of Payments
# Parameters:
    s_Ult     = 0.5                    # sigma-parameter of Ultimate ~ LogN(sigma)
    s_Inc     = 0.25                   # sigma-parameter of Incurred ~ LogN(sigma)
    r_t_c_max = 1.5                    # t_c_max / t_c_mean
    r_t_c_std = 2. / 3.                # t_c_std / t_c_mean
    r_t_p_max = 2.                     # t_p_max / t_p_mean
    r_t_p_std = 2. / 3.                # t_p_std / t_p_mean
     
    # Simulated Reported Lag: assume that t_r is beta-distributed within the range (0, 3*t_rep) and mean= t_rep and std = t_rep      
    t_r_max = t_rep * 3
    t_r = get_beta_rvs(0, t_r_max, t_rep, t_rep, size=1)[0]

    t_s = t_sub - t_r
    Reported = t_s >= 0.
    
    if Reported:
        # Simulated Closure Lag post Reporting:
        t_c_mean = 3 * max(1., t_paid - t_rep)                              # mean closure lag post reporting date
        t_c_max  = t_c_mean * r_t_c_max                                     # set upper bound
        t_c_std  = np.sqrt(t_c_mean * (t_c_max - t_c_mean)) * r_t_c_std     # set std to 2/3 of upper bound
        t_c      = get_beta_rvs(0, t_c_max, t_c_mean, t_c_std, size=1)[0]    
        Closed = t_c <= t_s
        
        # Simulated Incurred dates between Reported and Closure:
        Ultimate        = mean * lognorm.rvs(s=s_Ult)                       # Ultimate loss amount
        f_I             = 1.
        if t_c > 0: f_I = max(0., 2 * (1 - (t_inc-t_rep)/t_c) - 1)          # IBNER factor: f_I = I(t_rep) / I(t_clo)
        n_I_tot         = 1 + poisson.rvs(max(0, n_inc-1))                  # total nr of adjustments
        t_I             = t_c * np.random.rand(n_I_tot)                     # adjustment times 
        t_I.sort()
        t_I[0]          = 0.                                                # date of first estimate 
        t_I[n_I_tot-1]  = t_c                                               # date of final estimate (ultimate)
        x_I_0           = lognorm.rvs(s=s_Inc, size=n_I_tot)                # random estimates of Incurred
        x_I_0          /= x_I_0[n_I_tot-1]                                  # relative to ultimate
        x_I             = ((t_c-t_I) * x_I_0 + t_I) / t_c                   # gradually converge towards ultimate
        x_I            *= ((t_c-t_I) * f_I   + t_I) / t_c                   # apply IBNER-adjustment

        I_known         = t_I <= t_s                                        # dates <= t_s
        I_max_i         = I_known.argmin() - 1                              # index of latest known date 
        t_I_hi          = np.roll(t_I, - 1)                                 # get t(i+1)                              
        t_I_hi[I_max_i] = t_s                                               # upper integration limit t_s
        I_last          = x_I[I_max_i]                                      # latest known value
        t_i             = t_r + t_s                                         # default incurred lag
        if I_last > 0:
            t_i        -= ((t_I_hi - t_I) * x_I * I_known).sum() / I_last   # adjustment for development
        N_I             = I_known.sum()
        
        # Payment dates between Reported and Closure:
        t_p_max         = t_c
        t_p_mean        = t_c / r_t_p_max
        t_p_std         = np.sqrt(t_p_mean * (t_p_max - t_p_mean)) * r_t_p_std
        n_p_tot         = 1 + poisson.rvs(max(0, n_paid-1))    
        t_P             = get_beta_rvs(0, t_p_max, t_p_mean, t_p_std, size=n_p_tot, sort_true=True)
        x_P             = np.random.rand(n_p_tot)
        x_P.sort()
        x_P[n_p_tot-1]  = 1.
        
        P_known         = t_P <= t_s                                        # dates <= t_s
        P_max_i         = P_known.argmin() - 1                              # index of latest known date 
        t_P_hi          = np.roll(t_P, -1)                                  
        t_P_hi[P_max_i] = t_s
        P_last          = x_P[P_max_i]
        IP_last         = max(I_last, P_last)
        t_p             = t_r + t_s                                         # default incurred lag
        if IP_last > 0:
            t_p        -= ((t_P_hi - t_P) * x_P * P_known).sum() / IP_last  # adjustment for development
        N_P             = P_known.sum()

        Inc             = Ultimate * IP_last
        Paid            = Ultimate * P_last
    else:
        Closed = False
        Inc    = mean
        Paid   = 0.
        t_i    = t_r + (t_inc-t_rep)
        t_p    = t_r + (t_paid-t_rep)
        N_I    = 0
        N_P    = 0
    return Reported, Closed, Inc, Paid, t_r, t_i, t_p, N_I, N_P
        
# ********************************************************

class Red_Claim(object):
    def __init__(self, Date_Sub=date_sub, Date_Acc=today, Year=sub_year, Closed=False, Inc=100., Paid=0., Lag_R=0., Lag_I=0., Lag_P=0., Nr_I=1, Nr_P=2): 
        self.kind       = "Red_Claim"
        self.Date_Sub   = Date_Sub
        self.Date_Acc   = Date_Acc
        self.N_years    = None
        self.Year       = Year
        self.Reported   = False
        
        self.N_L        = 1
        self.Closed     = Closed
        self.Inc        = Inc
        self.Paid       = Paid
        self.Claim_Freq = 1
        self.Clos_Freq  = Closed
        self.Lag_R      = Lag_R
        self.Lag_I      = Lag_I
        self.Lag_P      = Lag_P
        self.Nr_I       = Nr_I
        self.Nr_P       = Nr_P
        
        self.Max_Lag    = None
        self.sum_I      = None
        self.sum_P      = None
        self.N_Lag_R    = None
        self.I_Lag_I    = None
        self.I_Lag_P    = None
        self.sum_Nr_I   = None
        self.sum_Nr_P   = None
        
        self._complete_stats() 
        
    def _complete_stats(self):
        self.Max_Lag  = (self.Date_Sub - self.Date_Acc).days / days_per_year
        self.Reported = self.Lag_R <= self.Max_Lag 
        
        self.Lag_R    = max(0., min(self.Lag_R, self.Max_Lag))
        self.Lag_I    = min(self.Lag_I, self.Max_Lag)
        self.Lag_P    = max(0., min(self.Lag_P, self.Max_Lag))

        if self.Closed: 
            self.Paid = self.Inc
            self.Clos_Freq = 1
        self.sum_I    = self.Inc
        self.sum_P    = self.Paid
        self.N_Lag_R  = self.N_L * self.Lag_R
        self.I_Lag_I  = self.Inc * self.Lag_I
        self.I_Lag_P  = self.Inc * self.Lag_P
        self.sum_Nr_I = self.Nr_I
        self.sum_Nr_P = self.Nr_P
        
    def _init_random(self, mean, t_rep, t_inc, t_paid, n_inc, n_paid, Date_Sub=date_sub, Year=sub_year):
        self.Date_Sub = Date_Sub
        self.Date_Acc = dt.date(Year, 1, 1) + dt.timedelta(days=int(((dt.date(Year+1, 1, 1) - dt.date(Year, 1, 1))).days * np.random.rand()))
        self.Year = Year
        t_sub = (self.Date_Sub - self.Date_Acc).days / days_per_year
        Reported, Closed, Inc, Paid, t_r, t_i, t_p, N_I, N_P = get_reduced_claim(mean, t_rep, t_inc, t_paid, t_sub, n_inc, n_paid)
        self.Reported = Reported
        
        self.Closed   = Closed
        self.Inc      = Inc
        self.Paid     = Paid
        
        self.Lag_R    = t_r
        self.Lag_I    = t_i
        self.Lag_P    = t_p
        self.Nr_I     = N_I
        self.Nr_P     = N_P

        self._complete_stats() 
        
    def get_red_stats(self):
        return np.array([
            self.Claim_Freq,
            self.Clos_Freq,
            self.Inc,
            self.Paid,
            self.Lag_R,
            self.Lag_I - self.Lag_R,
            self.Lag_P - self.Lag_R,
            self.Nr_I,
            self.Nr_P
            ])

    def Print(self):
        print('Nr years =', self.N_years) 
        print('Sub Date =', self.Date_Sub) 
        print('Ref year =', self.Year)
        print('Acc Date =', self.Date_Acc) 
        print('Count    =', self.N_L)
        print('Closed   =', self.Closed)
        print('Incurred =', self.Inc  / Units_I) 
        print('Paid     =', self.Paid / Units_I) 
        print('Rep Lag  =', self.Lag_R)
        print('Inc Lag  =', self.Lag_I)
        print('Paid Lag =', self.Lag_P)
        print('Nr Inc   =', self.Nr_I)
        print('Nr Paid  =', self.Nr_P)
        print()
        
class Red_Ann(Red_Claim):
    def __init__(self): 
        self.kind       = "Red_Annual"

        self.Date_Sub   = None
        self.Year       = None
        
        self.Date_Acc   = None
        self.N_years    = 1
        self.N_L        = 0
        self.Closed     = 0
        self.Inc        = 0.
        self.Paid       = 0.
        
        self.sum_I      = 0.
        self.sum_P      = 0.
        self.N_Lag_R    = 0.
        self.I_Lag_I    = 0.
        self.I_Lag_P    = 0.
        self.sum_Nr_I   = 0
        self.sum_Nr_P   = 0
        
        self.Claim_Freq = 0.
        self.Clos_Freq  = 0.
        self.Lag_R      = 0.
        self.Lag_I      = 0.
        self.Lag_P      = 0.
        self.Nr_I       = 0
        self.Nr_P       = 0

    def _complete_stats(self):
        if self.N_years > 0:
            self.Claim_Freq = self.N_L / self.N_years
            self.Clos_Freq  = self.Closed / self.N_years
        if self.N_L > 0:
            self.Inc   = self.sum_I    / self.N_L
            self.Paid  = self.sum_P    / self.N_L
            self.Lag_R = self.N_Lag_R  / self.N_L
            self.Nr_I  = self.sum_Nr_I / self.N_L
            self.Nr_P  = self.sum_Nr_P / self.N_L
        if self.sum_I > 0:
            self.Lag_I = self.I_Lag_I  / self.sum_I
            self.Lag_P = self.I_Lag_P  / self.sum_I

    def _aggregate(self, L):        
        self.N_L      += L.N_L
        self.Closed   += L.Closed
        self.Inc      += L.Inc
        self.Paid     += L.Paid
        
        self.sum_I    += L.sum_I
        self.sum_P    += L.sum_P
        self.N_Lag_R  += L.N_Lag_R
        self.I_Lag_I  += L.I_Lag_I
        self.I_Lag_P  += L.I_Lag_P
        self.sum_Nr_I += L.sum_Nr_I
        self.sum_Nr_P += L.sum_Nr_P
        
        self._complete_stats()
        
    def add_Claim(self, L):
        Add_OK = False
        try:
            if L.kind in ["Red_Claim", "Red_Annual"]:
                if self.N_L == 0:
                    self.Year = L.Year
                    self.Date_Sub = L.Date_Sub
                Add_OK = L.Reported and (self.Year == L.Year) and (self.Date_Sub == L.Date_Sub)
        except:
            print('Exception error: adding claim failed !')
        if Add_OK: self._aggregate(L)
        return Add_OK
            
class Red_Overall(Red_Ann):
    def __init__(self): 
        self.kind       = "Red_Overall"

        self.Date_Sub   = None
        self.Year       = set()
        self.Red_Ann    = {}
        
        self.Date_Acc   = None
        self.N_years    = 0
        self.N_L        = 0
        self.Closed     = 0
        self.Inc        = 0.
        self.Paid       = 0.
        
        self.sum_I      = 0.
        self.sum_P      = 0.
        self.N_Lag_R    = 0.
        self.I_Lag_I    = 0.
        self.I_Lag_P    = 0.
        self.sum_Nr_I   = 0
        self.sum_Nr_P   = 0
        
        self.Claim_Freq = 0.
        self.Clos_Freq  = 0.
        self.Lag_R      = 0.
        self.Lag_I      = 0.
        self.Lag_P      = 0.
        self.Nr_I       = 0
        self.Nr_P       = 0
    
    def add_Claim(self, L_Yr):
        Add_OK = False
        try:
            if L_Yr.kind in ["Red_Annual", "Red_Overall"]:
                if self.N_L == 0:
                    self.Date_Sub = L_Yr.Date_Sub
                Add_OK = self.Date_Sub == L_Yr.Date_Sub
        except:
            print('Exception error: adding annual claim failed !')
        if Add_OK: 
            if L_Yr.Year in self.Year:
                self.Red_Ann[L_Yr.Year]._aggregate(L_Yr)
            else:
                self.Year.add(L_Yr.Year)
                self.Red_Ann[L_Yr.Year] = L_Yr
                self.N_years = len(self.Year)
                self._aggregate(L_Yr)
        return Add_OK

# ******************************************************

