#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:49:51 2021

@author: iair
"""

import datetime
import emcee
import logging
import numpy as np
import os, sys   
import pysynphot
import time
import warnings

from astropy.table import Table
from lightcurve_fitting import filters, lightcurve, models
from lightcurve_fitting.bolometric import calculate_bolometric
from lightcurve_fitting.lightcurve import LC
from matplotlib import pyplot as plt
from multiprocessing import Pool

cdbsbase = '/home/arcavi'
ncpu = 20 
realizations = 100

os.environ['PYSYN_CDBS']=cdbsbase+'/data/cdbs' 
os.environ["OMP_NUM_THREADS"] = "1"

logging.getLogger().setLevel(logging.ERROR)

##############################################################################

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

##############################################################################

def get_bandpassfile(filt):

    if len(filt) == 2: # Swift Filter
        bandpassfile = cdbsbase+'/data/filter_curves/UVOTfilters_Breeveld2011/UV'+filt+'_UVOT.txt'
    elif filt == filt.lower(): 
        bandpassfile = cdbsbase+'/data/cdbs/comp/nonhst/sdss_'+filt+'_005_syn.fits'
    elif filt in ['R','I']:
        bandpassfile = cdbsbase+'/data/cdbs/comp/nonhst/cousins_'+filt.lower()+'_004_syn.fits'
    elif filt in ['J','K']:
        bandpassfile = cdbsbase+'/data/cdbs/comp/nonhst/bessell_'+filt.lower()+'_003_syn.fits'
    elif filt in ['H']:
        bandpassfile = cdbsbase+'/data/cdbs/comp/nonhst/bessell_'+filt.lower()+'_004_syn.fits'
    else:
        bandpassfile = cdbsbase+'/data/cdbs/comp/nonhst/johnson_'+filt.lower()+'_004_syn.fits'
        
    return bandpassfile

##############################################################################

def log_flat_prior(theta):
    temperature, radius_in_rsun = theta
    if temperature > 1000 and temperature < 100000 and radius_in_rsun > 10 and radius_in_rsun < 1e6:
        return 0 # log(1)
    else:
        return -np.inf  # log(0)
    
def log_logflat_prior(theta):
    temperature, radius_in_rsun = theta
    if temperature > 1000 and temperature < 100000 and radius_in_rsun > 10 and radius_in_rsun < 1e6:
        return (-np.log10(temperature)) + (-np.log10(radius_in_rsun)) 
    else:
        return -np.inf  # log(0)
    
def log_flat_t_logflat_r_prior(theta):
    temperature, radius_in_rsun = theta
    if temperature > 1000 and temperature < 100000 and radius_in_rsun > 10 and radius_in_rsun < 1e6:
        return 0 + (-np.log10(radius_in_rsun)) 
    else:
        return -np.inf  # log(0)

##############################################################################

def log_likelihood(theta):
    temperature, radius_in_rsun = theta
    
    if temperature < 1000 or temperature > 100000 or radius_in_rsun < 10 or radius_in_rsun > 1e6:
        return -np.inf
    
    chi2 = []
    normalization = []

    bb = pysynphot.BlackBody(temperature) # For 1 Solar Radius at 1 kpc    
    
    for filtername in filternames:

        bp = bandpasses[filtername]        
        obs = pysynphot.Observation(bb, bp, binset=bp.wave)
        magsystem = magsys[filtername]
        expected_mag = obs.effstim(magsystem) - 2.5*np.log10((radius_in_rsun**2)*((1000.0/distance_in_pc)**2))
        
        # Convert to flux for chi2 measurement:
        measured_flux = 10**(-0.4*mags_pysynphot[filtername])
        measured_fluxe = abs(measured_flux*(-0.4)*np.log(10)*mages_pysynphot[filtername])
        expected_flux = 10**(-0.4*expected_mag) 
        chi2.append(((expected_flux-measured_flux)/measured_fluxe)**2)
        normalization.append(np.log(2 * np.pi * measured_fluxe**2))
    
    return -0.5 * np.sum(normalization + chi2)
    
##############################################################################

def log_posterior(theta):
    return log_logflat_prior(theta) + log_likelihood(theta)

##############################################################################
 
def mute():
    sys.stdout = open(os.devnull, 'w') 

##############################################################################

distance_in_pc = 10
dm = 5*np.log10(distance_in_pc/10)

if __name__ == '__main__':
    
    ndim = 2  # number of parameters in the model
    nwalkers = 16  # number of MCMC walkers
    burnin = 200 # burn in steps
    steps = 400 # steps after burn in  

    n = datetime.datetime.now()
    n_str = str(n).replace('-','').replace(' ','-').replace(':','').split('.')[0]
    
    path = 'runs/'
    
    allfilternames = {'landolt': [['B','V','R','I'],['U','B','V','R','I'],['W2','M2','W1','U','B','V','R','I']],
                      'sloan'  : [['g','r','i'],['u','g','r','i'],['W2','M2','W1','u','g','r','i']]}
        
    merrs = [0.05,0.1]
    
    temperatures = np.linspace(5000,80000,num=16)
    radius_in_rsun = 1000
    
    print(n_str)
    print()
    
    for filtsys in allfilternames:
        
        unifiedfilternames = np.array([])
        for f in allfilternames[filtsys]:
            unifiedfilternames = np.append(unifiedfilternames,f)
        unifiedfilternames = list(set(unifiedfilternames))
                
        # Get all the bandpasses we'll need for BB fits:
         
        # For LIGHTCURVE_FITTING
        filter_objects = []
        for filtname in unifiedfilternames:
            if len(filtname) == 2:
                filtname = 'UV'+filtname    
            filter_objects.append(filters.filtdict[filtname])
        for f in filter_objects:
            f.read_curve()
         
        # For PYSYNPHOT
        wls = []
        bandpasses = {}
        magsys = {}
        for filtname in unifiedfilternames:
            bp = pysynphot.FileBandpass(get_bandpassfile(filtname))
            bandpasses[filtname] = bp
            wls.append(bp.avgwave())
            if filtname == filtname.lower():
                magsys[filtname] = 'ABMag'
            else:
                magsys[filtname] = 'VegaMag'
            
        for merr in merrs:
        
            for temperature in temperatures:
                
                # Simulate the data - LIGHTCURVE_FITTING:    
                
                print('')
                print('Simulating {}K data in flux space'.format(temperature))

                L_nu = models.blackbody_to_filters(filter_objects, temperature/1e3, radius_in_rsun/1e3)
                true_bb_mags_griffin = {}
                true_bb_mags, _ = lightcurve.flux2mag(L_nu.T, zp=[f.M0 for f in filter_objects])
                true_bb_mags += dm
                for i,f in enumerate(unifiedfilternames):
                    true_bb_mags_griffin[f] = true_bb_mags[i]                
                
                # Simulate the data - PYSYNPHOT:    
                    
                print('Simulating {}K data in mag space'.format(temperature))
                bb = pysynphot.BlackBody(temperature)
                true_bb_mags_pysynphot = {}
                for f in unifiedfilternames:
                    bp = bandpasses[f]
                    obs = pysynphot.Observation(bb, bp, binset=bp.wave)
                    magsystem = magsys[f]
                    true_bb_mags_pysynphot[f] = obs.effstim(magsystem) - 2.5*np.log10((radius_in_rsun**2)*((1000.0/distance_in_pc)**2))
                
                for realization in range(realizations):
                
                    mags_griffin = {}
                    mages_griffin = {}    
                    mags_pysynphot = {}
                    mages_pysynphot = {}
                    
                    # Simulate magnitude errors and add them to simulated data
                    
                    e = np.random.normal(scale=merr,size=len(unifiedfilternames))
                    mag_errs = {unifiedfilternames[i]:e[i] for i in range(len(e))}
                    for f in unifiedfilternames:
                        mags_griffin[f] = true_bb_mags_griffin[f] + mag_errs[f]
                        mages_griffin[f] = merr
                        mags_pysynphot[f] = true_bb_mags_pysynphot[f] + mag_errs[f]
                        mages_pysynphot[f] = merr
                                  
                    for filternames in allfilternames[filtsys]:
   
                        print('{}, merr={}, {}K, Realization {}:'.format(''.join(filternames), merr, temperature, realization+1))    
                
                        results_filename = 'results_{}_{}_{}_{}.txt'.format(n_str,''.join(filternames),merr,realization+1)  
                    
                        # Fit the data - LIGHTCURVE_FITTING:
                            
                        print(' Fitting with lightcurve_fitting method...', end="")    
                        st = time.time()
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                        
                            # make LC object:
                            d = {'MJD':[], 'mag': [], 'dmag': [], 'filt': [], 'source': [], 'nondet': []}
                            for key in filternames:
                                d['MJD'].append(0)
                                d['mag'].append(mags_griffin[key])
                                d['dmag'].append(merr)
                                d['filt'].append(key)
                                d['source'].append(0)
                                d['nondet'].append(False)
                            lc = LC(d)
                            lc.filters_to_objects()
                        
                            with HiddenPrints():
                                t = calculate_bolometric(lc, 0, path.replace('/','-griffin'), nwalkers=nwalkers, burnin_steps=burnin, steps=steps) 
                            ftemp = t['temp_mcmc'][0]*1e3
                            ftempl = (t['temp_mcmc'][0]-t['dtemp0'][0])*1e3
                            ftemph = (t['temp_mcmc'][0]+t['dtemp1'][0])*1e3
                            fradius = t['radius_mcmc'][0]*1e3
                            fradiusl = (t['radius_mcmc'][0]-t['dradius0'][0])*1e3
                            fradiush = (t['radius_mcmc'][0]+t['dradius1'][0])*1e3
                            
                            s = '{} {} {} {} {} {} {} {} {}\n'.format(temperature,merr,''.join(filternames),ftempl,ftemp,ftemph,fradiusl,fradius,fradiush)
                            with open(path+'flux-method_'+results_filename,'a') as f: 
                                f.write(s)

                        print(' {:.2f}s'.format(time.time()-st))
                        
                        # Fit the data - PYSYNPHOT: 
                            
                        print(' Fitting with pysynphot method...', end="")  
                        st = time.time()
            
                        # initialize walkers
                        tempguess = np.max([5000,temperature+np.random.normal(scale=10000)])
                        initial_guess = [tempguess, 1000]
                        initial_spread = [10000, 200]
                        starting_positions = np.random.randn(nwalkers, ndim)*initial_spread+initial_guess
                        for i in range(len(starting_positions[:,0])):
                            if starting_positions[i,0] < 1000:
                                starting_positions[i,0] = 1000
                            if starting_positions[i,1] < 10:
                                starting_positions[i,0] = 10 
                        
                        with Pool(ncpu,initializer=mute) as pool:
                        
                            with HiddenPrints(), warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
                                state = sampler.run_mcmc(starting_positions, burnin)
                            
                            sampler.reset()
                            with HiddenPrints(), warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                state = sampler.run_mcmc(state, steps)
                        
                        chain_table = Table(sampler.flatchain, names=['T','R'])
                        
                        samples = sampler.chain[:, :, :].reshape((-1, ndim))
                        results = zip(*np.percentile(samples, [16, 50, 84],axis=0))
                        lresults = list(results)
                        results_line = np.reshape(lresults,[1,6])
                        
                        s = '{} {} {} {} {} {} {} {} {}\n'.format(temperature,merr,''.join(filternames),*results_line[0])
                        with open(path+'mag-method_'+results_filename,'a') as f: 
                            f.write(s)

                        print(' {:.2f}s'.format(time.time()-st))

    print()
    print(n_str)