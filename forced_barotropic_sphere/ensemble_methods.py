import numpy as np
import spharm
import random
from tqdm import tqdm
import xarray as xr

from forced_barotropic_sphere.sphere import Sphere
from forced_barotropic_sphere.solver import Solver
from forced_barotropic_sphere.forcing import Forcing

#cython C methods
#import sys
#sys.path.append("../")
#import pyximport
#pyximport.install(setup_args={"script_args" : ["--verbose"]})
#import bm_methods.bm_methods as bm_methods
#import forced_barotropic_sphere.bm_methods as bm_methods

def integrate_ensemble(nlat, nlon, dt, T, ofreq, ics=None, forcing_type='gaussian', n_ens = 5, linear=True, vortpert=0., thetapert=0., forcingpert=0.):
    """
    Method to generate an ensemble run of the forced barotropic vorticity equation, if desired these can share the same forcing term, or have a weak, ensemble-varying perturbation applied to the forcing. IC pertubations can also be applied to each member.
    """
    st= Sphere(nlat,nlon)
    F = Forcing(st,dt,T)
    
    #generate single forcing run to be used by each ensemble member
    if forcing_type=='gaussian': 
        F.generate_gaussianblob_tseries()
    if forcing_type=='stochastic_eddy':
        F.generate_stocheddy_tseries()
        
    if ics.any()==None:
        ics = np.array([np.zeros((nlat,nlon)), np.zeros((nlat,nlon))])
        
    for ee in range(n_ens): #generate each ensemble member individually and run barotropic model
        ics_e = np.array([ics[0] + np.random.normal(loc=0., scale = vortpert, size= ics[0].shape),
                                              ics[1] + np.random.normal(loc=0., scale = thetapert, size=ics[1].shape)])
            
        st= Sphere(nlat,nlon)
        st.set_ics(ics_e)
        
        F_e = F
        
        if ee==0: #'control run'
            solver = Solver(st, forcing=F, ofreq=ofreq) # no noise applied to forcing of control run
            slns = solver.integrate_dynamics(linear=linear)
            
        else: #each perturbed member
            F_e.forcing_tseries += generate_forcingnoise(F.forcing_tseries, forcingpert)
            solver = Solver(st, forcing = F_e, ofreq=ofreq)
            slns = xr.concat([slns,solver.integrate_dynamics(linear=linear)], "ens_mem")
            
    slns = slns.assign_coords(ens_mem= range(n_ens))
    
    return slns

def generate_forcingnoise(forcing, pert= 1e-11):
    """Generate small white noise perturbations to a forcing instance to simulate small scale processes for each ensemble mem."""
    return np.random.normal(loc=0., scale = pert, size= forcing.shape)



##+++bimodal specific functions+++
def fit_KDE(ensemble):
    """fit a KDE to a distribution and find the critical points"""
    #TODO: check with our bm conditions? dependent on ensemble size, potentially useful for FP/FN experiments, but we'll likely
    #use much larger ensembles for synthetic study
    bw = 0.45730505192732634 * np.std(ensemble)
    means,mins = bm_methods.root_finding(ensemble, bw)
    #binary_ensemble = np.zeros(ensemble.shape, dtype=int)
    bimodal= False
    if len(means) == 2:# and (4 < len(np.where(ensemble < mins[0])[0]) < 46) \
    #and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[0], ensemble, bw)) < 0.85) \
    #and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[1], ensemble, bw)) < 0.85):
        #binary_ensemble[ensemble > mins[0]] = 1
        bimodal=True
    return bimodal#, binary_ensemble

    
def find_bimodality(slns):
    """find all isntances in slns in which the ensemble is bimodal using KDE estimation + brute force root finding"""
    #TODO: this isn't pretty. no need to check all locations, especially if we end up being zonally symmetric.
    slns_arr = np.array(slns)
    bm_arr = np.array(slns.sel(ens_mem=0))
    for tt in tqdm(range(slns_arr.shape[1])):
        #we'll try and be clever for how we search for bm, only do where spread > 50% percentile
        spread_tt = np.std(slns_arr[:,tt], axis=0)
        spread_min = np.percentile(spread_tt, 75)
        
        for yy in range(slns_arr.shape[2]):
            for xx in range(slns_arr.shape[3]):
                if (spread_tt[yy,xx] < spread_min) | (spread_min<1e-5):
                    bm_arr[tt,yy,xx] = False
                    continue
                bm_arr[tt,yy,xx] = fit_KDE(slns_arr[:,tt,yy,xx])
    return bm_arr
    
        
    
        
        

 