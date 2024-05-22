import numpy as np
import random
from tqdm import tqdm
import xarray as xr

import cython_routines as cr

from model.sphere import Sphere
from model.solver import Solver
from model.forcing import Forcing
from utils import config, constants

#cython C methods
#import sys
#import pyximport
#pyximport.install(setup_args={"script_args" : ["--verbose"]})
#import bm_methods.bm_methods as bm_methods
#import forced_barotropic_sphere.bm_methods as bm_methods
d2s = 86400

def integrate_ensemble(st, T, **kwargs):
    """
    Function to generate an ensemble of forced barotropic model,
    these are initialized with the same forcing term, which decorrelates after about a week.
    """
    M = config.M
    nlon = 2*M + 1
    nlat = M + 1
    
    n_ens         = kwargs.get('ensemble_size', config.DEFAULT_ENS_SIZE)
    ens_vpert     = kwargs.get('ensemble_vortp', 0.)
    ens_tpert     = kwargs.get('ensemble_thetap', 0.)
    ens_fpert     = kwargs.get('ensemble_forcep', 0.)
    share_forcing = kwargs.get('share_forcing', True)
        
    slns=[] #storage for each ensemble sln.
    
    if share_forcing:
        Si = Forcing(sphere=st).evolve_rededdy() #get starting position for red eddy forcing
        kwargs['red_eddy_start'] = Si #add start, then ensemble members decouple over time
    
    for ee in tqdm(range(n_ens)): #generate each ensemble member individually and run barotropic model
        
        ics_e = np.array([st.vortp  + np.random.normal(loc=0., scale = ens_vpert, size= st.vortp.shape),
                          st.thetap + np.random.normal(loc=0., scale = ens_tpert, size= st.thetap.shape)])
            
        st_e= Sphere(base_state=st.base_state)
        st_e.set_ics(ics_e)
        
        solver = Solver(st_e, T=T, **kwargs)
        slns.append(solver.integrate_dynamics())
        

    slns = xr.concat(slns,dim= "ens_mem")
    
    return slns

#+++
def generate_ensemble_forcing(nlat, nlon, dt, T, n_ens = 5):
    
    """
    This generates an ensemble of the red eddy forcing function
    with A=1; used to test how the forcing decorrelates over time
    """
    
    Nt = T//dt
    f = np.empty((n_ens,Nt+2,nlat,nlon))
    
    for ee in tqdm(range(n_ens)): #generate each ensemble member individually          
        st= Sphere(nlat,nlon)
        
        if ee==0: #'control run'
            F = Forcing(st,dt,T)
            Si = F.generate_rededdy_start()   
            f[ee,:] = F.generate_rededdy_tseries(A=1, Si=Si)
            
        else: #each perturbed member
            Si = F.generate_rededdy_start()
            f[ee,:] = F.generate_rededdy_tseries(A=1, Si=Si)
                        
    return f
   

##+++bimodal specific functions+++
def fit_KDE(ensemble):
    """fit a KDE to a distribution and find the critical points"""
    #TODO: check with our bm conditions? dependent on ensemble size, potentially useful for FP/FN experiments, but we'll likely
    #use much larger ensembles for synthetic study
    bw = 0.45730505192732634 * np.std(ensemble)
    Ms,ms = cr.bm_methods.recursive_rootfinding(np.min(ensemble),np.max(ensemble),
                                                tol=bw/10., fargs=(ensemble,bw))
    return Ms,ms
    #means,mins = bm_methods.root_finding(ensemble, bw)
    #binary_ensemble = np.zeros(ensemble.shape, dtype=int)
#     bimodal= False
#     if len(means) == 2:# and (4 < len(np.where(ensemble < mins[0])[0]) < 46) \
#     #and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[0], ensemble, bw)) < 0.85) \
#     #and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[1], ensemble, bw)) < 0.85):
#         #binary_ensemble[ensemble > mins[0]] = 1
#         bimodal=True
#     return bimodal#, binary_ensemble

    
def find_bimodality(slns):
    """find all isntances in slns in which the ensemble is bimodal using KDE estimation + brute force root finding"""
    #TODO: this isn't pretty. no need to check all locations
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
    
    
    
#+++ Generate additional noise functions+++
def generate_randomnoise(forcing, pert= 1e-12):
    """Generate small white noise perturbations to a forcing instance to simulate small scale processes for each ensemble mem."""
    return np.random.normal(loc=0., scale = pert, size= (forcing.Nt+2,len(forcing.sphere.glat),len(forcing.sphere.glon)))

def generate_eddynoise(forcing, Apert = 1e-12):
    """ Genereate noise perturbations in the form of eddies, rather than pure white noise at each gridpoint
    """
    noise_tseries = np.zeros((forcing.Nt+2,len(forcing.sphere.glat),len(forcing.sphere.glon)))
                                   
    stir_lat = 40. #degrees
    stir_width = 10. #degrees
    lat_mask = np.exp(- ((np.abs(forcing.sphere.glats)-stir_lat)/stir_width)**2 ) #eddy stirring location

    decorr_timescale = 2*d2s #1 days, faster decorrelation timescale
    Q = np.random.uniform(low=-Apert,high=Apert, size = (forcing.Nt+2,forcing.sphere.nspecindx)) #amplitude for each spectral wavenumber

    stirwn = np.where((8<=forcing.sphere.specindxn) & (forcing.sphere.specindxn<=12),1,0) #force over a larger set of wavenumbers

    Si = Q[0,:]*stirwn
    for tt in range(1,forcing.Nt+1):
        Qi = Q[tt,:]*stirwn
        Si = Qi*(1-np.exp(-2*forcing.dt/decorr_timescale))**(0.5) + np.exp(-forcing.dt/decorr_timescale)*Si

        noise_tseries[tt,:]= forcing.sphere.to_grid(Si)*lat_mask #apply lat mask for midlats.

    return noise_tseries

        
    
        
        

 