import numpy as np
import spharm
import random
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt    


from forced_barotropic_sphere.sphere import Sphere
from forced_barotropic_sphere.forcing import Forcing
from forced_barotropic_sphere.solver import Solver


d2s = 86400


def ensemble_forcing_sampling(nlat, nlon, dt, n_ens = 5, produce_plots=False):
    """
    Function designed to test how well a certain number of ensemble members samples a forcing. 
    Used to test whether more ensemble members are needed or not.
    
    inputs:
    nlat, nlon (int): number of latitude,longitude for sphere
    dt (int): time between integrations in seconds
    n_ens (int): number of ensemble members to integrate the forcing
    """
    T = 15*365*d2s #climatology run (10 yrs)
    Nt = T//dt
    st= Sphere(nlat,nlon)
    F = Forcing(st,dt,T)
    Si = F.generate_rededdy_start()
    fcli = F.generate_rededdy_tseries(A=1, Si=Si)
    
    T = 1*d2s #1 day run
    Nt = T//dt
    fens = np.empty((n_ens,Nt+2,nlat,nlon)) #ensemble samples
    
    for ee in tqdm(range(n_ens)): #generate each ensemble member individually and run barotropic model            
        st= Sphere(nlat,nlon)
        F = Forcing(st,dt,T)
        #note we initialize a different Si for each member so that we don't have to wait for members to decorrelate
        Si = F.generate_rededdy_start()
        fens[ee,:] = F.generate_rededdy_tseries(A=1, Si=Si)
        
        
    if produce_plots:
        
        fig,axs = plt.subplots(3,1,figsize=(15,5))
                        
    return fcli, fens


def climatological_spread(nlat,nlon,dt,A=5e-12, produce_plots=False): 
    
    #many shorter runs
    Nt = 7884//52*4 #number of integration steps (7884 = 1 yr)
    T= Nt*dt #total integration time
    print('integrating for ', T/86400, ' days, with a dt of ', dt/86400, ' days')
    
    ofreq=2
    nr = 500
    vort_std = np.empty((nr,nlat,nlon))
    theta_std = np.empty((nr,nlat,nlon))

    slns = []
    for nn in tqdm(range(nr)):
        st = Sphere(nlat,nlon, U=0.)
        F = Forcing(st,dt,T)
        Si = F.generate_rededdy_start()
        F.generate_rededdy_tseries(A=A, Si=Si)

        sln = Solver(st, forcing=F, ofreq=ofreq).integrate_dynamics(temp_linear=True, vort_linear=True)
        slns.append(sln.isel(time=slice(7884//52//ofreq*2,Nt//ofreq))) #we remove the first 2 week spin up period
        
    sln= xr.concat(slns, dim='runs')
    sln = sln.std(['runs','time'])
    
    if produce_plots:
        fig,axs = plt.subplots(3,1,figsize=(4,12))
        
        #plot 1 
        lats = [12,15,20]
        for ll in lats:
            ll=int(ll)
            axs[0].plot(sln.x.values,sln.theta.isel(y=ll), label= 'lat= '+str(round(sln.y.values[ll])))
            axs[0].axhline(sln.theta.isel(y=ll).mean(), linestyle='--', color='k',alpha=0.5)

        axs[0].set_xlim(0,360)
        axs[0].set_xlabel('lon')
        axs[0].set_ylabel('std(theta)')
        axs[0].legend()
        
        #plot 2
        cf=axs[1].contourf(sln.x.values,sln.y.values,sln.theta)
        axs[1].set_ylabel('lat')
        axs[1].set_xlabel('lon')
        plt.colorbar(cf, ax=axs[1], label = 'std(theta)')
        
        #plot 3
        axs[2].errorbar(sln.y.values,sln.theta.mean('x'), yerr=sln.theta.std('x'),color='k', ecolor='r')
        axs[2].set_ylabel('std(theta)')
        axs[2].set_xlabel('lat')
        axs[2].set_xlim(-90,90)
        axs[2].set_ylim(0.)
        plt.tight_layout()
        
        plt.show()
        
    return sln
        

    
        
    
        
        

 