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


def ensemble_forcing_sampling(M, dt, n_ens = 5, produce_plots=False):
    """
    Function designed to test how well a certain number of ensemble members samples a forcing. 
    Used to test whether more ensemble members are needed or not.
    
    inputs:
    nlat, nlon (int): number of latitude,longitude for sphere
    dt (int): time between integrations in seconds
    n_ens (int): number of ensemble members to integrate the forcing
    """
    T = 10*365*d2s #climatology run (10 yrs)
    Nt = T//dt
    st= Sphere(M)
    F = Forcing(st,dt,T)
    Si = F.generate_rededdy_start()
    fcli = F.generate_rededdy_tseries(A=1, Si=Si)
    
    T = d2s//6 #1 day run
    Nt = T//dt
    
    
    st= Sphere(M)
    F = Forcing(st,dt,T)
            
    nruns = 200
    #sampNum = range(5,n_ens+5,5)
    n_ens = [5,10,20,30,40,50,70,90,110,130,150,200,250,350,500]
    
    
    
    toplevel = np.empty((nruns,len(n_ens),st.nlat,st.nlon)) #ensemble samples
    
    for ei,ee in tqdm(enumerate(n_ens)):
        for nr in range(nruns):
            intlevel = np.empty((ee,Nt+2,st.nlat,st.nlon))

            for nn in range(ee):
                Si = F.generate_rededdy_start()
                intlevel[nn,:] = F.generate_rededdy_tseries(A=1, Si=Si)

            toplevel[nr,ei,:] = intlevel[:,-2].std(axis=0) #select random time and find std of ensemble
    
        
        
    if produce_plots:
        
        fig,axs = plt.subplots(3,1,figsize=(15,5))
                        
    return fcli, toplevel,n_ens


def climatological_spread(M, dt, A=5e-12, base_state='solid', temp_linear = True, vort_linear=True, produce_plots=False, **kwargs): 
    """
    Derive the climatology through an ensemble of 
    """
    
    #many shorter runs
    T = 365*d2s #total integration time
    Nt= T//dt #total integration time
    print('integrating for ', T/86400, ' days, with a dt of ', dt/86400, ' days')
    
    ofreq=2
    nr = 2
    
    st= Sphere(M)
    
    vort_std = np.empty((nr,st.nlat,st.nlon))
    theta_std = np.empty((nr,st.nlat,st.nlon))

    slns = []
    for nn in tqdm(range(nr)):
        st = Sphere(M, base_state=base_state)
        F = Forcing(st,dt,T)
        Si = F.generate_rededdy_start()
        F.generate_rededdy_tseries(A=A, Si=Si)

        sln = Solver(st, forcing=F, ofreq=ofreq, **kwargs).integrate_dynamics(temp_linear=temp_linear, vort_linear=vort_linear)
        slns.append(sln.isel(time=slice(7884//52//ofreq*2,Nt//ofreq)).std('time')) #remove the first 2 week spin up period
        
    sln= xr.concat(slns, dim='runs')
    
    if produce_plots:
        fig,axs = plt.subplots(3,1,figsize=(4,12))

        #plot 1 
        lats = [12,15,20]
        for ll in lats:
            ll=int(ll)
            axs[0].plot(sln.x.values,sln.theta.isel(y=ll).mean('runs'), label= 'lat= '+str(round(sln.y.values[ll])))
            axs[0].axhline(sln.theta.isel(y=ll).mean(['runs','x']), linestyle='--', color='k',alpha=0.5)

        axs[0].set_xlim(0,360)
        axs[0].set_xlabel('lon')
        axs[0].set_ylabel('std(theta)')
        axs[0].legend()

        #plot 2
        cf=axs[1].contourf(sln.x.values,sln.y.values,sln.theta.mean('runs'))
        axs[1].set_ylabel('lat')
        axs[1].set_xlabel('lon')
        axs[1].set_ylim(0,90)
        plt.colorbar(cf, ax=axs[1], label = 'std(theta)')

        #plot 3
        axs[2].errorbar(sln.y.values,sln.theta.mean(['x','runs']), yerr=sln.theta.mean('x').std('runs'),color='k', ecolor='r')
        axs[2].set_ylabel('std(theta)')
        axs[2].set_xlabel('lat')
        axs[2].set_xlim(0,90)
        axs[2].set_ylim(0.)
        plt.tight_layout()

        plt.show()

    return sln
        
# def climatological_spreadv2(nlat,nlon,dt,A=5e-12, produce_plots=False): 
    
#     #many shorter runs
#     Nt = 7884*10 #number of integration steps (7884 = 1 yr)
#     T= Nt*dt #total integration time
#     print('integrating for ', T/86400, ' days, with a dt of ', dt/86400, ' days')
    
#     ofreq=2
#     nr = 50
#     vort_std = np.empty((nr,nlat,nlon))
#     theta_std = np.empty((nr,nlat,nlon))
    
#     st = Sphere(nlat,nlon, U=0.)
#     F = Forcing(st,dt,T)
#     Si = F.generate_rededdy_start()
#     F.generate_rededdy_tseries(A=A, Si=Si)

#     sln = Solver(st, forcing=F, ofreq=ofreq).integrate_dynamics(temp_linear=True, vort_linear=True)

#     slns = []
#     nsamp=10000
#     print(sln.time.size)
#     for nn in tqdm(range(nr)):
#         #print(np.random.randint(0, sln.time.size, nsamp))
#         #print(sln.isel(time=np.random.randint(7884//52//ofreq*2, sln.time.size, nsamp)))
#         #raise ValueError()
#         si=sln.isel(time=np.random.randint(7884//52//ofreq*2, sln.time.size-1, nsamp)).std('time')
#         #print(si)
#         slns.append(si)
#         #slns.append(sln.isel(time=slice(7884//52//ofreq*2,Nt//ofreq)).std('time')) #we remove the first 2 week spin up period
#         #slns.append(sln.isel(time=Nt//ofreq-1)) #we remove the first 2 week spin up period

        
#     sln= xr.concat(slns, dim='runs')
#     #sln = sln.std(['time'])
    
#     return sln
    
        
    
        
        

 