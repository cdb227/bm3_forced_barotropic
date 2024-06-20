import numpy as np
import xarray as xr
from tqdm import tqdm


##physical constants
s2d = 1/3600/24    # seconds to days
s2h = 1/3600       # seconds to hours
a = 6371e3         # Radius of the earth in m
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
R = 287.           # Dry gas constant in J/K/kg
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees


def calculate_trajectories(ds, x0, t0=0., rvs=False):
    '''Calculates horizontal trajectories in time. Specify winds ``ds``, and the initial
    coordinates ``x0`` and times ``t0`` for the trajectories.
    rvs = True traces parcels back in time.
    Integrates trajectories until bounds of ds'''
    
    t0 = [t0] 
    T = ds.time.data[-1]-t0
    if rvs:
        T = t0-ds.time.data[0]


    dt=ds.time[1].data-ds.time[0].data 
    tstep = dt #by default we'll use a dt of 1/5 our integration dt

    Nt = int((T / tstep).item(0))
    dt = tstep.astype('d')

    # To calculate trajectories, we use time-evolving winds
    dts = np.arange(0, T, tstep)
    if len(dts)>Nt:
        dts=dts[:-1]
    #print(T, dts, dt, Nt)
    #print (ts.shape, dts.shape)
    if rvs:
        dts*=-1
        dt*=-1
        tstep*=-1

    x0 = np.array(x0)
    t0 = np.array(t0)
    
    Ntraj = x0.shape[0]
 
    # Set up arrays for holding trajectory data
    xs = np.zeros((Nt, 2, Ntraj), 'd')
    ts = np.full ((Nt, Ntraj), t0)
    
    xs[0, :] = x0.transpose()
    ts       += dts.reshape(-1, 1)
    
    print('Integrating %d trajectories for %s.' % (Ntraj, T*s2d)) 
    for n in range(Ntraj):
        print ('  %d. From %g E, %g N at %s.' % (n + 1, xs[0, 0, n], xs[0, 1, n], str(ts[0, n])))
    
    tmin = np.min(ts)
    tmax = np.max(ts)
    if (tmin < ds.time.data[0]) or ( tmax > ds.time.data[-1]): 
        raise ValueError("time not covered by the dataset ")
        
    # Convert cartesian velocities to angular velocities
    lamdot = r2d * ds.u / (a * np.cos(d2r * ds.y))
    phidot = r2d * ds.v /  a
    
    #fixes wraparound issue
    #lamdot = xr.concat([lamdot, lamdot.isel(lon=slice(0,1)).assign_coords(lon=[180])], dim='lon')
    #phidot = xr.concat([phidot, phidot.isel(lon=slice(0,1)).assign_coords(lon=[180])], dim='lon')
    
    # This is a helper function that interpolates the gridded wind
    # to a specific place and time
    def interp_winds(xi, ti):
        #x = xr.DataArray(((xi[0, :] + 180) % 360) - 180, dims = 'n')
        x= xr.DataArray(  xi[0, :]                    , dims = 'n')
        y = xr.DataArray(  xi[1, :]                    , dims = 'n')
        tm  = xr.DataArray(  ti[:]                       , dims = 'n')

        u = lamdot.interp(x = x, y = y, time = tm,kwargs={"fill_value":None}).data
        v = phidot.interp(x = x, y = y, time = tm,kwargs={"fill_value":None}).data

        return np.array([u, v])
                         
    # This loop calculates the trajectories. Each step, the horizontal
    # wind is interpolated to the current position of the fluid parcel (X0)
    # The fluid parcel is then moved by a distance dx = u * dt, except
    # that a more accurate estimate is used for dx (fourth-order Runge-Kutta)
    for n in tqdm(range(Nt - 1)):
        x = xs[n, :, :]
        t = ts[n, :]

        dx1 = dt * interp_winds(x,           t            )
        dx2 = dt * interp_winds(x + 0.5*dx1, t + 0.5*tstep)
        dx3 = dt * interp_winds(x + 0.5*dx2, t + 0.5*tstep)
        dx4 = dt * interp_winds(x +     dx3, t +     tstep)

        xs[n+1, :, :] = x + 1/6. * (dx1 + 2.*dx2 + 2.*dx3 + dx4)
        
    print("Completed %d of %d timesteps." % (n + 2, Nt))

    # Wrap longitudes to lie between -180 E and 180 W
    xs[:, 0, :] = ((xs[:, 0, :]) + 180) % 360 - 180
    
    # Return trajectory timesteps and lon,lat positions
    return ts, xs



def ens_calculate_trajectories(ds, x0, t0=0., rvs=False, tstep_factor = 2):
    '''Calculates horizontal trajectories in time. Specify winds ``ds``, and the initial
    coordinates ``x0`` and times ``t0`` for the trajectories.
    rvs = True traces parcels back in time.
    Integrates trajectories until bounds of ds'''
    
    t0 = [t0] 
    T = ds.time.data[-1]-t0
    if rvs:
        T = t0-ds.time.data[0]

    dt=ds.time[1].data-ds.time[0].data
    tstep = dt*tstep_factor #by default we'll use a dt of 1/5 our integration dt

    Nt = int((T / tstep).item(0))
    dt = tstep.astype('d')

    # To calculate trajectories, we use time-evolving winds
    dts = np.arange(0, T, tstep)
    #print(dts)
    if len(dts)>Nt:
        dts=dts[:-1]
        
    if rvs:
        dts*=-1
        dt*=-1
        tstep*=-1

    x0 = np.array(x0)
    t0 = np.array(t0)
    
    Ntraj = x0.shape[0]
 
    nens = len(ds.ens_mem)
    # Set up arrays for holding trajectory data
    xs = np.zeros((Nt, 2, nens, Ntraj), 'd')
    ts = np.full ((Nt, nens, Ntraj), t0)
    
    for n in range(nens):
        xs[0, :,n,:] = x0.transpose()
        ts[:,n,:]       += dts.reshape(-1, 1)
    
    print('Integrating %d trajectories for %s.' % (Ntraj, T)) 
    #for n in range(Ntraj):
    #    print ('  %d. From %g E, %g N at %s.' % (n + 1, xs[0, 0, n], xs[0, 1, n], str(ts[0, n])))
    
    tmin = np.min(ts)
    tmax = np.max(ts)
    if (tmin < ds.time.data[0]) or ( tmax > ds.time.data[-1]): 
        raise ValueError("time not covered by the dataset ")
        
    # Convert cartesian velocities to angular velocities
    lamdot = r2d * ds.u / (a * np.cos(d2r * ds.y))
    phidot = r2d * ds.v /  a
    
    #fixes wraparound issue
    #lamdot = xr.concat([lamdot, lamdot.isel(lon=slice(0,1)).assign_coords(lon=[180])], dim='lon')
    #phidot = xr.concat([phidot, phidot.isel(lon=slice(0,1)).assign_coords(lon=[180])], dim='lon')    
    
    # This is a helper function that interpolates the gridded wind
    # to a specific place and time
    def interp_winds(xi, ti):
        #x = xr.DataArray(((xi[0, :] + 180) % 360) - 180, dims = ['ens_mem','n'])
        x= xr.DataArray(  xi[0, :]                    , dims = ['ens_mem','n'])
        y = xr.DataArray(  xi[1, :]                    , dims = ['ens_mem','n'])
        tm  = xr.DataArray(  ti[:]                       , dims = ['ens_mem','n'])
        
        #print(lat, lon, tm)
        u = lamdot.interp(y = y, x = x, time = tm,kwargs={"fill_value": None}).data
        v = phidot.interp(y = y, x = x, time = tm,kwargs={"fill_value": None}).data

        return np.array([u, v])
                         
    # This loop calculates the trajectories. Each step, the horizontal
    # wind is interpolated to the current position of the fluid parcel (X0)
    # The fluid parcel is then moved by a distance dx = u * dt, except
    # that a more accurate estimate is used for dx (fourth-order Runge-Kutta)
    for n in tqdm(range(Nt - 1)):
        x = xs[n, :, :]
        t = ts[n, :]

        dx1 = dt * interp_winds(x,           t            )
        dx2 = dt * interp_winds(x + 0.5*dx1, t + 0.5*tstep)
        dx3 = dt * interp_winds(x + 0.5*dx2, t + 0.5*tstep)
        dx4 = dt * interp_winds(x +     dx3, t +     tstep)

        xs[n+1, :, :] = x + 1/6. * (dx1 + 2.*dx2 + 2.*dx3 + dx4)
        
    print("Completed %d of %d timesteps." % (n + 2, Nt))

    # Wrap longitudes to lie between -180 E and 180 W
    xs[:, 0, :] = ((xs[:, 0, :]) + 180) % 360 - 180
    
    # Return trajectory timesteps and lon,lat positions
    return ts, xs



