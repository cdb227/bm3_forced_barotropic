import numpy as np
import xarray as xr
import random

import cartopy                   
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point

import matplotlib as mpl          
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt    
import matplotlib.animation as manim
import matplotlib.ticker as mticker
import matplotlib.path as mpath

s2d = 1/86400.

#+++Simple plot modifications+++#
def add_cyc_point(var):
    """
    Add cyclic point to xarray object with coordinate x
    """
    lon_idx = var.dims.index('x')
    wrap_data, wrap_lon = add_cyclic_point(var.values, coord=var.coords['x'], axis=lon_idx)
    return wrap_data, wrap_lon

def make_ax_circular(ax):
    """
    Make an axes circular, useful for polar stereographic projections
    """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center) 
    ax.set_boundary(circle, transform=ax.transAxes)
    return ax

def add_gridlines(ax):
    """
    Add gridlines to plot
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels = False
    #gl.xlabel_bottom = False
    gl.xlines = False
    gl.ylocator = mticker.FixedLocator([45,60,75])
    gl.xlocator = mticker.FixedLocator([])
    #gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    return ax


#+++ Plotting routines for single model runs
def plot_overview(sln, levels=[None,None], var=['zetap','theta'], proj=ccrs.NorthPolarStereo()):
    
    """
    Plot vorticity and theta (or any two variables of interest) on 2 panel plot
    """
    label_dict = {'vort': r"$\zeta$", 'vortp': r"$\zeta'$", 'theta':r"$\theta$", 'thetap':r"$\theta'$"}
    unit_dict = {'vort': r'(x$10^5$ s$^{-1}$)', 'vortp': r'(x$10^5$ s$^{-1}$)', 'theta': r"(K)", 'thetap': r"(K)"}
                  
    assert set(var).issubset(label_dict.keys()),  "please choose two from the following: " +str(list(label_dict.keys()))
    
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': proj}, figsize=(7,5), sharex=True, sharey=True)
    sln = xr.concat([sln, sln.isel(x=slice(0,1)).assign_coords(x=[360])], dim='x')
    
    axs[0].set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
        
    #temporary addition due to cartopy's bug with certain contourf levels not showing
    cmap = plt.colormaps['bwr']
    norm = BoundaryNorm(levels[0], ncolors=cmap.N, clip=True)
           
    cf=axs[0].pcolormesh(sln.x,sln.y,sln[var[0]]*1e5, transform=ccrs.PlateCarree(), cmap = cmap, norm=norm)
    axs[0].set_title(label_dict[var[0]])
    plt.colorbar(cf,ax=axs[0],orientation='horizontal', label = unit_dict[var[0]])
    
    
    axs[1].set_title(label_dict[var[1]])
    cf= axs[1].contourf(sln.x,sln.y,sln[var[1]], levels=levels[1], extend='both',cmap = 'RdBu_r',
                        transform=ccrs.PlateCarree())
    cbar=plt.colorbar(cf,ax=axs[1],orientation='horizontal', label = unit_dict[var[1]])
    plt.setp(cbar.ax.get_xticklabels()[::2], visible=False)

    
    for ax in axs:
        #make axes circular, add gridlines, wind vectors and title
        thin=4
        if proj==ccrs.NorthPolarStereo():
            make_ax_circular(ax)
        add_gridlines(ax)
        ax.text(0.5, -0.1, 't = {:.0f} days'.format(sln.time.data*s2d), horizontalalignment='center',
             verticalalignment='top', transform=ax.transAxes)
        q=ax.quiver(sln.x.data[::thin],sln.y.data[::thin], sln.u.data[::thin,::thin], sln.v.data[::thin,::thin],
                    width=0.01, transform=ccrs.PlateCarree(), scale=60, scale_units='inches')
        ax.quiverkey(q, X=0.9, Y=1.0, U=10,label='10 m/s', labelpos='N')
    
    return axs


#+++Plotting routines for ensembles+++#
def plot_theta_ensspread(sln, levels=None, proj = ccrs.NorthPolarStereo(), ax=None, colorbar=True):
    """
    Plot ensemble spread (defined as 1 std of ensemble) of theta
    """
    if ax==None:
        f = plt.figure(figsize = (5, 5))
        ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    wrap_data, wrap_lon = add_cyc_point(sln.theta.std('ens_mem'))
    ax.set_title(r"$\theta'$")
        
    cf= ax.contourf(wrap_lon, sln.y.values, wrap_data, extend='max', levels=levels,transform=ccrs.PlateCarree(), cmap = 'Blues')
    if colorbar:
        plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'Ens. Std. (K)')
    ax=make_ax_circular(ax)
    ax=add_gridlines(ax)
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    return ax

def plot_zeta_ensspread(sln, levels=None, proj = ccrs.NorthPolarStereo(), ax=None, colorbar=True):
    """
    Plot ensemble spread (defined as 1 std of ensemble) of zeta
    """
    if ax==None:
        f = plt.figure(figsize = (5, 5))
        ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    wrap_data, wrap_lon = add_cyc_point(sln.vortp.std('ens_mem'))
    ax.set_title(r"$\zeta'$")
    
     #temporary addition due to cartopy's bug with certain contourf levels not showing
    cmap = plt.colormaps['bwr']
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        
    cf= ax.pcolormesh(wrap_lon, sln.y.values, wrap_data*1e5, transform=ccrs.PlateCarree(), cmap = cmap, norm=norm)
    if colorbar:
        plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'Ens. Std. ($\times$10$^5$ s$^{-1}$)')
    ax=make_ax_circular(ax)
    ax=add_gridlines(ax)
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    return ax

def plot_ensemble_overview(slns, levels=[None,None,None], proj=ccrs.NorthPolarStereo(), perturbation=[False,False]):
    """
    Plot vorticity and theta of randomly selected member and then the ensemble spread of theta
    """
    fig, axs = plt.subplots(1,3, subplot_kw={'projection': proj}, figsize=(9,5))
    sln = slns.isel(ens_mem=random.randint(0,len(slns['ens_mem'])-1)) #select random member to plot
    
    plot_vort(sln,ax=axs[0], levels=levels[0], perturbation=perturbation[0])
    add_windvecs(axs[0], sln)
    
    plot_theta(sln,ax=axs[1], levels=levels[1], perturbation=perturbation[1])
    add_windvecs(axs[1], sln)
    
    plot_theta_ensspread(slns, ax=axs[2], levels=levels[2])
    return fig,axs

    
def plot_bm_occurrence(slns,bmocc, levels=np.arange(0,0.101,0.02), proj=ccrs.NorthPolarStereo()):
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    frac = np.mean(bmocc,axis=0)
    cf= ax.contourf(slns.x.values, slns.y.values, frac, extend='max', levels=levels,transform=ccrs.PlateCarree(), cmap = 'Greens')
    plt.colorbar(cf,orientation='horizontal', label = 'bm fraction')
    ax=make_ax_circular(ax)
    ax=add_gridlines(ax)
    return f,ax

####
#+++animations+++#
####
    
def ensspread_point_animation(ds, xy, levels, proj=ccrs.NorthPolarStereo(), fr=4, filename='./images/ensspread_point.gif'):
    """
    Plot the evolution of ensemble spread of the temperature field and the ensemble at a discrete point
    """
    
    skip = int(1. / (fr * (ds.time[1] - ds.time[0])))
    if skip < 1: skip = 1
   
    frames = ds.time[::skip]
    
    fig = plt.figure(figsize=(5,3.5))
    ax1 = plt.subplot(1,2,1, projection = proj) #axes for map
    ax2 = plt.subplot(1,2,2) #axes for discrete point
    
    plot_theta_ensspread(ds.sel(time=0),ax=ax1, levels=levels)
    ax1.scatter(x=xy[0], y=xy[1], transform = ccrs.PlateCarree())
    
    ax2.set_xlim(0,frames[-1]*s2d)
    
    ym,yM = (np.min(ds.theta.sel(x=xy[0], y=xy[1], method='nearest')) -1, np.max(ds.theta.sel(x=xy[0], y=xy[1], method='nearest'))+1)
    ax2.set_ylim(ym,yM)
    ax2.yaxis.tick_right()
    ax2.set_xlabel('time (days)')
    ax2.set_ylabel('Temperature (K)')
    
    z=0
    def anim(t): 

        ax1.cla()
        ax2.cla()
        
        plot_theta_ensspread(ds.sel(time=t),ax=ax1, levels=levels, colorbar=False)
        ax1.scatter(x=xy[0], y=xy[1], transform = ccrs.PlateCarree(), color = 'tab:red', zorder=200)
        
        idx = np.where(frames==t)[0][0]
        z=idx+1
        
        X= frames[:z]*s2d
        ax2.plot(X, np.array(ds.theta.sel(time=frames[:z]).sel(x=xy[0], y=xy[1], method='nearest')).T, color='tab:blue')
        ax2.set_ylim(ym,yM)
        ax2.set_xlim(0,frames[-1]*s2d)
        ax2.yaxis.tick_right()
        ax2.set_xlabel('time (days)')
        ax2.set_ylabel('Temperature (K)')
        
        plt.ion()
        plt.draw()
    anim = manim.FuncAnimation(fig, anim, frames, repeat=False)
    
    anim.save(filename, fps=6, codec='h264', dpi=120)
    
    
def ensspread_animation(ds, xy, levels, proj=ccrs.NorthPolarStereo(), fr=4, filename='./images/ensspread_evolution.gif'):
    """
    Plot the evolution of ensemble spread of the temperature and vorticity field
    """
    
    skip = int(1. / (fr * (ds.time[1] - ds.time[0])))
    if skip < 1: skip = 1
   
    frames = ds.time[::skip]
    
    fig = plt.figure(figsize=(6,6))
    ax1 = plt.subplot(2,2,1, projection = proj) #axes for map
    ax2 = plt.subplot(2,2,2, projection = proj)
    ax3 = plt.subplot(2,2,3)#axes for discrete point
    ax4 = plt.subplot(2,2,4)
    
    plot_theta_ensspread(ds.sel(time=0),ax=ax1, levels=levels[0])
    ax1.scatter(x=xy[0], y=xy[1], transform = ccrs.PlateCarree())
    
    plot_zeta_ensspread(ds.sel(time=0), ax=ax2, levels=levels[1])
    ax2.scatter(x=xy[0], y=xy[1], transform = ccrs.PlateCarree())
    
    ax3.set_xlim(0,frames[-1]*s2d)
    ax4.set_xlim(0,frames[-1]*s2d)
    
    #ym1,yM1 = (np.min(ds.theta.sel(x=xy[0], y=xy[1], method='nearest')) -1, np.max(ds.theta.sel(x=xy[0], y=xy[1], method='nearest'))+1)
    ym1, yM1 = (0, np.max(ds.theta.sel(x=xy[0], y=xy[1], method='nearest').std('ens_mem')))
    ax3.set_ylim(ym1,yM1)
    ax3.yaxis.tick_right()
    ax3.set_xlabel('time (days)')
    ax3.set_ylabel('Temperature (K)')
    
    #ym2,yM2 = (np.min(ds.vort.sel(x=xy[0], y=xy[1], method='nearest')), np.max(ds.vort.sel(x=xy[0], y=xy[1], method='nearest')))
    ym2,yM2= (0, np.max(ds.vort.sel(x=xy[0], y=xy[1], method='nearest').std('ens_mem')))
    ax4.set_ylim(ym2,yM2)
    ax4.yaxis.tick_right()
    ax4.set_xlabel('time (days)')
    ax4.set_ylabel(r'Zeta (s$^{-1}$)')
    ax4.yaxis.set_label_position("right")
    
    z=0
    def anim(t): 

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        
        plot_theta_ensspread(ds.sel(time=t),ax=ax1, levels=levels[0], colorbar=False)
        ax1.scatter(x=xy[0], y=xy[1], transform = ccrs.PlateCarree(), color = 'tab:red', zorder=200)
        
        plot_zeta_ensspread(ds.sel(time=t),ax=ax2, levels=levels[1], colorbar=False)
        ax2.scatter(x=xy[0], y=xy[1], transform = ccrs.PlateCarree(), color = 'tab:red', zorder=200)
        
        idx = np.where(frames==t)[0][0]
        z=idx+1
        
        X= frames[:z]*s2d
        #ax3.plot(X, np.array(ds.theta.sel(time=frames[:z]).sel(x=xy[0], y=xy[1], method='nearest')).T, color='tab:blue')
        #ax4.plot(X, np.array(ds.vort.sel(time=frames[:z]).sel(x=xy[0], y=xy[1], method='nearest')).T, color='tab:blue')
        ax3.plot(X, np.array(ds.theta.sel(time=frames[:z]).sel(x=xy[0], y=xy[1], method='nearest').std('ens_mem')))
        ax4.plot(X, np.array(ds.vort.sel(time=frames[:z]).sel(x=xy[0], y=xy[1], method='nearest').std('ens_mem')))
        
        ax3.set_ylim(ym1,yM1)
        ax3.set_xlim(0,frames[-1]*s2d)
        ax3.yaxis.tick_right()
        ax3.set_xlabel('time (days)')
        ax3.set_ylabel('Temperature Std. (K)')
        
        ax4.set_ylim(ym2,yM2)
        ax4.set_xlim(0,frames[-1]*s2d)
        ax4.yaxis.tick_right()
        ax4.set_xlabel('time (days)')
        ax4.set_ylabel(r'Zeta Std. (s$^{-1}$)')
        ax4.yaxis.set_label_position("right")
    
        
        plt.ion()
        plt.draw()
    anim = manim.FuncAnimation(fig, anim, frames, repeat=False)
    
    anim.save(filename, fps=6, codec='h264', dpi=120)
    
    
    
def overview_animation(ds, times, xs, ts=None, filename = './images/overview.gif', step=3600*2):
    proj=ccrs.NorthPolarStereo()
    dt=step
    frames = np.arange(times[0], times[1], dt)
    
    if times[0] < ds.time.data[0] or times[1] > ds.time.data[-1]:
        raise ValueError('You are trying to animate a time period there is no data for.')
        
    skip = 3
    x = ds.x.data[::skip]
    y = ds.y.data[::skip]
        
    plt.ioff()
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': proj}, figsize=(5,3.5),sharex=True,sharey=True, dpi=120)
    #fig.clf()

    axs[0].set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())

    make_ax_circular(axs[0])
    
    ds = xr.concat([ds, ds.isel(x=slice(0,1)).assign_coords(x=[360])], dim='x')
    
    #use to get colorbar
     #temporary addition due to cartopy's bug with certain contourf levels not showing
    cmap = plt.colormaps['bwr']
    norm = BoundaryNorm(np.linspace(-1.5,1.5,6), ncolors=cmap.N, clip=True)      
    cf=axs[0].pcolormesh(ds.x.data,ds.y.data,ds.vortp.interp(time=0).data*1e5, transform = ccrs.PlateCarree(), 
            cmap=cmap,norm=norm)
    plt.colorbar(cf, ax=axs[0], label= r"$\zeta$' (s$^{-1}$)", orientation='horizontal', shrink=0.9)
    
    templevs = np.arange(255,300,5)
    
    cf=axs[1].contourf(ds.x.data,ds.y.data,ds.theta.interp(time=0).data, transform = ccrs.PlateCarree(), 
            levels = templevs, cmap='RdBu_r', extend='both')
    cbar=plt.colorbar(cf, ax=axs[1], label= r"$\theta$ (K)", orientation='horizontal', shrink=0.9)
    plt.setp(cbar.ax.get_xticklabels()[::2], visible=False)
    u = ds.u.interp(time=0).data[::skip, ::skip]
    v = ds.v.interp(time=0).data[::skip, ::skip]

    q=axs[0].quiver(x, y, u - u.mean(axis=1)[:,None], v, transform = ccrs.PlateCarree(), 
                    color = '0.2', units='inches', scale=50., width=0.01, pivot = 'mid',zorder=10)
    
    axs[0].quiverkey(q, X=0.9, Y=1.0, U=10,label="u' 10 m/s", labelpos='N')
    
    q=axs[1].quiver(x, y, u, v, transform = ccrs.PlateCarree(), 
                    color = '0.2', units='inches', scale=100., width=0.01, pivot = 'mid',zorder=10)
    axs[1].quiverkey(q, X=0.9, Y=1.0, U=10,label='U 10 m/s', labelpos='N')
    

    def anim(t):
        u = ds.u.interp(time=t).data[::skip, ::skip]
        v = ds.v.interp(time=t).data[::skip, ::skip]
        
        plt.ioff()
        title = '{:.2f} days'.format(t*s2d)
        
        for ax in axs:
            ax.cla()
            make_ax_circular(ax)
            ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())
            # Set the plot title
            ax.set_title(title, fontsize=9)
            #ax.quiverkey(q, X=0.9, Y=1.0, U=10,label='10 m/s', labelpos='N')

        
        cf=axs[0].pcolormesh(ds.x.data,ds.y.data,ds.vortp.interp(time=t).data*1e5, transform = ccrs.PlateCarree(), 
            cmap=cmap,norm=norm)
        cf=axs[1].contourf(ds.x.data,ds.y.data,ds.theta.interp(time=t).data, transform = ccrs.PlateCarree(), 
                    levels =templevs, cmap='RdBu_r', extend='both')

        q=axs[0].quiver(x, y, u - u.mean(axis=1)[:,None], v, transform = ccrs.PlateCarree(), 
                        color = '0.2', units='inches', scale=50., width=0.01, pivot = 'mid',zorder=10)

        axs[0].quiverkey(q, X=0.9, Y=1.0, U=10,label="u' 10 m/s", labelpos='N')

        q=axs[1].quiver(x, y, u, v, transform = ccrs.PlateCarree(), 
                        color = '0.2', units='inches', scale=100., width=0.01, pivot = 'mid',zorder=10)
        axs[1].quiverkey(q, X=0.9, Y=1.0, U=10,label='U 10 m/s', labelpos='N')
            

        
        if ts != None:
            Ntraj = xs.shape[2]
            for i in range(Ntraj):
                ind = np.where(ts[:, i] < t)[0]
                if len(ind) > 0:
                    ax.plot( xs[ind      , 0, i] ,  xs[ind      , 1, i],  'r', lw=2., transform = ccrs.PlateCarree(),)
                    ax.plot([xs[ind[0]   , 0, i]], [xs[ind[0]   , 1, i]], 'kx', transform = ccrs.PlateCarree(),)
                    ax.plot( xs[ind[25::50], 0, i] ,  xs[ind[25::50], 1, i],  'k+', transform = ccrs.PlateCarree(),)

                    if len(ind) < ts.shape[0]:
                        ax.plot([xs[ind[-1]  , 0, i]], [xs[ind[-1]  , 1, i]], 'ro', transform = ccrs.PlateCarree(),)

        plt.ion()
        plt.draw()

    anim = manim.FuncAnimation(fig, anim, frames, repeat=False)
    
    anim.save(filename, fps=12, codec='h264', dpi=120)
    plt.ion()
    

    
def animate_quiver(ds, times, xs, ts=None, filename = 'traj.gif', step=3600*2):
    
    dt=step
    frames = np.arange(times[0], times[1], dt)
    
    if times[0] < ds.time.data[0] or times[1] > ds.time.data[-1]:
        raise ValueError('You are trying to animate a time period '
                        'there is no data for.')
    skip = 3
    x = ds.lon.data[::skip]
    y = ds.lat.data[::skip]
        
    plt.ioff()
    f = plt.figure(3, figsize = (5, 3.5), dpi = 200)
    f.clf()
    ax = plt.subplot(1, 1, 1, projection = ccrs.NorthPolarStereo())
    ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())

    make_ax_circular(ax)

    theta = xr.concat([ds.theta, ds.theta.isel(lon=slice(0,1)).assign_coords(lon=[180])], dim='lon')
    
    #use to get colorbar
    cf=ax.contourf(theta.lon.data,theta.lat.data,theta.interp(time=0).data, transform = ccrs.PlateCarree(), 
            levels =np.arange(245,307,7), cmap='coolwarm', extend='both')
    plt.colorbar(cf, ax=ax, label='Temperature (K)')

    def anim(t):
        u = ds.u.interp(time=t).data[::skip, ::skip]
        v = ds.v.interp(time=t).data[::skip, ::skip]
        
        plt.ioff()
        ax.cla()
        make_ax_circular(ax)
        ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())
        
        cf=ax.contourf(theta.lon.data,theta.lat.data,theta.interp(time=t).data, transform = ccrs.PlateCarree(), 
                    levels =np.arange(245,307,7), cmap='coolwarm', extend='both')

        ax.quiver(x, y, u, v, transform = ccrs.PlateCarree(), 
                    color = '0.2', units='inches', scale=100., width=0.01, pivot = 'mid')
        
        title = '{:.2f} days'.format(t*s2d)
        
        # Set the plot title
        ax.set_title(title, fontsize=9)
        if ts != None:
            Ntraj = xs.shape[2]
            for i in range(Ntraj):
                ind = np.where(ts[:, i] < t)[0]
                if len(ind) > 0:
                    ax.plot( xs[ind      , 0, i] ,  xs[ind      , 1, i],  'r', lw=2., transform = ccrs.PlateCarree(),)
                    ax.plot([xs[ind[0]   , 0, i]], [xs[ind[0]   , 1, i]], 'kx', transform = ccrs.PlateCarree(),)
                    ax.plot( xs[ind[25::50], 0, i] ,  xs[ind[25::50], 1, i],  'k+', transform = ccrs.PlateCarree(),)

                    if len(ind) < ts.shape[0]:
                        ax.plot([xs[ind[-1]  , 0, i]], [xs[ind[-1]  , 1, i]], 'ro', transform = ccrs.PlateCarree(),)

        plt.ion()
        plt.draw()

    anim = manim.FuncAnimation(f, anim, frames, repeat=False)
    
    anim.save(filename, fps=12, codec='h264', dpi=240)
    plt.ion()
    
        
        

 