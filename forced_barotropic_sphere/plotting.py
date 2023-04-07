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



#+++Plotting Vorticity+++#
def plot_vort(sln, levels=None, proj = ccrs.NorthPolarStereo(), perturbation=False, ax=None, colorbar=True):
    """
    Plot absolute vorticity on NPS projection, with gridlines. Perturbation=True will plot only the perturbation values
    Can be used to plot on existing axes/figure as well
    """
    if ax==None:
        f = plt.figure(figsize = (5, 5))
        ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    
    
    #temporary addition due to cartopy's bug with certain contourf levels not showing
    cmap = plt.colormaps['bwr']
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    if perturbation:
        wrap_data, wrap_lon = add_cyc_point(sln.vortp)
        ax.set_title(r"$\zeta'$")
    else:
        wrap_data, wrap_lon = add_cyc_point(sln.vort)
        ax.set_title(r"$\zeta$")
    cf= ax.pcolormesh(wrap_lon, sln.y.values, wrap_data*1e5, transform=ccrs.PlateCarree(), cmap = cmap, norm=norm)
    if colorbar:
        plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'(x$10^5$ s$^{-1}$)')
    make_ax_circular(ax)
    add_gridlines(ax)
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    return ax

#+++Plotting Theta+++#
def plot_theta(sln, levels=None, proj = ccrs.NorthPolarStereo(), perturbation=False, ax=None, colorbar=True):
    """
    Plot temperature on NPS projection, with gridlines. Perturbation=True will plot only the perturbation values
    Can be used to plot on existing axes/figure as well
    """
    if ax==None:
        f = plt.figure(figsize = (5, 5))
        ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    if perturbation:
        wrap_data, wrap_lon = add_cyc_point(sln.thetap)
        ax.set_title(r"$\theta'$")
    else:
        wrap_data, wrap_lon = add_cyc_point(sln.theta)
        ax.set_title(r"$\theta$")
    cf= ax.contourf(wrap_lon, sln.y.values, wrap_data, levels=levels, extend='both', transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    if colorbar:
        plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'(K)')
    add_gridlines(ax)
    make_ax_circular(ax)
    
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    return ax


#+++Plotting Wind Vectors+++#
def add_windvecs(ax, sln, thin=3, zanom=True):
    """
    Add wind vectors to an existing figure
    """
    if zanom:
        q=ax.quiver(sln.x.values[::thin],sln.y.values[::thin], (sln.u - sln.u.mean('x')).values[::thin,::thin], (sln.v - sln.v.mean('x')).values[::thin,::thin], width=0.0075, transform=ccrs.PlateCarree(), scale=40, scale_units='inches')
    else:
        q=ax.quiver(sln.x.values[::thin],sln.y.values[::thin], sln.u.values[::thin,::thin], sln.v.values[::thin,::thin], width=0.005, transform=ccrs.PlateCarree(), scale=50, scale_units='inches')
        
    ax.quiverkey(q, X=0.9, Y=1.0, U=10,
             label='10 m/s', labelpos='N')
    return ax

def plot_overview(sln, levels=[None,None], proj=ccrs.NorthPolarStereo(), perturbation=[False,False], colorbar=[True,True]):
    """
    Plot both vorticity and theta on 2 panel plot
    """
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': proj}, figsize=(7,5))
    plot_vort(sln,ax=axs[0], levels=levels[0], perturbation=perturbation[0],colorbar=colorbar[0])
    add_windvecs(axs[0], sln)
    plot_theta(sln,ax=axs[1], levels=levels[1], perturbation=perturbation[1],colorbar=colorbar[1])
    add_windvecs(axs[1], sln)
    return fig,axs




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


#+++animations+++#

def overview_animation(ds, levels=[None,None], proj=ccrs.NorthPolarStereo(), perturbation=[False,False], fr=4, filename='./images/overview.gif'):
    """
    Plot an animation of the evolution of the vorticity and temperature field for a single model run
    """
    
    skip = int(1. / (fr * (ds.time[1] - ds.time[0])))
    if skip < 1: skip = 1
   
    frames = ds.time[::skip]
    
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': proj}, figsize=(5,3.5))
    
    plot_vort(ds.sel(time=0),ax=axs[0], levels=levels[0], perturbation=perturbation[0])
    add_windvecs(axs[0], ds.sel(time=0))
    plot_theta(ds.sel(time=0),ax=axs[1], levels=levels[1], perturbation=perturbation[1])
    add_windvecs(axs[1], ds.sel(time=0))
    
    def anim(t): 
        axs[0].cla()
        axs[1].cla()
        plot_vort(ds.sel(time=t),ax=axs[0], levels=levels[0], perturbation=perturbation[0], colorbar=False)
        add_windvecs(axs[0], ds.sel(time=t))

        plot_theta(ds.sel(time=t),ax=axs[1], levels=levels[1], perturbation=perturbation[1], colorbar=False)
        add_windvecs(axs[1], ds.sel(time=t))

        #axs.set(xlabel = r'x [$L_d$]', ylabel = r'y [$L_d$]', title = 'PV, t = %.1f' % t)
        plt.ion()
        plt.draw()
    anim = manim.FuncAnimation(fig, anim, frames, repeat=False)
    
    anim.save(filename, fps=4, codec='h264', dpi=120)
    
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
    
    
    
        
        

 