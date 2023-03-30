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
def plot_vort(sln, levels=None, proj = ccrs.NorthPolarStereo(), perturbation=False, ax=None):
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
    plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'(x$10^5$ s$^{-1}$)')
    make_ax_circular(ax)
    add_gridlines(ax)
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    return ax

#+++Plotting Theta+++#
def plot_theta(sln, levels=None, proj = ccrs.NorthPolarStereo(), perturbation=False, ax=None):
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
    plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'(K)')
    add_gridlines(ax)
    make_ax_circular(ax)
    
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    return ax


#+++Plotting Wind Vectors+++#
def add_windvecs(ax, sln, thin=4, zanom=True):
    """
    Add wind vectors to an existing figure
    """
    if zanom:
        q=ax.quiver(sln.x.values[::thin],sln.y.values[::thin], (sln.u - sln.u.mean('x')).values[::thin,::thin], (sln.v - sln.v.mean('x')).values[::thin,::thin], width=0.005, transform=ccrs.PlateCarree(), scale=50, scale_units='inches')
    else:
        q=ax.quiver(sln.x.values[::thin],sln.y.values[::thin], sln.u.values[::thin,::thin], sln.v.values[::thin,::thin], width=0.005, transform=ccrs.PlateCarree(), scale=50, scale_units='inches')
        
    ax.quiverkey(q, X=0.9, Y=1.0, U=10,
             label='10 m/s', labelpos='N')
    return ax

def plot_overview(sln, levels=[None,None], proj=ccrs.NorthPolarStereo(), perturbation=[False,False]):
    """
    Plot both vorticity and theta on 2 panel plot
    """
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': proj}, figsize=(7,5))
    plot_vort(sln,ax=axs[0], levels=levels[0], perturbation=perturbation[0])
    add_windvecs(axs[0], sln)
    plot_theta(sln,ax=axs[1], levels=levels[1], perturbation=perturbation[1])
    add_windvecs(axs[1], sln)
    return fig,axs

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


#+++Plotting routines for ensembles+++#
def plot_theta_ensspread(slns, levels=None, proj = ccrs.NorthPolarStereo(), ax=None):
    """
    Plot ensemble spread (defined as 1 std of ensemble) of theta
    """
    if ax==None:
        f = plt.figure(figsize = (5, 5))
        ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    wrap_data, wrap_lon = add_cyc_point(slns.theta.std('ens_mem'))
    ax.set_title(r"$\theta'$")
        
    cf= ax.contourf(wrap_lon, slns.y.values, wrap_data, extend='max', levels=levels,transform=ccrs.PlateCarree(), cmap = 'Blues')
    plt.colorbar(cf,ax=ax,orientation='horizontal', label = r'Ens. Std. (K)')
    ax=make_ax_circular(ax)
    ax=add_gridlines(ax)
    return ax

    
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
    
    
        
        

 