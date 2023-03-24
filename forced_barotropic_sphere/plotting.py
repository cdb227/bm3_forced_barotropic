import numpy as np
import xarray as xr

import cartopy                   
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point

import matplotlib as mpl          
import matplotlib.pyplot as plt    
import matplotlib.animation as manim
import matplotlib.ticker as mticker
import matplotlib.path as mpath



#+++Simple plot modifications+++#
def add_cyc_point(var):
    lon_idx = var.dims.index('x')
    wrap_data, wrap_lon = add_cyclic_point(var.values, coord=var.coords['x'], axis=lon_idx)
    return wrap_data, wrap_lon

def make_ax_circular(ax):
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center) 
    ax.set_boundary(circle, transform=ax.transAxes)
    return ax

def add_gridlines(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels = False
    #gl.xlabel_bottom = False
    gl.xlines = False
    gl.ylocator = mticker.FixedLocator([60])
    gl.xlocator = mticker.FixedLocator([])
    #gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    return ax



#+++Plotting Vorticity+++#
def plot_vortp(sln, levels=None, proj = ccrs.NorthPolarStereo()):
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    cf= ax.contourf(sln.x.values, sln.y.values, sln.vortp, extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf,orientation='horizontal', label = r'(s$^{-1}$)')
    ax= make_ax_circular(ax)
    ax= add_gridlines(ax)
    return f,ax


def plot_vort(sln, levels=None, proj = ccrs.NorthPolarStereo()):
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    cf= ax.contourf(sln.x.values, sln.y.values, sln.vort, extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf,orientation='horizontal', label = r'(s$^{-1}$)')
    ax= make_ax_circular(ax)
    return f,ax

#+++Plotting Theta+++#
def plot_theta(sln, levels=None, proj = ccrs.NorthPolarStereo()):
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    cf= ax.contourf(sln.x.values, sln.y.values, sln.theta, extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf,orientation='horizontal', label = r'(K)')
    ax.gridlines()
    ax= make_ax_circular(ax)
    return f,ax

def plot_thetap(sln, levels=None, proj = ccrs.NorthPolarStereo()):
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    cf= ax.contourf(sln.x.values, sln.y.values, sln.thetap, extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf,orientation='horizontal', label = r'(K)')
    ax= make_ax_circular(ax)

    return f,ax

def plot_thetadev(sln, levels=None, proj = ccrs.NorthPolarStereo()):
    """plots theta deviation from zonal mean"""
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    cf= ax.contourf(sln.x.values, sln.y.values, sln.theta - sln.theta.mean('x'), extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf,orientation='horizontal', label = r'(K)')
    ax= make_ax_circular(ax)
    return f,ax

#+++Plotting Wind Vectors+++#
def add_windvecs(f,ax, sln, thin=1):
    """
    Add wind vectors to an existing figure
    """
    q=ax.quiver(sln.x.values[::thin],sln.y.values[::thin], sln.u.values[::thin,::thin], sln.v.values[::thin,::thin], width=0.005, transform=ccrs.PlateCarree(), scale=50, scale_units='inches')
    ax.quiverkey(q, X=0.9, Y=1.0, U=10,
             label='10 m/s', labelpos='N')
    return f,ax

def add_windvecs_zanom(f,ax, sln, thin=1):
    """
    Add wind vectors to an existing figure, with zonal mean subtracted
    """
    
    q=ax.quiver(sln.x.values[::thin],sln.y.values[::thin], (sln.u - sln.u.mean('x')).values[::thin,::thin], (sln.v - sln.v.mean('x')).values[::thin,::thin], width=0.005, transform=ccrs.PlateCarree(), scale=50, scale_units='inches')
    ax.quiverkey(q, X=0.9, Y=1.0, U=10,
             label='10 m/s', labelpos='N')
    return f,ax

#+++plotting both vorticity and theta+++#
def plot_vortp_theta(sln, levels=None, proj=ccrs.NorthPolarStereo(), extent = [-179.9,179.9,29,90]):
    fig, axs = plt.subplots(1,2,
                        subplot_kw={'projection': proj},
                        figsize=(7,5))
    vortp,lons = add_cyc_point(sln.vortp)
    cf= axs[0].contourf(lons, sln.y.values, vortp, extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf, ax = axs[0],orientation='horizontal', label = r'(s$^{-1}$)')
    axs[0].set_title(r"$\zeta$'")
    
    theta,lons = add_cyc_point(sln.theta)
    cf= axs[1].contourf(lons, sln.y.values, theta, extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'RdBu_r')
    plt.colorbar(cf, ax= axs[1],orientation='horizontal', label = r'(K)')
    axs[1].set_title(r"$\theta$")
    
    for i in range(len(axs)):
        axs[i] = add_gridlines(axs[i])
        axs[i].set_extent(extent, crs=ccrs.PlateCarree())
        axs[i] = make_ax_circular(axs[i])
        fig,axs[i] = add_windvecs_zanom(fig,axs[i],sln,thin=2)
        axs[i].text(0.5, -0.1, 't = {:.2f} days'.format(sln.coords['time'].values/86400), horizontalalignment='center',
         verticalalignment='top', transform=axs[i].transAxes)
    return fig,axs



#+++Plotting routines for ensembles+++#
def plot_theta_ensspread(slns, levels=None, proj = ccrs.NorthPolarStereo()):
    """plots ensemble spread (defined as 1 std of ensemble) of theta"""
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 30, 90], crs=ccrs.PlateCarree())
    cf= ax.contourf(slns.x.values, slns.y.values, slns.theta.std('ens_mem'), extend='both', levels=levels,transform=ccrs.PlateCarree(), cmap = 'Blues')
    plt.colorbar(cf,orientation='horizontal', label = r'Ens. Std. (K)')
    ax=make_ax_circular(ax)
    ax=add_gridlines(ax)
    return f,ax
    
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
    
    
        
        

 