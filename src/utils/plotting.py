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

from matplotlib.collections import LineCollection

import seaborn as sns

s2d = 1/86400.
d2r = np.pi / 180.

#+++Simple plot modifications+++#
def add_cyc_point(ds):
    """
    Add cyclic point to xarray object with coordinate x
    """
    ds=xr.concat([ds, ds.isel(x=slice(0,1)).assign_coords(x=[180])], dim='x')
    ds=xr.concat([ds.isel(x=slice(0,1)).assign_coords(x=[-180]), ds], dim='x')
    return ds

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

def sanitize_lonlist(longitudes):
    """
    This fixes an issue when moving across meridians with line plots 
    """
    for i in range(1, len(longitudes)):
        diff = longitudes[i] - longitudes[i - 1]
        if diff > 180:
            longitudes[i] -= 360
        elif diff < -180:
            longitudes[i] += 360
    return longitudes

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
    #plt.ioff()
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': proj}, figsize=(7,5), sharex=True, sharey=True)
    #sln = xr.concat([sln, sln.isel(x=slice(0,1)).assign_coords(x=[180])], dim='x')
    sln = add_cyc_point(sln)
    
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
    #plt.ion()
    return fig,axs

def plot_energy(sln, normalize_to_ic = True):
    """
    Plot globally integrated perturbation energy and enstrophy
    """                 
    fig, axs = plt.subplots(1,2, figsize=(7,5), sharex=True)

    cosphi = np.cos(d2r * sln.y)
    dys = sln.coords['time'] * s2d

    up = sln.u - sln.u.mean('x')
    KE = ((up**2 + sln.v**2) * cosphi).mean(['x', 'y'])
    KE.coords['time'] = dys

    EN = ((sln.vortp**2) * cosphi).mean(['x', 'y'])
    EN.coords['time'] = dys
    
    if normalize_to_ic:
        KE = KE / KE[0]
        EN = EN / EN[0]

    KE.plot(ax=axs[0])
    axs[0].set_title('Perturbation Energy')  
    axs[0].set_xlabel('time (days)')

    EN.plot(ax=axs[1])
    axs[1].set_title('Perturbation Enstrophy')
    axs[1].set_xlabel('time (days)')
        
    return axs

#+++Plotting routines for ensembles+++#
def plot_theta_ensspread(ds,t, levels=np.arange(0,10,2), filename='./espread.png', 
                         trjs=None,ts=None, proj = ccrs.NorthPolarStereo()):
    """
    Plot ensemble spread (defined as 1 std of ensemble) of theta.
    
    Parameters:
    ds : (xarray.Dataset) The dataset containing the ensemble data.
    t : (int) The time index to plot.
    levels : Contour levels for variances
    filename: str
    trjs : Trajectories locations
    ts : Time steps for trajectories
    proj : Projection for the plot. Default is ccrs.NorthPolarStereo().
    """
    
    f = plt.figure(figsize = (5, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())
    make_ax_circular(ax)
    
    background = ds.theta.sel(ens_mem=0).sel(time=0)
    theta = add_cyc_point(ds.theta)
    theta = theta.std('ens_mem').sel(time=t)
    
    #use to get colorbar
    cm = sns.color_palette("light:seagreen", as_cmap=True)
    normcm = mpl.colors.BoundaryNorm(levels, cm.N)

    cf=ax.contourf(theta.x.data, theta.y.data, theta.data, transform = ccrs.PlateCarree(), levels=levels,cmap=cm, extend='max', norm=normcm)
    
    plt.colorbar(cf, ax=ax, label='Std(Temp) (K)',shrink=0.8)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                  linewidth=2, color='k', linestyle='--',alpha=0.8)
    gl.xlabels = False
    #gl.xlabel_bottom = False
    gl.xlines = False
    gl.ylocator = mticker.FixedLocator([60])
    ax.text(20, 70, 'ice edge', transform=ccrs.PlateCarree(), 
       ha='center', va='bottom', fontsize=12, color='k')
    
    btlev= np.arange(255,300,5)
    norm = plt.Normalize(btlev[0], btlev[-1])
    cmap = plt.cm.coolwarm
    
    ax.contour(background.x.data, background.y.data, background.data,
          cmap=cmap, levels=btlev, transform=ccrs.PlateCarree(),zorder=10, alpha=0.75, linestyles='--')
    
    ax.text(0.5, -0.1, 't = {:.2f} days'.format(theta.time.item(0)/86400), horizontalalignment='center',
         verticalalignment='top', transform=ax.transAxes)
    ds.theta.load()
    
    def fix_lon(xs):
        xsup = np.mod(xs, 360)
        xsup[np.where(xsup>180)[0]]-=360
        return xsup
    
    if trjs is not None:
        ntr = trjs.shape[2]
        for i in range(ntr):
           
            txs = fix_lon(trjs[:,0,i])
            tys = trjs[:,1,i]
 
            scol = ds.theta.sel(ens_mem=i).interp(time=0,
                   x=txs[-1],y=tys[-1], kwargs={"fill_value":None}).item(0)
            ax.scatter(trjs[-1, 0, i], trjs[-1, 1, i], c= scol, norm=norm,cmap=cmap, marker='o',transform=ccrs.PlateCarree(), zorder=10, edgecolors='k')
            
            
            #locations for ploting
            xsc = xr.DataArray(txs.reshape(-1), dims=['location'])
            ysc = xr.DataArray(tys.reshape(-1), dims=['location'])
            tsc = xr.DataArray(ts[:,i].reshape(-1), dims=['location'])
            cols = ds.theta.sel(ens_mem=i).interp(time=tsc,x=xsc,y=ysc,kwargs={"fill_value":None}).values
            
            txs= sanitize_lonlist(txs)
            points = np.array([txs, tys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, cmap=cmap,norm=norm, transform=ccrs.PlateCarree())
            lc.set_array(cols)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            
            ax.plot(trjs[0:1, 0, i], trjs[0:1, 1, i], 'kx', ls='', mew=2., transform=ccrs.PlateCarree(), zorder=20) #final point
            #ax.plot(trjs[20::20, 0, i], trjs[20::20, 1, i], 'k+', ls='',  transform=ccrs.PlateCarree(),mew=1.)
            
    f.savefig(filename, dpi=300)
            
    return ax

    

 
