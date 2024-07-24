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

from matplotlib.animation import FuncAnimation


from matplotlib.collections import LineCollection

import seaborn as sns

from utils.plotting import add_cyc_point, make_ax_circular, sanitize_lonlist, add_gridlines

s2d = 1/86400.
d2r = np.pi / 180.    



   
def animate_thetaens(ds, times, xs=None, ts=None, tlevs = np.arange(0,12,2),
                     filename = 'espread.gif', dt=3600*2):
    """
    Animate the spread of theta, including trajectories if desired.
    
    Parameters:
    - ds: xarray Dataset containing the data
    - times: tuple with start and end times for the animation
    - xs: array with trajectory data
    - ts: array with time steps for trajectories
    - tlevs: levels for theta contour
    - filename: output file name for the animation
    - dt: time step in seconds for animation frames
    """
    
    frames = np.arange(times[0], times[1], dt)
    
    if times[0] < ds.time.data[0] or times[1] > ds.time.data[-1]:
        raise ValueError('You are trying to animate a time period '
                        'there is no data for.')

    plt.ioff()
    f = plt.figure(3, figsize = (5, 3.5), dpi = 200)
    f.clf()
    ax = plt.subplot(1, 1, 1, projection = ccrs.NorthPolarStereo())
    ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())

    make_ax_circular(ax)
    
    #ds=xr.concat([ds, ds.isel(lon=slice(0,1)).assign_coords(lon=[180])], dim='lon')
    #ds=xr.concat([ds.isel(lon=slice(0,1)).assign_coords(lon=[-180]), ds], dim='lon')
    ds= add_cyc_point(ds)
    background = ds.theta.sel(ens_mem=0).sel(time=0)
    theta = ds.theta.std('ens_mem')
    
    #use to get colorbar
    cm = sns.color_palette("light:seagreen", as_cmap=True)
    normcm = mpl.colors.BoundaryNorm(tlevs, cm.N)

    cf=ax.contourf(theta.x.data, theta.y.data, theta.interp(time=times[1]).data, transform = ccrs.PlateCarree(), 
            levels=tlevs,cmap=cm, norm=normcm)
    
    plt.colorbar(cf, ax=ax, label='Std(Temp) (K)',shrink=0.8)
    
    btlev= np.arange(250,300,5)
    norm = plt.Normalize(btlev[0], btlev[-1])
    cmap = plt.cm.coolwarm
    
    ax.contour(background.x.data, background.y.data, background.data,
           cmap=cmap, levels=btlev, transform=ccrs.PlateCarree(),linestyles='--', alpha=0.75,zorder=10)
        
    
    def anim(t):
        #print(t/frames[-1])
        plt.ioff()
        ax.cla()
        make_ax_circular(ax)
        ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())
        
        cf=ax.contourf(theta.x,theta.y, theta.interp(time=t).data, transform = ccrs.PlateCarree(), 
                    cmap=cm, levels=tlevs,norm=normcm)
        
        ax.contour(background.x.data, background.y.data, background.data,
           cmap=cmap, levels=btlev, transform=ccrs.PlateCarree(),zorder=10,linestyles='--', alpha=0.75)
        
        title = '{:.2f} days'.format(t*s2d)
        
        # Set the plot title
        ax.set_title(title, fontsize=9)
        if ts is not None:
            Ntraj = xs.shape[2]
            for i in range(Ntraj):
                ind = np.where(ts[:, i] < t)[0]

                if len(ind) > 0:

                    col = ds.theta.sel(ens_mem=i).interp(time=t,
                             x=xs[ind[-1],0,i], y=xs[ind[-1],1,i]).item(0)

                    ax.plot( sanitize_lonlist(xs[ind      , 0, i]) ,  xs[ind      , 1, i],
                            c=cmap(norm(col)), lw=2., transform = ccrs.PlateCarree(),)
                    ax.plot( xs[ind, 0, i] ,  xs[ind      , 1, i],  c=cmap(norm(col)), lw=2.,
                            transform = ccrs.PlateCarree(),zorder=20)
                    ax.plot([xs[ind[0] , 0, i]], [xs[ind[0], 1, i]], 'kx', zorder=20,transform = ccrs.PlateCarree(),)

                    if len(ind) < ts.shape[0]:
                        #initial point
                        col = ds.theta.sel(ens_mem=i).interp(time=0, x=xs[-1,0,i],y=xs[-1,1,i]).item(0)

                        ax.scatter([xs[ind[-1]  , 0, i]], [xs[ind[-1]  , 1, i]], c= col, norm=norm, cmap=cmap,
                         transform=ccrs.PlateCarree(), zorder=30)

        plt.ion()
        plt.draw()
        
    frames =  np.append(frames, [frames[-1]] * 5)

    anim = manim.FuncAnimation(f, anim, frames, repeat=True,repeat_delay=2000)
    
    anim.save(filename, fps=12, codec='h264', dpi=200)
    plt.ion()
    
    
    

def overview_animation(ds, times, xs, ts=None, filename='./images/overview.gif', dt=3600*2):
    
    """
    Create an overview animation of the ensemble spread in theta and zeta
    
    Parameters:
    - ds: xarray Dataset containing the data
    - times: tuple with start and end times for the animation
    - xs: array with trajectory data
    - ts: array with time steps for trajectories (optional)
    - filename: output file name for the animation
    - step: time step in seconds for animation frames
    """
        
    # Define projection and frame step
    proj = ccrs.NorthPolarStereo()
    frames = np.arange(times[0], times[1], dt)

    # Ensure the requested times are within the dataset's time range
    if times[0] < ds.time.data[0] or times[1] > ds.time.data[-1]:
        raise ValueError('You are trying to animate a time period for which there is no data.')

    skip = 3  # Step size for downsampling u,v data
    x = ds.x.data[::skip]
    y = ds.y.data[::skip]
    
    u = ds.u[:,::skip,::skip]
    v = ds.v[:,::skip,::skip]
        
    # Prepare the figure and axes
    plt.ioff()
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': proj}, figsize=(8, 4), sharex=True, sharey=True, dpi=200)

    # Set up each axis
    for ax in axs:
        ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())
        make_ax_circular(ax)
    
    # Extend the dataset to handle the wrap-around at 360 degrees longitude
     #ds = xr.concat([ds, ds.isel(x=slice(0, 1)).assign_coords(x=[360])], dim='x')
    ds = add_cyc_point(ds)
    
    # Define colormap and normalization for color scales
    cmap = plt.cm.bwr
    norm = BoundaryNorm(np.linspace(-1.5, 1.5, 6), ncolors=cmap.N, clip=True)

    templevs = np.arange(255, 300, 5)
    def init():
        """Initial plot setup."""
        # Initial plot for vorticity
        cf1 = axs[0].pcolormesh(ds.x.data, ds.y.data, ds.vortp.isel(time=0).data * 1e5, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
        plt.colorbar(cf1, ax=axs[0], label=r"$\zeta$' (s$^{-1}$)", orientation='horizontal', shrink=0.9)

        # Initial plot for potential temperature
        cf2 = axs[1].contourf(ds.x.data, ds.y.data, ds.theta.isel(time=0).data, transform=ccrs.PlateCarree(), levels=templevs, cmap='RdBu_r', extend='both')
        plt.colorbar(cf2, ax=axs[1], label=r"$\theta$ (K)", orientation='horizontal', shrink=0.9)

        # Initial wind vectors
        ui = u.isel(time=0).data
        vi = v.isel(time=0).data

        q1 = axs[0].quiver(x, y, ui - ui.mean(axis=1)[:, None], vi, transform=ccrs.PlateCarree(), color='0.2', units='inches', scale=100, width=0.02, pivot='mid', zorder=10)
        axs[0].quiverkey(q1, X=0.9, Y=1.0, U=10, label="u' 10 m/s", labelpos='N')

        q2 = axs[1].quiver(x, y, ui, vi, transform=ccrs.PlateCarree(), color='0.2', units='inches', scale=100, width=0.02, pivot='mid', zorder=10)
        axs[1].quiverkey(q2, X=0.9, Y=1.0, U=10, label='U 10 m/s', labelpos='N')

    def anim(t):
        #print(t/frames[-1])
        """Update the plot for a given time step."""
        for ax in axs:
            ax.cla()  # Clear axis
            make_ax_circular(ax)
            ax.set_extent([-179.9, 179.9, 20, 90], crs=ccrs.PlateCarree())
            ax.set_title(f'{t / 86400:.2f} days', fontsize=9)  # Set title with time in days

        # Update vorticity plot
        cf1 = axs[0].pcolormesh(ds.x.data, ds.y.data, ds.vortp.sel(time=t).data * 1e5, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

        # Update potential temperature plot
        cf2 = axs[1].contourf(ds.x.data, ds.y.data, ds.theta.sel(time=t).data, transform=ccrs.PlateCarree(), levels=templevs, cmap='RdBu_r', extend='both')

        # Update wind vectors
        ui = u.sel(time=t).data
        vi = v.sel(time=t).data

        q1 = axs[0].quiver(x, y, ui - ui.mean(axis=1)[:, None], vi, transform=ccrs.PlateCarree(), color='grey', units='inches', scale=100, width=0.02, pivot='mid', alpha=0.5, zorder=10, headlength=3, headaxislength=3, headwidth=2)
        axs[0].quiverkey(q1, X=0.9, Y=1.0, U=10, label="u' 10 m/s", labelpos='N')

        q2 = axs[1].quiver(x, y, ui, vi, transform=ccrs.PlateCarree(), color='grey', units='inches', scale=100, alpha=0.5, width=0.02, pivot='mid', zorder=10, headlength=3, headaxislength=3, headwidth=2)
        axs[1].quiverkey(q2, X=0.9, Y=1.0, U=10, label='U 10 m/s', labelpos='N')

        # Plot trajectories if provided
        if ts is not None:
            for i in range(xs.shape[2]):
                ind = np.where(ts[:, i] < t)[0]
                if len(ind) > 0:
                    axs[1].plot(sanitize_lonlist(xs[ind, 0, i]), xs[ind, 1, i], 'k',
                                lw=2,alpha=0.4,zorder=20,transform=ccrs.PlateCarree())
                    axs[1].plot(xs[ind[0], 0, i], xs[ind[0], 1, i], 'kx',zorder=25, transform=ccrs.PlateCarree())
                    axs[1].plot(xs[ind[25::50], 0, i], xs[ind[25::50], 1, i], 'k+',zorder=25, transform=ccrs.PlateCarree())
                    if len(ind) < ts.shape[0]:
                        axs[1].plot(xs[ind[-1], 0, i], xs[ind[-1], 1, i], 'ro',markersize=3,zorder=25,transform=ccrs.PlateCarree())

    # Initialize the plot and create the animation
    init()
    anim_func = FuncAnimation(fig, anim, frames=frames, repeat=False)
    anim_func.save(filename, fps=12, codec='h264', dpi=200)
    plt.ion()