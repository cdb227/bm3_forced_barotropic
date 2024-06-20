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
    print(frames)
    def anim(t): 
        print(t)

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
    
    
from matplotlib.animation import FuncAnimation
    

def overview_animation(ds, times, xs, ts=None, filename='./images/overview.gif', step=3600*2):
    # Define projection and frame step
    proj = ccrs.NorthPolarStereo()
    frames = np.arange(times[0], times[1], step)

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
        print(t/frames[-1])
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
    

    
def animate_thetaens(ds, times, xs, ts=None, tlevs = np.arange(0,12,2),
                     filename = 'espread.gif', step=3600*2, mod=False):
    """
    animate spread of theta, include trajectories as well if desired
    """
    
    dt=step
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
    Ntraj = xs.shape[2]
    
    btlev= np.arange(250,300,5)
    norm = plt.Normalize(btlev[0], btlev[-1])
    cmap = plt.cm.coolwarm
    
    ax.contour(background.x.data, background.y.data, background.data,
           cmap=cmap, levels=btlev, transform=ccrs.PlateCarree(),linestyles='--', alpha=0.75,zorder=10)
        
    
    def anim(t):
        print(t/frames[-1])
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
    