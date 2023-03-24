# barotropic_model READ ME

## 1. Introduction
The purpose of this project is to create an atmosphere toy model to solve forced barotropic vorticity equations, within the context of ensemble forecasts. It has several different settings which can be modified, such as using linear or nonlinear dynamics, different forcing scenarios, etc. One of the mains goals of this project is to explore how different forcing scenarios or ensemble initialisations can lead to bimodality in ensemble forecasts.

## 2. Model Dynamics
To spatially solve for vorticity, we're using spherical harmonics through the **spharm** Python package. One disadvantage of spectral methods is the presence of Gibbs osciallations, which are apparent in the topography field. The integration scheme is RK4.

The barotropic vorticity equation is:
$$\frac{\zeta}{dt} = \frac{\bar u_A}{a} \zeta_\lambda - v\beta + r_s\zeta + \mathcal{F}$$
>$$D $$
This equation is then expanded into:
> <img src="readmeimages/Eqn2.gif" /> 
Linearized, this equation is:
> <img src="readmeimages/Eqn3.gif" /> 
Since we want to solve for the vorticity tendancy, we want to set everything to the right side of the equation:
> <img src="readmeimages/Eqn4.gif" /> 

## 3. Code Documentation
The main script where we're solving the equations from **Section 2** is **barotropic.py**. This script initializes all the variables we need and the linear dynamics (nonlinear dyamics pending). This script also includes plot and output saving and diagnostics. **forcing.py** is where we set up the topographic forcing. The 'simple' topography (*topography_simple*) is created by a gaussian function. The 'real' topography (*topography_real*) is read in from a file using a function from **xarray_IO.py**. **xarray_OI.py** contains functions fro reading in netCDFs, creating variables and dimenstions for output netCDFs, and eventually writing the netCDF output. The **namelist.py** script should be the only script you should need to edit. The main veriables you should be changing in **namelist.py** are *topo_case* depending out which topography you want to use (real or simple), the *plot_freq* and *output_freq* variables for how often you want plot and data output. You could also change the location of the 'simple' topography with *topo_clatd* and *topo_clond*. *plot_tools.py* includes plotting functions used in the **barotropic.py** script. **spectral.py** includes the functions needed for using spherical harmonics and is where we calculate the coriolis effect. Finally, **runscript.py** is what you want to use to actually run the model. This is also where you decide if you want to run the linear or nonlinear dynamics version. Each script bascially has a singular class in it, that has the same name as the script. We're using object oriented programming so each classe structures data into objects, which can have variables and functions in them. A benefit of using classes is that we can have a singular class but run the same functions over different instances, so it avoids duplication, keeps the code cleaner, and makes it easier to change inputs in the namelist. We can initialize one class within another class, for example when we initialize the plot_tools class within the barotropic class, now we have access to the structure and functions within plot_tools. 

**Bugs and unwanted features:** We had to initialize the plot_tools class 3 different times in the barotropic class so that the plots showed up and saved correctly. We also had to be careful about flipping the latitude field of the input zonal wind field and the topography field. The latitude direction is important for calculations in spherical harmonics code.

## 4. Code Walkthrough
To run the code as is, just run **runscript.py**. We only coded out the linear dynamics but once we code out the nonlinear dynamics you can specify this in the **runscript.py** on line 18: *model.integrate_linear_dynamics()*  or  *model.integrate_nonlinear_dynamics()*

To change the 'real' topography field change *dfile_topo* in the **namelist.py**. The main veriables you should be changing in **namelist.py** are *topo_case* depending out which topography you want to use (real or simple), the *plot_freq* and *output_freq* variables for how often you want plot and data output. You could also change the location of the 'simple' topography with *topo_clatd* and *topo_clond*. 

For diagnostic plotting of variables at the final time step uncomment: lines 202 and 203 in **barotropic.py**

## 5. Example Output
This is an example of a streamfunction plot at time-step 71 using real topography:
> <img src="readmeimages/output.png"/> 
(The units should be in m2/s, this image was created as a test and had placeholder units). Our model does a pretty good job in the Northern Hemisphere matching the orographically-forced model in Held et al., (2002) but our stream function patterns aren't as angles as gthe ones in Fig. 2b in Held et al., (2002). In the real world, it seems like the most positive and negative areas of the streamfunction are more over the oceans, so maybe having only an atmospheric component and not having any thermal forcing in our toy model is why the toy model results don't match reality. 

## 6. Future Improvements

1. Add nonlinear dynamics 
2. Add thermal forcing
3. Add diffusion

## 7. References





Held, Isaac M., et al. “Northern Winter Stationary Waves: Theory and Modeling.” Journal of Climate, vol. 15, no. 16, 2002, pp. 2125–2144. JSTOR, www.jstor.org/stable/26249392. Accessed 15 Dec. 2020.








