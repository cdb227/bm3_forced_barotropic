# barotropic_model READ ME

## 1. Introduction
The purpose of this project is to create an atmosphere toy model to solve forced barotropic vorticity equations, within the context of ensemble forecasts. It has several different settings which can be modified, such as using linear or nonlinear dynamics, different forcing scenarios, etc. One of the mains goals of this project is to explore how different forcing scenarios or ensemble initialisations can lead to bimodality in ensemble forecasts.

## 2. Model Dynamics
To spatially solve for vorticity, we're using spherical harmonics through the **spharm** Python package. One disadvantage of spectral methods is the presence of Gibbs osciallations, which are apparent in the topography field. The integration scheme is RK4.

The linearized, forced barotropic vorticity equation we are solving for can be written as follows:
$$\frac{\partial\zeta '}{\partial t}= - \bar{u}\frac{\partial\zeta '}{\partial x} - v'\gamma + F_\zeta - D_\zeta $$
(derivation below)

## 3. Code Documentation

**Bugs and unwanted features:** We had to initialize the plot_tools class 3 different times in the barotropic class so that the plots showed up and saved correctly. We also had to be careful about flipping the latitude field of the input zonal wind field and the topography field. The latitude direction is important for calculations in spherical harmonics code.

## 4. Code Walkthrough


## 5. Example Output


## 6. Future Improvements


## 7. References

Held, Isaac M., et al. “Northern Winter Stationary Waves: Theory and Modeling.” Journal of Climate, vol. 15, no. 16, 2002, pp. 2125–2144. JSTOR, www.jstor.org/stable/26249392. Accessed 15 Dec. 2020.

### Notes
#### Deriving Vorticity Equation
$$\require{cancel}$$ 
Beginning with an unforced system, we can write the evolution of absolute vorticity as:

$$\frac{D(F+\zeta)}{Dt} = 0 $$

which expands to

$$\frac{\partial\zeta}{\partial t} + u\frac{\partial\zeta}{\partial x} + \cancel{ u\frac{\partial f}{\partial x} } + v \frac{\partial\zeta}{\partial y} + v\frac{\partial f}{\partial y}=0$$

where the second term cancels since the zonal gradient of $f$ is zero. <br>
Defining $\beta= \frac{\partial f}{\partial y}$ and linearizing about zonal mean flow results in

$$\frac{\partial\zeta '}{\partial t} + \bar{u}\frac{\partial\zeta '}{\partial x} + v' (\frac{\partial \bar\zeta}{\partial y} + \beta)=0$$
which, since, 
$$\bar\zeta = \cancel{ \frac{\partial \bar v}{\partial x}} - \frac{\partial \bar u}{\partial y}$$
(zonal gradient of zonal average is zero)<br>
can expanded and simplified as follows:
$$\frac{\partial\zeta '}{\partial t} + \bar{u}\frac{\partial\zeta '}{\partial x} + v'(\beta -\frac{\partial^2 \bar u}{\partial y^2}) = 0 $$
$$\frac{\partial\zeta '}{\partial t} + \bar{u}\frac{\partial\zeta '}{\partial x} + v'\gamma = 0 $$
with $\gamma = (\beta -\frac{\partial^2 \bar u}{\partial y^2})$, the meriodinal gradient in absolute vorticity <br>

This is the basis of the equation we are trying to solve. Rather than freely evolving, we now also add a forcing term $F_\zeta$ (orography, Rossby waves, etc. ) as well as a dissipation term (friction). Rearranging to solve for the vorticity tendency gives:
$$\frac{\partial\zeta '}{\partial t}= - \bar{u}\frac{\partial\zeta '}{\partial x} - v'\gamma + F_\zeta - D_\zeta $$








