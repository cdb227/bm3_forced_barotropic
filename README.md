# bm3_forced_barotropic README

## 1. Introduction
The purpose of this project is to create an atmosphere toy model to solve the forced barotropic vorticity equations and apply this model within the context of ensemble forecasts. It has several different settings which can be modified, such as using linear or nonlinear dynamics, different forcing scenarios, the number of ensemble members and how the initial conditions are perturbed. One of the mains goals of this project is to explore how different forcing scenarios or ensemble initialisations can lead to bimodality in ensemble forecasts.

## 2. Model Dynamics
To spatially solve for vorticity, we're using spherical harmonics through the **spharm** Python package. One disadvantage of spectral methods is the presence of Gibbs osciallations, which are apparent in the case of rapidly changing vorticity fields. The integration scheme is RK4.

The linearized, forced barotropic vorticity equation we are solving for can be written as follows:
$$\frac{\partial\zeta '}{\partial t}= - \bar{u}\frac{\partial\zeta '}{\partial x} - v'\gamma - r_s\zeta' + F_\zeta $$
(derivation below)

$F_\zeta$ includes both predictable forcing (steady) and unpredictable forcing. We may decompose this into $F_\zeta'$ to represent the predictable portion of forcing (shared among ensemble members) and $F_\zeta''$ to represent more random processes (such as convection) that is not shared between members.

In the case of eddy stirring $F_\zeta'$, similar to Linz et al. (2018) we will represent stochastically forced eddies which take the form
$$F_\zeta' = A \exp{ \left[ - \left( \frac{( |\phi| - \phi_o)}{\Delta\phi} \right)^2 \right]}\text{Re}\left[ \tilde{W} (t)\exp(ik\lambda) \right]$$
where we set the stirring amplitude $A=8\times10^{-10}$, the meridional width of eddy stirring as $\Delta\phi=10^\circ$, the stirring latitude as $\phi_o=40^\circ$, the zonal wavenumber as $k=6$. $\tilde{W}(t)$ represents complex white noise with unit variance. 

#### Advection-Diffusion Model
The advection diffusion model as used in Linz et al (2018) is as follows:

$$\frac{\partial\theta}{\partial t} = -\mathbf{v}\cdot\nabla\theta - \frac{\theta - \theta_{eq}}{\tau} - \kappa\nabla^8\theta$$
where $\mathbf{v}$ is determined by the solved vorticity equation, $\kappa$ is the hyper-diffusion coefficient and \tau is the thermal relaxation timescale. 
The equilibrium temperature is set to be
$$ \theta_{eq} = \theta_0 - \Delta\theta\sin^2\phi$$


<p align="center">
  <img src="https://github.com/cdb227/bm3_forced_barotropic/blob/main/images/evo_compressed.gif" alt="animated" />
</p>
An example evolution of our model. Winds are stirred stochastically, stirring the mean temperature field, creating the "flower pattern" (with the number of petals is equal to zonal wavenumber of the forcing). If the stirring amplitude is not large enough or if the memory is too low, the temperature field relaxes back to the equilibrium temperature (as seen at t=0).

#### Ensemble Run

The following animation depicts an ensemble run, where purely random white noise is applied to each gridpoint of each ensemble member at t=0 of the vorticity field. White noise is drawn from a Gaussian distribution with $\sigma= 1e-6$. Otherwise, runs are identical (including a shared forcing term). 

<p align="center">
  <img src="https://github.com/cdb227/bm3_forced_barotropic/blob/main/images/ensspread_point.gif" alt="animated" />
</p>

While this causes spread for a brief period of time (3 weeks or so), this is ultimately driven by the IC perturbation magnitude, rather than how that IC perturbation develops. Hence why the ensemble members reconverge after a sufficient amount of dampening time. Even including the nonlinear terms into the vorticity equation, the result seems to be about the same. In order to create the nonlinearity we require, we need to introduce a feedback produced by the temperature that affects the vorticity evolution?



## 3. Code Documentation

**Bugs and unwanted features:**<br>
**Things to check:** Advection from anamolous meridional wind of mean temperature gradient $v'\frac{\partial \bar\theta}{\partial y}$: $$\frac{\partial \bar\theta}{\partial y} = \frac{\partial \theta_{eq}}{\partial y}$$?
<br>
Hyperdiffusion term only applied to smallest harmonic? <br>
Spatially covariance in ensemble perturbations?


## 4. Future Improvements
Implement some sort of boundary layer paramterization <br>
Step function to represent sea ice field?


## 5. Derivations

#### Deriving Vorticity Equation

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

This is the basis of the equation we are trying to solve. Rather than freely evolving, we now also add a forcing term $F_\zeta$ (orography, Rossby waves, etc. ) as well as a dissipation term $r_s\zeta'$, where $r_s$ represents the frictional dissipation rate. Rearranging to solve for the vorticity tendency gives:
$$\frac{\partial\zeta '}{\partial t}= - \bar{u}\frac{\partial\zeta '}{\partial x} - v'\gamma + F_\zeta - r_s\zeta' $$

#### Reproducing L18 climatology
<!-- ![Figure 1b of Linz et al (2018), representing the climatology produced by their advection-diffusion model](images/L18_fig1b.PNG) -->

<!-- ![An integration of our model](images/L18_singlerun.png) -->



## 6. References

Linz, Marianna, Gang Chen, and Zeyuan Hu. "Large‐scale atmospheric control on non‐Gaussian tails of midlatitude temperature distributions." Geophysical Research Letters 45.17 (2018): 9141-9149.

Barnes, Elizabeth A., et al. "Effect of latitude on the persistence of eddy‐driven jets." Geophysical research letters 37.11 (2010).

Held, Isaac M., et al. “Northern Winter Stationary Waves: Theory and Modeling.” Journal of Climate, vol. 15, no. 16, 2002, pp. 2125–2144. JSTOR, www.jstor.org/stable/26249392.










