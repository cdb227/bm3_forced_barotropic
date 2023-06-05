# bm3_forced_barotropic README

## 1. Introduction
The purpose of this project is to create an atmosphere toy model to solve the forced barotropic vorticity equations and apply this model within the context of ensemble forecasts. It has several different settings which can be modified, such as using linear or nonlinear dynamics, different forcing scenarios, the number of ensemble members and how the initial conditions are perturbed. One of the mains goals of this project is to explore how different forcing scenarios or ensemble initialisations can lead to bimodality in ensemble forecasts.

## 2. Model Dynamics
To spatially solve for vorticity, we're using spherical harmonics through the **spharm** Python package. One disadvantage of spectral methods is the presence of Gibbs osciallations, which are apparent in the case of rapidly changing vorticity fields. The integration scheme is RK4.

The linearized, forced barotropic vorticity equation we are solving for can be written as follows:
$$\frac{\partial\zeta '}{\partial t}= - \bar{u}\frac{\partial\zeta '}{\partial x} - v'\gamma - r_s\zeta' + F $$
(derivation below)

$F$ includes both predictable forcing (steady) and unpredictable forcing. We may decompose this into $S$ to represent the predictable portion of forcing (shared among ensemble members) and $\hat{S}$ to represent more random processes (such as convection) that is not shared between members.

In the case of eddy stirring $S$, similar to Vallis et al. (2003) we will represent red eddies with an Ornstein‐Uhlenbeck stochastic process which takes the form

$$ S_{mn}^{i} = Q^{i} ( 1 - e^{-2dt/\tau} )^{1/2} + e^{-dt/\tau}S_{mn}^{i-1} $$

Where $n$ represents the total spectral wavenumber and $m$ the zonal spectral wavenumber. $Q^i$ is a real number chosen uniformly at random between $(-A,A)$, $A$ being the stirring amplitude ($10^{-11}$). $\tau$ is the decorrelation time of the stirring. Only total wavenumbers $8\le n \le 12$ are stirred. 

A latitude mask of  $\exp{ \left[ - \left( \frac{( |\phi| - \phi_o)}{\Delta\phi} \right)^2 \right]}$ is applied to represent stirring originating from the midlatitudes ($\Delta\phi=10^\circ$ and $\phi_o=40^\circ$).

Following Vallis et al. (2004), a decorrelation timescale for stirring of $\tau=$2 days is used with a frictional timescale of $1/r_s =$ 7 days. This is a good representation for baroclinic eddies.


#### Advection-Diffusion Model
The advection diffusion model used is as follows:

$$\frac{\partial\theta}{\partial t} = -\mathbf{v}\cdot\nabla\theta - \frac{\theta - \theta_{eq}}{\tau_t} - \kappa\nabla^8\theta$$

where $\mathbf{v}$ is determined by the solved vorticity equation, $\kappa$ is the hyper-diffusion coefficient (0.1 day damping) and $\tau_t$ is the thermal relaxation timescale (8 days). 

The equilibrium temperature is set to be:
$\theta_{eq} = \theta_0 - \Delta\theta\sin^2\phi$

With $\theta_0 = 300$K and $\Delta\theta\= 45$ K. 


#### Example Run

<p align="center">
  <img src="https://github.com/cdb227/bm3_forced_barotropic/blob/main/images/evo.gif" alt="animated" />
</p>
An example 2-week integration of the model. Winds stir the temperature field. If the stirring amplitude is not large enough or if the eddies decorrelate too quickly, the temperature field relaxes back to the equilibrium temperature (as seen at t=0).

#### Ensemble Run

In the context of ensemble forecasts, we can run multiples of these models in unison, with perturbations to the IC or forcing scenarios. Ultimately, we would like to see spread develop from the start of the forecast and then saturate, representing an approach to climatology.

The following animation depicts an ensemble run (10 members), where purely random white noise is applied to each gridpoint of each ensemble member at t=0 of the vorticity field. White noise is drawn from a Gaussian distribution with $\sigma= 1e-12$. This represents IC uncertainty. Furthermore, an unpredictable forcing $\hat{S}$ is included in the form of \mathcal{N}(\mu=0,\,\sigma=1e-12) drawn and applied to each timestep for each ensemble member independently. 

<p align="center">
  <img src="https://github.com/cdb227/bm3_forced_barotropic/blob/main/images/ensspread_evolution_wn.gif" alt="animated" />
</p>


Spread seems relatively small, where in reality we would expect forecasts to diverge to a much larger extent over this time period. We can increase the perturbation magnitude, this should have a smaller amplitude than our predictable forcing (the red eddies). True white noise at every gridpoint doesn't seem realistic? 

In reality noise should have some spatial/temporal covariance which would increase ensemble spread? One crude example is introducing $\hat{S}$ to have a similar form as $S$, but with a more rapid decorrelation timescale (1 day) and smaller amplitude (1e-12). This results in the following forecast:

<p align="center">
  <img src="https://github.com/cdb227/bm3_forced_barotropic/blob/main/images/ensspread_evolution_rn.gif" alt="animated" />
</p>

This produces more spread, but the magnitudes still seem low. Furthermore, we still get extended periods of higher spread and lower spread, rather than a saturation? 


In order to create the nonlinearity we require for bimodality to form, we need to introduce a feedback produced by the temperature that affects the vorticity evolution?


## 3. Code Documentation

**Bugs and unwanted features:**<br>
**Things to check:** Advection from anamolous meridional wind of mean temperature gradient $v'\frac{\partial \bar\theta}{\partial y}$: $\frac{\partial \bar\theta}{\partial y} = \frac{\partial \theta_{eq}}{\partial y}$?
<br>
Hyperdiffusion term only applied to smallest harmonic? <br>
Spatial covariance in ensemble perturbations?


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

<!-- #### Reproducing L18 climatology -->
<!-- ![Figure 1b of Linz et al (2018), representing the climatology produced by their advection-diffusion model](images/L18_fig1b.PNG) -->

<!-- ![An integration of our model](images/L18_singlerun.png) -->



## 6. References

Vallis, Geoffrey K., et al. "A mechanism and simple dynamical model of the North Atlantic Oscillation and annular modes." Journal of the atmospheric sciences 61.3 (2004): 264-280.

Linz, Marianna, Gang Chen, and Zeyuan Hu. "Large‐scale atmospheric control on non‐Gaussian tails of midlatitude temperature distributions." Geophysical Research Letters 45.17 (2018): 9141-9149.

Barnes, Elizabeth A., et al. "Effect of latitude on the persistence of eddy‐driven jets." Geophysical research letters 37.11 (2010).

Held, Isaac M., et al. “Northern Winter Stationary Waves: Theory and Modeling.” Journal of Climate, vol. 15, no. 16, 2002, pp. 2125–2144. JSTOR, www.jstor.org/stable/26249392.












