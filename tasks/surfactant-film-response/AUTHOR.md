# Surfactant film response, revision 3

The calibration cell has reflection symmetry through the film. The driven measurements fix surface shear viscosity. The fluctuation measurements in calibration have a uniform temperature. In a cell with a temperature gradient, thermal stress at different depths drives distinct combinations of height and surfactant density. A force in a real Fourier mode can induce a response in spatial quadrature. The upper and lower liquids make opposite contributions to this coupling. The revision 2 control uses the full coupled deterministic response. It applies the equilibrium fluctuation relation at the film temperature in every preparation.

## Derivation

Use complex amplitudes with spatial and temporal derivatives ik and −iω. For one liquid, measure distance s from its rigid wall toward the film, in units of 1/k. Let W denote bulk velocity in this inward direction. Incompressibility gives U = iW′ for the longitudinal velocity. Eliminating the pressure from Stokes' equation gives (∂²s − 1)² W = 0. The wall conditions W(0) = W′(0) = 0 leave the basis s cosh s − sinh s and s sinh s. At the film, W(a) and W′(a) are known from the two interface velocities, with a = kh. The required tangential and normal tractions are iμk(W″+W) and μk(3W′−W‴) in these local coordinates.

In global x, z coordinates, define b = 2μk/(sinh²a − a²), S = sinh a cosh a. The resistance matrix for one liquid is

```text
             [ b (S − a)       −i ε b a² ]
R(x,z)  =    [ i ε b a²         b (S + a) ]
```

Here ε = +1 for the lower layer and −1 for the upper. The transverse resistance is μk coth a. Sum upper and lower matrices. Write U, V, W for the surface velocities in global x, y, z. The height follows −iωH = W. Conservation of the insoluble species gives (0.18 k² − iω)c = −ikU. Eliminating c and H adds the following diagonal impedances, in x, y, z order.

```text
k² [ηs + 0.42 + 0.8/(0.18 k² − iω)],   k² ηs,   1.3 k²/(−iω)
```

Inverting the resulting matrix against the applied traction gives the measured velocity. The transverse response with nonzero forcing is strictly monotone in ηs, so it fixes the unknown even though calibration cannot distinguish the two closures. Each hidden group changes only stated apparatus controls.

The independent reference integrates the first-order bulk equations for (U,W,σxz,σzz) with matrix exponentials, using the unknown wall tractions as degrees of freedom. It solves both wall-to-film matching conditions and both surface balances alongside the density and height equations. Mirror reflection of the cell reverses both cross responses and leaves the diagonal entries invariant.

## Thermal preparation

Put q = (c,h), with stiffness K = diag(0.8, 1.3 k²), and let R be the x–z drag above with the surface viscosity terms. Include the static viscous surface stress in R. The kinematic matrix B = diag(−ik, 1) maps (U,W) to the material rate of q. Surfactant diffusion adds mobility F = diag(0.18 k²/0.8, 0). Eliminating velocity gives deterministic relaxation dq/dt = −M K q, where M = B R⁻¹ B† + F.

A unit boundary velocity determines the bulk strain rate at each depth. In one layer the dissipation form per unit depth is 4μk² U*U + τ*xz τxz/μ, where τxz = μ(∂z U + ikW). Replace each product by its two-by-two Gram matrix for the pair of unit boundary velocities. The thermal drag Rₜ is the sum over layers of this local form integrated against T(z)/T₀, plus the film's viscous drag at unit temperature. Then the normalized noise covariance is N = B R⁻¹ Rₜ R⁻¹ B† + F. With G = (M K − iωI)⁻¹ the spectrum of the two measured fields is G N G†. At uniform temperature N = M and the time-antisymmetric density–height spectrum vanishes. With a temperature gradient the lag can be nonzero.

The reference propagates the covariance of the fluctuating total-stress bulk state from each wall. White shear stress enters the U derivative with normalized variance T/μ. The random difference of normal stresses enters the σxz derivative with variance 4Tμk². Eliminating the unknown wall tractions enforces the given interface velocity and gives a random excess film traction. A joint system for U, W, c and h combines the liquid tractions, surface viscous stress noise and conserved diffusive current noise. This checks the oracle's virtual-dissipation quadrature independently.

## Reproduction

Run `python scripts/validate_surfactant_film_response.py` from the repository root. Add `--generate` to regenerate the data. Seed 31873 fixes the 39 driven preparations and 12 isothermal spectra. Each response is rounded to 0.0001. The standard uncertainty is one meter step divided by √12. The validator reproduces the meter readings and compares the two calculations at the fitted viscosity and both ends of its allowed interval. It checks reversal under reflection, the equilibrium cross lag, Hermitian symmetry and quadrature convergence.

Hidden groups use seed 31879. Each group's RMS error must be below the larger of 2.5% of its reference RMS and 0.0002 (two meter steps). Both completed controls fit 1.136897 for a true viscosity of 1.137, with reduced χ² of 1.0975. The oracle's largest relative error is below 0.0025%. The isothermal control passes all driven groups and density power. Its height power, cross lag and in-phase cross spectrum have relative errors of 9.29%, 100% and 31.71%.

Revision 1 measured driven responses alone. Its frozen source is in commit `981c19d8`. Revision 2 retains those preparations and adds the spontaneous fluctuation readout.

## Revision 3

A travelling temperature pattern correlates two optical channels at different lateral wavevectors and frequencies. The relevant local temperature is its complex Fourier coefficient (a−ib)/2 at the probe difference. The spatially averaged background contributes only to diagonal wavevector correlations.

For probe j, write uⱼ for the longitudinal unit-boundary flow, Sⱼ for its vertical shear rate ∂z uⱼ + ikⱼ wⱼ and nⱼ for its lateral unit direction. With x = n₁·n₂, the bulk cross-dissipation integrand is

    2 μ k₁ k₂ (1 + x²) u₁* u₂ + μ x S₁* S₂.

The horizontal normal strains project on nⱼ nⱼ; the normal strain along z has the opposite trace. The film's two-dimensional traceless shear and dilation give a cross drag k₁ k₂ [0.42 + ηs(2x²−1)]. Its conservative surfactant flux has cross mobility 0.18 k₁ k₂ x/0.8. Multiply each contribution by the Fourier coefficient of its local temperature. Propagate the resulting noise with the first and second probe's own causal Green functions on the left and conjugate right.

The independent reference instead integrates a two-wave covariance for the random total-stress bulk states and eliminates their wall tractions. The validator also interchanges the two channels and checks complex conjugation. Both implementations reproduce all earlier response groups. The completed collinear control accounts for temperature and frequency mixing but aligns both strain fields before computing their covariance. This is exact for every calibration observation and each revision 2 diagnostic. On the oblique height, density and density–height groups it has 96.3%, 368.7% and 310.9% relative RMS errors. The full solution stays below 0.0032%. The meter floor and percentage tolerance are unchanged. Revision 2 is archived in commit `c20dc106`.

The target is three independent unhinted GPT-5.6 Luna High trials. From the root, run

```bash
python3 scripts/run_science.py tasks/surfactant-film-response --agent oracle --trials 1
python3 scripts/run_science.py tasks/surfactant-film-response --agent oracle --trials 1 --solution-model scripts/surfactant_film_response_baseline.py
python3 scripts/run_science.py tasks/surfactant-film-response --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3 --restricted-network
```
