# Surfactant film response, revision 1

The calibration cell has reflection symmetry through the film. Fitting a surface viscosity in this cell leaves a physical question open. What velocity will a lateral force produce perpendicular to the film when one plate is brought closer? A force in a real Fourier mode can induce a response in spatial quadrature. The upper and lower liquids make opposite contributions to this coupling. The scalar impedances in the starter retain the finite-wall correction to each diagonal drag but miss the coupling.

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

## Reproduction

Run `python scripts/validate_surfactant_film_response.py` from the repository root. Add `--generate` to regenerate the public and private records intentionally. Seed 31873 sets the 39 calibration preparations. Each response is rounded to 0.0001. The standard uncertainty is one quantization step divided by √12. The validator reproduces the stored meter readings and checks independent solutions at the fitted viscosity and the ends of its allowed interval. It also checks reversal under reflection and cancellation in a symmetric cell.

Hidden groups use seed 31879. Each relative RMS error is the Euclidean norm of the prediction error divided by the norm of its group's reference signal. At 2.5%, the accepted absolute RMS error for the cross responses is about 0.00047, several meter steps. The oracle and shortcut both fit 1.13690 for a true value of 1.137, with reduced χ² of 1.2212. The oracle's largest hidden error is 0.0022%. Omitting longitudinal–normal traction coupling gives 100% on each cross response and 41.0% on the mixed response.

The target is three independent unhinted GPT-5.6 Luna High trials. From the root, run

```bash
python3 scripts/run_science.py tasks/surfactant-film-response --agent oracle --trials 1
python3 scripts/run_science.py tasks/surfactant-film-response --agent oracle --trials 1 --solution-model scripts/surfactant_film_response_baseline.py
python3 scripts/run_science.py tasks/surfactant-film-response --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3 --restricted-network
```
