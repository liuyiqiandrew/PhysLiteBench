# Acoustic cavity pressure, revision 1

This candidate uses exact linear acoustic fields but tests whether their nonlinear mechanical readout requires a change in the mean fluid state. The supplied model treats the Eulerian mean density as unchanged and rectifies the equation of state on its exact first-harmonic density. It fits optical density calibration exactly. A closed cavity must also satisfy mean local momentum balance at fixed total mass. The resulting mean density redistribution changes the wall force.

The apparatus is explicit about the nonlinear mass and momentum equations, the isothermal barotropic equation of state, fixed rigid walls, total mass, constant longitudinal viscosity and force per current volume. Its thermostat removes dissipated heat without applying momentum. The periodic steady state is taken before the forcing amplitude tends to zero. This avoids importing a pressure formula from an open acoustic beam, a moving boundary, an adiabatically heated gas or a prescribed force per unit mass.

Calibration is optical density at the wall, not a generic mechanical pressure sensor. A mechanical transducer's first-harmonic signal would also contain viscous normal traction. The hidden readout is the mean full normal traction; its viscous mean vanishes for these preparations, as derived below. This calibration clarification was made before any task data or evaluation was generated; the original prototype is preserved separately.

## Fields and force

Let k=m*pi/L, omega=r*c0*k, and use the exp(-i*omega*t) convention. At first order the velocity is V*sin(kx), and density is R*cos(kx), where

    V = drive/[eta*k² + i*(B*k²/omega-rho0*omega)],
    R = -i*rho0*k*V/omega.

Both controls compute these complete viscous fields. The optical calibration reads abs(R). It is strictly decreasing in positive eta, so the parameter is globally identifiable on the allowed interval. A grid objective check and noiseless recovery at the interval endpoints and interior values support the implementation of that fit.

At second order, R and V are in temporal quadrature. Their mean product therefore vanishes at every position. Mean continuity and the impermeable walls give mean(u2)=0. Averaging the conservative momentum equation gives

    mean(p2) + rho0*mean(u1²) - eta*d(mean(u2))/dx = C.

The last term is zero for this single standing mode. With p''=(5-1)*B/rho0², the equation of state is

    mean(p2) = c0²*mean(rho2) + 0.5*p''*mean(rho1²).

Fixed total mass imposes the integral of mean(rho2) equal to zero, hence

    C = rho0*abs(V)²/4 + p''*abs(R)²/8.

At either fixed wall u1=0, so C is the mean excess normal wall force per area. The supplied closure instead predicts p''*abs(R)²/4. It has ordinary positive nonzero signals, uses every control, preserves total mass and equal endwall forces, and gives exact first-order calibration. Its defect is the local mean-state assumption; it does not satisfy the complete mean momentum balance. At resonance the source/correct ratio is 4/3. A new viscosity fit cannot repair that factor while retaining the calibrated density dynamics.

A trial retaining this mean-state approximation is a physical-model failure. A trial deriving the needed redistribution but making a factor, phase, perturbation-order or implementation error must be classified separately. At fixed fitted viscosity the causal physical repair is to replace the force readout with the constrained mean-state result, leaving the first-harmonic dynamics and calibration unchanged.

## Independent reference and validation

The private reference discretizes the first-harmonic fields on a staggered finite-volume grid. It then solves discrete mean continuity and local momentum balance for the mean velocity and density, with a separate total-mass row. Mean wall traction is extrapolated from the resulting stress. This calculation uses neither the oracle expression for C nor its analytic mode amplitude. The default 256/512 grids use Richardson extrapolation; an additional 384/768 calculation checks refinement.

Across 108 domain corners and 24 random controls, the oracle/reference error is below 1.47e-7 relative; refinement changes are below 1.18e-7. Mass and mean momentum residuals are below 1.8e-14, and external drive work equals viscous dissipation within 2.66e-13. Mean velocity is below 1.51e-13. Resonance, drive scaling and opposite-wall symmetry checks pass. The source's equal endwall forces do not establish its local momentum balance.

Calibration has 24 distinct settings repeated twelve times, for 288 readings. The optical instrument uncertainty is fixed sigma=.0002, independent of the response and unknown parameter. The private true viscosity is .12; its checked-in fit is .1199981877 with reduced chi-square .90283. Public and private data copies are identical. Public model construction stores None and supplies an ordinary fit stub; no fitted truth is exposed.

The standard prediction gate is .04 group normalized RMS, with 3% parameter and reduced-chi-square1.5 gates. Nominal correct hidden errors are below 3.03e-5; shortcut force errors are .33337, .49325 and .28743. Hidden force signals span .03816–4.94786 in the specified coefficient units. All 256 noise realizations pass calibration and correct-model predictions; all reject the shortcut. Maximum correct hidden error is 6.96e-5 and minimum shortcut force error is .28735. Local fresh-copy pytest gives oracle7/7 and shortcut4pass/3 intended force failures, both in under one second. These are scientific controls; model difficulty has not been measured for this revision.

The exact neutral instruction, pinned dependencies and 600/60-second agent/verifier limits are retained. No physical correction formula or solver mandate appears in the public README. Root owns Docker controls and future fresh trials. Reproduce the scientific report from the repository root with:

    uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/acoustic-cavity-pressure-r1/scripts/validate_acoustic_cavity_pressure.py

Only `--generate` intentionally replaces both calibration copies. Seeds are recorded in metadata. The report, local controls and provenance live in the stage's results directory.

## Scope and primary background

A broad archive-name and author-text audit found no previous fixed-mass acoustic-fluid mean-pressure task. Related surface-wave tasks concern solid incremental boundary tractions; the elastic-resonator task concerns stored oscillator energy; electric-dipole force concerns optical momentum transfer. They share general mechanics but do not use this missing second-order fluid mean-state constraint. The rejected temperature-dependent adsorption proposal was not built because its thermal forcing was not determined by the proposed material data.

R. T. Beyer, [Acoustic Radiation Pressure of Plane Compressional Waves](https://doi.org/10.1103/RevModPhys.25.653), Rev. Mod. Phys.25,653(1953), discusses fixed-position versus particle-following averages and mechanical force. Here the force is derived directly from the specified closed-cavity mass and momentum equations; no universal radiation-pressure or wave-energy relation is assumed.
