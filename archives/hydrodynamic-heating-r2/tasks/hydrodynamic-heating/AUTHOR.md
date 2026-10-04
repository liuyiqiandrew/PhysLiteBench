# Viscous charged-fluid local calorimetry, revision 2

This is a staged scientific revision. It has not been evaluated by a model. The canonical revision-1 task, its original conditional result and its fresh three-trial result remain unchanged. The fresh revision-1 batch passed 1/3: two submissions retained local Drude heating and one correctly used the full nonlocal current. Those reviews are in `results/retained-three-physics-trial-reviews.json` at repository level.

Revision 2 changes the material and measured preparation. An incompressible transverse Newtonian charged fluid has a known nonzero viscosity, stationary no-slip walls and a calorimeter covering a specified depth window. Both completed controls solve the entire viscous Maxwell boundary problem. The source assigns local work against friction and viscous force to heat; the oracle measures local internal thermal production. This is a physical observable change with an exact whole-slab calibration degeneracy. It is not a retry of the old task or a tighter gate. The neutral instruction, 600/60-second limits and 4% prediction threshold are retained.

## Apparatus and energy balance

Let p²=n0 q²/m, gamma=.06 and nu=.04. With time convention exp(-i omega t), transverse current J_y and kx=omega sin(theta),

    (gamma - i omega - nu Laplacian) J = p² E,
    E'' + (omega²-kx²) E = -i omega J.

Pressure does not enter this incompressible transverse shear. Both y velocity and y current vanish at the stationary walls. The fluid/background calorimeter collects thermalization of momentum relaxation as well as Newtonian viscous heat. Optical field amplitudes are linear; heating is their leading quadratic mean. Thermal conduction after deposition is excluded by the local operational readout.

The instantaneous viscous dissipation is eta/2 times the squared symmetric velocity gradient. Period averaging gives the positive local heat

    q_heat = [gamma |J|² + nu (|J'|² + kx² |J|²)] / (2 p²).

The supplied source instead evaluates the local power against the dissipative force,

    q_source = [gamma |J|² - nu Re(J* (J''-kx² J))] / (2 p²).

The source's field and derivative computations are exact solutions of its stated constitutive equations. Their difference is a divergence,

    q_heat - q_source = nu/(2 p²) d_z Re(J* J').

Its full-slab integral vanishes because J=0 at both ends. It need not vanish over an interior window. Thus positive, accurate total optical absorption does not determine local heat deposition. All scored signals are nonzero. The tested domain controls have positive output for both closures; mathematical passivity and the physical meaning of local heat are separately checked.

The factor 1/2 is the harmonic time average. Dividing by incident normal Poynting flux cos(theta)/2 yields the predictor normalization; there is no averaging over depth. Including the kx contribution to the symmetric strain is necessary at oblique incidence. The validator checks its full tensor contraction explicitly.

## Controls and independent reference

The two bulk dispersion roots satisfy

    (s-omega²)(gamma-i omega+nu s) = i omega p².

Four anchored exponential modes enforce the two radiating Maxwell conditions and two no-slip conditions. The independent verifier instead integrates a real eight-component first-order Maxwell/momentum boundary-value system. It does not use the modal roots or modal matching code. A pre-packaging complex-valued `solve_bvp` prototype failed adaptive mesh convergence in one case; the explicit real/imaginary formulation resolves that numerical issue. No evaluated source used the unsuccessful prototype solver.

Existing validation checks boundary residuals, optical 1-R-T balance, the window stress-flux identity, nonnegative strain heating, angle reversal, quadrature refinement and tighter independent BVP tolerance. It also checks reference/current-density units and the exact whole-slab control equivalence. The private reference only sees the independent finite boundary solve.

The calibration contains 256 readings from 16 whole-slab settings, with a fixed instrument sigma of .0001, seed 110713. The true plasma frequency is 1.03. Sigma is independent of the response and unknown parameter. Grouped weighted least squares is exactly equivalent to fitting all records. Thirteen noiseless parameter profiles across [.85,1.15] check global identification; the 256-noise validation uses seed 110719. Both controls are fitted and scored for every noise draw. No hidden threshold is adjusted after a model trial.

The final report is `../../results/hydrodynamic-heating-r2-validation.json` relative to this task directory, and the final source manifest is `../../results/hydrodynamic-heating-r2-source-provenance.json`. The report records exact control errors, parameter ranges, 256 fit outcomes, convergence, local pytest output and hashes. Oracle and shortcut are tested in isolated temporary copies. Model difficulty remains unknown until root evaluates a frozen package.

## Reproduction

From the stage root, run:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_hydrodynamic_heating.py

Use `--regenerate` only to intentionally regenerate both calibration copies with the fixed seed. Root owns Docker controls and model runs. No live task files or earlier evidence are edited by this stage.

## Primary background

The transverse viscous charged-fluid electromagnetic equations and additional hydrodynamic wall conditions are discussed in [Forcella et al., Electromagnetic properties of viscous charged fluids](https://arxiv.org/abs/1406.1356). The particular local heat readout, normalization and stress-energy balance above are derived explicitly for this task; the paper is background rather than an undocumented apparatus assumption.
