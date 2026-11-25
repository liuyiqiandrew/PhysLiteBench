# Magnetic tracer R5 proposal and derivation

The harmonic apparatus and calibration remain intact. A stationary periodic
magnetic channel resolves the joint inertial/contact limit through spatial
density harmonics. The public finite-mass SDE, order of limits, potential,
magnetic profile and detector fully define the preparation.

The completed shortcut uses one white thermal contact with the same low-frequency
drag and temperature. It correctly fits harmonic calibration, the earlier R4
harmonic readouts, and equilibrium channel preparations. In a two-temperature
channel the frozen fast covariance varies with the local magnetic field. The
noise-induced drift depends on this covariance, so spatial density is not the
single-temperature Boltzmann density despite magnetic forces doing no work.

With balanced fast state z=(sqrt(m)*v,sqrt(m*kappa/zeta)*f), define its OU drift A
and covariance S from AS+SA^T+Q=0. Let P select velocity and C=-P A^-1. The slow
generator has drift C[:, :2] F + C'(x) S[:,0] and diffusion sym(C S P^T).
The oracle solves the stationary one-dimensional periodic forward equation.
The independent reference instead expands the full finite-mass Kramers equation
in Fourier modes in x and Hermite functions in all four fast variables and
extrapolates three positive masses. It never calls the effective drift formula.

Qualification target: three fresh unhinted GPT-5.6 Luna High trials, with the
unchanged 600/60-second limits and 2.5% prediction tolerance. Preserve R4's 3/3
result; this scientific revision introduces a distinct allowed preparation.
Scientific and Docker controls and the noise study must pass before trials.
