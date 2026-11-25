# Finite-extensible polymer stress, revision 1

The completed source is the Peterlin constitutive approximation to the supplied
nonlinear molecular spring. It solves a positive, self-consistent conformation
tensor and its mechanical stress. Hookean calibration is exactly shared with
the molecular model. Finite-extension stress requires the distribution of
spring forces, which the closure replaces with a mean spring coefficient.
Agent difficulty has not been measured.

## Apparatus and mechanical readout

Two overdamped beads with independent drag `zeta` and thermal forces are joined
by `U=-L² log(1-Q²/L²)/2`, with known spring constant `H=1`. The relative-coordinate
equation is

`dQ = [kappa Q - 2Q/(zeta*(1-Q²/L²))]dt + sqrt(4T/zeta)dW`,

where `kappa=diag(rate,-rate)`. The factors of two arise from the two bead drags
and independent noises. The finite-length boundary is natural and inaccessible;
the stated potential supplies confinement without resetting or rupture. The
Hookean drift is stable over the entire allowed drag/rate domain:
`2/zeta-|rate| >= 2/4.8-.25 > 0`.

The solvent-subtracted planar mechanical stress per molecular number density is
the force dipole minus isotropic Brownian stress. The measured difference is
therefore `H <(Qx²-Qy²)/(1-Q²/L²)>`; the isotropic contribution cancels. Positive
normal stress denotes tension. Planar stress is force per line length; dividing
by molecules per area yields energy per molecule. This fixes the sign and
normalization without assigning stress to a single conformation observable by
analogy. Out-of-plane confinement does no measured work.

The unknown bead drag is in `[3.2,4.8]`; its private true value is 4.1. All lengths,
energies and times are in fixed reference units. `temperature` supplies the
thermal energy `T`, not a separately unknown Boltzmann conversion.

## Exact distribution and valid completed approximation

Because the prescribed velocity gradient is symmetric, the exact stationary
connector density inside the disk is

`P(Q) proportional to (1-Q²/L²)^(L²/(2T))*exp[zeta*rate*(Qx²-Qy²)/(4T)]`.

It gives zero configuration-space probability current for the stated dynamics.
This does not imply zero rheometer power: flow stress is conjugate to the
maintained solvent deformation, and `rate*(Sigma_xx-Sigma_yy)` is nonnegative.
Let `Wi=zeta*rate/4`, `b=L²/T`. The oracle integrates the angular dependence
analytically with modified Bessel functions and uses radial Gauss–Jacobi
quadrature for the second moments. Exact steady second-moment balance gives
the stress difference `2Wi*trace(C)`.

The source preaverages the nonlinear spring coefficient. It solves

```
Cxx = T/(f-2Wi),  Cyy = T/(f+2Wi),
f = 1/[1-trace(C)/L²],
stress = f*(Cxx-Cyy).
```

The scalar root is unique on `f>max(1,2|Wi|)`: the root function
`1-1/f-[1/(f-2Wi)+1/(f+2Wi)]/b` is strictly increasing, negative at its lower
endpoint, and positive at infinity. Thus covariance is positive and mean-square
extension remains below `L²`. This is the standard coherent mean-force model,
which relaxes individual finite-extension statistics; it is not a numerical
regularization of the exact molecular potential. Its own constitutive residual
is checked directly. It does not preserve the exact finite-spring equilibrium
extension, which is an expected consequence of preaveraging, not an undisclosed
extra measurement.

For a Hookean connector both models are exactly the same Gaussian process, with
`Cxx=T/(1-zeta*rate/2)` and `Cyy=T/(1+zeta*rate/2)`. Six signed nonzero rates and
three temperatures provide 18 distinct settings, each repeated eight times.
Every nonzero response is strictly monotonic in positive drag (with sign set by
the rate). Full-range objective profiles and noiseless endpoint recovery verify
identifiability. The fixed .0005 instrument sigma is independent of response
and drag. Public and private calibration copies are identical.

## Independent reference and controls

The private reference does not use the radial/Bessel reduction or the stress
moment identity. It integrates the actual nonlinear force dipole in nested
Cartesian coordinates over the disk. A separate OU Lyapunov solve checks the
Hookean calibration. The validator also evaluates the singular force with an
independent radial weight, verifies stationary probability current, stress
reversal, positive flow power, equilibrium isotropy, finite-extension bounds
and the source's own steady conformation equations.

The public finite-spring domain has `b in [5,15]`. At the boundary the probability
scales as distance to power `b/2`, so the first and second force moments have
integrable exponents at least 1.5 and .5. The direct Cartesian calculation agrees
with the oracle within `6.31e-8` relative over 56 corner/rate-sign/interior cases;
80-to-128-point refinement is below `5.71e-8`. The separate radial force check
agrees within `1.56e-13`. No hidden signal is close to zero: their magnitudes lie
between 1.061 and 6.317 energy units.

All 256 noise realizations pass the calibration and parameter gates for both
completed controls. All correct hidden predictions pass; all three finite-spring
groups reject the source at the unchanged .04 group normalized-RMS gate. Nominal
source errors are .583, .580 and .643. Worst noisy correct error is .000282;
smallest noisy source error is .5803. Local isolated controls are recorded in
`results/fene-stress-r1-local-controls.json`.

The public fit is an ordinary unfinished fit with `drag=None`; both completed
controls fit the same positive drag using bounded weighted least squares. The
neutral instruction permits replacement of any implementation. The public
apparatus specifies the force law, thermostat, preparation and measured stress,
without naming the closure correction or prescribing a solver. Private true
values, hints and references are excluded from the agent image.

Reproduce scientific checks with:

```sh
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/fene-stress-r1/scripts/validate_fene_stress.py
```

Use `--generate` only for intentional calibration replacement. Seeds and hashes
are in the stage's reports. The preserved prototype contains the initial screen
and an outside-domain large-`b` quadrature overflow; the final prototype uses a
controlled direct-radial limit check. These development artifacts are labeled
and are not task data, grading references or model outcomes.

## Overlap and causal classification

A broad archive/staging/task author-file search for Peterlin, FENE,
finite-extensibility and nonlinear-spring found no existing packaged family.
The archived active-trap task concerns non-Gaussian propulsion; the elastic-ring
task concerns stiff versus rigid equilibrium measures. The broad theme of a
statistical closure is related, but neither is this nonlinear molecular-force
approximation or rheometric measurement.

A trial that retains Peterlin force preaveraging has the intended physical
failure. An agent deriving the correct molecular distribution but making a
force-quadrature, drag-factor, sign or coding mistake has a mathematical or
implementation failure instead. Inspect discarded diagnostics and the complete
trajectory before classifying a final submission. Scientific separation alone
does not establish Luna failure rates.

Primary background: [On the Peterlin approximation for finitely extensible
dumbbells](https://doi.org/10.1016/S0377-0257(96)01497-8) and
[Validity of a macroscopic description in dilute polymeric solutions](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.62.1441).
