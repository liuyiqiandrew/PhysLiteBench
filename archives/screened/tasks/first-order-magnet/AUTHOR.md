# First-order magnet, revision 1

This is an independent candidate. It preserves `magnetic-equilibrium` revision 1
and replaces its unstable-root shortcut with a locally stable, field-aligned
metastable phase. It is not evidence of agent difficulty until Harbor trials and
source reviews are complete.

## Thermodynamics and calibration

For the stated infinite-range Hamiltonian, the thermodynamic free energy per
spin at magnetization m is

    f(m) = -J*m^2/2 - Q*m^4/4 - h*m
           + T*[(1+m)/2*log((1+m)/2) + (1-m)/2*log((1-m)/2)].

Here Q=1.2 is known and only J is fitted. All energies, including T with kB=1,
use one specified unit. Magnetization is dimensionless. Differentiating gives

    m = tanh((J*m + Q*m^3 + h)/T),
    f''(m) = T/(1-m^2) - J - 3*Q*m^2.

Canonical equilibrium is the global minimum of f, not an arbitrary stable
solution of this self-consistency equation. The apparatus explicitly allows full
equilibration and excludes phase-transition lines, so it does not ask for a
history-dependent state or an undefined mixture at exact coexistence.

Calibration uses T=1.6, 2.0, 2.5 and both field signs, with 16 magnitudes from
0.12 to 0.9 at each temperature. Every setting has two independent measurements.
For every allowed J in [0.3,0.8], free energy is strictly convex at these
calibration temperatures. The minimum curvature is at least
2*sqrt(3*Q*1.6)-3*Q-0.8=0.4. Thus the root near zero is the unique equilibrium
state and the shortcut is exactly calibration-equivalent. Nonzero fields make
magnetization sensitive to J, so this agreement does not leave J unidentified.

Data use J=0.55, seed 9213, and constant sigma equal to 0.006 times the largest
noiseless calibration magnitude. The noise-validation seed is 19213. Both data
copies are identical; only `--generate` replaces them.

## Hidden distinction and controls

Hidden temperatures are 0.70, 0.78, and 0.84, with both signs of fields having
12 magnitudes from 0.003 to 0.015. The small root reached from zero has the
correct field sign, a tiny equation residual, and strictly positive free-energy
curvature. An ordered state of the same sign has lower free energy. The shortcut
therefore cannot be repaired merely by enforcing bounds, following the field
sign, or rejecting unstable stationary points.

The oracle finds all stationary points by partitioning magnetization at zeros
of f''. These follow from the quadratic in y=m^2,

    3*Q*y^2 + (J-3*Q)*y + (T-J) = 0.

On each resulting interval f' is monotone. It brackets its roots and compares
free energies. In the convex calibration regime it uses one bracket. The
independent reference scans free energy for local minima and refines each with
bounded minimization; it does not solve self-consistency roots. The standalone
validator also enumerates competing minima independently on a dense root grid,
checks field reversal and convexity, and verifies both the shortcut-to-equilibrium
energy gap and the gap to every competing minimum.

Local controls require reduced calibration chi-square below 1.5 and fitted J
within 3%. Every hidden temperature group requires absolute magnetization RMSE
below 0.025. There is no division by small field or small magnetization.

The checked-in calibration yields J=0.54817838 and chi-square 0.96394 for both
controls. Oracle hidden errors are at most 0.000315; shortcut errors are
0.9231–0.9315. Oracle and independent-reference agreement is within 3.1e-8 on
calibration and 8.2e-9 on hidden inputs, well below the threshold.

The 256-noise validation fits both controls and evaluates all hidden groups on
every realization, rather than checking only fitted-parameter extrema. All
calibration and parameter checks pass. The maximum J error is 0.682%, maximum
oracle hidden RMSE is 0.000652, and minimum shortcut hidden RMSE is 0.9205.
Across every sampled fit and hidden state, the shortcut remains field aligned,
its residual is below 5e-17, its curvature exceeds 0.1105, and its free-energy
excess exceeds 0.00965. The closest competing minimum remains more than 0.00571
above equilibrium. These are measured margins over this validation set, not
exhaustive bounds over all possible Gaussian noise draws.

## Reproduction and independent review

Run `scripts/validate_first_order_magnet.py` from the repository root in the
pinned NumPy/SciPy environment. Its report is
`jobs/first-order-magnet-validation/summary.json`. Ordinary validation preserves
data; `--generate` is for intentional changes before a revision is frozen.
The completed shortcut is `scripts/first_order_magnet_baseline.py`.

A teammate independently checked the Hamiltonian, entropy, curvature, branch
selection, calibration convexity, units, and numerical controls. Their report is
`results/first-order-independent-review.json`. They identified the
need to exclude exact coexistence even at nonzero field; the public README now
states that condition. The hint explains the distinction between local and
global equilibrium without revealing parameters or hidden settings.
