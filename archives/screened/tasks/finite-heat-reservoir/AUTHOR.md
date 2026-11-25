# Finite heat reservoir, revision 1

The completed shortcut treats the two internal oscillators as a canonical heat bath at temperature E/3. It has a correct normalized Boltzmann density and accurate moment quadrature. Harmonic calibration measures only <q²>=E/(3k), which this shortcut matches exactly. Its inferred total energy is correct.

The apparatus is instead isolated at fixed total energy. Ergodicity is stated explicitly, with weak nonlinear mixing and equilibration before neglecting coupling energy; purely harmonic coupling would not justify that assumption. Integrating the probe momentum and the two internal-mode phase spaces leaves position weight (E−U(q))^(3/2), supported where U<=E. This finite-reservoir distribution has different higher moments even in a harmonic potential. A pure single-well quartic potential keeps the accessible surface connected.

The oracle integrates in q. The reference changes variables to potential energy; independent tests use Beta-distributed energy partitions for pure powers and direct uniform-sphere sampling of a harmonic constant-energy surface. Kurtosis errors use absolute RMS limit .025. For the harmonic apparatus the correct kurtosis is 2.25; the canonical value is 3. These differences are large relative to calibration noise and integration error.

Finite-reservoir phase-space statistics are described in [Finite thermal reservoirs and the canonical distribution](https://www.sciencedirect.com/science/article/abs/pii/S0378437117304533). This apparatus, calibration and tests are independently constructed.

Validation command: `uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_finite_heat_reservoir.py --noise-trials 256`. Add `--generate` only to intentionally regenerate public and private measurements together. Peer review and measured Luna scores pending.
