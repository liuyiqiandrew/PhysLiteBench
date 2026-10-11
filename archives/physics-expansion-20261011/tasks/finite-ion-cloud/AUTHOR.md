# Finite ionic cloud, revision 1

This is a finite canonical gas with genuine two-dimensional logarithmic
interactions, not a three-dimensional cylindrical mean-field cell.
Calibration measures a regime where all ions occupy the outer cloud.
At stronger coupling the number of outer ions is correlated, even though
permutation symmetry gives each labeled ion a fractional outer probability.

For k ions at scale R, configurational entropy contributes 2k log(R/a).
The central, bound/free and free/free energies together contribute
Gamma*k*(k+1)*log(R/a)/2. Thus the leading partition exponent and pressure are

    A_k = 2*k-Gamma*k*(k+1)/2,
    phi = max_(k=0,...,N) A_k/(2*N).

Pressure is continuous at adjacent-sector ties Gamma=2/j and vanishes at
Gamma>=2. N stays finite before the radius limit. The fixed disk has no
translational pressure. No image force or charge reservoir is implicit.

The completed shortcut is a product distribution for localization. Each
ion independently belongs to the outer cloud with probability p=n/N, so
K is binomial and E[K(K+1)]=(1-1/N)*n^2+2n. Its variational exponent is

    A_H(n) = (2-Gamma)*n-Gamma*(1-1/N)*n^2/2.

Maximizing over 0<=n<=N gives the supplied response. For N=1 it is exact.
It retains finite-N pair counting but omits the anticorrelated localization
that fixes the total free count away from ties. Both models have the exact
all-free pressure 1-Gamma*(N+1)/4 and the same first threshold 2/N.
A real-valued relaxation of the exact integer maximum was rejected as a
control: its error can occur for a single ion and is not a clean correlation
mistake. That rejected concept has never been evaluated.

The private reference integrates ordered logarithmic radii rather than
maximizing sectors. Angular pair-distance factors are bounded above by
constants times the larger radius and bounded below on separated angular
sectors. These constants do not grow with R. The full and ordered-radial
partition functions therefore have the same leading logarithmic exponent.
A triangular matrix exponential evaluates the nested radial integrals.
Pressures at log(R/a)=3200,6400,12800,25600 are extrapolated in inverse-log
powers; refinement doubles all four lengths. This also handles exact ties,
where the partition function acquires a polynomial prefactor. Tests at and
within .001 of every transition have maximum limit error below 4.7e-7.
The elementary finite-radius one-ion integral agrees to 4.5e-16.

Independent scrambled Sobol integration retains every angular interaction
in the full Gibbs weight at log(R/a)=40 and 80. Six cases, four scrambles
of 2^18 points each, cover all-free, partially bound, fully bound and a tie.
The finite logarithmic partition ratios agree with the radial comparison
within sampling and finite-size errors. At a tie this ratio is not the
infinite-radius pressure: the logarithmic prefactor still contributes.
These checks support the asymptotic argument; they are not grading truth.

True reference coupling g=1.12. Calibration has 24 settings, ten independent
replicates each, sigma .001, seed 620001. Gamma<=.42 throughout calibration
and the allowed parameter range, below the first threshold .5. Weighted
linear regression uniquely recovers g. The fitted value is 1.11981305955,
with reduced chi-square 1.12996. The hidden 4% relative RMS gate separates
oracle errors below .132% from shortcut errors of 30.95%, 68.90%, 85.80%.
Shared one-ion anchors pass both controls. Noise validation uses 256 draws,
seed 620009; its worst oracle error is .642%, safely below the gate.
There are also 160 domain checks and 41 noiseless parameter recoveries.

The public specification fixes the Hamiltonian, ensemble, limit order,
pressure derivative, units, parameter range and API. Diagnosis, hidden
cases, hint and reference remain outside the ordinary agent image.

Primary physics sources:
- Mallarino, Tellez and Trizac, *The contact theorem for charged fluids*,
  equations 37–40, https://arxiv.org/pdf/1410.6499.
- Burak and Orland, https://arxiv.org/pdf/cond-mat/0509126.
- Varghese, Vemparala and Rajesh,
  https://www.imsc.res.in/~rrajesh/Publications/pe_inequivalence.pdf.
Their Manning parameter is N*Gamma/2 in this task's convention.

Validate from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/physlite-uv-cache \
uv run --offline --no-project --python 3.13 --with numpy==2.3.3 \
  --with scipy==1.16.3 --with pytest==8.4.2 \
  python scripts/validate_finite_ion_cloud.py
python3 scripts/run_science.py tasks/finite-ion-cloud --agent oracle --trials 1
python3 scripts/run_science.py tasks/finite-ion-cloud --agent oracle --trials 1 \
  --solution-model scripts/finite_ion_cloud_baseline.py
```

Only explicit `--generate` regenerates data. Freeze the package after
review and controls, then run exactly three gpt-5.6-luna High plain trials.
Disable built-in web search; shell networking remains available. Inspect
all final source and trajectories before classifying any failure.
