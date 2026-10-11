# Counterion cloud, revision 2

Revision 1's three agents all recovered the monovalent Manning limit.
That full revision is archived. Revision 2 adds separately conserved
monovalent and trivalent ions. It tests selective condensation, using a
completed mean-valence PB closure that already includes condensation.

Write lambda=u*tau0, xi=lB*lambda, and g=1-f+f/3. Total ion number per rod
length is lambda*g; f is the fraction of neutralizing charge supplied by
trivalent ions. Both species share one potential, but have different
Boltzmann factors and separately fixed populations.

In Bjerrum units, let v1 and v3 be the numbers of uncondensed ions per
length. The coefficient of log(R/a) in the PB free energy is

    E=(v1+3*v3)^2-2*(v1+v3),
    0<=v1<=(1-f)*xi, 0<=v3<=f*xi/3.

The first term is the residual rod's electrostatic energy; the second is
the radial entropy of free particles. Minimization selects trivalent
condensation first. With xeff=v1+3*v3 and vfree=v1+v3, the four regimes are:

| Condition | xeff | vfree |
|---|---|---|
| xi<=1/3 | xi | xi*g |
| xi>1/3, (1-f)*xi<=1/3 | 1/3 | 2*(1-f)*xi/3+1/9 |
| 1/3<(1-f)*xi<=1 | (1-f)*xi | (1-f)*xi |
| (1-f)*xi>1 | 1 | 1 |

The last regime has all trivalents condensed and monovalents partly
condensed. Both species cannot be partly condensed simultaneously. Pure
species limits follow continuously, with empty intermediate regimes.

The finite-cell virial identity gives

    phi_R=(1-a^2/R^2)*(1-xi/(2*g)+pi*a^2*n_total(a)/(lambda*g)).

Applying the same identity to the condensed layer gives the limiting
pressure phi=(vfree-xeff^2/2)/(xi*g)=-min(E)/(2*xi*g). This includes the
electrostatic contribution of the outer cloud, not just its ideal number.
The public apparatus fixes the canonical finite-cell-then-dilution order.

The shortcut replaces the species by ions of mean valence zbar=1/g,
preserving both total ion count and charge. It implements the complete
single-species PB limit at effective coupling xi/g. This is exact at pure
species endpoints and whenever 3*xi<=1. It misses the different valence
populations selected by condensation in mixtures.

Calibration contains 48 settings with five independent replicates,
sigma .001, seed 619001. All have u<=.15, so 3*xi<=.7875 over the full
parameter interval. The shared law 1-xi/(2*g) identifies tau0 uniquely by
weighted linear regression. True tau0 is 1.3 nm^-1. Hidden groups test
partial trivalent condensation, exhausted trivalents with free monovalents,
and exhausted trivalents with partly condensed monovalents. Pure-species
anchors and radius variation check shared limits. The group error is RMS
error divided by truth RMS, with a 4% gate. Noise validation uses 256 draws
with seed 619009. Exact fitted errors are retained in the validation JSON.

The oracle uses the four-regime formula. The private limit reference
minimizes the bounded convex free energy numerically, without those
branches. This is an independent implementation of the limiting law.
The independent physical check solves the finite-cell coupled PB boundary
value problem using log densities and the electric field, imposing both
species counts. Log radii 80,160,320 establish dilution convergence in all
four regimes. Dense quadrature checks each species population and the
finite-cell contact identity at log radius 12. Random/boundary and
transition checks compare the two limit implementations; 41 noiseless
fits establish parameter recovery. See the saved validator report.

The public image contains only the apparatus, starter, calibration and
interface test. Author derivation, hint, solution and private truth remain
outside it. The mixture concept review is
results/expansion-counterion-mixture-concept-review.md. Multicomponent PB
context: Deserno and Holm, *Cell model and Poisson-Boltzmann theory*,
https://arxiv.org/abs/cond-mat/0112096. The mixture limit above was checked
by derivation and finite-cell numerics.

```bash
UV_CACHE_DIR=/tmp/physlite-uv-cache uv run --offline --no-project --python 3.13 \
  --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 \
  python scripts/validate_counterion_cloud.py
python3 scripts/run_science.py tasks/counterion-cloud --agent oracle --trials 1
python3 scripts/run_science.py tasks/counterion-cloud --agent oracle --trials 1 \
  --solution-model scripts/counterion_cloud_baseline.py
```

Only `--generate` replaces the calibration and private truth. After peer
source review and fresh Docker controls, freeze exactly three plain
`gpt-5.6-luna` High trials, concurrency three, built-in web search disabled.
Shell networking remains available. Retain every outcome and review final
source and trajectories before classifying failures. Do not rerun an
unchanged completed revision.
