# Finite ionic cloud, revision 3

Revisions 1 and 2 each passed all three Luna High trials and remain archived.
Revision 3 extends the finite cell to three ions. Its completed pair-cluster
control retains the exact one-ion and two-ion responses, so calibration
now includes both. The omitted physics is consistency of the three relative
angles around a triangle. Pairwise forces do not imply independent pair
configurations in equilibrium.

The public apparatus remains a neutral finite canonical gas with genuine
two-dimensional logarithmic interactions, hard inner and outer circles,
and a fixed uniformly charged disk. The full pair Hamiltonian, temperature
convention, units, range and pressure derivative at fixed N,a,T,Gamma are
explicit. The disk contributes no translational ideal-gas pressure. There
is no dilution limit or implicit three-body interaction.

Order the logarithmic radii as s, s+x, s+x+y, with L=log(R/a). Write

    W_d(theta) = (1+exp(-2d)-2exp(-d)*cos(theta))^(Gamma/2),
    c_m(d) = average_theta W_d(theta)*exp(-i*m*theta).

The exact angular factor for three ions is

    A(x,y) = average_(theta,phi) W_x(theta)*W_y(phi)*W_(x+y)(theta+phi)
           = sum_(integer m) c_m(x)*c_m(y)*c_m(x+y).

The oracle evaluates this convolution with a real Fourier transform of
positive pair weights. It keeps the zero, positive, negative and Nyquist
modes with their correct weights. The two-ion oracle integrates the exact
hypergeometric pair factor, and the one-ion response is elementary.

The completed control keeps only m=0, giving H(x)*H(y)*H(x+y), where
H(d)=hyp2f1(-Gamma/2,-Gamma/2;1;exp(-2d)). This is the conditional-angle
Bethe or pair-cluster partition approximation with uniform one-body
marginals. It retains all pair energies and radial correlations, while
neglecting the consistency of pair angles on the loop. It is not a
realizable Gibbs trial distribution of three positions, not the variational
bound used in revision 2, and not a model formed by deleting one pair force.
Its positive auxiliary partition function and pressure implement that
physical closure completely. Each pair tends toward opposite angles;
three pairs sharing three positions cannot do so independently.

After the common-radius integral, let C=6-6*Gamma, B=4-3*Gamma, D=2-Gamma.

    Z proportional to integral_(x,y>=0,x+y<=L) A(x,y)*exp(B*x+D*y)
                                             *(exp(C*(L-x-y))-1)/C dx dy.

The derivative Z' replaces the last quotient by exp(C*(L-x-y)). The
finite-annulus pressure is (1-exp(-2L))*Z'/(6Z). exprel handles C=0.
The control uses the same integral with its product-pair angular factor.

The reference integrates the positive microscopic Gibbs weight over two
linked angles, two radial gaps and the common radius. It uses no Fourier
convolution, and no analytic angular or common-radius reduction for N=3.
Radial and angular refinement check its frozen hidden truth. At Gamma=2,
the Vandermonde determinant gives an independent exact anchor:

    Z_N proportional to product_(j=0,...,N-1) integral_a^R r^(2j+1-2N) dr.

Equivalently, the exact three-ion angular factor is
(1+r^2)*(1+t^2)*(1+r^2*t^2)-2*r^2*t^2, r=exp(-x), t=exp(-y).
The missing loop term is explicit. Ideal-gas, domain, noise and convergence
checks supplement this anchor. Numerical errors at weak coupling and a
narrow annulus are reported separately from the much smaller errors at
the discriminating strong-coupling preparations.

Calibration has 48 preparations, five replicates each, sigma .001 and seed
622001. True g=1.12. Both controls agree exactly on the one-ion and two-ion
physical responses and use the same weighted scalar fit. The prediction
gate is 4% group RMS error divided by truth RMS; parameter error must be
below 3% and reduced calibration chi-square below 1.5. Noise validation
uses 256 draws with seed 622009. The saved validation JSON records fit,
noise margins, all reference refinements and eighteen determinant checks.
The fitted coupling is 1.11993343604, with reduced chi-square 1.06875.
Oracle target-group errors are at most .034%; the completed closure misses
10.98%, 19.00%, and 29.54%. Across noise draws, the largest oracle error is
.447% and the smallest target closure error is 10.79%. Hidden reference
refinement changes pressure by at most 8.12e-10. Across the broader domain,
the reference/oracle disagreement is at most 1.38e-5; the largest oracle
refinement change is 7.17e-6 at weak coupling. These absolute errors are
well below the 4% grading gate and do not drive the hidden separation.

The pair-cluster interpretation follows the general loop expansion of
Xiao and Zhou, https://arxiv.org/abs/1104.3631; its specialization here is
derived above. The finite-cell contact setting is described by Mallarino,
Tellez and Trizac, https://arxiv.org/pdf/1410.6499. This is a classical
angular frustration problem. The archived triad-photon-phase candidate
also uses pair calibration and a cycle, but tests a quantum exchange phase.

Author notes, hint, reference and private cases are excluded from the agent
image. The full apparatus and API remain public.

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

Data regeneration requires explicit `--generate`. After review and fresh
Docker controls, freeze inputs and grading for exactly three gpt-5.6-luna
High plain trials. Disable built-in web search; shell networking remains
available. Inspect final code and trajectories before classification.
