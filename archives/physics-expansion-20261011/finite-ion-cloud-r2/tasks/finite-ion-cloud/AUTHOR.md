# Finite ionic cloud, revision 2

Revision 1 measured the infinite-dilution pressure and passed all three
Luna High trials. It is preserved in the expansion archive. Revision 2
measures pressure at a specified finite outer radius and limits N to one
or two ions. The physical issue is the relative-angle correlation, which
changes finite-cell pressure even though every one-body angle is uniform.

The public Hamiltonian is a genuine two-dimensional logarithmic Coulomb
gas in a homogeneous dielectric. Neutrality, both hard walls, the finite
canonical ensemble, pressure derivative and accessible-area normalization
are explicit. No large-N or dilution limit is taken. Absolute radius drops
out at a fixed radius ratio, but the ratio R/a remains observable.

For two ions, order their logarithmic radii as t1=s, t2=s+d, with
0<d<L=log(R/a) and 0<s<L-d. Write A=4-3*Gamma and B=2-Gamma. Integrating
the overall rotation and relative angle gives

    H(d) = <(1+exp(-2d)-2exp(-d)cos(theta))^(Gamma/2)>_theta
         = hyp2f1(-Gamma/2,-Gamma/2;1;exp(-2d)),
    Z(L) proportional to integral_0^L H(d)*exp(B*d)
                                      *(exp(A*(L-d))-1)/A dd,
    Z'(L) proportional to integral_0^L H(d)*exp(B*d+A*(L-d)) dd,
    phi = (1-exp(-2L))*Z'(L)/(4Z(L)).

At A=0 the quotient is L-d; exprel implements this continuously.
R-independent angular and indistinguishability factors cancel. The
one-ion answer follows from Z=integral_0^L exp((2-Gamma)*t) dt, with the
same finite-annulus factor and denominator 2 rather than 4.

The completed approximation takes independent uniform azimuths conditional
on the radii. Optimizing its radial Gibbs variational distribution gives
the angular mean potential log(max(r1,r2)/a). It retains radial correlations
and finite-N pair counting, but its angular Boltzmann factor is H=1.
A triangular matrix exponential evaluates that radial partition function
exactly. This is a valid completed physical approximation. Uniform angular
marginals do not justify independent relative angles: the Gibbs weight
favors ions on opposite sides. Averaging the potential before computing
the Gibbs distribution loses this correlation and overestimates pressure.
Unlike revision 1, restoring the integer free-ion count does not repair it.

Calibration uses only N=1, so both responses agree exactly for the entire
allowed coupling interval. There are 24 preparations, ten replicates each,
noise sigma .001 and seed 621001. True g=1.12; weighted nonlinear least
squares yields 1.11973766763 with reduced chi-square .79621. The hidden
4% relative RMS gate separates oracle group errors below .052% from
approximation errors 13.18%, 23.33%, 37.54%. One-ion anchors pass both.
Across 256 independent noise draws (seed 621009), the largest oracle error
is .170% and the smallest target approximation error is 13.04%.

The independent reference directly integrates the positive Gibbs weight
over radial gap, common radius and relative angle with Gauss-Legendre
quadrature. It uses neither hyp2f1 nor exprel. Its 96-node rule is checked
against 160 nodes; the maximum tested error against the analytic-angle
oracle is 3.41e-10, decreasing to 3.24e-11. Frozen hidden values agree under
refinement to 1.73e-15. Validation also includes 128 domain cases, 41
noiseless fits, the zero-coupling ideal gas and eighteen inner/outer contact
checks. For N=2, (B_outer-B_inner)/Z=4-3*Gamma; the largest residual is
3.74e-13. The pressure is computed from the outer mechanical contact.

The contact identity is consistent with Mallarino, Tellez and Trizac,
*The contact theorem for charged fluids*, equation 34,
https://arxiv.org/pdf/1410.6499. The explicit two-ion integrals above are
derived from this task's Hamiltonian, not copied from a mean-field formula.

Public files disclose the complete apparatus and API; this derivation,
the hint, hidden cases and reference remain outside the agent image.

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

Only explicit `--generate` regenerates data. After review and fresh Docker
controls, freeze all inputs and grading for exactly three gpt-5.6-luna High
plain trials. Built-in web search is disabled; shell networking remains
available. Review every final source and trajectory before classification.
