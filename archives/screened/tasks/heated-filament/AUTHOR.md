# Heated filament — revision 1

This backup tests whether spatially independent thermal reservoirs are mistaken for independent modal reservoirs. Uniform-temperature calibration observes mixed-sensor spectra and identifies one drag coefficient. The completed shortcut assigns every mode its correct local-temperature average but omits correlations between modes. It remains exact for any single-mode sensor under every allowed temperature profile.

## Physical derivation

Let phi_n=sqrt(2/pi)*sin(n*x), u=sum q_n*phi_n, and T_mn=integral T(x)*phi_m(x)*phi_n(x) dx. The stated local force balance gives

    gamma*dq_n/dt = -n^2*q_n + xi_n,
    <xi_m(t)*xi_n(t')> = 2*gamma*T_mn*delta(t-t').

For a sensor with weights w_n, define h_n=w_n/(n^2+i*gamma*omega). The two-sided angular-frequency spectrum is

    S_Q(omega) = 2*gamma*h^dagger*T*h.

The temperature matrix is positive because T(x)>0. Deterministic modal relaxation is diagonal. Although all filament modes are forced, higher modes cannot feed back into the first six and the measured projection is exact without a dynamical truncation approximation.

The shortcut retains only T_nn and returns 2*gamma*sum T_nn*|h_n|^2. It is the internally consistent model of independently forced modes at their individual effective temperatures. All supplied inputs affect that model. Uniform T makes the off-diagonal elements vanish, so calibration cannot distinguish the two models. Nonuniform T correlates modal forces. Signed mixed sensors then measure interference terms that survive frequency averaging.

Calibration contains 128 independent observations (32 distinct preparations repeated four times), with uniform temperatures, four mixed sensors and frequencies 0,0.12,0.3,0.45. For every allowed gamma<=1.8, each modal contribution gamma/(n^4+gamma^2*omega^2) increases strictly with gamma: n^4-gamma^2*omega^2>=0.3439. Thus the calibration parameter is uniquely identifiable. Both completed models use the same bounded weighted fit. Seed20371; uncertainties are 0.8% of each noiseless spectrum.

## Independent reference and validation

The private reference solves the spatial Dirichlet adjoint equation

    (-partial_x^2+i*gamma*omega)*chi=w,
    S_Q=2*gamma*integral T(x)*|chi(x)|^2 dx.

It uses a centered second-difference tridiagonal solve and Richardson extrapolation of 256/512 intervals, independently of modal force covariance. The author validator also evaluates the continuous Dirichlet Green function with split quadrature on either side of its derivative discontinuity. It checks uniform-bath fluctuation-dissipation, single-mode exactness, spatial reflection, positivity of the temperature matrix, and agreement of the integrated two-sided spectrum with the stationary Lyapunov covariance C_mn=2*T_mn/(m^2+n^2).

Run from the repository root:

    OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_heated_filament.py

Use `--generate` only to regenerate calibration intentionally. The default performs256 noise fits plus hidden sensitivity at their observed fitted-parameter extrema; it does not claim exhaustive hidden Monte Carlo testing. Reports are `results/heated-filament-science.json`, `jobs/heated-filament-validation/summary.json`, and `results/heated-filament-local-controls.json`.

The checked-in fit is gamma1.0995441, reduced chi-squared1.08474. Across256 realizations the fitted range is [1.0975921,1.1022673]. Oracle hidden errors are below0.00036; shortcut errors are0.3530,0.4217,0.3613 in the three mixed-sensor groups. Both pass the separate single-mode anchors. The grading tolerance0.025 exceeds calibration uncertainty and reference error by a wide margin. Independent reference disagreement is below5.3e-12 absolute on hidden inputs and1.3e-11 at selected allowed-range corners; direct Green quadrature agrees below4.7e-19. Local controls: oracle8passed; shortcut5passed/3hiddenfailed in about0.33seconds each. Harbor controls subsequently passed as expected; reviewed Luna outcomes are recorded below.

Background: Fontana and Bellon, [Linking fluctuation and dissipation in spatially extended out-of-equilibrium systems](https://arxiv.org/abs/2208.08356), studies the relationship between spatial temperature, dissipation and mechanical fluctuations. This is background for the local-bath framing; the task's exact modal covariance and independent spatial/Green references establish its own predictions.

## Reviewed evaluation

The frozen pair at `jobs/matrix-hardening-r1-20261002-081016/index.json` produced **3/3 plain passes and 3/3 hinted passes** for gpt-5.6-luna at high reasoning effort. All trials completed without exceptions. Both control outcomes were as expected.

Every unhinted submission explicitly recovered the full projected modal covariance from the local noise law and correctly included cross terms between the signed complex sensor responses. The three hinted submissions made the same physical correction. All six fitted the drag correctly and had maximum normalized hidden error below0.000360. Public trajectories and final sources establish physical success; there are no failed trials requiring causal repairs. This family does not qualify for the requested low-pass-rate catalog and should be archived as a completed screen. All six reviews and source hashes are in `results/brownian-hardening-trial-reviews.json`.
