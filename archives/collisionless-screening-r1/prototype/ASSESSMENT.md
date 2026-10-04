# Collisionless screening prototype

This directory is author-only and contains no task or model-agent evaluation. The candidate outline and overlap assessment are in ../physics-outlines-20261004-plasma/ASSESSMENT.md. Final packaging is not authorized by this file.

With F(u)=15(1−u²)²/16 on [−1,1], define zeta=omega/(k v0). Direct division gives

H_PV=(15/4)[2 zeta²−4/3+(zeta³−zeta)log|(zeta−1)/(zeta+1)|].

The completed standing-response approximation is E/Eext=1/[1−n0 H_PV/(k v0)²]. For an exp[i(kx−omega t)] drive switched on as exp(eta t) from the remote past, eta>0, the causal boundary value is H=H_PV+i pi F'(zeta) inside the support. Outside support and at zero frequency the controls coincide exactly. The source must compute H_PV independently, not discard the imaginary component of a computed causal response.

For charge −1 the linearized kinetic equation is f1_t+i k v f1=E f0'. Combining its causal Duhamel solution with i k Eind=−integral f1 gives

E(t)=Eext(t)−n0 integral_0^infinity tau phi(k v0 tau) E(t−tau) d tau.

Here phi(s)=integral F(u)exp(−i s u)du=15 j2(s)/s², with phi(0)=1. A finite switch-rate forcing therefore has transfer [1+n0 integral tau phi(k v0 tau)exp[(i omega−eta)tau]d tau]^-1. The reference evaluates this causal time integral independently and extrapolates three positive eta values. It inserts no resonant jump or complex logarithm.

The maximum 18-case reference error is 9.55e−6 and becomes 1.21e−6 when eta values are halved. The nominal calibration cases are exactly equivalent; the hidden in-phase gaps range from .16078 to 2.86755. The independent principal-value quadrature checks the analytic source expression to 1.89e−15. These are deterministic science checks only; no noise validation or empirical difficulty claim is made.

The first run used the oscillatory quadrature routine at zero frequency and emitted a cycle-limit warning. Its exact script and report are preserved as initial-check.py and initial-report.json. The final code uses ordinary infinite-interval quadrature at zero frequency; it finishes without that warning and leaves the reported worst errors unchanged. No physical formula changed.

The distribution is single-peaked and stable, but may support undamped collective modes outside the velocity support. This is why a future public protocol must specify the remote-past adiabatic selection and why calibration frequencies must be separated from dielectric zeros. No interchange with an arbitrary suddenly switched finite-time steady state is claimed. For n0 in [.6,1.4], k>=1.5, v0>=.9 and |zeta|>=1.5 the positive source denominator is at least .57086; inside hidden ratios .2 to .65 it remains positive, at least .73001.

The peer concept review confirms the sign and kernel. A trial that recognizes the causal selection and then uses the wrong i0 sign must be classified separately from one that retains the standing-response approximation.
