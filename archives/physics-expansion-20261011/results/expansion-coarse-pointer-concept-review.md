# Coarse pointer readout: concept review

Verdict: approve the displacement-coupling version for development. It is minimal, fully specified in principle, and has a valid completed quantum-instrument shortcut with exactly shared calibration. It is distinct from the closest archived quantum-work and qubit-control tasks. No task implementation or model evaluation was performed; the checks below are independent calculations for this concept.

## Apparatus and exact channel

Use pointer position `q` and known displacement `d` in millimeters, conjugate momentum satisfying `[q,p]=i`, and the prescribed impulsive unitary `U=exp(-i*d*sigma_z*p)`. The qubit begins in `|+x>`. The initial pointer is the pure state

```
psi_sigma(q) = (2*pi*sigma^2)^(-1/4) exp[-q^2/(4*sigma^2)],
```

so its position variance is exactly `sigma^2`. This fixes the factor-of-two convention, zero position mean, and absence of an unmentioned pointer phase or mixed-state preparation. The impulse is followed by pointer-position measurement and the specified qubit analyzer, with no extra free evolution or feedback. Only the Boolean event `q>=h` is retained. The readout is the joint probability per prepared shot, not an analyzer probability normalized by the accepted fraction.

Let `Phi` denote the standard normal CDF. The accepted branch maps qubit density matrix entries as

```
rho++ -> a+ rho++,     a+ = Phi(( d-h)/sigma),
rho-- -> a- rho--,     a- = Phi((-d-h)/sigma),
rho+- -> c rho+-,      c = exp[-d^2/(2*sigma^2)] Phi(-h/sigma).
```

These follow from the translated pointer amplitudes `psi(q-d)` and `psi(q+d)`. The overlap exponent is `d^2/(2*sigma^2)`, not `d^2/(8*sigma^2)`; the latter applies if the symbol denotes the full separation between pointer centers.

For analyzer state `cos(theta/2)|+z> + exp(i*phi) sin(theta/2)|-z>`, the accepted joint probability is

```
P = (a+ + a-)/4 + cos(theta)*(a+ - a-)/4
    + sin(theta)*cos(phi)*c/2.
```

The rejection branch replaces both CDF tails by their complements and has coherence coefficient `exp[-d^2/(2*sigma^2)] Phi(h/sigma)`. Summing the branches preserves trace and leaves the expected unconditional pointer-induced dephasing. At `d=0`, both branches only multiply the original state by their classical acceptance probabilities.

## Valid shortcut and calibration

The shortcut uses the binary effects `E_A=diag(a+,a-)` and `I-E_A`, with Kraus operators equal to their positive square roots. This defines a complete, positive, trace-preserving instrument when branches are combined. The accepted off-diagonal coefficient is `c_short=sqrt(a+*a-)`; populations and acceptance probabilities remain exactly correct for every parameter and preparation. Cauchy-Schwarz applied to the two accepted pointer wavefunctions gives `0<=c<=c_short`, strictly for finite threshold and nonzero displacement.

Thus the shortcut assumes the square-root instrument associated with the observed binary effects instead of the instrument selected by the stated pointer coupling. This is a physical measurement-model assumption, not an incorrect probability normalization. Distinct POVMs and instruments, including binary square-root implementations, are treated explicitly by [Reitzner and Bouda, sections 1.3 and 2.2](https://link.springer.com/article/10.1007/s11128-024-04468-w). No specialized result is needed to derive this task: translating a Gaussian and applying the Born rule suffices.

For `h>d>=0`, both `a+` and `a-` increase strictly with positive `sigma`, since their normal-CDF arguments are negative and approach zero as the width grows. Acceptance `(a++a-)/2` therefore identifies the common width globally on `[.6,1.0]`; diagonal-analyzer joint probabilities are also shared exactly. In contrast, acceptance at `h=0` is always one half for the proposed initial state and cannot identify width by itself. This edge case must not become the only informative calibration.

The phrase “maximum coherence” should be restricted to the nondemolition, population-preserving class of diagonal instruments. A general measure-and-prepare instrument can create other coherences, so no unrestricted extremality claim is needed. Also, discarding a classical record is not a quantum eraser. Even a binary projector on this same pointer followed by tracing it produces the integrated branch above. The physical distinction is between the specified pointer instrument and the direct binary square-root instrument on the qubit.

## Independent reference and separation

An independent reference can integrate the outgoing joint wavefunction directly:

```
P = integral_h^infinity |[cos(theta/2)*psi(q-d)
         + exp(-i*phi)*sin(theta/2)*psi(q+d)]/sqrt(2)|^2 dq.
```

This avoids the closed overlap coefficient and the reduced density-matrix formula. Check acceptance with the spin-summed density, complementary outcomes, probability bounds, and quadrature refinement. If using a finite integration endpoint, the omitted joint probability is bounded by the corresponding tail of `[psi(q-d)^2+psi(q+d)^2]/2`; no Monte Carlo reference or large pointer Hilbert space is necessary.

I evaluated this integral adaptively for 96 settings: 32 corners/analyzer choices and 64 random preparations in `sigma in [.6,1.0]`, `d in [0,1.8]`, `h in [-2,2]`, and the full proposed angle ranges. Maximum absolute discrepancy from the closed expression was `3.33e-16`; probability and Cauchy-Schwarz bounds passed. For `sigma=.8`, `d=1.4`, `h=-2`, and the `+x` analyzer, the true joint probability is `.5508016060354646`, while the shortcut gives `.8830461815187303`, a `60.32%` relative error. For `d=1` at the same width and threshold, the nonzero probabilities are `.7010606371720047` and `.9463950067301139`. Strong separation does not require destructive-interference zeros or tiny acceptance probabilities.

## Novelty and scope

The archive search found no existing pointer-instrument task. I reviewed the quantum-work revisions, quantum-calorimetry, paired-spin-noise, and qubit-control r9 descriptions. Quantum-work substitutes a single energy-change observable for two projective energy records; qubit-control concerns initial system-bath correlations or coherent oscillator dynamics. Here both controls implement valid instruments with the same observed effects, but their state changes differ because the apparatus resolves a continuous pointer. That is a distinct physical assumption despite the shared use of qubit coherence and operational measurements.

Proceed with the stated displacement apparatus, an explicit analyzer ket, and joint-probability readout. Keep the explanatory instrument comparison private. Scientific validity and separation do not predict whether the target model will make the intended error; later trial classification must distinguish an assumed square-root instrument from mistakes in Gaussian integration, angle conventions, or normalization.
