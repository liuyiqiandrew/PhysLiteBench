# Accelerated detector pair, revision 1

The task separates single-worldline Unruh thermality from spatial correlations
between two accelerated detectors. Both controls predict every isolated
detector rate exactly. The completed approximation replaces the sampled
Minkowski vacuum by a homogeneous inertial thermal scalar bath at temperature
a/(2*pi). That is a definite positive field state, but its correlations
between separated points differ from those sampled by the prescribed pair.

The public apparatus fixes the canonical scalar Wightman normalization,
worldlines, transverse separation, common proper clock, identical real
coupling phases, initial states, and resolved final state. The readout is the
stationary second-order transition rate into gg, rather than total loss from
the initial Bell state. The latter also includes transitions to ee. No
switching, finite-time, recoil, or frequency-shift convention is left for
the agent to choose.

## Physical response

With the Fourier convention G(omega)=integral exp(i*omega*s) W(s) ds, the
downward local spectrum is

```
G_local = omega/[2*pi*(1-exp(-2*pi*omega/a))].
```

At zero acceleration this is omega/(2*pi). Along the pair's worldlines,
the cross correlator is

```
W_cross(s) = -1/[4*pi²*(4*sinh²(a*(s-i0)/2)/a²-L²)].
```

Its Fourier spectrum equals the local spectrum times

```
f = sin[(2*omega/a)*asinh(a*L/2)]
    /[omega*L*sqrt(1+(a*L/2)²)].
```

The resolved symmetric/antisymmetric rates are mu²*G_local*(1+/-f).
Each individual lowering matrix element has magnitude 1/sqrt(2), giving
one local term and one signed cross term, with no extra factor two. The
thermal-bath shortcut instead uses f=sinc(omega*L). It therefore includes
collective interference, all spatial inputs, and the exact local occupation
factor. The missing physics is the accelerated spatial field correlation.
At a=0 the two correlation functions coincide for every separation.

## Independent reference and validation

The independent reference numerically Fourier transforms the pulled-back
vacuum correlator. It shifts the integration line downward by a positive
c inside its analytic strip, then includes the factor exp(omega*c). With
c=min(1/abs(omega),pi/a), that factor remains moderate; the inertial case
uses c=1/abs(omega). Real cosine and imaginary sine transforms on the
positive half-line reconstruct the full spectrum. The reference contains
neither the analytic cross factor nor the Planck formula. Refinement changes
both integration tolerance and contour depth.

Across 88 domain and boundary cases, unit-coupling rates agree within
7.5e-15. Hidden reference changes are below 4.4e-13 in group relative norm.
Direct integration at positive and negative frequencies independently checks
the KMS relation at moderate acceleration, within 2.3e-16 absolute. Further
checks verify positivity, dimensional scaling, and that the two Bell rates
sum to twice the local rate. None of these checks substitutes KMS for the
upward integral.

Calibration has 224 readings at 28 settings: 16 single-detector settings and
12 inertial-pair settings, each repeated eight times. All share a scalar
mu², which multiplies known nonzero coefficients; weighted least squares
therefore has a unique minimum. The true value is .00011, fixed Gaussian
sigma is 2e-8, and data seed is 611401. Both controls fit .000110000058617
with reduced chi-square .980597.

The noise study uses 256 datasets and seed 611409. All calibration and
parameter checks pass. Maximum oracle prediction error is .01243%; the
weakest shortcut diagnostic group remains above 40.6%. Each group uses
prediction error norm divided by true rate norm, below 4%. Parameter error
must be below 3%, and reduced calibration chi-square below 1.5. Diagnostic
rates are nonzero and the distinction is much larger than noise or numerical
uncertainty. Frozen hidden truth keeps grading fast; ordinary validation
recomputes it.

Run `scripts/validate_accelerated_detector_pair.py` in the pinned numerical
environment. Only `--generate` replaces both calibration copies and hidden
truth. The report is
`results/expansion-accelerated-detector-pair-r1-validation.json`.

## Evaluation and related work

After source review and both Docker controls, evaluate exactly three frozen,
unhinted gpt-5.6-luna High trials under the standard 600/60-second limits.
Inspect source and trajectories. Extending the single-detector thermal
equivalence to the full inertial spatial bath is a physical failure. Deriving
the vacuum cross correlator correctly but making a Fourier-sign, factor,
or quadrature error needs a separate classification.

The archived circular-detector candidate studied the global temporal
correlator on one rotating worldline. This task has exact local response
and tests a pair's off-diagonal spatial correlations. It does not reuse
that local temporal closure.

Primary context and independent analytic confirmation are
[Hu and Yu, Entanglement dynamics for uniformly accelerated two-level atoms](https://arxiv.org/abs/1501.03321),
especially their scalar-field cross spectrum. The task uses only the resolved
leading-order transition rate; it does not require their full entanglement
dynamics or the electromagnetic-field variant of the problem.
