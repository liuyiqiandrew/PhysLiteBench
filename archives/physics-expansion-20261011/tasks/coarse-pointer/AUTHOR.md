# Coarse pointer, revision 1

The public apparatus fixes a particular measurement instrument, including
its pure pointer state, unitary displacement, discarded pointer, timing,
and joint readout. A binary POVM alone would not fix the state update.

The outgoing spinor is `(psi(q-d),psi(q+d))/sqrt(2)`. Integrating over
`q>=h` gives accepted populations `a_plus/2,a_minus/2`, with
`a_plus=Phi((d-h)/w)` and `a_minus=Phi((-d-h)/w)`. The accepted coherence
is `c/2`, where `c=exp(-d^2/(2*w^2))*Phi(-h/w)`. Projection onto the
specified analyzer gives

```
P_joint = (a_plus+a_minus)/4 + cos(theta)*(a_plus-a_minus)/4
          + sin(theta)*cos(phi)*c/2.
```

The completed shortcut uses the positive square root of each binary POVM
effect as its Kraus operator. It is a valid trace-preserving instrument
when acceptance and rejection are combined, with accepted coherence
`sqrt(a_plus*a_minus)/2`. It reproduces every acceptance probability and
all diagonal analyzer probabilities, but it is not the instrument made by
the stated pointer. Discarding a fine classical record does not restore
quantum coherence. Even a binary projection on the same pointer followed
by tracing it gives the physical integrated channel.

Calibration has 18 settings repeated 12 times, with independent Gaussian
noise of standard deviation 0.0005. The true width is 0.83 mm. Every
threshold exceeds the displacement, so both Gaussian tails strictly
increase with width. The calibration identifies width globally on its
allowed interval. Hidden cases use nonzero transverse analyzer components
and wider accepted intervals; shared anchors include zero coupling,
diagonal analyzers, and acceptance-only readouts.

The oracle evaluates the closed Gaussian integrals. The private reference
instead integrates the squared analyzer projection of the full translated
pointer spinor. The validator checks adaptive quadrature refinement, a
4097-point position-grid integration, probability bounds, positivity of
both outcome maps, complementary analyzer outcomes, and the total pointer
overlap. It also checks 41 noiseless fits and 256 independent calibration
noise draws. Ordinary runs read checked-in data; `--generate` intentionally
replaces public and private measurements and hidden truth together.

Calibration seed: 614001. Noise seed: 614009. Domain seed: 614013.
Prediction groups require relative Euclidean error below 4%; the parameter
gate is 3%, and reduced calibration chi-square must be below 1.5. The
validation report records numerical error and separation from noise.
The standard 600/60-second agent/verifier limits and pinned dependencies
are unchanged. Only `environment/` enters the public Docker image.

Run `scripts/validate_coarse_pointer.py` with the repository's pinned
NumPy/SciPy environment. Docker controls use `scripts/run_science.py` with
`--agent oracle`, then with `--solution-model scripts/coarse_pointer_baseline.py`.
The revision is frozen before three unhinted gpt-5.6-luna High trials.
Count an intended failure only if the retained code assumes the binary
square-root instrument. An algebra, Gaussian-width, normalization, or
analyzer-convention error is a different failure class.
