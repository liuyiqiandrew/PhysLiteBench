# Ring current fluctuations

Consider a continuous-time exclusion process on a periodic ring of an even number
of sites (L), containing exactly (L/2) particles. Each site holds at most one
particle. Initially, all configurations with this fixed particle number are
equally likely. There are no reservoirs or other interactions.

An occupied site attempts a jump to its right neighbor at rate

\[
 r_+=D\exp(E/L)
\]

and to its left neighbor at rate

\[
 r_-=D\exp(-E/L).
\]

The attempt succeeds only if the destination is empty. The periodic bond joining
the last and first sites follows the same rule. The rate prefactor (D) is common
to every experiment and is measured in inverse microscopic time units. The clock
is fixed across experiments; it is not rescaled when (D) changes. The field
(E) is dimensionless.

Let (J_+(t)) and (J_-(t)) count all successful right and left jumps,
respectively, across **all** bonds up to microscopic time (t). Blocked attempts
are not counted. Define

\[
 Q_t=\frac{J_+(t)-J_-(t)}{L}.
\]

For a dimensionless counting parameter \(\lambda\), the requested observable is
the scaled cumulant generating function

\[
 \psi(E,\lambda;D)=
 \lim_{\substack{L\to\infty\\L\ \mathrm{even}}}
 L\left[\lim_{t\to\infty}\frac{1}{t}
 \log\mathbb E\!\left(e^{\lambda Q_t}\right)\right].
\]

Take the long-time limit at each fixed (L) before the large-ring limit. The
expectation is over this fixed-particle-number process. The counting parameter
weights trajectories in this expectation; it does not change the physical jump
rates. Return the leading asymptotic value defined above, in inverse microscopic
time units. Ring size and observation duration are not prediction inputs.

The public input domain is

\[
 0\le E\le12,\qquad -2E-2\le\lambda\le2,
 \qquad0.8\le D\le1.2.
\]

The unknown parameter is (D). Each record in `data/calibration.json` contains
`input`, `value`, and `sigma`. The input has keys `field` ((E)) and `bias`
(\(\lambda\)). Values are independent noisy estimates of the stated asymptotic
observable, with independent Gaussian measurement errors of known standard
deviation `sigma`. Repeated inputs are independent measurements. The reported
errors are in the same units as \(\psi\).

Implement this interface in `model.py`:

```python
model = Model()
model.fit(records)              # returns model and stores model.diffusivity
values = model.predict(inputs) # NumPy array, shape (len(inputs),)
```

`diffusivity` must be finite and lie in `[0.8, 1.2]`. `predict` accepts a list of
`{"field": E, "bias": lambda_value}` dictionaries and returns finite raw
\(\psi\) values. It must also accept an empty list and return an empty array.
You may change any prediction function or helper while preserving this API.
