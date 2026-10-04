# Spatial diffusion, revision 1

The public apparatus fixes the stochastic process through a uniform-temperature
bath, mechanical drag, fluctuation-dissipation relation, and overdamped limit.
A drag profile alone would not fix an Ito/Stratonovich convention. Here the
physical position process in Ito form is

    dx = D'(x) dt + sqrt(2 D(x)) dW,  D(x) = D0 (1 + a cos(q x)), q = 2 pi / L.

Its density obeys `dp/dt = d/dx [D(x) dp/dx]`. The uniform density is stationary;
there is no potential or variable accessible volume. The completed shortcut
instead describes zero-drift Ito diffusion, `dp/dt = d²/dx² [D(x) p]`. It has
stationary density proportional to `1/D(x)`. Both are mathematically valid
stochastic models, but only the former describes the specified apparatus. See
[Hottovy, Volpe, and Wehr](https://arxiv.org/abs/1112.2607) for the physical
small-inertia limit with position-dependent friction.

## Calibration and implementation

For `a = 0`, the measured cosine moments are `c_n(t) = c_n(0) exp(-D0 q² n² t)`.
The decay identifies the one unknown diffusivity without distinguishing the two
processes. Calibration contains two independent copies of 100 experiments, with
seed 9210 and Gaussian sigma equal to 0.006 times the largest absolute noiseless
calibration value. Public and private copies are generated together only by the
validator's `--generate` option.

For the physical model, the moments satisfy

    dc_n/dt = -D0 q² [n² c_n + a n(n-1)c_(n-1)/2 + a n(n+1)c_(n+1)/2].

The `c_0` coefficient in the first row vanishes. The oracle diagonalizes this
symmetric tridiagonal matrix using 36 nonconstant cosine modes. For the shortcut,
both neighbor coefficients are `a n²/2`, including the nonzero `c_0` source. Its
stationary moments are `r^n`, with `r = -a/(1+sqrt(1-a²))`. Rescaling the transient
moments by `1/n` gives the same symmetric matrix; thus the supplied predictor is
a completed, stable solution of the shortcut's assumed equation.

The verifier uses a separate periodic finite-volume flux discretization and BDF
integration on 512 cells. It has no dependence on the model's Fourier code.
Noisy calibration uses reduced chi-square < 1.5 and relative parameter error <
3%. Each hidden contrast group requires absolute cosine-moment RMSE < 0.03.
This is an absolute dimensionless tolerance, not a relative percentage. It is
well above calibration uncertainty and reference discretization error, and well
below the shortcut's physical error. The three groups include both signs of
the drag modulation and an initially uniform ensemble.

## Validation

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 \
  --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 \
  scripts/validate_spatial_diffusion.py
```

The initial 256-noise validation used seed 19210: calibration passed for all
256, with maximum parameter error 0.713%. The checked-in calibration fits to
D0 = 0.5506982353 micrometer²/s, chi-square 0.95234, against true D0 = 0.55.
Oracle hidden RMSE is at most 0.0000478; shortcut RMSE ranges 0.1579–0.2453.
At the noise sample's fitted-parameter extrema, oracle RMSE remains below
0.000266 and shortcut RMSE remains above 0.1575. This is a parameter-extrema
sensitivity check, not an exhaustive hidden-prediction Monte Carlo claim.

The independent finite-volume error decreases by four per grid doubling:
1.37e-5 at 256 cells, 3.42e-6 at 512, and 8.54e-7 at 1024. Probability is positive
and mass error is below 1.8e-13. A separate finite-volume zero-drift Ito solver
agrees with the completed shortcut to 1.08e-6. Uniform equilibrium stays uniform.
The local public/private pytest harness gives 7 passed for the oracle, and
4 passed/3 hidden failures for the completed shortcut.

Harbor confirms the oracle/shortcut control separation. GPT-5.6 Luna high
scored 2/3 unhinted and 3/3 with only `hint.md` appended to its instruction.
The target of 0/3 unhinted was not reached. See the
[reviewed evaluation results](../../DASHBOARD.md#candidate-expansion-matched-physics-hints).
