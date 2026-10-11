# Disordered wave chain, revision 1

The apparatus distinguishes phase disorder between measurements from loss of
coherence during scattering. The completed shortcut is the valid incoherent
intensity transport process, including arbitrarily many reflected rays. Its
error is replacing a fixed coherent sample by that process before averaging.

For a single delta barrier, continuity and the derivative jump give
`t=k/(k+i*b)`, `r=-i*b/(k+i*b)` and power `T1=1/(1+(b/k)^2)`.
With `lambda=(b/k)^2`, adding incoherent intensities gives
`T_inc,N=1/(1+N*lambda)`. The shortcut is positive, bounded and reciprocal;
it neither drops repeated reflections nor confuses amplitude with power.
For two coherent barriers, averaging the exact multiple-reflection expression
over one uniform phase also gives `1/(1+2*lambda)`. Thus all N=1 and N=2
calibration is exactly shared, for every permitted strength and wave number.
For three barriers the exact mean is
`1/((1+lambda)*sqrt(1+4*lambda))`, already different from the shortcut.

The full chain retains coherent multiple scattering, including fluctuations
in the transmission of each partial chain. Randomizing only the next gap
does not replace that partial chain by its average transmission. This changes
longer-chain transport even though each individual gap phase is uniform.
The explicit separation distribution is redrawn for each experiment's k;
this is not a claim about sweeping k through one fixed disorder realization.
The large fixed offset in each separation changes no stationary phase.

## Independent representations

Write `x=acosh(2/T-1)` and `a=acosh(2/T1-1)`. A transfer-matrix product with
one fresh uniform gap phase has
`cosh(x_new)=cosh(x)*cosh(a)+sinh(x)*sinh(a)*cos(phi)`.
The private reference recursively integrates this Markov kernel against
`sech(x/2)^2`. It uses a shrinking finite interval, 1200 spatial grid cells,
96 angular nodes and cubic interpolation. Doubling both resolutions changes
hidden answers by at most 7.10e-12 relative group norm. It does not use the
oracle's conical-function spectral representation.

The oracle diagonalizes the angular-average operator with
`P(s,a)=P_{-1/2+i*s}(cosh(a))`. It evaluates

    mean(T_N) = integral_0^infinity
        2*pi*s*tanh(pi*s)/cosh(pi*s) * P(s,a)^N ds.

The conical function is itself an angular integral of
`cos(s*log(cosh(a)+sinh(a)*cos(theta)))/sqrt(cosh(a)+sinh(a)*cos(theta))`.
The implementation uses 160 radial nodes through s=14 and 192 angular nodes.
The spectral tail is bounded by 4.54e-18 using |P|<=1. Doubling both orders
and extending the endpoint to 16 changes the 88 tested domain answers by
at most 1.75e-14. This is a numerical domain check, not a claimed uniform
roundoff bound. The two independently represented deterministic answers
agree within 1.25e-12 absolute on that screen. Exact N=0,1,2,3 identities
and passive probability bounds also pass.

For a separate microscopic check, the validator multiplies the actual
complex delta-jump and free-propagation matrices for 262,144 independent
configurations at each of six preparations. It uses neither radial recursion
nor spectral inversion. Its mean transmissions agree within 2.04 standard
errors, and its mean logarithms agree with `N*log(T1)` within six standard
errors. These stochastic checks are not grading truth. Hidden values are
frozen deterministic recursion results.

Primary context: Ng and Englert, [One-dimensional transport revisited](https://arxiv.org/abs/1212.1951),
provides the uniform-phase transfer recurrence and its Legendre treatment.
The paper's transfer law, averaging operator and spectral construction were
checked against the independently developed prototype. The public apparatus
supplies the wave equation and physical boundary conditions, not this solver.
Direct transfer matrices or converged ensemble integration are valid methods.

## Calibration and controls

There are 192 readings: twelve independent repetitions of sixteen N=1,2
settings. Fixed instrument sigma=.001; true strength=.83 mm^-1; calibration
seed 613001 and noise seed 613009. Every calibration mean strictly decreases
with positive strength, proving global identification on [.6,1.0]. Forty-one
noiseless fits including both endpoints recover within 2.90e-8 absolute.
Both controls fit .8298243791550319 with reduced chi-square 1.04498.

The ordinary gates are 3% parameter error, reduced chi-square 1.5, and 4%
relative Euclidean error per hidden group. The shared anchors pass both
controls. Shortcut errors are 17.60%, 92.73% and 29.67% in the short-chain,
long-chain and weak-barrier groups. The largest oracle group error is 0.0636%.
All 256 noise realizations pass calibration and parameter checks. The worst
oracle group error is 0.157%, and the smallest diagnostic shortcut error is
17.49%. These margins are much larger than calibration and numerical error.

Reproduce with the repository's pinned Python/NumPy/SciPy environment:
`python scripts/validate_disordered_wave_chain.py`. Only `--generate`
regenerates public/private calibration and hidden truth. The completed control
is `scripts/disordered_wave_chain_baseline.py`; the saved numerical report is
`results/expansion-disordered-wave-chain-r1-validation.json`. Standard 600/60
second limits and dependency pins are retained. No model difficulty claim
precedes the frozen batch.

Retaining incoherent intensity addition because random gaps supposedly erase
all interference is a physical error. Correct coherent transfer calculations
that fail through sampling, normalization, interpolation, or implementation
must be classified separately. The source and emitted trajectory determine
that classification. The closest archived coherent-array and film-transmission
candidates concern different physical omissions; the concept review records
that scope rather than claiming an exhaustive novelty theorem.
