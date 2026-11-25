# Disordered wave chain: concept review

Verdict: approve for packaging. Static coherent multiple scattering with a random phase ensemble is distinct from the existing film-transmission flux-normalization task, coherent-array dynamic-correlation task, and quantum-residence delay/dwell task. This is concept approval, not evidence of model difficulty.

Use identical lossless leads and specify the delta barrier by the jump condition `psi'(x_j+) - psi'(x_j-) = 2 b psi(x_j)`, with positive unknown `b` in inverse-length units. At wave number `k`, write `r=(b/k)^2`, so the single-barrier transmission is `T1=1/(1+r)`. Each gap phase is independent and uniform. A phase is held fixed throughout every repeated reflection in one scattering experiment and is redrawn only between realizations. If phases are produced by random gap lengths, their distribution is prepared separately at each requested wave number.

The actual incoherent intensity-transport process is a clean completed shortcut: sum all reflected intensities while discarding phase information on propagation. Its transmission is `1/(1+N r)`, lies in `(0,1]`, and exactly matches the coherent ensemble mean at both `N=1` and `N=2`. Calibration on these counts at several wave numbers identifies positive `b` uniquely. Do not add the coherent mean-log-transmission law as a calibration branch of this dephased process; that law describes a different ensemble observable.

For the coherent chain, its resistance variable `R=(1-T)/T` obeys the exact one-step law

```
R' = R+r+2Rr + 2 sqrt[R(1+R)r(1+r)] cos(theta),
theta uniform on [0,2pi).
```

Phase integration gives `E[T'|T]=T/(1+r T)`. Hence `E[T2]=1/(1+2r)` but

```
E[T3] = 1 / [(1+r) sqrt(1+4r)].
```

The nonlinear map is strictly concave, so the coherent mean is strictly below the dephased prediction for every nonzero barrier and `N>=3`. Averaging static phases does not make the phases on repeated visits independent. This gives an accessible physical diagnosis without requiring a localization theorem or the spectral representation.

The radial transition `cosh(x')=cosh(x)cosh(a)+sinh(x)sinh(a)cos(theta)`, with `x=acosh(2/T-1)` and `a=acosh(2/T1-1)`, supplies a deterministic one-state integral kernel. Direct complex transfer multiplication supplies a separate numerical representation and flux-conservation checks. The optional Mehler-Fock expression is correctly normalized:

```
E[T_N] = 2pi integral_0^infinity
    q tanh(pi q)/cosh(pi q) [P_{-1/2+iq}(cosh(a))]^N dq.
```

The Legendre addition theorem and transform convention are documented in [Lenz's Mehler-Fock paper](https://arxiv.org/pdf/1612.04573). The observable transform is `2pi/cosh(pi q)`; the `N=0` expression equals one. Since the real conical function has magnitude at most one, cutting the integral at `Q` has absolute error at most `4(Q+1/pi)exp(-pi Q)`. Refine both the conical-function angular integral and the spectral integral; refine radial and phase grids separately in the kernel.

I read `staging/gpt6-expansion/random_phase_chain_prototype.py` and independently checked three parameter pairs spanning the proposed domain. Spectral results at `N=0,1,2,3` matched the closed expressions within `1.71e-14`; spectral versus radial-kernel results at `N=4,8,12` differed by at most `1.11e-12`. These are science checks of the prototype, not final package validation. Earlier echo and quantum-bus overlap rejections remain preserved in the expansion archive.

Proceed with the elementary scattering apparatus and mean transmission as the only observable. In later trial review, a source that correctly constructs the coherent transfer product but fails quadrature or sampling has a numerical failure; a source that equates phase-ensemble averaging with intensity-only propagation has the intended physical failure.
