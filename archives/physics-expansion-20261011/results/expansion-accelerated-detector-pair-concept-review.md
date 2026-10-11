# Accelerated detector pair: concept review

Independent review, 2026-10-11. **Approved for prototyping and packaging, subject to source review and numerical controls.** The proposed scalar-field apparatus is physically well defined and has an exact calibration loophole. No empirical difficulty claim is justified yet.

## Observable and preparation

Use a canonically normalized real massless scalar field in unbounded 3+1 Minkowski vacuum, with `hbar=c=1`, and two identical pointlike two-level detectors. Their equal real coupling amplitudes and common proper-time clock must be explicit. The transverse separation permits equal proper accelerations and a common stationary time; the proposed shift `x=(cosh(a*tau)-1)/a` only translates the usual hyperbola and has no spectral consequence. Give the `a=0` worldlines separately.

State the free detector Hamiltonian and unit transition matrix element of each `sigma_x`. Prepare the field and detectors initially uncorrelated. For a pair measurement the detector state is the pure symmetric state `(eg+ge)/sqrt(2)`, with its relative phase referred to the specified identical real couplings. External forces prescribe the motion; recoil, field backreaction, additional decay channels, and direct detector interactions are absent at the retained order.

The pair readout must be the **resolved transition rate into gg**. It is not the total loss of symmetric-state population: at positive acceleration, transitions from that state to ee also contribute to depletion. Define a proper-time stationary Golden-Rule rate from long smooth switching windows at leading order in `mu**2`, with weak coupling taken before the long-time limit. Switching transients and population depletion are excluded. This avoids the contradiction of evolving a fixed nonzero coupling for infinite time while still treating the initial state as unchanged. Lamb shifts do not alter this leading transition-rate coefficient.

## Normalization and physical formulas

Starting with the canonical Minkowski two-point function, its stationary pullback for transverse separation L is

```
W_L(s) = -a**2/(16*pi**2)
         / [sinh(a*(s-i0)/2)**2 - (a*L/2)**2].
```

Let `F_L(lambda)=integral exp(i*lambda*s)*W_L(s) ds`. Positive lambda corresponds to downward detector transitions with this Fourier convention. Direct contraction of the two symmetric-state matrix elements gives

```
R(S -> gg) = mu**2 * [F_0(omega) + F_L(omega)].
```

There is no additional factor of two. For coincident detectors this equals twice a single excited detector's downward rate; for vanishing cross correlation it equals one such rate, since S contains one excitation.

For positive acceleration the proposed closed formulas are correct:

```
F_0(omega) = omega/[2*pi*(1-exp(-2*pi*omega/a))],
f = F_L(omega)/F_0(omega)
  = sin[(2*omega/a)*asinh(a*L/2)]
    / [omega*L*sqrt(1+(a*L/2)**2)].
```

The single upward rate is `mu**2*F_0(omega)*exp(-2*pi*omega/a)`. At `a=0`, use `F_0(omega)=omega/(2*pi)`, zero upward rate, and `f=sin(omega*L)/(omega*L)`. These limits avoid a removable division by zero. The small-L limit gives f=1; the cross spectrum is even under changing the sign of omega apart from the common thermal factor. These are useful normalization, positivity, and detailed-balance checks.

[Hu and Yu, Entanglement dynamics for uniformly accelerated two-level atoms](https://arxiv.org/html/1501.03321) gives the canonical normalization, transverse-acceleration geometry, and the same diagonal/off-diagonal spectra in equations 13–18. Their collective-state population equations also confirm the resolved symmetric-to-ground coefficient. The formulas above can be derived directly from the public worldlines and Wightman function; they need not be supplied publicly.

## Exact calibration loophole and separation

The completed shortcut is a genuine auxiliary physical system: an inertial pair at separation L in a homogeneous thermal scalar bath at `T=a/(2*pi)`. Its local spectrum equals the accelerated detector's local spectrum, while its normalized cross spectrum is `sin(omega*L)/(omega*L)`. Thus all single-detector calibration at varied acceleration/gap is exact. At zero acceleration the finite-separation pair calibration is also exact. This second calibration family checks the collective normalization rather than leaving it arbitrary. Any nonzero downward setting identifies the sole unknown `mu**2` by weighted linear least squares.

The distinction is not whether the accelerated vacuum satisfies a thermal condition: the two accelerated probes share the relevant stationary KMS time. The inappropriate step is replacing its spatial correlations with those of a homogeneous inertial thermal bath. A local temperature alone does not determine the cross spectrum.

Illustrative points within the proposed domain give large separation without near-zero pair rates. The values below are direct evaluations of the proposed formulas, not model trials or final hidden inputs.

| a | omega | L | Correct rate divided by mu² | Shortcut relative error |
|---|---|---|---|---|
| 10 | .5 | 1.5 | .309108 | .822789 |
| 8 | 1 | 1 | .328010 | .642287 |
| 6 | 1 | 2 | .260082 | .371417 |
| 10 | 2 | 1 | .479789 | .349008 |
| 4 | .5 | 3 | .155618 | .564930 |

Choose fixed instrument noise independently of the unknown coupling. Verify repeated-noise gain fits and group separation before freezing. Very small-acceleration upward rates may underflow physically; they should not be used alone as relative-error denominators. Downward symmetric rates avoid that issue. Calibration can include exact zero excitation readings, but those carry no gain information.

## Independent contour reference

Numerical Fourier integration of the pulled-back Wightman function is an appropriate independent reference if it does not use the closed cross factor. The lower analytic strip is `-2*pi/a < Im(z) < 0`. For `0<c<2*pi/a`, contour deformation gives

```
F_L(lambda) = exp(lambda*c)
              * integral exp(i*lambda*s)*W_L(s-i*c) ds.
```

Both the downward shift and the prefactor sign matter. A choice such as `c=min(1/omega,pi/a)` avoids exponentially large prefactors as acceleration approaches zero. For the inertial branch use `W_L(z)=-1/[4*pi**2*(z*z-L*L)]`. Do not insert a finite real-time regulator and treat its distorted spectrum as exact.

Check contour-height invariance, integration refinement/tails, the inertial limit, coalescing-detector limit, large separation, and positive semidefiniteness of the two-detector spectrum (`abs(f)<=1`). Direct upward quadrature can lose relative accuracy through cancellation when acceleration is small. It is valid to obtain tiny upward rates from KMS after independently integrating the downward spectrum, but that construction is not itself an independent numerical test of KMS. An absolute tolerance must accompany relative comparisons near zero. A finite integral window alone needs a tail estimate or independent extension test.

## Archive overlap and difficulty risk

The relevant archive is `archives/screened/circular-detector-r1`. It uses the same scalar detector framework but tests the full temporal response along one circular trajectory against a local worldline-interval closure. This proposal keeps every single-detector response exact and tests the two-worldline spatial correlation. Its prepared entangled state and collective readout are essential new physics, so it is more than relabeling the circular trajectory task. Other archived correlated-bath tasks also use cross correlations, but not this mismatch between exact local thermality and a different spatial spectrum.

A knowledgeable solver can derive or recognize the cross factor directly from the stated interval. Strong physical validity and separation do not imply that the target model will fail it. Preserve the distinction between choosing the inertial thermal spatial spectrum (physical modeling error) and selecting the correct Wightman pullback but mishandling a Fourier sign, contour, prefactor, or numerical integral (mathematical/numerical error).
