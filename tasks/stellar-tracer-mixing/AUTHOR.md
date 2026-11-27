# Stellar tracer mixing

## Revision 1 – isolated changes

Angular momentum L is conserved throughout. In a slow change the radial
action is conserved too. The isochrone action for energy E < 0 is
Jr = m/√(−2E) − (L + √(L² + 4mb))/2. Invert this expression at (1,1) to
obtain the initial energy and specify the phase-space density at the end of
the slow preparation. [Binney (2014)](https://doi.org/10.1093/mnras/stu297)

At an instantaneous switch, position and velocity are continuous. To find
the previous energy at a given final radius add the old potential minus
the new potential to the final energy. For each switch average the previous
density over one radial oscillation in the *new* field at fixed final (E,L).
An unbound intermediate orbit contributes zero after escape. Two pulses
separated by mixing generally leave a measurable change even if the potential
returns to its initial shape.

The six-dimensional phase volume at fixed E,L is 8π² L Tᵣ dE dL. The position
solid angle gives 4π, the velocity azimuth gives 2π, and integrating over
both signs of radial velocity gives the full radial period Tᵣ. In
s = √(r²+b²), the two turning points are s = c ± d, with c = −m/(2E) and
d² = (c−b)² + L²/(2E). Write s = c + d cos θ. The time weight is
s dθ/(cπ) and Tᵣ = 2πm/(−2E)^(3/2). The oracle integrates in θ.

The private implementation instead integrates dr/|vᵣ| in physical radius,
evaluates Jr = ∫ |vᵣ| dr/π, and numerically inverts the initial action for
each angular momentum. The validator compares 56- and 80-point quadratures
and checks the static limit. Static measurements are rounded deterministically.
Their log slopes identify p. The completed control evolves every step
adiabatically, conserving both actions even at an abrupt switch. It predicts
static and slow preparations correctly. On the one-switch, return-cycle, and
two-switch groups its largest fractional errors are 0.364, 7.92, and 1.55.
Each group permits 2% maximum relative count error. The oracle stays within
0.006% in all groups.

Run `python scripts/validate_stellar_tracer_mixing.py` from the repository
root. The script reads committed data. Passing `--generate` recomputes it.

## Revision 2 – repeated weak cycles

Revision 2 adds the pulse train and retains the revision 1 readings. At fixed L the canonical phase volume is constant per unit radial action J. Write C = (L + √(L² + 4mb))/2, E = −m²/[2(J+C)²] and radial frequency Ω = m²/(J+C)³. Along a perturbation with coupling (a,c), the potential derivative at fixed position is V = Φ(a − cb/s), with s = √(b² + r²). The sudden switch gives a first-order action step ε(V−⟨V⟩)/Ω. The slow return leaves this action unchanged.

Thus the action variance per scaled exposure is B(J) = Var(V)/Ω². Before aperture loss, the one-cycle transfer preserves canonical phase volume: uniform phase-space density stays uniform under Hamiltonian evolution and phase averaging. The probability current of a uniform action population therefore has zero divergence, and the circular endpoint admits no outward flux. In the weak-pulse limit this fixes the mean action drift to B′/2. The density evolves by ∂τ f = ∂J(B ∂J f)/2, with regular zero-current behavior at J=0 and an absorbing upper endpoint J = m/√(2 binding_limit) − C. Escaping trajectories have already passed out of this finite binding window in the local diffusion limit.

The oracle integrates the orbital variance, then evolves the initial action density with a conservative cell-centered finite-volume operator and a symmetric tridiagonal eigensolver. The completed control reconstructs past orbits with centered backward action increments of the correct local variance. It solves ∂τ f = B ∂²J f/2, which drops the drift from the transfer operator. Both methods recover the static exponent 2.73 and reproduce the finite-switch observations. On mass, scale-length, and mixed pulse trains, the control's maximum relative count errors are 25.8%, 44.1%, and 79.8%.

The private reference evolves the finite-pulse process itself, without a drift law or a diffusion equation. For a given final J after slow restoration it uses the same J in the temporary field, samples the post-jump radial phase there, and reconstructs the exact pre-jump action. Interpolation deposits these weights in a sparse backward transfer matrix. Repeated application incorporates aperture losses after every pulse. Trains of 400 and 1,600 cycles, in both perturbation directions, give two estimates of the same limit. Linear extrapolation in pulse amplitude corrects the leading finite-step bias at the absorbing boundary. The validator repeats the calculation with 144 and 576 cycles and an independent spatial grid. The three groups permit 2% maximum relative error. The oracle's worst difference from the finite-pulse reference is 0.055%.

## Revision 3 – coherent pulse pairs

Revision 3 adds instantaneous restoration after a fixed hold time T and keeps all previous readings. During a cycle the two jumps act on the same orbital history. At first order their action increments have opposite signs and sample V at two radial phases separated in time by T. The action variance per scaled exposure becomes the time average of [V(t)−V(t+T)]² / Ω². For a circular orbit it vanishes. It also vanishes at actions where ΩT is an integer multiple of 2π. Near such an action the variance vanishes at least quadratically, so the limiting continuous diffusion cannot cross it in finite exposure.

The oracle converts the eccentric anomaly ψ to a uniform time angle via θ = ψ + (d/c) sin ψ for s = c + d cos ψ. It advances θ by ΩT, inverts for the new ψ and integrates the squared difference. At a resonant action the initial density persists. At other actions it solves the conservative diffusion in the interval between neighboring resonances, the circular endpoint or the detector, whichever is encountered first. Flux vanishes at each internal resonance. The independent reference follows finite orbits in the temporary potential: at the second switch it reconstructs the temporary energy and sign of radial velocity, goes back T to the first switch, and recovers the preceding action at unchanged position and velocity. Iterating this exact transfer gives an independent weak-pulse limit. The validator repeats the transfer at two discretizations. The largest discrepancy from the oracle among the coherent readings is 0.058%.

The completed control for this revision uses the correct prior solution for slow restoration. For a fixed hold it assumes that the radial phases at the two jumps have become independent. This doubles the one-jump variance and removes its dependence on T. Its worst count errors are 6.77% and 9.64% in the two coherent groups under the same 2% tolerance.

In both private controls the aperture mesh resolves narrow boundary layers near zero coupling. For layers thinner than 10⁻⁷ of the action interval the absorbing half-line solution controls the limit. Independent checks evaluate the fraction of unperturbed density at distances of 0.1, 1 and 3 diffusion widths from the aperture. This also covers the near-static regime from revision 2.

Independent physics review confirmed the time-angle weight, the canonical drift constraint and the inaccessible interior resonances. The validator also checks an entire radial-period hold against the unchanged action density on that orbit.
