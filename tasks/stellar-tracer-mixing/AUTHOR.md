# Stellar tracer mixing, revision 1

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
