# Spin ring fringe, revision 1

The spin Hamiltonian conserves the parity of the number of up spins. Let
`n_j = (1+Z_j)/2`. The Jordan–Wigner map attaches the string
`prod_{l<j} (1-2n_l)` to the lowering operator. A hop or pair term on
neighboring sites cancels the string in the interior. Closing the ring
multiplies the fermion boundary bond by `-P`, where `P=(-1)^sum(n)`.
The even spin sector therefore uses antiperiodic fermions. The odd sector
uses periodic fermions. Both sectors occur in the initial Gibbs state.

Within a momentum pair (k,-k), define `xi=h-2J cos(k)` and
`z=2i d exp(i phi) sin(k)`. In the vacuum and ordered pair basis the
even Hamiltonian has rows `[-xi, conj(z)]` and `[z, xi]`. The two odd
states have zero energy. At an unpaired momentum 0 or pi the empty and
occupied states have energies -xi/2 and xi/2. Summing these offsets reproduces
the physical constant -Nh/2 on both momentum grids.

On each grid form the block traces of the ordered drive times
`exp[(i T - beta) H_prep]`, and of the Gibbs weight alone. Compose blocks by
adding their parities modulo two. Keep the required global parity on each
grid. Sum the two selected complex traces and divide by the sum of the two
selected partition functions. The analyzer takes one quadrature of this
normalized amplitude. Keeping the amplitudes preserves the phase of a full
rotation in each many-body sector.

## Controls and reference

The shortcut uses unrestricted fermions on a single periodic momentum grid.
Calibration turns off exchange and pairing everywhere, so no term samples the
boundary bond. Both implementations recover the visibility by weighted linear
regression. Seed 7319 generates 24 records with independent noise of standard
deviation 0.002. The validator checks visibility recovery for 256 additional
noise realizations using seed 7320.

The private reference constructs the original spin matrix for small rings
and diagonalizes each Hamiltonian directly. For large rings in the hidden
cases, each pulse is diagonal in a product basis. A field pulse contributes
an on-site phase. A pulse with h=0 and d=-J contributes a bond phase in an
in-plane spin basis rotated by phi/2. Inserting the basis at each time slice
leaves a one-dimensional transfer matrix. A column records one spin sign per
slice. The trace of its Nth power closes the **spin** ring. The validator checks
the two references on the same six-site pulse sequence and compares every
frozen truth to the independently implemented momentum-space solution. A
single nearly half-turn on each X–X bond gives a size-dependent return phase
even when the one-site spin expectation vanishes.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_spin_ring_fringe.py`
from the repository root. Use `--generate` only when intentionally regenerating
fixtures. Reduced calibration chi-square must be below 2.5 and visibility
error below 0.005. Each hidden family has an absolute signal RMSE limit of
0.008 (four calibration standard errors). At the generating visibility the
oracle and spin references agree to numerical precision. With the fitted
visibility the largest family RMSE is 0.000270. The periodic approximation
gives RMSE 0.0772 for thermal quenches, 0.8163 for full-ring returns, and
0.1750 for pulse sequences at finite temperature.

Fermionic trace phases in general paired systems are discussed in
[Klich, J. Stat. Mech. P11006 (2014)](https://arxiv.org/abs/1403.7824).
