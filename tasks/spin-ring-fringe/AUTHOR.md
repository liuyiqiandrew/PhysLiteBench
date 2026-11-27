# Spin ring fringe, revision 3

Independent preparations on two XY rings are compared by a product of
physical site swaps. For the selected spins S, the spin trace of that gate
is tr(ρ₁,S ρ₂,S). In an orthogonal Pauli basis this is the sum of products
of the two expectation values divided by 2^|S|. The same identity holds for
the Hermitian Majorana monomials of the selected spin Hilbert space.

## Finite spin-ring solution

Start the Jordan–Wigner string at site zero. Denote the two Majoranas there
and at subsequent sites by aⱼ and bⱼ, so Zⱼ=−iaⱼbⱼ. In parity p=∏Zⱼ the
Hamiltonian becomes i∑aᵢ Mᵢⱼ bⱼ. The diagonal of M is h/2, the X bond
contributes −Jx at (j+1,j), and the Y bond contributes −Jy at (j,j+1).
Multiply both terms on the closing bond by −p. In the Gaussian ensemble on
this grid before projection, the real contractions Gᵢⱼ=⟨−iaᵢbⱼ⟩ come from
replacing each singular value E of M by tanh(βE). The partition weight is
the product of 2 cosh(βE).

For increasing site sets A,B of equal size k define the ordered Hermitian
monomial with coefficient (−i)^k (−1)^[k(k−1)/2] multiplying all the a's
and then all the b's. Its Gaussian expectation is det(G[A,B]). Inserting
the parity string gives the complementary minor times (−1)^[sum(A)+sum(B)].
Each spin-sector trace therefore uses these two minors with relative sign p
and the appropriate Gaussian partition weight. Sum the two sectors, divide
by the trace of the identity, and multiply matching moments of the copies.
Unbalanced monomials have zero expectation for this bipartite coupling.

The local Hilbert space of the selected spins has its own Majorana basis.
A selected Majorana acquires Z factors from **all unselected sites to its
left** when expressed in ring Majoranas. Products cancel these factors in
pairs. At each unselected site with an odd number of selected generators
to its right, include both ring Majoranas in the determinant. The remaining
sign is common to the two copies and squares to one in the overlap. A
contiguous selection along the chosen ordering has no uncanceled gap factors
for the even products. The completed control uses restricted mode correlations,
which give the exact visibility and connected-interval readout. For separated
spins it misses the operators supported in the gaps. The distinction is
discussed in [Fagotti and Calabrese (2010)](https://arxiv.org/abs/1003.1110).

## Validation

The reference diagonalizes the original spin matrix and traces out unused
sites on generic rings through nine spins. On larger hidden rings the
Hamiltonians are commuting X–X or Y–Y Ising bonds; transfer between the
selected spin values uses a classical 2 by 2 matrix raised to each gap
length. This retains correlations between distant selected spins in a
nearly ordered ring. Direct and transfer traces also agree on small rings.
Five independent spot checks including an odd 29-site ring and a selection
crossing the closing bond agree within 7e−16. The high-temperature overlap
is exactly 2^(−|S|) when either copy is maximally mixed.

Seed 92711 generates the noisy calibration. At the fitted visibility
0.8170504 the reduced chi-square is 1.090. Both controls pass this test.
The largest oracle family RMS error is 0.000026; the mode restriction
gives 0.260, 0.042 and 0.274 on anisotropic, frustrated and commuting rings.
The per-family limit is 0.008 dimensionless fringe units, over five
calibration standard errors. The validator checks 256 further noise
realizations with seed 92712.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_spin_ring_fringe.py`.
Pass `--generate` to rebuild the checked-in data intentionally. R3 permits
1,800 seconds of solving and 180 seconds of verification. R1 and R2 and their
original run evidence are preserved in git and the evaluation ledger.
