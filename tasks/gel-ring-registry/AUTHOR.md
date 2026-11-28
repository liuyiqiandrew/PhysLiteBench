# Ring polymers in a two-pore gel

The comparison measures topology after a ring relaxes on a channel graph. A circuit followed by its reverse contributes no length to the final image, but the tip can change conformation on both traversals. Two rings with the same taut cyclic sequence can therefore have different conformations and seams at closure. The graph idealizes a phantom filament confined to narrow channels. The growth chemistry and two-ring selection specify this device. For the first-passage factorization on a free group, see [Lalley, The Local Limit Theorem for Random Walks on Free Groups](https://algo.inria.fr/seminars/sem98-99/lalley.pdf).

## Summing erased paths

Let Q_j be the two-state transition matrix for continuing in direction j, including its conformation switch. Let s be the closure probability vector and π the initial conformation distribution. Write j′ for the reversed direction. A held filament's reduced route lives on the tree of reduced words. Reaching a particular neighbor i either takes one direct step or starts with a step j≠i and first returns along j′. The matrix F_i of probabilities for first reaching neighbor i before closure solves

    F_i = Q_i + Σ_{j≠i} Q_j F_{j′} F_i.

The probability vector for eventually closing at the current vertex is d=(I−Σ_j Q_j F_{j′})⁻¹s. Thus the probability of a held reduced sequence w is πᵀ F_w d, with F_w the ordered product of the step matrices.

Release changes this sequence into t u t⁻¹ for a removable stem t and cyclically reduced core u. Start with D=dπᵀ. Adding a final stem letter j sends its inner trace weight X to F_{j′} X F_j. Grow the stem without adding a reversal. At the final join exclude stems ending in either the inverse of u's first letter or u's last letter. The resulting boundary matrix S then gives the rooted core weight tr(F_u S). The reference solution evaluates every allowed seam position, groups equal cyclic images and applies the two preparations' independent weights. A periodic image has as many distinct seam sequences as its least period.

The completed approximation retains the freely sliding registration, full contour lengths and two-state dynamics on every visible traversal. At each canceled return it uses the correct row sums of the return matrix but puts the resulting tip back in its entering conformation. This loses the conformation memory carried by invisible excursions. Its maximum errors are 0.0262 on seam readings, 0.0103 on windings and 0.0220 on mixed readouts. The grading tolerance is 0.004.

## Independent checks

The private calculation sums unrestricted returns in each branch with the edge to its parent removed. It adds stem contributions by their exact lengths and enumerates one canonical necklace at a time, traversing only its distinct rotations. All 24 readings agree within 3 × 10⁻¹⁵ with the oracle. After sorting by taut length and gating the first seam, each registration has probability above 0.00079. The homogeneous length-two public check counts all five ordered image pairs by hand.

A third check assigns two explicit chemical states to every reduced route on a finite pore tree and solves the sparse absorbing Markov chain. It then tightens each terminal image literally. On the first seam setting, truncating the tree at depths 5, 7 and 9 gives 0.73013014, 0.72980601 and 0.72973780. The infinite-tree calculation gives 0.72972603. This comparison does not use first-passage matrices, seam weights or rotation multiplicities from either calculation.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_gel_ring_registry.py` to regenerate the predictions in memory and repeat the controls and tree check.
