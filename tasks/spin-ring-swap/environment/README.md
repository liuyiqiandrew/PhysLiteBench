# Gap-parity records on critical spin rings

Two independent identical rings each contain N spin-1/2 sites and have periodic XX exchange H = −sum_j (S⁺_j S⁻_(j+1) + S⁻_j S⁺_(j+1)), including the closing bond. Prepare each ring in its ground state in the sector with exactly N/2 up spins. Take N to infinity through multiples of 32.

Choose m = 2 or 3 disjoint exchange arcs. Each `arcs` entry [a,b] denotes the sites from Na inclusive to Nb exclusive. The arcs are sorted around the ring with 0 ≤ a < b ≤ 1. Endpoints are on the grid j/16, and all arc and gap lengths are at least 1/16 of the ring. Gap j follows arc j and ends at the next arc. Gap m−1 wraps from the last arc back to the first.

## Measurement

For gap j let g_j = (−1)^(n_j1 + n_j2), where n_j1 and n_j2 are the numbers of up spins in that gap in the two rings. A monitor can measure a product of these g values without indicating its factors. An entry [indices, sign] of `records` selects the given sign ±1 for the product of all g_j in the index list. All records refer to the same pair of rings. They are ideal projective measurements. Retain pairs satisfying every record; the apparatus provides no individual gap value unless implied by the record. There are zero, one or two consistent independent records. Indices are zero-based, distinct within a list and may occur in both records.

For each entry −1 of `phase_plates`, rotate the second ring in the corresponding arc so that every up spin receives phase −1 and every down spin phase +1. Entry +1 applies the identity. On the retained state measure the product of ordinary two-qubit SWAP operators between the two rings at every site in all exchange arcs. In the physical spin basis the gate interchanges the two spin states with unit amplitude for all four inputs. The gap monitors, rotations and SWAP act in that order. No sites in a gap are exchanged.

Let C_N be the expectation of this signed SWAP measurement conditional on the specified records. For each arc separately measure the same exchange on that single contiguous arc in independent, unmonitored copies with all phase plates set to +1. Let D_N be the product of those single-arc expectations. Return V lim(C_N/D_N), taking the ratio at the same N. The visibility V is common to all records. The normalized result may exceed one. A blank phase-reference record `{"blank": true, "phase": phi}` reads V cos(phi).

## Example

Use `arcs=[[0,0.125],[0.375,0.75]]` and `phase_plates=[1,1]`. With `records=[]`, the answer divided by V is 1.376781325. With `records=[[[0],1]]` it is 1.878710168. Reversing that recorded sign gives 0.874852482. The two readings distinguish samples whose up-spin parities in the interior gap agree or differ. A detector with `records=[[[0,1],1]]` measures only the product of the two gap signs.

## Interface and checks

Implement `Model().fit(records)`, returning self, and `model.predict(inputs)`, returning a finite one-dimensional NumPy array in input order, including an empty array for empty input. Set `model.visibility` in [0.65,0.95]. Rows in `data/calibration.json` contain an input and the corresponding value. The starter fits the visibility and uses the unmonitored Gaussian fermion result for the macroscopic arcs. It treats the phase plates and records as spectators.

Maximum absolute error must be below 0.000002 on calibration and below 0.003 on each group of single, product and overlapping parity monitors. Run `python -m pytest -q test_public.py` from /app to check calibration.
