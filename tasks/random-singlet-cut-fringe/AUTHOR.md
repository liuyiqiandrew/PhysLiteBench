# Carrier echo across a random-singlet cut

One carrier spin encounters the left partners of an energy-ordered set of singlets and visits their right partners in reverse order. Each partner is individually unpolarized. The retained information in the two-spin state changes a Ramsey coherence and can survive averaging over uncertain geometry and formation scales. The flow of a two-cut random XX chain defines the disorder ensemble. For the fixed-point background, see [Fagotti, Calabrese and Moore, Phys. Rev. B 83, 045110 (2011)](https://doi.org/10.1103/PhysRevB.83.045110).

## Carrier exchange

For a bra arm u and ket arm v let E describe their carrier map after contracting all pairs that formed later. Let S be the two-spin singlet density and A and B the outbound and return exchange gates. A new outer pair acts on E through the linear map

    E_new(X) = Tr_pair[B_v (E ⊗ I_pair)(A_v (X ⊗ S) A_u†) B_u†].

A carrier map has four output ket/bra and four input ket/bra indices when the paired indices are individually flattened. These sixteen complex entries transform linearly when an outer pair is added. The innermost carrier map is the pair of turnaround unitaries acting on the two Ramsey arms. For formation order 1,…,N the required matrix product applies the Nth pair first in this contraction. A common carrier analyzer F yields g_uv=Tr[F E(ρ_initial)]. The normalized fringe is 2 Re(exp(−iλ)g₀₁)/(g₀₀+g₁₁), with each g already averaged over the same accepted disorder ensemble.

For one selected singlet and two complete swaps on arm 1 the carrier and two chain spins undergo minus the three-cycle permutation. Its expectation on |↑⟩ times a singlet is +1/2. Fully reading and discarding the z orientation of that singlet gives zero. Replacing the two partners by independent unpolarized spins gives −1/4. If the final carrier is selected up, the idle and swap arms have acceptance 1 and 1/2 and the singlet gives a fringe of 2/3.

## Disorder and validation

With two initial central spins the untouched interior exchange is activated at μ=ln(1+x), where x is an initial unit-mean exponential. Each undisturbed boundary has a rate-one outer crossing. An early one-sided crossing leaves behind a boundary density E₁∗Eₖ, where Eₖ(x)=k exp(−kx) and k=exp(μ). If the other boundary also crosses before the internal activation, the new common density is E₁∗E₁∗Eₖ. The activation releases the pending Eₖ phase. With a single central spin its two unmodified boundaries race at unit rate. An ordinary regenerated common cut has the two-stage waiting law with rates (3±√5)/2. The public helper solves this earlier instrument's backward equation for general ordered matrix weights.

The oracle contracts each singlet and the carrier indices by tensor multiplication. An independent reference constructs each local map by inserting matrix units on the carrier in a literal eight-dimensional carrier-plus-pair calculation. It transports the final carrier measurement forward through the disorder, representing the common-cut density directly as (a+b x)exp(−x). The public helper and its waiting stages are absent from the reference. The worked example, single-sided limit, conditional carrier readout and fixed-parameter decoherence controls are checked by `OPENBLAS_NUM_THREADS=1 python scripts/validate_random_singlet_cut_fringe.py`. Add `--generate` to write the stored fixtures.

Twelve empty scans fit one common visibility. Eighteen held-out probes cover outbound-only exchanges, coherent returns and joint carrier/herald selections. Both the calibrated independent-partner starter and the z-resolved singlet control reproduce the outbound-only group and fail the two return groups. The solve and verifier allowances are 1,800 and 180 seconds.
