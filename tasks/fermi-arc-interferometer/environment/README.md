# Fermi arc interferometer

A one-dimensional ring of circumference one holds noninteracting spinless fermions. A Ramsey probe measures whether the total particle number in a union of arcs is even or odd. The source makes a coherent superposition of low-energy excitations at one edge of a filled set of orbitals. Predict the parity contrast relative to independent reference arcs.

## Source

Position x is in [0,1) and orbital k has normalized wavefunction exp(2πikx), with signed integer k. Use N → infinity through positive multiples of 64. The measured sample contains M=N+d particles, where `detuning` is the integer d in [−2,2]. All states are pure Slater determinants built in strictly increasing orbital order. Relative phases refer to this ordering at measurement. At each N, the four basis determinants all occupy orbitals 0,...,M−3. Their last two occupied orbitals are

| Component in `state` | Additional orbitals |
| --- | --- |
| 0 | M−2 and M−1 |
| 1 | M−2 and M |
| 2 | M−2 and M+1 |
| 3 | M−1 and M |

The four `state` entries are [magnitude, angle] with angles in radians. Magnitudes are in [0,1] with at least one positive entry. Each complex coefficient is magnitude × exp(i angle). Normalize their coherent sum to unit norm. The basis determinants are orthonormal. Beam phases are given at measurement, with no subsequent free propagation.

## Parity detector

Each entry of `arcs` is [a,b,delta_a,delta_b]. At finite N it counts particles in the oriented interval [a+delta_a/N, b+delta_b/N), with positions understood modulo one. The arcs are ordered and disjoint for sufficiently large N, with nominal endpoints on the grid j/32. There are one to three arcs. Each arc and each gap between neighboring arcs around the ring has width at least 1/16. The offsets are in [−0.75,0.75].

Let P(S)=exp(iπ times the particle number in S). The probe reads the parity of the total number in the union S_N. Its normalization is measured on separate reference rings. For each nominal arc [a,b), take the parity expectation on its unshifted interval in the filled N-particle state with occupied orbitals 0,...,N−1. Multiply these reference means to get D_N. Return

    visibility × lim (⟨prepared state | P(S_N) | prepared state⟩ / D_N).

Use a common N for all factors before taking the limit. The reference uses zero offsets and zero detuning even when the sample differs. The ratio can have magnitude greater than one. For `blank=true` the phase reference reads visibility × cos(phase).

## Worked measurement

Set `arcs=[[0,0.25,0,0]]` and `detuning=0`. Give the last two basis states equal magnitudes and the first two magnitude zero. With both phases zero the answer divided by visibility is 0.125. With relative phase π it is 0.5. The two preparations have identical occupation probabilities. Their limiting normalized parity matrix in the last two basis states has both diagonal entries 5/16 and both off-diagonal entries −3/16. A separate ground-state check uses disjoint arcs [[0,0.125,0,0],[0.375,0.75,0,0]], detuning zero and only state component 0. Its visibility-free ratio is 1.306562965.

## Interface and checks

Implement `Model().fit(records)`, returning self, and `model.predict(inputs)`, returning a finite one-dimensional NumPy array with one real result per input, including an empty array for an empty list. Expose the fitted `model.visibility` in [0.65,0.95]. Calibration records in `data/calibration.json` contain `input` and `value`. The public starter fits this visibility. Its approximate physics treats the arc parities as independent and keeps the single-arc filled-sea oscillation for each interval, regardless of the source excitations.

The maximum absolute error must be below 0.000002 on calibration and below 0.003 in each of the filled droplet, coherent edge doublet and general coherent packet measurements. The doublet uses the last two basis states. Run `python -m pytest -q test_public.py` in /app for the calibration check.
