# Delta-box wall force, revision 1

The task separates the force on a confining wall from a scale-invariant energy-to-pressure conversion. The supplied predictor solves the correct defect-modified quantum spectrum and canonical populations. It then uses 2*mean_energy/length for the wall force. That identity is exact in the defect-free calibration but fails when the physical delta strength remains fixed as the wall is displaced.

The public apparatus states the full one-particle Hamiltonian, hard walls, fixed physical defect strength, a defect pinned at the origin, Gibbs preparation, and force on the right wall. It gives no virial relation or force-derivative recipe. The neutral instruction permits any solver. Known sigma is fixed instrument noise. The weak preparation contact is removed before measurement, excluding its force without changing the stationary Gibbs mixture. The hint is private.

## Physics

For length L, mass m and repulsive defect strength g, an even state is

    psi(x)=A*sin[k*(L/2-|x|)],
    q=k*L/2, q*cot(q)=-m*g*L/2,
    A²=2/[L-sin(k*L)/k].

Its force on either wall is the boundary momentum stress

    F=k²/[m*(L-sin(k*L)/k)].

Odd states vanish at the defect and have k=2*n*pi/L, with force k²/(m*L). Each mode is normalized on the complete interval and receives its canonical probability. The two walls move by half the length change under a symmetric displacement, so minus the free-energy derivative with respect to L equals the force per wall, without an extra factor of two.

At fixed physical g the exact relation is

    F=(2*mean_energy-g*mean_contact_density)/L.

The source omits the contact virial, while retaining its effect on all energy levels and populations. In a pure even state the relative shortcut excess is -sin(2*q)/(2*q), with maximum about 21.72%. This explains the chosen moderate-strength regime. Very large repulsion and high temperature reduce the gap; the task does not claim a uniform shortcut error at every allowed control. The selected hidden groups retain a substantial force signal and a 20.29–21.72% pointwise prototype gap.

Calibration with g=0 is exactly shared, including quantum excited-state thermal populations. Varying lengths and temperatures identifies the common mass. This is separate from the earlier confined-fluid, elastic-metric and optical momentum tasks: here the spectrum and Gibbs ensemble are already correct, while the pressure conversion assumes an absent scaling symmetry.

The even root is solved as an offset d from the free root q0=(n+.5)*pi, using (q0+d)*tan(d)=m*g*L/2. This monotone equation includes g=0 continuously and avoids skipping roots at tiny positive strengths. The change was made before the full validation and freeze.

## Independent verification

The oracle uses analytic parity eigenstates and normalized wall stress. The private reference instead discretizes the full interval, puts g/dx on the central site of a tridiagonal Hamiltonian, and computes its low eigenspaces. Its wall gradient follows directly from the boundary eigenvector amplitudes. A 511/1023-point Richardson estimate removes the leading grid error. It imports no parity roots or oracle functions. Refinement to 2047 points changes force by at most 1.34e-9 over the domain corners.

A separate fixed-g displacement of canonical free energy agrees with boundary stress to 9.03e-11. Further checks cover wavefunction normalization, the derivative jump at the defect, the contact Hellmann–Feynman identity, the contact virial, positive spectral gaps, and near-zero strength from 1e-18 to 1e-9. Increasing 32 retained levels to 48 changes no tested force at floating-point precision. The smallest sampled level gap is 1.17549. Oracle/reference differences are at most 9.80e-10 at domain corners and 3.43e-10 for the hidden inputs. A cold hidden reference evaluation takes 0.208 seconds.

Primary background: Pedram and Vahabi, [Exact solutions of a particle in a box with a delta function potential](https://doi.org/10.1119/1.3373925), American Journal of Physics 78,839 (2010), [author manuscript](https://arxiv.org/abs/1001.0311). For the relation between boundary derivatives and quantum stress, see Bobrov et al., [Virial theorem for an inhomogeneous medium, boundary conditions for the wave functions, and stress tensor in quantum statistics](https://doi.org/10.1103/PhysRevE.82.010102), Physical Review E82,010102(R) (2010). The task's force and reference are derived directly from the specified Hamiltonian.

## Calibration and controls

There are 144 calibration measurements: four lengths, three temperatures and twelve independent repeats. True mass is .63, sigma=.001, calibration seed 970031 and noise seed 970037. Public and private data are identical. The fitted mass is .6299986666335907, with reduced chi-square 1.008208. A 121-point profile over the full [.4,.9] mass interval has one minimum at each of three separated true masses, including .401 and .899; noiseless relative recovery errors are below 5.55e-9.

The standard thresholds are reduced chi-square<1.5, parameter error<3%, and hidden group normalized RMS error<.04. Nominal oracle errors are at most 2.12e-6. Shortcut errors are .21288, .21721 and .21637; both closures pass the defect-free off-calibration anchors.

All 256 independent noise trials pass calibration and parameter recovery. Every oracle passes all hidden groups, and every shortcut fails the three defect groups. Worst oracle error is 1.61e-5 and minimum shortcut error is .212866. Isolated local pytest reports oracle 8/8 and shortcut 5 passes with 3 intended hidden failures. Scientific validation does not establish model difficulty. This backup is staged outside canonical tasks and has no evaluation result.

Reports: [science](../../results/delta-box-force-validation.json), [local controls](../../results/delta-box-force-local-controls.json), [provenance](../../results/delta-box-force-source-provenance.json), and [prototype](../../results/delta-box-force-prototype.json).

From the task bundle root, reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_delta_box_force.py`. Add `--generate` only when intentionally regenerating both frozen calibration copies.
