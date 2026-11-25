# Magnetic bilayer, revision 1

The completed shortcut has correct damping, precession, anisotropy, exchange, and finite-wavelength interlayer magnetostatic coupling within each transverse component. It treats the energy from dynamic volume charge and dynamic surface charge separately. These positive energy blocks define a stable approximation. They miss the interference between sources of the same magnetic potential.

For a Fourier mode, let k=|wavevector|, s=ky/k, P=(1-exp(-k))/k, and Q=exp(-k*gap)*(1-exp(-k))^2/(2*k). In each unit-thickness film the transverse self-demagnetizing matrix is diag(s^2*(1-P), P). In coordinates (m_y,m_z), the coupling from the lower to upper film is

    N12 = Q * [[s^2, -i*s], [-i*s, -1]],
    N21 = conjugate_transpose(N12).

This follows from the Coulomb kernel exp(-k*|z-z'|)/(2*k), volume charge -i*ky*m_y, and surface charges -m_z at the lower face and +m_z at the upper face. The self mixed term cancels by reflection within a film; the interlayer mixed terms do not. The shortcut sets those mixed terms to zero. Its energy is the sum of the two principal charge-sector energies, so it remains positive. Both models reduce exactly to the same nontrivial magnetostatically coupled response for ky=0, as used in calibration.

The full positive transverse energy matrix A also includes the local restoring fields 0.45 and 0.70 and exchange 0.08*k^2. With J=diag([[0,-1],[1,0]],[[0,-1],[1,0]]) and damping alpha, the linear Gilbert generator is gyro_rate*(J-alpha*I)*A/(1+alpha^2). Its energy derivative is -gyro_rate*alpha*|A*m|^2/(1+alpha^2). Thus both controls decay rather than becoming unstable. Dropping the charge interference is a physical approximation, not a wrong sign or non-Hermitian matrix.

The public specification explicitly defines a lowest-thickness-mode projection and the Fourier readout. The oracle is exact within that prescribed approximation; it does not silently discard higher modes from a three-dimensional problem. The private reference solves the piecewise Poisson equation for both films, the spacer, and the exterior. It enforces potential continuity and derivative jumps from the actual surface magnetization, averages the resulting field, then solves implicit Gilbert dynamics by a matrix exponential. It does not use the reduced demagnetization entries above. A third calculation directly integrates the volume and surface charge Coulomb energy.

Calibration contains 192 measurements with fixed standard deviation 0.00003, seed 35021, and true gyromagnetic rate 0.93. The fixed uncertainty carries no response information. Both controls fit 0.9299953261 with reduced chi-square 1.00567046. The calibration objective has one minimum over the full allowed interval for screened interior and near-boundary true rates. All 256 additional noise draws pass calibration and parameter checks. For every draw, all oracle hidden groups pass and all shortcut groups fail.

The three hidden groups use oblique waves, both signs of ky, both layer readouts, and real or complex initial Fourier amplitudes. Each group uses RMS prediction error divided by RMS reference signal, with a 0.04 limit. Checked-in oracle errors are below 0.000034; shortcut errors are 0.0776–0.1338. Across 256 noise draws, oracle error is below 0.00582 and shortcut error above 0.0767. The tolerance is well above calibration uncertainty propagation and numerical discrepancies.

Oracle and independent Poisson dynamics agree within 4.9e-17 in hidden predictions. Matrix comparisons are within 3.4e-16 for the Poisson calculation and 1.7e-16 for direct Coulomb quadrature. Domain checks cover positive energies, damped eigenvalues, the exact Gilbert energy identity, signed-wavevector conjugation, layer reflection, zero-time output, large separation, and superposition. These checks hold for both completed controls where physically applicable. Local verification gives 7/7 oracle passes and 4 passes plus 3 intended hidden failures for the shortcut.

Related magnetic bilayer physics is discussed by den Teuling et al., [Dipolar-exchange spin waves in thin bilayers](https://arxiv.org/abs/2504.00631), Low Temperature Physics 51, 998–1003 (2025), [published paper](https://doi.org/10.1063/10.0038642). Here the projected model and its independent reference follow directly from the stated magnetostatic boundary problem.

Reports: [science](../../results/magnetic-bilayer-validation.json), [local controls](../../results/magnetic-bilayer-local-controls.json), and [root peer review](../../results/root-hardening-physics-review.json), key `magnetic-bilayer-r1`. Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_magnetic_bilayer.py`. Calibration regeneration requires `--generate`.

The instruction is the approved neutral instruction. Agent evaluation remains pending; successful scientific controls do not establish an agent failure rate.
