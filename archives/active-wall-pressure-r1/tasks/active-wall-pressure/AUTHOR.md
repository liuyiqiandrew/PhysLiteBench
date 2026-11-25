# Active wall pressure, revision 1

This staged task separates the exact stationary state of active particles from a constitutive expression for the force on an orienting wall. The completed source solves the full position/orientation problem, including the spatial and angular derivatives of the entire wall potential. It uses every control and computes a nonzero pressure. Its remaining approximation applies the torque-free midplane stress to that state. The physical observable is the actual coating reaction, so the angular transfer at the wall also enters the pressure.

The unknown is a common particle number per unit y-length, called loading. A normalized one-particle stationary density psi obeys integral dx dtheta psi=1. Multiplication by loading converts a mean force per particle into pressure per wall length. The stochastic equations and all remaining coefficients are public. In particular, rotational mobility is Dr/T, not an independent unknown, and the spatial derivative of the orientational interaction acts on translation. The prescribed self-propulsion is part of the ideal overdamped model; direct solvent/support loads are outside the calibrated transducer output.

Define rho(x)=integral psi dtheta, m(x)=integral cos(theta) psi dtheta and q(x)=integral cos(theta)^2 psi dtheta. Let tau=−Dr*V_theta/T denote angular drift. At steady state the zero positional flux gives

    integral_0^infinity dx integral dtheta V_x psi
        = T*rho(0) + v*integral_0^infinity m(x) dx.

The integrated first angular moment gives

    Dr*integral_0^infinity m dx
        = v*q(0) − T*m'(0) − integral_0^infinity dx integral dtheta sin(theta)*tau*psi.

Both potential derivatives vanish at the midplane. The far tails vanish under the quartic confinement. Combining these identities yields the pressure conversion used by the oracle. The source retains the first two terms of the second identity, evaluated from the correct anisotropic finite-channel state. It does not additionally assume an isotropic far-away bulk, replace the joint state by a Gaussian, or discard the orienting input from the kinetics.

Writing c_n(x)=integral exp(−in theta) psi dtheta, reflection in theta makes these moments real. The oracle's additional term per unit loading is

    (v/Dr) * integral_0^infinity Dr*h*A(x)/(2*T) * [c_1(x)−c_3(x)] dx,
    A(x)=s^4/(ell^4+s^4), s=max(abs(x)−L,0).

The coating displacement changes its full interaction by −V_x times that displacement. This establishes the force sign and includes the force associated with its orientational energy. At h=0 the extra angular term vanishes identically for every speed, temperature and separation. Thus the completed source and oracle agree exactly on all calibration controls, even in a narrow anisotropic channel. A positive calibration slope makes loading globally identifiable throughout [.8,1.2]. This is a physical stress closure error, not an incomplete stationary solver or a missing numerical factor.

The source and oracle solve angular Fourier moments n=−18,...,18, with centered conservative spatial flux and zero flux at a remote numerical boundary. They use Richardson extrapolation from 129 and257 x points. The private reference instead builds a positive nearest-neighbor Markov generator on position and orientation grids. Its Scharfetter–Gummel rates use potential differences and the active translational bias, and the wall force is integrated directly from V_x. Richardson extrapolation of161×48 and321×96 grids removes the leading error. The reference neither computes the midplane stress nor inserts the angular-transfer formula. Its refinement to321×96 and641×192 agrees within3.35e−7 relative at the strongest cold alignment cases.

Science is recorded in `results/active-wall-pressure-r1-validation.json` within this stage. There are288 independent calibration readings,12 distinct controls repeated24times. The true loading is1.07; both constructors start at None. Sigma=.001 is a fixed absolute instrument uncertainty, independent of loading and response. Seeds141031 and141037 generate the checked-in sample and256 noise checks. The fitted loading is1.0696888894, with relative error.0002908 and reduced chi-square.81415. The maximum numerical calibration bias is6.03e−5 sigma. The source/oracle calibration bases are numerically identical, their minimum slope is.14737, and the Fisher information is2.2675e7.

The three diagnostic shortcut errors are.31565,.34682,.32659, with unchanged torque-free anchors. All256 fits meet parameter and calibration gates; every oracle passes and every shortcut fails all three diagnostic groups. The maximum oracle hidden error is.0005572 and the minimum shortcut diagnostic error is.31511, at the.04 RMS-relative prediction gate. The maximum noise-study parameter error is.0005566 and chi-square1.2561. The gate is well above all calibration/numerical errors and was not tightened to force separation.

Independent oracle/reference agreement is within1.26e−6 relative over24 supported corners. Oracle spatial refinement changes results by at most1.58e−6. Normalization error is below2.3e−16; reconstructed joint densities have only roundoff-scale negative minima, no lower than−2.7e−14. A passive v=0 partition integral independently checks the thermal limit for h=−1,0,2, within7.63e−7 relative. All tested physical pressures are positive. The older quadratic-wall exploratory scripts and results are preserved in prototype/; the smoother quartic potential was chosen before calibration generation to reduce interface discretization error, with the same mechanism.

`results/active-wall-pressure-r1-local-controls.json` contains actual isolated local pytest runs: oracle7/7 in4.50seconds; source4passes plus exactly3intended hidden failures in4.17seconds. Agent/verifier limits remain600/60seconds. There are no model evaluations yet, and these scientific controls do not establish model difficulty.

Reproduce from the repository root:

```
uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/active-wall-pressure-r1/scripts/validate_active_wall_pressure.py
```

Use `--generate` only for deliberate calibration regeneration. The neutral instruction permits replacing any implementation while retaining the interface. Docker copies only public files. The separate completed source control is `scripts/active_wall_pressure_baseline.py` within the stage.

The archive audit in `prototype/archive-overlap.json` records the broad AUTHOR scan. Directly checked neighbors were active-trap (non-Gaussian position statistics), dumbbell-stress (affine stretching versus corotation), hydrodynamic-heating (local energy conversion from exact nonlocal fields), and hot-brownian-rotation (frequency-dependent thermal-stress weighting). None tests wall torque dependence of active mechanical pressure. The shared themes of exact internal mathematics and incorrect physical readout are acknowledged.

Primary context: Solon et al., [Pressure is not a state function for generic active fluids](https://arxiv.org/abs/1412.3952), Nature Physics11,673–678(2015), DOI10.1038/nphys3377, derives the role of wall torques in active mechanical pressure. The specific finite-channel moment identity, joint potential and independent force calculation used here are given above.
