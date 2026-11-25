# Hot Brownian rotation r1

Status: staged and scientifically validated; no model evaluations yet. No canonical task or shared status file was changed.

The completed source has the exact unsteady rotational hydrodynamic impedance and exact static heated noise. Calibration therefore validates the nonuniform-temperature reduction at zero frequency. Hidden torque spectra test whether one effective bath temperature survives at all frequencies. The physical fluid's local thermal stresses do not generally permit that contraction.

All 256 noise realizations pass calibration and physical predictions; all reject the completed shortcut. Maximum oracle error 3.30e-5, minimum shortcut diagnostic error .1052, tolerance .025. Independent radial stress dynamics agree within 4.57e-7 relative on hidden cases. Local controls: oracle 9/9; shortcut 6 pass/3 intended failures. Exact static and uniform-bath anchors pass both.

The archive audit explicitly compares unsteady-sphere, heated-filament, thermal-bodies r11, hydrodynamic-fluctuations, gyroscopic-noise and entropy-anomaly. None used this frequency-dependent scalar noise contraction, though the local-reservoir background is shared. Empirical difficulty remains unknown. Root owns Docker controls and three fresh unhinted frozen trials.
