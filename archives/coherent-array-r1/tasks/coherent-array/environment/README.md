# Scattered field from a trapped array

Five identical classical point scatterers have known centers x_j=j for j=0,...,4 and scalar displacements u_j along the line. All masses are one. Their ideal potential is

    U = (K/2) * [.6 * sum_j u_j^2
                 + coupling * sum_(j=0)^3 (u_(j+1)-u_j)^2].

There is no closing link or other interaction. The common unknown spring scale K lies in [.8,1.2]. Length is measured in L0, energy in E0, mass in M0, and time in sqrt(M0*L0^2/E0). The numerical temperature T is k_B times physical temperature divided by E0. K is measured in E0/L0^2.

Each scatterer remains in contact with an independent local heat bath throughout observation. In these units its motion is

    du_j = p_j dt,
    dp_j = [-partial U/partial u_j - .4*p_j]dt + sqrt(.8*T)dW_j,

where the W_j are independent standard Wiener processes. The full array is prepared in its stationary classical Gibbs state before every measurement. The bath temperature and damping remain fixed during each measurement.

A negligibly weak single-scattering optical probe sees identical known scalar scattering amplitudes. An ideal phase-stable heterodyne reference records both quadratures of the scattered complex field. After removal of the known single-particle amplitude and optical carrier, the recorded field is

    A_q(t) = sum_j exp[i*q*(j+u_j(t))] / sqrt(5).

The output is Re <A_q(t) conjugate(A_q(0))>, averaged over stationary preparations and bath noise. It includes the mean field. The probe causes negligible force or heating, and there is no detector filtering or additional noise contribution to this correlation.

Each input contains `temperature` in [.04,.12], `coupling` in [0,1], `wavevector` q in [1,2*pi], and `delay` t in [0,4]. Outputs are dimensionless field correlations in the normalization above.

## Data and interface

`data/calibration.json` contains independent measurements with keys `input`, `value`, and `sigma`. The fixed instrument standard deviation is .003 for every reading; repetitions are independent. Prediction inputs have the same schema and allowed ranges.

Implement `Model.fit(records)` to estimate the common K, store it as `self.stiffness`, and return `self`. `Model.predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in the listed input order.
