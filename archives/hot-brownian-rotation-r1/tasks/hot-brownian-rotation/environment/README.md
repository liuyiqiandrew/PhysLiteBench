# Fluctuating torque on a heated sphere

A rigid sphere of known radius `radius` is immersed in an unbounded incompressible Newtonian liquid. An ideal support fixes its center and orientation. The liquid satisfies no slip at the surface and is at rest on average far away. A noiseless support transducer measures one Cartesian component of the torque exerted by the liquid on the sphere. The clamp and transducer add no fluctuating torque.

Use classical linear fluctuating hydrodynamics as the exact model on the stated scales. Fluid density is one. The dynamic viscosity `viscosity` is constant in space and independent of temperature; its unknown value lies in [.7,1.4]. Retain the fluid's linear time-dependent inertia. Neglect convective inertia, compressibility, thermal expansion and buoyancy.

An external heat supply maintains the known steady radial temperature

    T(r) = ambient + rise*radius/r,  r >= radius.

Heat advection and fluctuations of this prescribed temperature are negligible. The heat supply exerts no fluctuating torque. The liquid is locally in thermal equilibrium. In units with k_B=1 its symmetric, zero-mean Gaussian thermal stress s has covariance

    <s_ij(r,t) s_kl(r',t')>
      = 2*viscosity*T(r)*(delta_ik*delta_jl + delta_il*delta_jk)
        *delta^3(r-r')*delta(t-t').

The liquid momentum equation contains the divergence of this stress in addition to pressure and Newtonian viscous stress. Fluid velocity is divergence-free. These assumptions specify the ideal continuum model throughout the given temperature and frequency range.

After stationarity, the measurement is the two-sided connected torque spectrum

    S(omega) = integral_-infinity^infinity exp(i*omega*t)
               <delta torque_x(t)*delta torque_x(0)> dt.

It has units of torque squared times time. Angular frequency `omega` is nonnegative in the stated inverse-time units. No deterministic torque is applied during measurement. Every record uses a fresh stationary preparation of the same liquid.

Each input contains the known `radius` in [.8,1.2], `ambient` in [.8,1.2], `rise` in [0,2*ambient], and `omega` in [0,120]. The same viscosity applies to all experiments. Calibration records in `data/calibration.json` contain `input`, measured `value`, and fixed instrument uncertainty `sigma=.02`; errors are independent Gaussian variables.

Implement `Model.fit(records)`, returning self and storing the inferred value as `viscosity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` containing the spectra in input order. Run `python -m pytest -q test_public.py`.
