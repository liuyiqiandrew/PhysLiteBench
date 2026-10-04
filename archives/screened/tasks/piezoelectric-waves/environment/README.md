# Acoustic waves in an insulating piezoelectric bulk

An infinite homogeneous, lossless piezoelectric solid supports small-amplitude
plane acoustic waves. Its mass density is 6000 kg/m^3. The solid has no free
electric charge or mobile charge carriers, and there are no electrodes,
surfaces, imposed electric fields, or electrical connections. Use the
electroquasistatic limit: magnetic induction and electromagnetic retardation
are negligible at the acoustic frequencies. The material is initially at rest
with zero strain and electric field. Ignore damping, dispersion, and nonlinear
effects. Mechanical and electric responses remain coupled during the wave.

The linear constitutive equations in the fixed crystal axes x,y,z are

    stress_ij = lambda*trace(strain)*delta_ij + 2*mu*strain_ij
                - sum_k e_kij * E_k
    D_k = sum_l epsilon_kl * E_l + sum_ij e_kij * strain_ij.

Here strain_ij=(partial_i u_j + partial_j u_i)/2 is tensor strain, stress is in
Pa, E in V/m, and electric displacement D in C/m^2. The shear modulus mu is
20 GPa. The unknown first Lame coefficient lambda lies in [24,50] GPa and is
the same in all experiments. These elastic coefficients are defined at fixed
electric field. The dielectric tensor at fixed strain is
diag(8e-9,8e-9,1e-8) F/m. The only nonzero piezoelectric coefficients are

    e_xxz = e_xzx = e_yyz = e_yzy = 12 C/m^2
    e_zxx = e_zyy = -7 C/m^2
    e_zzz = 18 C/m^2.

The first index of e labels electric displacement and the last two label
strain. These coefficients, the density, and all propagation directions are
known exactly. The ideal linear constitutive model applies throughout the
stated range.

For a given unit vector direction n, consider the three acoustic plane-wave
branches proportional to exp(i*k*(n dot x - v*t)), with positive k. Label their
positive phase speeds in increasing order: branch 0 is slowest, branch 2 is
fastest. At a degeneracy, repeated speeds retain their repeated branch labels.
A detector measures the phase speed in m/s. Every input contains direction,
a three-element unit vector in the crystal axes, and branch, an integer 0,1,2.

Calibration measures the fastest branch along the z axis and both slower
branches for directions in the xy plane. Independent Gaussian measurement
errors have the listed standard deviations. One unknown lambda applies to
all records and all requested directions.

Implement Model.fit(records), returning self and storing lambda in
Model.lame_parameter in GPa. Implement Model.predict(experiments), returning a
finite NumPy array of shape (len(experiments),) in m/s. Records in
data/calibration.json have input, value, and sigma. Run python -m pytest -q
test_public.py.
