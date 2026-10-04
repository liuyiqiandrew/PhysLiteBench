# Soret-Dufour relaxation, revision 1

The starter and completed shortcut retain the measured Soret composition response but model heat as Fourier conduction alone. They solve that triangular coupled system correctly. Isothermal composition calibration is exactly insensitive to the reciprocal heat current, because the bath removes it, and identifies the diffusivity.

Use conjugate entropy forces Xq=grad(1/T) and Xc=-grad[(muA-muB)/T]. The measured species flux gives Lcc=nD*c0(1-c0)/R and Lcq=nD*c0(1-c0)*S*T0². Zero magnetic field and microscopic reversibility require Lqc=Lcq. Thus q=-kappa gradT−nD R T0² S gradc. With fixed-composition heat capacity C, conservation gives C*Tt=-divq and n*ct=-divj. Equal partial enthalpies avoid heat-flux/energy-storage ambiguity. The stated kappa is the coefficient at uniform composition, not the zero-diffusion-flux conductivity.

The symmetric Onsager matrix is positive definite over the entire fitted D range. At its upper bound the squared cross coefficient is .187 of the diagonal product. The correct modal diffusion matrix is [[kappa/C,nD R T0²S/C],[D*c0(1-c0)*S,D]], with consistent SI/mm units. Insulated and impermeable endpoints impose zero current for both fields, satisfied by the cosine modes.

Hidden tasks measure temperature amplitudes after the distributed bath is removed. Errors are RMS normalized by the group's RMS reference amplitude, limit .04. The source shortcut predicts no temperature from a pure composition preparation; this is a physical omission, not inaccurate fitting or propagation. The oracle uses continuum cosine modes; the independent reference reconstructs Onsager currents and conservative finite-volume modes. Real-space current-divergence integration, grid convergence, conservation and entropy-production positivity are checked independently.

Validation: `uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_soret_dufour.py --noise-trials 256`. Opt-in generation uses `--generate`. Concept reviewed by physics_review; final source review and measured scores pending.

Related primary evidence for reciprocal heat/species coefficients: [Heat and matter transport in binary liquid mixtures](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.2133). The task uses the explicitly stated linear constitutive model and independently derived coefficients.
