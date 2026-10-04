# Tube dispersion, revision 1

The shortcut replaces the steady Poiseuille profile by its cross-sectional mean and propagates a one-dimensional Gaussian with molecular diffusivity. This is mathematically exact for the zero-flow calibration and fits D correctly. In flow, an initially uniform radial population does not remove accumulated correlations between axial displacement and radial velocity history.

For stationary reflecting radial Brownian motion, axial variance is sigma0²+2Dt+2∫(t-s)Cov[u(r_s),u(r_0)]ds. In a round tube u=2U(1-r²/a²). Neumann J0 radial modes have roots j_n of J1, decay D*j_n²/a², and velocity coefficients -8U/j_n² in the area-normalized basis. The oracle uses this converged transient series. It includes both the short-time Var(u)*t² limit and the long-time extra diffusivity U²a²/(48D).

The independent reference discretizes conservative radial diffusion on annuli and integrates first/second axial moments. It uses no Bessel functions. Independent direct real-space moment ODE checks and refinement comparisons supplement the analytic short/long-time identities. The detector measures number-weighted axial position moments; there is no first-passage or flux weighting. All tracer molecules remain counted in the axially unbounded apparatus.

Hidden variance errors are RMS normalized by the group's RMS reference variance, with limit .04. Calibration recovery tolerance is 3%, and reduced chi squared must be below 1.5. Standard 256 noise fits and parameter-extrema predictions check robustness. Validation: `uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_tube_dispersion.py --noise-trials 256`; add `--generate` only for deliberate data regeneration. Concept reviewed by physics_review; final source review pending.

The axial-moment formulation follows [Aris, On the dispersion of a solute in a fluid flowing through a tube](https://doi.org/10.1098/rspa.1956.0065). This benchmark uses an independently derived finite-time series and annular reference.
