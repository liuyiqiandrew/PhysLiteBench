# Ionic current loops, revision 1

A dilute ternary electrolyte is locally electroneutral in a periodic square. The electrostatic potential must be a scalar periodic field. Charge relaxation imposes divergence-free charge current, but does not set that current to zero locally. Different cation mobilities and crossed concentration gradients allow internal current loops.

Write g=sum(z_i D_i c_i) and kappa=sum(D_i c_i). The potential satisfies div(kappa grad phi)=-laplacian(g). The completed shortcut instead sets grad phi=-grad(g)/kappa locally. It conserves local charge and each integrated species amount, and is exact for the one-dimensional reflection-symmetric calibration. In two dimensions its proposed potential gradient generally has nonzero curl. The physical task is to enforce electrostatic integrability while allowing current circulation.

The shared D factors out of the potential equation and multiplies all observed concentration slopes. It is identifiable by a linear weighted least-squares fit. Measurements are instantaneous Fourier slopes at known positive concentration profiles, avoiding an extra time-integration error. Calibration has both cation profiles even in x, making the one-dimensional periodic current exactly zero. Hidden profiles have crossed or oblique gradients and measure the mixed Fourier responses.

The oracle solves a periodic elliptic potential equation by preconditioned CG with 49-point Fourier derivatives on each axis. The independent reference constructs conservative finite-volume face conductances and solves a sparse potential system on a 96 by 96 grid. It then uses those same faces for all ionic fluxes. Calibration uses an independent exact one-dimensional reduction. The validator checks refinement to 192 cells, integrated mass, local charge-rate cancellation and electric curl.

The tolerance is relative RMS 0.04 of each group signal, floor 1e-6. Current numerical and parameter errors are well below it: nominal oracle maximum 0.00165; at observed noise-fit extrema the maximum is 0.00778. The shortcut error stays above 0.076 at those extrema. Seed 16222 generated the calibration; 256 additional draws use seed 26222. Every parameter estimate is within 3%; 254/256 noisy draws pass the chi-square threshold 1.5, exceeding the predefined 99% validation criterion. The checked-in calibration passes for both controls. See results/ionic-current-loops-science.json.

During initial scientific screening, one candidate phase-shifted preparation had only 2.9% separation, below the preselected 4% tolerance. Before any agent evaluation it was replaced by a crossed-gradient preparation with stronger separation. The physical equations, tolerance, calibration, and public parameter domain were unchanged.

Reproduce with:

```
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_ionic_current_loops.py
```

The planned evaluation is three plain and three hinted gpt-5.6-luna high trials at the repository's unchanged agent and verifier time limits. Difficulty results are pending. The hint explains current continuity and the periodic potential constraint without revealing fitted parameters or hidden measurements.
