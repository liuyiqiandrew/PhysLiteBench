# Floating-contact noise, revision 2

Revision 1 tested whether the island potential fluctuates at all. All three plain and all three hinted trials repaired that closure correctly. Its complete source, validator, control, science report, manifest and six reviewed trials are preserved in `archives/floating-contact-noise-r1/`; all original jobs remain unchanged.

Revision 2 gives the starter the correct stationary potential and full low-frequency covariance projection. Calibration measures both mean currents and covariances, so it checks that charge feedback and identifies the common mixing angle. The remaining approximation applies the same linear charge-response map to the intrinsic third cumulant evaluated at the mean island potential. This conserves total current fluctuations at every order, but misses how intrinsic shot noise changes with the fluctuating potential. The omitted noise–potential correlation contributes at leading order to the long-time third cumulant.

The public apparatus now states the macroscopic semiclassical counting limit instead of revision 1's linear Langevin limit. Counting time tends to infinity at finite capacitance before taking the leading macroscopic-capacitance term. Internal thermalization is fast and heat removal fixes zero temperature; there is only one single-channel floating contact, and scattering is energy independent. These assumptions specify the voltage-probe counting statistics and remove charging-energy, floating-temperature and order-of-limits ambiguities.

Write T_ij=|S_ij|², w=exp(s), and z=exp(s_probe). At positive bias, the two energy windows have respectively the single-particle and single-hole generating factors

    F0 = T00 + T01*w + T20*w/z
    F02 = T11 + T01*w + T21*z.

The stationary potential saddle requires F0=F02. Choosing the branch normalized at s=0 gives the physical cumulant generating function per unit bias

    log[(T00+T11)/2 + T01*w
        + sqrt((T11-T00)^2+4*T20*T21*w)/2].

The oracle evaluates its third derivative analytically. Current and noise continue to use the separately checked scattering-current matrix and charge-response projection. Bias reversal changes the sign of odd cumulants; exchanging the counted external lead also changes their sign.

The independent reference closes the probe channel with a lossless reflector of phase phi, computes the coherent two-terminal transmission, and uniformly averages its Bernoulli cumulants. This averages the **log generating function**, not a probability generator of a quenched phase mixture. Its equivalence to the voltage probe applies here because there is exactly one single-channel probe and energy-independent scattering. It would generally fail for multiple probes or channels. See [Förster, Samuelsson, Pilgram and Büttiker, Voltage and dephasing probes: a full counting statistics discussion](https://arxiv.org/abs/cond-mat/0609544), and [Pilgram et al., Full counting statistics for voltage and dephasing probes](https://arxiv.org/abs/cond-mat/0512276).

The 160 calibration records use seed 20021; 256 independent noise draws use seed 30021. Hidden groups measure third cumulants at weak, intermediate and stronger probe coupling, with both bias signs and both external leads. Each group's RMS error is normalized by its RMS physical signal; the limit is .04. The completed shortcut is numerically exact for its projected-fixed-potential approximation. The independent reference uses 1024 reflector phases, with 2048-point refinement. Validation also checks public parameter/direction corners, the effective transmission probability bounds, scattering unitarity, odd-cumulant signs, analytic-CGF derivatives by Cauchy integration, and the disconnected Bernoulli limit.

Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_floating_contact_noise.py --noise-trials 256`. Add `--generate` only when deliberately regenerating both calibration copies. The report distinguishes the 256 calibration fits from hidden sensitivity checks at their fitted-parameter extrema. Local isolated tests give oracle 7 passed and shortcut 4 passed / 3 intended hidden failures.

Final validation recovers alpha=.5098766173 with reduced chi-square1.01545. Nominal oracle third-cumulant errors are .000804/.00140/.00516; the shortcut gives .213/.625/3.748. All 256 calibration fits pass, and the fitted-parameter extrema give oracle error at most .02722 versus shortcut error at least .2082. Independent phase-average agreement is1.67e-15; 1024-to-2048 phase refinement is1.12e-16; the Cauchy derivative check differs by at most3.22e-13. The source physics review and evaluated batch results are recorded separately in results.
