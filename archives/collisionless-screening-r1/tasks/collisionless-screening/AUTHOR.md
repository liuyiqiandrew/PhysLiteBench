# Collisionless screening, revision 1

This staged candidate has no model-agent evaluations. It retains the exact neutral instruction, unrestricted implementation replacement, an unfinished public fit with `density=None`, and the standard 600-second agent / 60-second verifier limits. Root controls all eventual Docker/model runs.

The source computes a complete real standing-response kinetic susceptibility and solves Poisson self-consistency. The physical error is its nonabsorptive, time-symmetric treatment of resonant particles, despite the explicit preparation in time. It is not a routine that computes the correct causal complex response and discards its imaginary part. The source uses every input, including finite calibration drive frequencies. The readout is always the in-phase total field; the discrimination does not rely on an identically zero source quadrature.

## Physics and selection in time

Let F(u)=15(1−u²)²/16 on |u|<1 and zero outside. It is normalized, nonnegative and single-peaked. With z=omega/(k*v0), division of F'(u) by u−z gives

    H_PV(z)=(15/4)[2z²−4/3+(z³−z) log|(z−1)/(z+1)|].

The source transfer is [1−density*H_PV/(k*v0)²]^-1. This is the self-consistent response of the stated real principal-value kinetic closure; it is not claimed to solve the actual causal experiment. For an electron charge of minus one, the linearized kinetic and Poisson equations are

    f1_t + i*k*v*f1 = E_total*f0',
    i*k*E_induced = −integral f1 dv.

The public remote-past drive exp[(eta−i*omega)t] with eta>0 places the velocity denominator at u−z−i*eta/(k*v0). In the specified order of limits,

    H(z)=H_PV(z)+i*pi*F'(z),  |z|<1,

and H=H_PV outside support. The physical dielectric is 1−density*H/(k*v0)². The oracle returns its inverse's real part. For positive frequencies within support, F'(z)<0, so the dielectric has positive imaginary part and its inverse has negative imaginary part under the public exp[−i*omega*t] convention. The lost kinetic response changes the in-phase field after self-consistency, even though the source's real susceptibility is evaluated accurately.

The plasma is a specified prepared collisionless velocity distribution, not a Maxwellian thermal equilibrium. Its single maximum gives the stable one-component Vlasov setting. Compact support can support undamped collective modes outside the particle velocities. The apparatus therefore fixes the remote-past adiabatic selection, rather than assuming that transients from an arbitrary sudden switch decay. Taking linear amplitude before eta→0 also avoids replacing the defined response by nonlinear particle trapping at finite drive.

The public frequency windows keep both models away from poles and logarithmic endpoints. Outside support, H_PV=integral F(u)/(u−z)² du is positive and decreases for z>1. At z>=1.5 and maximum density/(k*v0)², the source denominator is at least .5708666. Within [.20,.65], H_PV increases to H_PV(.65); the denominator is at least .7300184. The physical dielectric cannot vanish there because its imaginary part is nonzero. At zero frequency both denominators exceed one. These are numerical-regularity choices declared before any model run, not hidden exclusions based on a scored failure.

## Exact calibration and independent reference

For static screening, H(0)=−5 and the transfer is 1/[1+5*density/(k*v0)²]. It strictly decreases throughout density [.6,1.4], establishing global identification. Finite-frequency calibration uses phase speeds outside support, where both controls also agree exactly. There are 18 unique settings: six static and twelve nonzero-frequency controls, each repeated independently sixteen times, for 288 records. The fixed instrument sigma is .0012; it depends on neither density nor response. Calibration seed219031 and noise seed219037 are fixed. The private true density1.07 differs from the disclosed interval midpoint; every constructor starts with None.

The verifier derives a causal memory equation from the original kinetic initial-value problem:

    E(t)=E_ext(t)−density*integral_0^infinity tau*phi(k*v0*tau)*E(t−tau) d tau,

where phi(s)=integral F(u)exp(−i*s*u)du=15*j2(s)/s². Thus finite positive switch rates have transfer

    [1+density/(k*v0)² * integral_0^infinity
        s*phi(s)*exp[(i*z−delta)*s] ds]^-1,

with delta=eta/(k*v0). The reference performs this time integral at delta=.012,.006,.003 and extrapolates the transfer quadratically to delta=0. It contains neither a principal-value logarithm nor a resonant jump. Halving all three rates checks the limiting calculation independently.

The final time integration uses composite 16-point Gauss–Legendre quadrature on intervals no wider than pi/2, ending at20/delta. Since |s*phi(s)|<=15/s²+45/s³+45/s⁴, the remaining exponentially weighted tail is bounded by [15/L²+45/L³+45/L⁴]*exp(−delta*L)/delta; it is below1e−11 for every rate used. Raising quadrature order to24 changes memory integrals by at most1.82e−15. Separate direct velocity quadrature verifies phi to7.78e−16 and the source principal value to2.92e−15.

The first packaged reference used weighted infinite-interval quadrature. Its domain validation passed but emitted one convergence-acceleration warning. The complete initial reference and reports are preserved in results/initial-quadrature. Only that integration method was replaced. Public physics, source, oracle, data, uncertainty and grading were unchanged; the full validator was rerun without data regeneration. The final calibration reference differs from the oracle by at most6.53e−5 of one instrument sigma, so numerical calibration bias is negligible. No model trials occurred during this change.

## Validation and limits

Seventy-two corner/random cases cover both density bounds, geometry bounds and all frequency windows. The largest in-phase oracle/reference discrepancy is1.82e−6; halving the switch rates changes the full complex response by at most3.05e−6. Frequency reversal gives complex conjugation, resonant response has the expected passive sign, the zero-density transfer is one, and static screening is exact. Thirty-three noiseless fits across the whole density interval recover it to6.58e−9 absolute. All scored source groups remain above the4% gate throughout those density samples.

The actual fit is1.0698469563 with reduced chi-square1.09116. The three source group errors are .35934,1.16901,2.99451. Physical in-phase signals in those groups are at least .1395, so this is not normalization by nearly zero answers. Both models pass static/nonresonant anchors.

All256 independent noisy calibrations pass their fit and parameter gates. Every oracle prediction passes, while every source fails all three diagnostic groups. Maximum oracle hidden error is .001667, minimum source diagnostic error .358446, maximum calibration chi-square1.29109. Prediction errors are group RMS divided by reference RMS, with gate .04. The parameter gate is3% and reduced chi-square gate1.5. Local isolated pytest returns oracle7/7 in .55seconds and source4passes/3intendedfailures in .56seconds. These are scientific controls, not empirical agent difficulty.

## Prior work and evaluation classification

The outline audit searched235 archive/stage/current AUTHOR or assessment files and read the closest eleven authors. Collisionless-trap and plasma-compression concern adiabatic actions and pressure anisotropy. Qubit r8 concerns cross-interval quantum bath response, and finite-band-reservoir concerns localized-mode populations after a contact quench. No kinetic resonant-response task was found. These shared broad themes are acknowledged; this task introduces a different classical kinetic preparation.

Primary background is Landau, [On the vibrations of the electronic plasma](https://www.princeton.edu/~vnd/Landau1946.pdf),1946, with the [author's reprint record](https://www.ufn.ru/ru/articles/1967/11/m/). The indexed primary passage on analytic continuation and the publisher abstract were accessible; direct full-PDF retrieval timed out. No claim is made that the complete paper was read. The compact velocity distribution, calibration and independent memory reference are constructed and derived here. The source/primary access history and two rejected alternative outlines are preserved under outlines/.

A later trial that retains the time-symmetric closure exhibits the intended physical error. A trial that recognizes the causal preparation but makes an i0-sign, logarithm, normalization or coding error must be classified separately. Causality is familiar physics, so these large scientific gaps do not establish that Luna will fail.

Reproduce from the repository root:

    uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/collisionless-screening-r1/scripts/validate_collisionless_screening.py

Only --generate replaces both calibration copies. Reports are results/collisionless-screening-r1-validation.json and results/collisionless-screening-r1-local-controls.json inside the stage. Source provenance and independent peer review are recorded separately. No canonical task or shared status file was changed.
