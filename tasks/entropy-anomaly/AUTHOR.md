# Entropy anomaly revision 7: a spatial calorimeter reveals contact population lag

This revision keeps the complete revision-6 fast-contact particle apparatus, public parameter domain, calibration values, fitted friction and 600/60-second limits. It adds an explicitly public smooth calorimeter detector and a corresponding measured-residence background subtraction. All 192 default-uniform calibration records are unchanged byte for byte. Prediction, parameter and chi-square gates remain .04, .03 and 1.5. Scientific validation and the requested three unhinted `gpt-5.6-luna` high trials are separate evidence.

## Prior evidence and selected physical distinction

Revision 6 has one reward pass, two agent timeouts and one clean intended physical failure. `6aaek3Y` fitted the completed mixed-contact shortcut and failed all three contact diagnostics; adding the finite-kinetic-contact hierarchy at its unchanged fitted friction repaired the failure without changing calibration. `Dvgx4xe` recognized that physics but deleted its model during a failed file replacement and timed out, leaving no submitted model; it is an implementation/timeout failure, not a physical failure. `VKPRXCy` left a reward-passing hierarchy implementation despite an agent timeout. Its integrated entropy agrees with the full oracle to 1.13e-13 on hidden cases and 5.23e-13 on 64 random interiors and six anchors, with independent paired finite-mass agreement within 7.92e-8.

The last implementation omitted the first mass correction to conditional contact position densities. That omission is legitimate for the spatially integrated revision-6 observable: both contact corrections are periodic derivatives with zero integral, and the proportional bath temperatures make their entropy coefficients spatially constant after division by temperature. It is not legitimate in a spatially weighted calorimeter. This revision's completed shortcut includes all correct revision-6 velocity/contact moments and all bulk predictions, and applies only that equal-local-position-contact approximation to the new readout. The full revision-6 task, scripts, controls, reports and every frozen/native/model trace are preserved in `archives/entropy-anomaly-r6`; earlier r5 evidence is also retained. Its 185-file preservation manifest verifies the complete archive.

## Complete public operation and parameter identification

The particle has two velocities and mass m on a periodic square, with no potential. In contact s, m dv=(-Gv+B R_clockwise v+F)dt+sqrt(2G T_s(x))dW, G=diag(gamma,gamma*drag_ratio), T_s(x)=T0(1+s delta)(1+a cos(kx)). The valve flips independently at the actual rate kappa/m, where kappa is the input `switch_rate` held fixed as m tends to zero. Position and velocity are continuous at a contact change; no instantaneous particle work, impulse or extra heat increment occurs. The magnetic field does no work. The finite-mass bath entropy increment and all units are stated publicly.

When optional `detector_phase=phi` is supplied, the actual calorimeter weights each bath entropy increment by w(x)=(1+cos(kx-phi))/2 at the particle's position. The phase lies in [-pi,pi]; no phase field means w=1, exactly the old bulk observable. Detection changes neither dynamics nor the contact law. At each positive mass bring the actual preparation to stationarity and measure its weighted entropy rate A_m and actual weighted residence R_m=E_actual[w(x)]. Independently prepare the same friction, drags, field, T0, delta, kappa and mass with forces and profile contrast set to zero, and measure the unweighted background bath entropy rate B_m. The reported rate is

    lim_{m->0} [A_m-B_m R_m].

Every requested input has its own matched background and actual residence. The background is not multiplied by a guessed uniform residence or by the leading overdamped density. There is no detector noise, missing contact energy, unreported switching impulse or undisclosed subtraction constant. The result is an excess local entropy signal and can have either sign. Positivity of the total entropy production does not imply positivity of this subtracted port readout.

Calibration omits the detector phase and uses delta=B=0. The calibrated entropy is proportional to 1/gamma, so weighted least squares identifies the same common gamma in [.7,1.6]. Neither the detector nor background adds a fitted parameter. The phase changes only the readout; all existing force, drag, field, temperature, contrast, wavenumber and kappa bounds are unchanged.

## Moment hierarchy and finite contact-position lag

Use the scaled fast velocity q=sqrt(m)*v. The exact leading conditional contact masses are rho/2, and the fast generator contains velocity relaxation, thermal forcing, magnetic rotation and switching kappa together. Raw velocity moments through degree four give the Hilbert hierarchy used in revision 6: leading degree-two and degree-four moments; first corrections of degrees one and three; and the finite degree-two correction. All tensor dimensions and thermal/contact correlations are retained. The leading summed second moment is rho*T(x)*I, so the leading positional density and currents are the same complete overdamped fields used in r6, rather than separate bath densities.

For the contact masses at the next order, continuity gives the label-antisymmetric correction

    a_+(x)=-partial_x M1_plus,x/(2 kappa),   a_-=-a_+.

Each a_s integrates to zero. The common O(m) correction to the position density requires a higher positional equation, but the measured-residence subtraction removes its contribution pointwise in the final observable. Indeed the leading local bath entropy density is ell*rho/m, with ell independent of position because both baths scale with the same T(x). In the matched uniform/no-force background the finite-mass entropy is exactly ell/m. The common mass correction contributes ell*rho2 to A_m and the same term to B_m R_m. Thus the oracle may use zero common correction while retaining a_+ and a_-, which have a nontrivial weighted consequence.

The local finite entropy coefficient is obtained by applying the degree-two relaxation system to its thermal population source 2 gamma_i T_s a_s, force/first-moment source, and spatial derivative of the degree-three correction. It then evaluates sum_s,i gamma_i*(M2_finite,s,ii/T_s-a_s). For uniform w, this reproduces exactly the r6 bulk value. For a port, integrate that local coefficient against w. The oracle does not insert an extra phenomenological port factor or fit a correction.

The completed source solves the same leading second/fourth, first and third kinetic-contact systems and positional problem, but projects a_s to zero in the finite heat stage. This assumes the local positional contact fractions are instantaneously equal through the first mass correction, even while the conditional velocity statistics retain correct finite persistence. It is a completed local fast-contact occupancy approximation. It becomes exact for bulk spatial integration, delta=0, uniform temperature profiles, or sufficiently rapid kappa, but it misses a small local population lag multiplied by an O(1/m) heat response. Source and oracle differ only in this physical projection, not in a tensor factor, solver stability, unfinished fit or modified calibration.

## Independent closed anchor and discriminating preparations

For B=0 and drag_ratio=1, direct scalar moment algebra gives

    correct_port-source_port
      = delta² gamma² / [(gamma+kappa)² (gamma+2kappa) (1-delta²)]
        * integral w partial_x²(T rho) dx.

Stationary constant-force transport satisfies partial_x(T rho)=Fx*rho-gamma*Jx. For the stated sinusoidal port this reduces to

    coefficient * Fx*k/2 * E_rho[sin(kx-phi)].

This checks the population-source coefficient, contact factor two, derivative sign and detector phase independently of the tensor hierarchy. The lag vanishes for zero x drive in the isotropic zero-field pure-gradient problem. Therefore the new discriminating groups use driven nonuniform profiles, rather than recycling those r6 pure-gradient preparations.

Three four-case port groups vary isotropic drive, unequal drag channels, and magnetic transfer of transverse drive into x transport. Their simple allowed settings include both force/phase reflection partners. At gamma=1.1 the source's group RMS discrepancies are about .0711, .1729 and .3639. The group error denominator is its true RMS signal; a small individual port signal is not used as a relative denominator. Two additional groups preserve bulk readouts and uniform-profile ports, which both controls must pass. Equal-contact port equality is checked independently during author validation. The .04 gate is unchanged and is far above finite-mass/reference and calibration errors.

Complementary detector phases give weights that sum to one, so their port excesses must sum to the bulk excess. Phase periodicity, uniform-profile half-bulk signals, contact relabeling and magnetic reflection provide additional readout checks. These identities also ensure the model is implementing the stated detector rather than an arbitrary input-dependent scaling.

## Independent finite-mass calorimetry and scientific validation

The production private reference retains the independent two-contact Fourier-Hermite Kramers solver. It solves the literal finite-mass SDE/contact equations for the actual preparation and separately solves the uniform/no-force background. It evaluates weighted Stratonovich bath entropy from each solved conditional second velocity moment, measures weighted residence from the actual finite-mass contact position marginals, and subtracts the actual separately measured background multiplied by that residence. It never imports the oracle, population-lag reduction, local leak coefficient or leading overdamped density. Three decreasing masses are extrapolated after subtraction. Mass, spatial basis and velocity basis refinements check convergence.

The validator retains all r6 normalization, current, leading conditional covariance, nonGaussian fourth moment, finite-mass energy/entropy, uniform-background and calibration anchors. Across the full 512 public-domain corners it checks density positivity, conditional leading covariance positivity, constant leading local leak, contact mass normalization and the full bulk identities. Existing kinetics and stability assumptions are unchanged. It additionally compares both controls with archived r6 on every old hidden input and 64 random bulk interiors, checks complementary ports, detector phase periodicity, equal-contact ports and exact uniform half-bulk reductions at three allowed friction values, and evaluates the scalar lag anchor across isotropic parameter corners. A weighted finite-mass energy-transport identity includes the derivative term partial_x(w/T_s); dropping that boundary transport term would falsely equate a local heat signal with local force work. The reference's measured-residence/background product is checked directly.

Noise validation uses 256 independent draws with the existing seed9161066. Every draw refits both completed models on the unchanged measurements, checks calibration/parameter gates, and checks the oracle's five groups. The source must fail all three port diagnostics while passing both anchor groups. Calibration noise is not propagated by changing physics or tolerances. Isolated public/private pytest runs retain the 60-second verifier budget and expect oracle9/9 and source6 pass/3 intended port failures. Fresh Docker controls and any frozen model trials are owned by the parent task and must be reported separately.

## Reproduction and preservation

Run `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_entropy_anomaly.py` from this candidate root with NumPy, SciPy and pytest installed. Existing calibration is read rather than regenerated, and must match the archived r6 bytes and its private copy. The distinct report is `results/entropy-anomaly-r7-validation.json`; controls are in `results/entropy-anomaly-r7-controls`. Self-contained proposal calculations and independent weighted finite-mass checks are preserved in `results/prototypes/entropy-anomaly-r7` and `results/entropy-anomaly-r7-spatial-port-proposal.json`.

The final report records noise extrema, timings, reference refinements, domain identities and task/script hashes. The public Docker image explicitly copies only the README, starter, calibration and public test. True gamma, hidden preparations, independent reference, oracle, optional hint and this derivation remain private. The public microscopic dynamics, detector law and measured-residence operation give all information needed to derive the correct prediction. Actual-source peer review and fresh scientific/Docker controls precede freezing; all three model trials and their exact failure classifications must be preserved even when a revision does not achieve the target.

## Recorded scientific controls

The revision-7 validator passed in 171.22 seconds; the cold independent hidden reference took 16.76 seconds. Oracle/reference discrepancy was at most 4.40e-7, halving mass changed it by at most 3.88e-7, and the velocity/spatial basis refinement changed it by at most 1.45e-10. All 512 domain corners passed, with minimum leading position density .07339 and minimum scaled conditional velocity covariance .36079. The scalar contact-population lag anchor agreed within 1.03e-12; weighted energy transport and measured-residence subtraction agreed within 2.89e-11. Bulk preservation across 82 preparations and three friction values was within 2.67e-15; complementary port sums agreed within 8.89e-16.

All 256 noise controls passed their intended classification: the oracle maximum diagnostic error was .004198, the source minimum diagnostic error was .06698, both controls' maximum anchor error was .003917, maximum relative friction error was .004361 and maximum calibration chi-square was 1.30776. Isolated verifier runs gave oracle9/9 in 18.33 seconds and source6 pass/3 physical port failures in 18.52 seconds, within the unchanged60-second limit. These results establish scientific separation and verifier feasibility; they do not claim any Luna outcome.
