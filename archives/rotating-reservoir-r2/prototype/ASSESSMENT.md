# Rotating reservoir: evidence and bounded hardening proposal

Status: bounded scientific proposal, not a task package. No new model or Docker runs. Canonical rotating-reservoir and all other tasks are unchanged. The fresh r1 outcome remains one physical pass and two physical failures; the earlier initial physical failure remains separate. The user now prioritizes hardening existing families.

## What the actual trials establish

`trial-review.json` records all three fresh trials, emitted public actions/messages, final sources, verifier metrics, native input blocks and 86 evidence hashes. All 16 current task files match the frozen batch. Native sessions use gpt-5.6-luna, high effort and CLI0.154.0; the three user blocks exactly match the approved plugin inventory, Oct4 environment and unrestricted neutral instruction. No private-task access or extra physics hint was observed.

Jyjcowq and zgWx9JS accurately fit drag0.6698903812 and leave the exact covariance and laboratory-power readout unchanged. Their public steps7 and8 respectively explicitly endorse the source's heat interpretation. Each fails all rotating prediction groups. Neither attempts a correct heat model and then regresses through arithmetic or implementation trouble.

fu84PgC initially endorses the source too, then public step8 identifies the moving reservoir's relative velocity. It replaces the heat function by `b*(trace(L C L.T)-2*T_b)`, with `L=(-Omega R,I)`, and keeps the full covariance mechanics. Its worst prediction error is8.08e-5. This is a real physical success, not a fitting accident. Almost identical fitted drag across the three submissions isolates the readout choice.

Consequently, changing rotation sign, drag anisotropy or trap axes is not a strong scientific revision. The existing six-dimensional Maxwell-memory prototype also still obeys the same stationary relative-work identity in rigid rotation. Its large numerical separation from laboratory power does not demonstrate a new reasoning demand. It is preserved and rejected here as insufficient hardening by itself.

## Proposed r2: a finite compliant contact in an affine moving bath

Retain the inertial trapped particle, stationary bath A, separate moving-bath thermostat, common unknown drag gamma and stationary mean calorimetry. Add one explicitly physical overdamped coupler at position q, attached to the main particle r by a known isotropic spring h=b/tau. Bath B acts on q, not directly on r. Its externally maintained incompressible flow is u(x)=A x, with A=Omega R+diag(s,-s). The full laboratory equations are

```
dr = v dt
 dv = [-K r - gamma v - h(r-q)]dt + sqrt(2 gamma T_a)dW_a
 dq = [A q + h(r-q)/b]dt + sqrt(2 T_b/b)dW_b .
```

The two noises are independent. The main mass is1; b=.9; tau, K, flow and temperatures are known. There is no extra pressure, hydrodynamic interaction or unspecified force. The q equation is the stipulated overdamped force balance at its own physical bath contact. The measured quantity remains the heat removed by bath B's thermostat, positive into B, after joint stationarity. Mean heat has no finite-window kinetic endpoint issue. The flow-maintaining actuator is a separate mechanical energy source.

The supplied source would solve this full six-dimensional Gaussian process exactly, including all affine stretching, noise and r–q correlations. It would already contain the successful r1 moving-frame correction, reporting the exact relative particle-to-contact work `-<z dot (v-A r)>`, where z=-h(r-q). This is a well-defined mechanical energy flux, commonly obtained by assigning the compliant bath's exchange to the particle point. Its error is treating that flux as the outer thermostat's heat. It is not an omitted deformation term in the state generator.

A pure rotation moves both spring ends without directly changing spring length. A symmetric velocity gradient stretches their finite separation and supplies elastic energy. Thus the old point-contact frame correction is necessary but no longer sufficient. This is a genuine apparatus change: the contact is spatially extended, the moving bath acts at a specified different coordinate, and its flow can deform the connector.

## Independent energy derivation and heat convention

The force on q from the spring is `F_q=h(r-q)=-z`. For the stated Langevin bath the Stratonovich thermal increment into B is `F_q dot (dq-Aq dt)`. Its mean is

`Q_B=<|z|²>/b - 2*T_b/tau`.

Let `U_s=h|r-q|²/2=tau|z|²/(2b)`. Transforming the actual bead equations gives

`dz=[A z-z/tau-(b/tau)(v-A r)]dt + sqrt(2bT_b)/tau dW`,

up to an immaterial reversal of the independent Wiener process. Itô energy balance gives

`d<U_s>/dt=(tau/b)<z dot A z> - <|z|²>/b - <z dot (v-A r)> + 2T_b/tau`.

At stationarity the physical heat equals the source plus `(tau/b)<z dot E z>`, where E is the symmetric part of A. The antisymmetric part contributes identically zero. This is connector work from the flow gradient, not a stationary endpoint-energy correction. The actuator power is `<z dot A q>`. With `Q_A=gamma*(<|v|²>-2T_a)`, the total first law is `Q_A+Q_B=<z dot A q>`.

The source already uses A, including its strain, in both its exact state and its particle-relative work. There is no unused control. The physical correction follows from locating the bath and applying energy balance, not increasing numerical resolution.

## Calibration and bounded numerical evidence

Calibration can include finite tau, anisotropic traps, both nonzero rotation signs and thermal biases, all with s=0. Then the models coincide for every gamma. The prototype checks116 such cases across29 gamma values. Maximum disagreement is1.10e-14. Include stationary-flow calibration as well: each Cartesian mode has conductance

`b/[k*tau²+(gamma+b)*tau+1+b/gamma]`.

For calibration tau<=.6, its derivative is strictly positive on gamma[.4,1.1], because `b/gamma²>.6`. A nonzero known temperature contrast globally identifies gamma. Other rotating records can therefore enrich calibration without carrying the entire identifiability claim.

The main prototype retains all45 screened settings, including zero strain and weak cases. Its six diagnostic cases have positive physical heat. An independent reference builds the actual (r,v,q) process and integrates propagated noise impulses over time. It agrees with the (r,v,z) Lyapunov oracle within1.07e-14 in heat and1.29e-14 in covariance. Heat and total-energy identities hold within3.95e-14.

All256 corners of the provisional domain gamma[.4,1.1], Omega±.6, strain±.4, tau[.2,.8], kx[1,1.6], ky[2.2,2.8], Ta,Tb[.8,1.4] are stable. Largest sampled drift real part is−.32535; smallest covariance eigenvalue is.23569. This is a corner screen, not a proof of stability throughout the continuum. The initial screen additionally includes tau1, and one diagnostic has tau.9; these exploratory points are retained and not claimed to lie in the proposed package domain. No strained case was removed for instability.

`limits-report.json` adds three ordinary diagnostic settings over41 gamma values:123 cases. Physical heats remain at least.25407, the source's relative-work signals at least.09845, and relative discrepancies range41.8–61.3%. Equilibrium covariance and heat checks are accurate to1.16e-14. As tau decreases to zero, both outputs approach the correct instantaneous-bath result and the finite-compliance discrepancy vanishes. These are deterministic author checks, not model evaluations. No noisy calibration, full-domain proof or final graded cases exist yet.

## Actual family overlap and limitations

The existing rotating-reservoir r1 and its unlaunched Maxwell-memory prototype are the direct predecessors. This proposal must be called a family revision, not a new independent family.

Archived dumbbell-stress passed3/3 plain and3/3 hinted. Its source omits strain transport of the covariance; this proposed source includes that transport exactly. The shared affine spring physics is substantial and is acknowledged. Archived thermal-bodies r11 passed3/3 after agents restored reservoir correlations; those are also already retained here. Thermal-bodies r12 passed3/3 and concerns finite-window heat variance versus spring work plus stored energy. The present observable is a stationary mean with persistent affine work on the connector; the discrepancy does not disappear at long times. FENE stress concerns nonlinear force preaveraging, which is absent here.

This is a meaningful candidate for hardening, but the fully explicit bath location may still lead a solver directly to the correct q-contact heat. The previous successes in spring-bath tasks limit any claim that it will be harder. Numerical margins alone do not establish difficulty. If a trial derives the correct q heat and fails a trace/sign or solver implementation, that must not be classified as the intended physical failure.

Primary background read for this review: Speck, Mehl and Seifert, *Role of External Flow and Frame Invariance in Stochastic Thermodynamics*, [arXiv0712.0232](https://arxiv.org/pdf/0712.0232), full five-page author manuscript, especially equations2–3 and12. It distinguishes flow work and relative bath heat. It is not a derivation of this mixed inertial/overdamped apparatus; that construction and its balance are derived above. No other new primary source was claimed read.
