# Adiabatic capture, revision 2

Author-only. This is a changing-path revision of the archived separatrix-capture task, not a new family. The source already solves the actual crossing condition and preserves the correct outgoing branch actions. It retains a locally self-similar area allocation for probabilities when the lobe proportions vary. Calibration includes capture and validates the complete physical correction solved by the r1 agents.

## Apparatus and limits

The isolated classical Hamiltonian is H=p²/2+q⁴/4−s*q²/2+delta*s^(3/2)*q. It fixes the absolute energy zero. The cubic ramp takes s from .05 to s_final, with delta(s)=delta_final+a*log(s/s_final). Inputs are delta_final in [.15,.25], a in [0,.12], s_final in [1.8,2.2]; the unknown J0 is in [.12,.16]. Each initial Hamiltonian is fully determined by these controls. A fresh connected orbit is selected with action uniform on [J0,1.2*J0] and independent orbital time uniform over a period. There is no bath, friction, feedback or equilibration after preparation.

The measured total Hamiltonian energy is averaged over the specified ensemble at each finite ramp duration before taking the slow limit. Different paths can have the same final Hamiltonian, but generally have different initial Hamiltonians under the same stated action-preparation law. This distinction is explicit in the preserved feasibility assessment. No quantum or single-trajectory limiting claim is made.

## Physics and source

Write A_i(delta) for the two separatrix lobe areas at s=1. The physical areas are S_i(s)=s^(3/2)*A_i(delta(s)). For a prepared J, both controls find the actual crossing s_c from sum(S_i(s_c))=2*pi*J, assign outgoing actions J_i=S_i(s_c)/(2*pi), invert each final branch area, and average the finite action interval.

The complete source sets P_i=S_i(s_c)/sum(S). The oracle uses P_i=S_i'(s_c)/sum(S'). The former is exact when a=0; it is a definite probability-allocation approximation, not a missing trajectory solver, wrong branch action, unused control, or sign/index error. Both models account for the full path and preserve the actual crossing branch areas. A later solver that recognizes growth rates but miscarries their derivative or a trajectory calculation must be classified separately from retaining this physical approximation.

For saddle c−c³=delta and h=sqrt(2*(1−c²)),

    A_L/R' = 2*sqrt(2) [asin(2*c/h) ± pi/2],
    S_i' = sqrt(s) [1.5*A_i+a*A_i'].

The full path tilt lies in [.15−.12*log(44),.25]. This places c in [−.35,.27] and gives saddle curvature 1−3*c²≥.6325. Left-lobe growth is positive directly. For B=1.5*A_R+a*A_R', the derivative A_R''=4/[(1−c²)*(1−3*c²)^(3/2)] is below9.1, while A_R'≤−3.28. Hence partial_delta B<−3.82 and partial_a B<0. The global minimum bracket at delta=.25,a=.12 is .1392709104>0. Both lobes grow; there is no late recrossing due to lobe shrinkage. The prepared interval starts far above the initial total separatrix action and lies below its final value throughout the bounded domain.

The classical growing-area probability rule and its crossing-time qualification are in [Stabel and Anglin, PDF pages1–2, Eq.1](https://arxiv.org/pdf/2207.02317). [Neishtadt, Sec.2.2–2.4](https://arxiv.org/pdf/1705.04347) gives the positive-flux/nondegenerate-saddle assumptions and ensemble interpretation. Actual access scopes are preserved in prototype/source-access-and-history.json and the independent prototype peer; neither record claims a complete review of the quantum paper or all66 pages of Neishtadt. The log-path calculations and bounds are our derivation.

## Calibration and grading

Calibration uses256 independent readings at a=0, delta_final=.2 and s_final=1.8, with fixed energy-instrument sigma=.001. It is a captured preparation, not a noncrossing or zero-response example. At a=0 the area and area-growth fractions coincide exactly. With alpha_i=A_i/sum(A) and J=J0*xi, the response derivative is the average of sum(alpha_i²*xi*omega_i)>0. This proves global identification of J0; branch frequencies are positive. Both controls use the same bounded monotonic inversion for the actual one-setting dataset. A general bounded-fit fallback does not restrict the public API.

The generation command intentionally uses the exact classical slow-limit calibration energy plus independent Gaussian noise. Finite-duration trajectory errors are reported separately and are not injected as measurement bias, counted as noise, or claimed to be below the uncertainty of the sample mean. The unknown .1437, random seeds and private controls stay outside the public image. Neither the constructor nor the public data sigma supplies the true parameter. Ordinary validation reads checked-in calibration; regeneration requires --generate.

There are three changing-path groups with two ordinary negative-energy values each, plus a two-setting constant-path anchor. The fixed Hamiltonian zero and sizeable total energies prevent cancellation-based normalization. Zero and small slopes remain in the public domain and correctly give equal or approaching predictions; no gap is claimed over every allowed input. Gates remain normalized RMS .04, parameter relative error .03, and reduced calibration chi-square1.5. The exact neutral instruction allows replacing any implementation helper. Agent600/verifier60/build600 limits and explicit public Docker copy are unchanged.

## Independent reference and scientific checks

The reference computes initial connected-orbit energies and periods, samples orbital time, and propagates the declared time-dependent Hamilton equations. It reads final q,p directly. Capture probabilities, outgoing branch-action allocation and final-area inversion do not enter propagation. The reviewed r1 initial-orbit solver and fourth-order splitting structure are retained, so independence is from the capture prediction, not a claim of unrelated initial-geometry code. The work integral includes the actual path derivative ds/dt*[-q²/2+(1.5*delta+a)*sqrt(s)*q].

Every scored input has a default duration512,32-action,256-phase reference, a doubled-duration check at the same grid, and a denser48-action,384-phase check at the same duration. These separate duration from ensemble effects. Sampling errors need not decrease monotonically for each finite grid. Additional edge preparations, a half-step run and a shifted phase grid check work balance and finite-ensemble sensitivity. The full-domain analytic/growth/quadrature checks include interior slopes and parameters. Exact calibration inversion is tested across the entire J0 interval, and256 independent calibration-noise draws check both completed controls against the fixed direct reference.

The finite reference is an approximation to the public slow limit. Its errors and per-case changes must remain comfortably below .04; they are not asserted to vanish or to be sigma-level calibration agreement. The local verifier tests run in fresh temporary app/test directories and enforce the actual60-second limit. These author checks are not Docker or model evaluations.

## Preserved history and reproduction

R1 is archived after3/3 physical passes. One of those scored passes also records an agent600-second timeout; its saved correct model passed all checks, and both facts remain preserved. Six earlier unstarted setup failures have null rewards and are distinct from those three model trials. No r1 source, plan, native artifact or outcome was changed for r2. See history/r1-and-prototype.json for exact archive hashes.

The feasibility copy preserves every original byte:108 screen variants, including six lost-double-well rejections and weak/zero-slope cases,16 direct runs, original author manifest and independent peer. The excluded a=.15 and negative-slope variants remain recorded. Its first representative refinements jointly changed duration/grid; the later supplements separated those effects, as the original assessment states. A weaker final-area-only approximation was screened but is not the completed task source.

The actual final source hashes, per-case comparisons, noise outcomes and local controls are in the staged results. No empirical difficulty is claimed. Reproduce science with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python -B scripts/validate_adiabatic_capture.py` from this stage. Add --generate only for intentional calibration regeneration. Any author failure is written separately with a timestamp; validation-runs.jsonl preserves each direct calculation. The independent final source/input/grading peer is attached before the top-level freeze.

## Completed evidence

The first full package scientific execution passed in302.01 seconds; no failed author attempt or repair was needed. All256 noise draws pass calibration/parameter gates and every oracle group; the maximum oracle group error is.002073, and the minimum source diagnostic error is.11496. Constant-path source anchors pass. The checked-in fit is J0=.1437217347, with relative error.0001513 and reduced chi-square1.06259.

All41 noiseless parameter recoveries across [.12,.16] are within1.38e−14. Source/oracle calibration differs by at most4.45e−16. The smallest diagnostic group gap over those strengths is.11518. The132 corner/interior static cases have maximum24→48 action-quadrature change1.27e−13 and area-derivative discrepancy1.34e−9. The source and physical model coincide at a=0.

Every one of the eight scored inputs has its own duration and ensemble refinements. Maximum relative direct errors are.002326 at default settings,.001054 at doubled duration, and.002446 at denser ensemble. Maximum duration and grid changes are.001273 and.000730. Six additional parameter/control edge preparations stay within.001380. At the selected changing-path case, a half-step changes the mean by7.10e−9 and a phase-grid shift by.000158. All34 direct results remain in validation-runs.jsonl.

The captured calibration's finite-duration biases are −.0004139 at duration512 and +.0002982 at1024, versus the exact slow-limit value. These are not negligible relative to the uncertainty of256 averaged readings; this is why exact slow-limit calibration generation is explicitly separated from trajectory checking. They are not treated as measurement noise or silently used to shift J0.

Fresh isolated local controls give oracle7/7 in20.35 seconds and completed shortcut4 passes plus exactly the three intended failures in20.13 seconds. Both are below the unchanged60-second verifier limit. This certifies the local harness on this host, not Docker startup reliability or model difficulty. The final private input audit verifies exact neutral instruction, public/private data equality, fixed independent sigma, None constructor, explicit image scope and absence of the true J0 from all public payloads.
