# Changing-path capture feasibility

The bounded study supports an adiabatic-capture r2 revision. This is the same separatrix-capture family as r1, with a changing asymmetry during the ramp. No task harness, model trial or Docker run was made. Empirical difficulty is unknown: two completed r1 solvers recovered the correct self-similar capture solution, and PfFcuLL explicitly named area growth rates.

The proposed domain is J0 in [.12,.16], final tilt in [.15,.25], path slope a in [0,.12], and final s in [1.8,2.2]. The prescribed tilt is delta(s)=delta_final+a*log(s/s_final). For different a, the final Hamiltonian can be identical. The initial Hamiltonians generally differ; each is fully specified and receives the same uniform-action/time-uniform-phase preparation law. This is not a claim of different outcomes from an identical initial Hamiltonian.

## Physical calculation and completed approximation

Let A_L(delta), A_R(delta) be the two separatrix areas for s=1. The physical lobe areas are S_i(s)=s^(3/2) A_i(delta(s)). For each prepared action J, both models find the actual crossing s_c from S_L(s_c)+S_R(s_c)=2*pi*J. Both assign the correct outgoing branch actions J_i=S_i(s_c)/(2*pi), invert the final branch areas, and integrate the finite initial action interval.

Only the probabilities differ. The physical model uses P_i=S_i'(s_c)/(S_L'+S_R')(s_c). The complete approximation instead uses P_i=S_i(s_c)/(S_L+S_R)(s_c), a locally self-similar area-allocation closure. It retains the full changing path, correct crossing time and branch actions. At a=0 it is exactly correct, including capture. A weaker endpoint-only area model was screened and retained in every result; it is not the proposed source.

The classical capture-rate rule is supported by the classical discussion and Eq. 1 on PDF pages 1–2 of [Stabel and Anglin](https://arxiv.org/pdf/2207.02317). [Neishtadt, Sec. 2.3–2.4](https://arxiv.org/pdf/1705.04347) gives capture probabilities at crossing and the finite-neighborhood/slow-limit interpretation. The quartic path, bounds and numerical checks below are our calculation. Only the stated sections were read; no full-paper review is claimed.

## Growing lobes on the whole proposed domain

The saddle c solves c−c³=delta. The complete tilt interval is [.15−.12*log(44), .25], contained in [−.304103,.25]. Since (−.35)−(−.35)³=−.307125 and .27−.27³=.250317, the saddle stays in [−.35,.27]. Thus 1−3*c²≥.6325: the saddle stays nondegenerate.

With h=sqrt(2*(1−c²)), direct differentiation of the lobe integral gives

    A_L' = 2*sqrt(2) [asin(2*c/h)+pi/2],
    A_R' = 2*sqrt(2) [asin(2*c/h)−pi/2],
    A_R'' = 4 / [(1−c²)*(1−3*c²)^(3/2)],
    S_i' = sqrt(s) [1.5*A_i+a*A_i'].

The left bracket is positive because A_L>0, A_L'>0 and a≥0. For the right bracket B=1.5*A_R+a*A_R', the outward rounded bounds on the whole c interval are A_R'≤−3.28 and 0<A_R''<9.1. Therefore partial_delta B≤1.5*(−3.28)+.12*9.1<−3.82, and partial_a B=A_R'<0. Its global minimum occurs at delta=.25,a=.12. The evaluated value is .1392709104; the analytic location of this bound is separate from the sampled minimum physical growth .1194622815. Both physical lobes grow even though their normalized fractions change.

For the representative a=.10 path, the left area fraction changes from .2167 initially to .8136 finally. Across the prepared crossing interval it is .5809–.6006, while the left growth fraction is .6898–.7085. There is no vanishing source, unused path control, or division by a small readout.

## Calibration and signal

A constant-tilt captured calibration at a=0, delta_final=.2 and s_final=1.8 exactly shares all r1 capture physics. Write alpha_i=A_i/sum(A), J=J0*xi. Its derivative with respect to J0 is the average of sum_i alpha_i²*xi*omega_i, which is strictly positive because each branch frequency omega_i is positive. This proves global injectivity across the proposed J0 interval. Seventeen noiseless inverse checks recover J0 to 1.38e−14. The sampled slope is 1.5516–1.5694, and the mean energy spans −1.0561 to −.9936. These are noiseless checks, not claims about arbitrary noisy fitting objectives.

The 72 static cases at a=.06,.08,.10,.12 give 6.561–17.595% relative source errors. Absolute physical energies are .8030–1.8231 and absolute source energies are at least .6809. All values use the Hamiltonian's fixed zero. There is no cancellation-based small-signal normalizer. The a→0 limit is preserved explicitly; it correctly removes the gap. A full package would still need fixed instrument noise, 256 calibration-noise checks and actual local controls.

## Independent dynamics and numerical scope

The direct reference constructs the initial action/time-uniform ensemble independently, then propagates Hamilton's equations under the declared controls. No capture probability, outgoing branch action or final-energy area inversion is used during propagation. The reference reuses the reviewed r1 initial-orbit solver and time-dependent fourth-order splitting structure; this is independent of the crossing calculation, not a claim of an unrelated preparation implementation.

The first 11 trajectory results and the five supplements are all preserved in trajectory-runs.jsonl. The first three representative runs jointly changed ramp duration and ensemble resolution; they must not be described as a pure duration check. The supplements isolate that question: at a fixed 32-action by 256-phase grid, relative energy errors for durations 256, 512 and 1024 are .3937%, .1587% and .1052%. At fixed duration 512, increasing the grid to 48 by 384 changes the energy by .0003311 (.0227% of the physical energy). A phase-offset change gives .0003156, and halving the integration step gives 7.41e−8. The energy-work residual decreases from 4.57e−6 to 2.84e−7 on the half-step check.

At the two refined edge cases, slowing the ramp from 512 to 1024 reduces relative errors from .2142% to .0662% and from .1923% to .1206%. Other bounded direct cases are within .233%; all were run with the actual changing tilt. Preparation energy and periodic-orbit closure checks are below 1.01e−11 and 8.45e−11. The reference is an ensemble convergence check, not a pointwise assertion about the basin selected by every individual trajectory.

The identical-final-Hamiltonian comparison at J0=.1437, delta_final=.2, s_final=2.2 gives physical means −1.609630894 (a=0) and −1.422497175 (a=.12). The direct duration-512 means are −1.608353558 and −1.419188558. The stronger source gives −1.179136808 for the latter, with a 17.108% error. Initial Hamiltonians differ as specified above.

The static domain check samples 105 paths with 65 control points each. Initial and final total separatrix actions in that sample are respectively at most .005479 and at least 1.06028, well outside the prepared band [.12,.192]. The 24→48 action-quadrature change is 4.45e−16 in three cases. Analytic area derivatives agree with centered finite differences to 1.34e−9. These finite checks are not a full future harness validation. The main script took 36.95 s, the supplemental script 18.75 s; individual direct runs were .48–8.09 s on this host.

## Preserved variants and limitations

screen-report.json retains all 108 initial variants: 102 evaluated and six rejected because the tilt left the double-well domain. It includes weak/zero-slope cases, negative slopes, the stronger a=.15 cases not selected for the safe box, and the weaker endpoint-only model. No outcome was discarded. The declared safe box was then checked separately.

This is a genuine path-dependent revision, not a new physics family. The source is a complete self-similar probability extension, with correct branch actions. A solver that recognizes the growth-rate law but implements a bad derivative or trajectory solver must later be classified separately from retaining the area-probability approximation. Known r1 successes make another physical pass plausible; no claim of benchmark hardness is justified yet.

All r1 task files, plans, trials and prototype files were read-only. The copied geometry and initial-ensemble modules are author snapshots, not agent-visible files. The proposed public outline contains the Hamiltonian, actual preparation, path and measured energy without the capture rule or a solver restriction. Packaging and model evaluation remain unapproved at this freeze.
