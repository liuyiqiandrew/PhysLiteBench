# Deforming crystal current, revision 2

This is a same-family revision of the archived r1, which passed 1/3 unhinted
trials and had two reviewed physical failures. That result is preserved. The
r1 passer recovered the proper moving-electrode charge rather than using a
polarization-density derivative. The new source includes that correction,
including the exact load divider and its geometry derivative. Its remaining
approximation is one-way electrical coupling: it relaxes the material at zero
macroscopic field before calculating the connected circuit's charge. The
specified finite load instead creates a field that changes that equilibrium.
This is a weak-electric-backaction approximation, exact at ideal short circuit
or vanishing electrical coupling. No r2 model evaluation has occurred.

## Energy, circuit and units

Use J=det(F), b=|F^(-T)e3|, n=F^(-T)e3/b and c=J*b^2. The face area is
S*a0^2*J*b and normal gap is N*a0/b. Thus Cb=C0*c with
C0=epsilon_b*a0*S/N. Write q=Q_upper/(q0*S), cL=C_L/C0 and
g=q0^2/(epsilon_b*a0*E0)=1. Each complete metallic node is neutral, so its
crystal and load faces carry opposite free charge. The intensive energy is

    U/(N*S*E0) = W0/E0 + g/2*((q-s3)^2/c + q^2/cL).

W0 already contains microscopic zero-macroscopic-field electrostatics. The
additional terms are the macroscopic background field and external capacitor
energy. This avoids double counting. The prescribed mechanical drive holds F;
the mechanically separate load has constant dimensional C_L during that drive.
Its normalized cL is held fixed in the preceding macroscopic size limit.
Keeping dimensional C_L fixed while increasing N at fixed lateral area would
instead short out the feedback. Six preserved dimensional checks and the
explicit counterexample distinguish these limits.

Charge minimization gives q=p*s3, p=cL/(c+cL), and adds
r*s3^2/2 to W0, with r=g/(c+cL). The physical internal equation is

    K*x + 4*|x|^2*x - h + r*s3*e3 = 0,  x=s-S0.

If H0=(K+4|x|^2)I+8xx^T, its drive tangent obeys
(H0+r*e3e3^T)*ds=dh-r_prime*s3*e3. Here
r_prime=-g*c_prime/(c+cL)^2. The measured response is
p*ds3+p_prime*s3, p_prime=-cL*c_prime/(c+cL)^2. Positive current enters the upper
crystal electrode; the load's upper face has the opposite charge. The source
uses the same charge relation and derivative, but its s and ds solve W0 alone.
No input is unused, and the difference is not a missing current-conversion
factor or numerical root.

Convexity holds over the continuous public box: H0 is at least K*I, and the
added electrical quadratic is positive. The joint state and charge equilibrium
is unique. Upper-triangular F with positive diagonal gives c>0. The analytic
component bounds from the preserved assessment are |x1|<=.0672,
|x2|<=.0549 and -.00125<=s3<=.48265. For the last bound write
s3=(D*.22+h3)/(D+r), with D=K+4|x|^2>=.8 and -.177<=h3<=.21012.
The selected negative center therefore remains inside its material cell.
Numerical corner/interior checks supplement these bounds; they do not replace
a continuous-domain proof of uniqueness or constitute an interval error proof.

## Calibration and independent reference

Both calibration files are unchanged bytes from r1 (SHA256
3b4d4c7c5bddcec37b92b34545ae02ff60a5cd7b76fb0caf8d9c472a7b1cd8cc).
Omitted load denotes ideal short circuit. There are 18 settings, nine
longitudinal stretches and both drive signs, repeated 16 times for 288 readings.
Sigma=.001 is a fixed instrument uncertainty. Private true K=1.06 and original
calibration seed 428711 remain outside the image. No data were regenerated.

For positive longitudinal calibration, x is axial and
K*x3+4*x3^3=.35*(z^2-1). The response .7*z/(K+12*x3^2) strictly decreases in K:
its derivative is -.7*z*(K-12*x3^2)/(K+12*x3^2)^3. Since |x3|<=.1113,
12*x3^2<.149<K over [.8,1.2]. This proves global identification; signed drives
share the same squared residuals. Both models agree exactly at every K.
Forty-one noiseless fits and 201-point loss profiles confirm the numerical
fitting implementation. This does not claim that every arbitrary noisy
objective is unimodal.

The verifier separately solves seven field/node unknowns: the three components
of s, lab-normal electric field e, top-minus-bottom voltage V, crystal charge qc
and load charge qL. Field units are q0/(epsilon_b*a0^2), voltage units
q0*N/(epsilon_b*a0), and charges use q0*S. Its equations are

    internal_force + g*e*F^T*n = 0,
    qc - s3 + J*b*e = 0,  V + e/b = 0,
    qL - cL*V = 0,  qc + qL = 0.

For ideal short, V=0 replaces the load law. The compensating reference variable
qL then represents transferred charge, not a finite capacitor face. Response
is a fourth-order finite difference of solved qc. The reference uses neither
the reduced divider nor the oracle's tangent/Hessian. It shares the specified
constitutive energy and exact planar macroscopic setting. It is not a
microscopic lattice-field simulation or an independent derivation of W0.

## Saved science and local controls

The packaged validation recomputes all 1,632 bounded domain cases: 64 F corners,
both K and finite-load endpoints, six rate basis directions, plus 96 random
interior controls. The maximum independent response error is 5.73e-12; 288
cases also halve the finite-difference step. All 34 distinct graded settings
(18 calibration, 12 loaded and 4 short anchors) are checked at steps 2e-4,
1e-4 and 5e-5. Their maximum oracle/reference error is 2.93e-13 and last-step
change 3.10e-12. Calibration numerical bias is 2.93e-10 sigma. State, field,
node-neutrality, drive-sign, linearity, frame and rigid-rotation checks pass.
The frame/rotation checks are author-only outside the upper-triangular schema.

Three finite-load groups at cL=.5,1,2 retain the predefined ordinary positive
mixed/longitudinal preparations. Across 41 K values their minimum normalized
source error is .303815; minimum physical/source signals are .127161/.210492.
The 1,632 general domain cases include small responses and sign changes. They
remain saved without filtering: no uniform large-gap claim is made for arbitrary
G. Short anchors, zero drives and weak responses are allowed. The preserved
prototype also records g->0 and cL->infinity limits and all out-of-domain probes.

There are 512 fresh noise samples per completed control: 256 at nominal K and
one at each of 256 endpoint-inclusive K values. Seed 613931 is independent of
the original data and bounded prototype. Both controls pass every calibration
and parameter gate. Every oracle passes every group; every shortcut passes the
short anchors and fails each finite-load group. Maximum reduced chi-square is
1.24550, parameter relative error .00031294 and oracle prediction error
.00029735. Minimum source diagnostic error is .303815 at the standard .04 gate.
This finite sample is not an all-noise guarantee.

On the unchanged calibration artifact both fits return K=1.06010455335 with
chi-square .823904. Oracle errors are below 9.38e-5; source finite-load errors
are .75147/.53806/.33905. Local isolated pytest gives oracle 7/7 in .603 s and
shortcut four passes plus three intended failures in .626 s. The entire
packaged scientific run took 28.07 s. The verifier budget remains 60 s and the
agent budget 600 s. No Docker or model run was used for these checks.

## History, scope and reproducibility

The bounded prototype, original author manifest, original failed execution,
checkpoint snapshot, float-only serialization repair and successful numerical
report are preserved exactly in prototype/. The first execution saved no
numerical results because JSON encoding failed on a NumPy integer; no success
is assigned retroactively. The successful second bounded run and first full
package run have separate source/report snapshots. Weak and sign-changing
cases are retained in both reports. The historical repair-start manifest's
status path was superseded, but its exact referenced status bytes remain in
the original checkpoint. The current prototype wrapper and original author
manifest each verify against their intended payloads.

The r1 passer solved electrode charge transport, so increasing deformation
coefficients or adding more charge components was rejected as hardening. This
revision changes the electrical preparation and reciprocal internal-state
feedback while retaining that solved readout. Piezoelectric-waves overlaps in
electrical constraints altering mechanics, but treats bulk wave/Gauss
constraints. Terminal-current-noise treats stochastic lead weighting with
already-correct kinetics. Neither is a duplicate finite-load crystal solve.
The preceding actual-trajectory assessment and its 32 evidence hashes are
preserved separately in results/family-history.json.

The r1 Vanderbilt proper-piezoelectricity paper is background for transported
electrode current, not a source for the new load construction. No new literature
access is claimed here. The finite-load model follows directly from the stated
cell charges, constitutive energy, Maxwell boundary law and node constraints.
The prototype and final package receive independent actual-file review.

The public image contains only README/model/public test/calibration data. The
true parameter, hidden groups, oracle, reference and author material are
private. Instruction is the exact neutral unrestricted text; there is no
required method or edit restriction. A failure preserving zero-field internal
relaxation under load is a physical closure error. Recognizing feedback but
miscomputing a sign, geometry derivative, field units or nonlinear solve is a
mathematical/implementation failure. Read the full emitted trajectory before
assigning a cause. Numerical margins do not establish empirical difficulty.

Reproduce with the pinned NumPy 2.3.3, SciPy 1.16.3 and pytest 8.4.2 environment:

```sh
python -B staging/deforming-crystal-current-r2/scripts/validate_deforming_crystal_current.py
```

This preserves a timestamped validator/report for each author execution and
never regenerates calibration data or launches models.
