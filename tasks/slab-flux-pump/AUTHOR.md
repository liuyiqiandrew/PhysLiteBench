# Slab flux pump

## Revision 1

A transport current remains on while an external magnetic field cycles. The same critical state must satisfy the induction specified at both slab faces. Adding an unbiased magnetic cycle to an independently prepared, static self-field fits the zero-bias calibration. It misses the redistribution of moving flux between the two faces and the resulting DC voltage.

## Critical state and voltage

With depth d = x + 1 from the left face, Ampère's law gives J_y = −∂B/∂d. Define F(B) = B + B|B|/2. The critical-state law becomes |∂F/∂d| ≤ j. Along a monotone ramp, each face supplies a cone of allowed values of F. Leave the old profile unchanged wherever it lies inside both cones. Elsewhere, clip to their intersection. This extends a critical front from each driven face and preserves the pinned interior. The face values always admit a profile: the largest allowed difference in F is 1.456, below the minimum slope budget 2j = 1.7.

For a field ramp with sign s, the branch entering from the left has F = F(h + i) − s j d. It terminates where it meets the old profile or the opposite moving branch. In a stationary core E = 0. If the two moving branches meet, the current reverses there. Continuity of E and the constraint on its direction again give E = 0 at the meeting point. Let L denote that depth. Integrating Faraday's law, ∂E_y/∂d = −∂B/∂t, from 0 to L gives the tap voltage. On this branch its time integral per unit h is

    (1 + |h + i|) [(h + i) − B(L)] / (s j).

Both the portions of a ramp with partial penetration and those with fronts that meet must be included. The oracle finds L by bisection against the profile at the previous reversal and integrates over the two legs with 4,096 midpoints per leg.

The completed shortcut solves the unbiased magnetic cycle and the transport ramp individually with the same field-dependent critical current. Its transport profile is antisymmetric and contributes zero to the spatial mean. The cyclic branch has zero net impulse. With transport present, the combined profile can have a different mean and a nonzero impulse.

## Scientific controls

All 12 calibration readings have zero bias and contain crest and trough measurements for six stated cycles. For a fully penetrated, positive high endpoint h with F(h) > j, the mean induction is

    [(1+h)³ − ((1+h)²−2j)^(3/2)] / (3j) − 1.

This is strictly decreasing in j through the calibrated interval, identifying the parameter. The validator compares this independent integral with both numerical solutions for a fully reversing cycle. Calibration uses deterministic eight-decimal rounding and no random noise.

The private reference advances B on cell centers, accumulates the change in cell flux on each time step and locates the zero of E directly from the inactive cells or the moving field extremum. It uses 16,384 cells for the endpoint readings and 4,800 cells and 1,280 steps per leg for impulses. Halving only the spatial or only the temporal resolution across all ten impulse cases changes the answer by at most 0.000061 or 0.000464, respectively, against an absolute tolerance of 0.008. For zero bias, reflection symmetry and periodicity give zero impulse. Reversing the bias at fixed applied field reverses the sign of the impulse: spatial reflection swaps the faces, and Faraday's law gives equal cycle integrals at the two faces of the periodic profile. A small biased cycle with an untouched core also gives zero. The validator checks all three cases.

| Control | Fitted j | Max calibration error | Crest error | Trough error | Impulse error |
|---|---:|---:|---:|---:|---:|
| Coupled critical state | 1.06300007 | 1.28 × 10⁻⁷ | 7.6 × 10⁻⁸ | 1.1 × 10⁻⁷ | 0.000351 |
| Superposed fields | 1.06300007 | 1.28 × 10⁻⁷ | 0.08941 | 0.10285 | 1.91806 |

Endpoint accuracy is graded to 0.003 in absolute induction units and impulse to 0.008. Both thresholds exceed the numerical discretization errors and resolve the change caused by the bias.

## Reproduction

The validator reads the checked-in records and recomputes the reference. Use `--generate` to rewrite the data deliberately.

```bash
python3 scripts/validate_slab_flux_pump.py
python3 scripts/run_science.py tasks/slab-flux-pump --agent oracle --trials 1
python3 scripts/run_science.py tasks/slab-flux-pump --agent oracle --trials 1 --solution-model scripts/slab_flux_pump_baseline.py
python3 scripts/run_science.py tasks/slab-flux-pump --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3 --restricted-network
```

## Revision 2

Steady rotation permits simultaneous flux cutting and depinning. The electric
field is normal to the local critical ellipse. The R2 control solves the
axial experiments correctly and assumes E is parallel to J during rotation.
The two directions coincide for a circular critical surface.

Write B = b exp(iθ) in the yz plane and measure distance u outwards from
the pinned front. For positive ω define K = (E/ω) exp(−iθ).
Ampère gives J = (b θ′ − i b′) exp(iθ), and Faraday gives (E/ω)′ = B,
with primes denoting u derivatives. Normality fixes
b dθ/db = −γ² Re K / Im K for cutting ratio γ. The critical ellipse then
fixes du/db. The oracle integrates these relations from the front to the
known surface amplitude and aligns the phase at the surface. At small b
the critical current tends to j, the distance tends to √3 b/j and
K tends to √3 b²/[j(2+i√2 γ)]. This follows by substituting a leading
power B proportional to u^(1+i√2 γ) into Maxwell's equations. Reversing
rotation conjugates B and reverses the sign of the conjugate E.

The independent solver integrates both Cartesian complex fields in physical
depth with an event at the specified surface amplitude. The validator checks
the front seed sensitivity and rotational symmetry. The accepted maximum
errors in B and E/|ω| are 0.001 and 0.0006 in the instrument units.
The distinction between current and electric field directions in critical
state laws is discussed by [Clem et al. (2011)](https://arxiv.org/abs/1103.1393).

## Revision 3

The laminate adds a static normal induction and a conducting sheet below the
surface. All collinear calibrations remain unchanged. The starter includes the
collinear solution and a dilute-field estimate of the rotating profile. The
completed shortcut solves the full bulk critical-state equations but evaluates
the Joule signal in the undisturbed fields of a homogeneous slab.

Use distance u increasing from the front towards the left surface. Write b
and e for the complex tangential magnetic and electric fields in (y,z), with
e in units of |ω|. In physical coordinates b′ = iJ and e′ = s b, with
s the sense of rotation. At the sheet e is continuous and the tangential
field jumps by b_coat − b_sub = iK. Hall current contributes to this jump,
and its scalar product with e vanishes. Thus the sheet power is G|e|².

Let n be the threading induction and c the cutting ratio. In local tangential
axes along b and perpendicular to b, the restricted critical ellipse has
squared axis ratio A = (n²+|b|²)/(n²+|b|²/c²). The normal to the full
ellipsoid has an x component even though Jₓ vanishes. In physical coordinates
Eₓ/|ω| = (c⁻²−1)n Re(b* e)/(n²+|b|²/c²), with the star denoting complex
conjugation. This field gives a DC potential difference during rotation.

The oracle integrates the bulk equations in log u with the regular variables
|b|/u and e exp(−i arg b)/u². It shoots for the substrate thickness
that reproduces the imposed tangential surface amplitude after crossing the
sheet and coating. In a threaded core the limiting in-plane ellipse is circular.
The leading solution has |b|/u = j / [√3(1+|n|)] and winding angle s√2 log u.
With no threading field the winding coefficient is s√2 c. The reference
integrates Cartesian fields in physical depth and eliminates Eₓ using the
inverse full three-dimensional ellipsoid. It quadratures the resulting Ex
separately along each side of the contact.

The validator checks both interface jump components, E continuity,
conjugation under joint reversal of n and s, Eₓ = 0 for c = 1, Joule power
positivity and front seed convergence. A separate real-vector Maxwell solver
was compared on three off-grid inputs at five depths, including both sides
of the sheet. The largest difference was 2.2 × 10⁻¹⁰. The calibrated
oracle's largest errors on the four laminate readouts are 1.1 × 10⁻⁸,
1.9 × 10⁻⁹, 3.5 × 10⁻¹⁰ and 2.1 × 10⁻¹⁰. The passive-sheet control misses
them by 0.04635, 0.00869, 0.00123 and 0.00102. Both controls match the
longitudinal calibration within 1.3 × 10⁻⁷. R3 uses 1,800 s for the solver
and 180 s for verification to allow independent numerical methods to finish.

## Revision 4

The film can carry a sheet current at zero local electric field. In that regime
it shields the substrate. The tangential induction on its coating side can be
nonzero, and the maximum such amplitude is κ. Propagate the coating from this
interface with E = 0. If its outer field at κ exceeds the imposed amplitude,
solve for the smaller interface induction. Otherwise the substrate is penetrated
and the magnetic jump also contains the critical film current κ e/|e|. The
sheet power is G|e|² + κ|e|. The two branches meet continuously at onset.

To find a bulk current, use axes 1 along the tangential induction of magnitude
b and 2 transverse to it. Write n for the normal field and D for total field
magnitude. In the Jₓ = 0 plane put t = n J₁ / D. Then t² + J₂² ≤ j_c² and
|t| ≤ c j_c |n|/b from the cutting planes. Maximize (E₁ D/n)t + E₂J₂.
The circle optimum may instead occur at a cutting bound. On the smooth side
Eₓ = −b E₁/n. At a corner write λ = E₂/J₂. Combining the two active normals
gives Eₓ = (n/b)(E₁ − λJ₁). A tip with E₂ = 0 at the interface is approached
from the moving material. The Cartesian reference solves for the two Lagrange
multipliers in the physical basis; the polar oracle clips the support point
in t and propagates the two coating boundary preparations independently.

The oracle and Cartesian reference agreed within 1.2 × 10⁻⁹ on 30 seeded
off-grid setups including 12 shielded films and probes within and beyond the
coating. The completed shortcut retains the normal-sheet, ellipsoidal-bulk law
of revision 3 for all preparations. Every collinear and normal-sheet control
therefore remains calibrated.

From virgin tangential field, the film can carry the required rotating current
until its magnitude reaches κ. Above that point the flux front enters the
substrate and the electric field at the sheet becomes finite. To check the
matching at onset, both shooting branches were compared on all 128 corners
and 64 seeded interior points of the seven independent parameter ranges.
The shielded branch was sampled at 12 interface inductions and the active
branch at 20 substrate thicknesses. Their minimum secant slopes were 0.537
and 0.110 and their limiting surface amplitudes agreed to 7.8 × 10⁻⁸.
The solver uses the common threshold state within 5 × 10⁻⁷ in drive amplitude,
well within the stated measurement accuracy. The supplied normal-sheet
helper and collinear ramp helper let solvers concentrate on the film.
