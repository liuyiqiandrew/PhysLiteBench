# Slab flux pump, revision 1

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
