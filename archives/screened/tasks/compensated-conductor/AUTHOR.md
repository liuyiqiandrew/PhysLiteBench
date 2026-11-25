# Compensated-conductor candidate, revision 1

This is a separate task from `hall-bar`; it preserves that task's revision and
evaluation history. Target: three GPT-5.6 Luna high-effort trials without a hint
and three with `hint.md` appended, using the same agent configuration.

Two carrier populations have charges +/-e and equal unknown mobility mu. Known
densities are independently prepared. The README defines a classical bulk
semiconductor limit with charge-conserving pair generation/recombination and
immobile doping. The measured quantity is local central-bulk jx, excluding thin
edge layers. These assumptions matter: separately conserved particles in a
sealed finite bar could not sustain the transverse particle fluxes of the
two-carrier Drude bulk. Pair generation/recombination accommodates that flux
near the edges without allowing electrical current through an insulating wall.
The role of these edge processes is discussed in
[Alekseev et al., Physical Review B 95, 165410](https://arxiv.org/abs/1612.02439).

Let S=n_positive+n_negative and D=n_positive-n_negative. Summing the two Drude
tensors gives s=e*mu*S/[1+(mu*B)^2] and
h=e*mu^2*B*D/[1+(mu*B)^2], with conductivity matrix [[s,h],[-h,s]]. The total
transverse current must vanish, so Ey=(h/s)*Ex and jx=(s+h^2/s)*Ex. The shortcut
adds only the two longitudinal diagonal conductivities and uses jx=s*Ex.

Calibration has equal carrier densities, so h=0 even at nonzero B. It spans
three equal-density preparations, four signed electric fields and nine magnetic
fields including zero. The zero-field records remove the reciprocal-mobility
ambiguity of a single nonzero-field conductance. A dense full-bound parameter
scan checks that the exact calibration loss has one minimum.

Hidden cases use positive-majority, negative-majority and partially compensated
preparations at both field signs. Their common transverse field is not zero.
The oracle eliminates Ey algebraically; the independent reference solves five
linear equations for the four carrier drift velocities and Ey. Those equations
are two separate vector Lorentz-force balances and one total-current constraint.

The private verifier requires reduced calibration chi-squared below 1.5 (216
measurements minus one fitted mobility), mobility error below 3%, and each hidden
case's current RMSE divided by reference RMS below 0.04. Measurement sigma is
0.006 times the largest noiseless calibration current. Calibration seed: 9312;
noise-validation seed: 19312. The independent validator checks both controls on
all hidden cases for each of 256 noisy calibrations. It also checks zero-field,
zero-drive, compensation and single-carrier limits, field and charge-exchange
symmetry, zero total transverse current, and Joule power equal to the sum of
the two carriers' frictional dissipation.

All 256 calibration and parameter checks pass. The largest oracle hidden error
is 0.414%, while the smallest shortcut error is 27.2%. Thus the 4% prediction
tolerance exceeds fitting noise while preserving a clear physical separation.

Run from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_compensated_conductor.py
python3 scripts/run_science.py tasks/compensated-conductor --agent oracle --trials 1
python3 scripts/run_science.py tasks/compensated-conductor --agent oracle --trials 1 --solution-model scripts/compensated_conductor_baseline.py
```

The validator reads checked-in calibration unless `--generate` is explicit.
Its report is `jobs/compensated-conductor-validation/summary.json`. Scientific
controls and measured agent difficulty are separate claims; Harbor batches are
managed by the parent task.
