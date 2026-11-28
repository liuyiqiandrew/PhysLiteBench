# Correlated gate charges in an avalanche photodiode

A short optical pulse creates one electron–hole pair at a known depth in a planar multiplication layer. The two contacts resolve groups of electrons and holes collected simultaneously. A second amplifier measures the charge induced while carriers are moving. Measure the joint spectrum of two charge integrators and the conditional current on the same avalanches.

## Carrier dynamics

The layer is 0≤x≤1. At time zero the primary pair is at `depth`. The electron travels right at `velocities[0]`=vₑ and the hole left at speed `velocities[1]`=vₕ. The drift speeds are constant in the operating range. A carrier disappears on reaching its collecting contact. Diffusion and trapping are negligible. Distinct charge carriers at the same point keep separate impact clocks.

An impact creates an electron and a hole at the incident carrier's position. The incident carrier continues in the same direction. Its energy and both new carriers' energies are reset, so each of these three must cover its own full `dead_space[c]` before it can impact again. Other carriers keep their energies. The primary pair also starts cold. Carrier type c=0 is an electron and c=1 is a hole.

The amplifier weighting field is uniform. In units of elementary charge its running induced charge Q starts at zero and obeys dQ/dt=vₑ Nₑ(t)+vₕ Nₕ(t), where N counts carriers still in flight. Collection has no extra jump in Q. The bias capacitor discharges by this same total charge during a shot. In the calibrated voltage range the log ionization coefficient drops linearly with Q. After a carrier has crossed its dead space, its instantaneous impact hazard per unit **distance** is

    a × ionization[c] × exp(−feedback × Q(t)).

The factor a is the same for both species; all other quantities are supplied. Conditional on the shared instantaneous field, the active carriers have independent impact hazards. Ineligible carriers can still induce current. The feedback affects impact rates while drift speeds and calibrated dead distances remain fixed. Recovery between shots restores Q=0.

The calibration isolates a single fully energized electron in a voltage-clamped layer with unit nominal ionization coefficient. Its `active_length` is the available distance after the initial dead flight. The recorded value is the probability of no impact, rounded to eight decimals. The unknown a is in [0.8,1.1].

## Terminal clocks and readout

`gain` G is the total number of electrons eventually collected and also the total number of holes. Each impact adds one pair. The prompt contact times are Tₑ⁰=(1−depth)/vₑ and Tₕ⁰=depth/vₕ. In a shot, Tₑ and Tₕ are the last electron and last hole arrival at their respective electrodes. Mₑ counts electrons arriving exactly at Tₑ, and Mₕ counts holes arriving exactly at Tₕ. Co-moving equal-sign carriers reach the same electrode simultaneously. The two timing electrodes register both clocks on the same avalanche.

`last_delays=[dₑ,dₕ]` selects the observed terminal times. A zero component selects the probability atom at the corresponding prompt time. For each positive component c, select T_c in [T_c⁰+d_c,T_c⁰+d_c+ε), then take the limit as ε decreases to zero. For two positive components this selects joint density per unit electron and hole time. For exactly one positive component it selects an atom on one electrode and a density on the other. Always impose the stated gain and both clock selections.

Both clocks select a single avalanche. Exactly one clock is prompt in these measurements. The selected opposite clock is a regular delayed density. All terminal packet heights are accepted (`multiplicities=[null,null]`), and `tag=null`.

## Joint charge spectrum

Two ideal integrators read different gates of the same displacement current. For `gates=[[u₁,v₁],[u₂,v₂]]`, their output is the charge vector F=(Q(v₁)−Q(u₁), Q(v₂)−Q(u₂)). The gate windows may overlap. A third integrator has `read=[A,B]` and reports Y=[Q(B)−Q(A)]/G on this same shot.

At `charges=[z₁,z₂]`, consider a circular acceptance aperture in the measured charge plane. Write P(r)=Pr(‖F−z‖₂<r) within the base ensemble selected by gain and terminal times. Take the terminal time limit before the charge aperture limit. With `observable="intensity"`, return the following according to `part`.

- `"point"` returns lim P(r) as r→0, the probability of the specified pair of charges occurring exactly.
- `"ridge"` returns lim P(r)/(2r), a probability density per unit Euclidean arclength along a regular one-dimensional spectral line through z. The quoted charge is away from point masses, ridge crossings and endpoints.
- `"area"` returns lim P(r)/(πr²), the ordinary joint density per unit area at a regular interior point, away from point masses and ridges.

With `observable="waveform"`, return lim E[Y | ‖F−z‖₂<r], imposing the same gain and terminal clocks. Every queried component has positive weight. A ridge can be horizontal, vertical or oblique. Its arclength is measured in the plane with the same induced-charge units on both axes. Charges given to 12 decimal places identify the indicated point or ridge.

For graded shots depth is in [0.3,0.65], electron speed in [0.8,1.3], hole speed in [0.75,1.2], ionization coefficients in [1.5,3], dead distances in [0.02,0.06] and feedback in [0.35,1.1]. Gain is 3 or 4. All integrator gates end before either primary carrier is collected. Positive delays are at most 1.4. The check below uses zero dead space and clamped bias.

## Worked shots

Use depth 0.4, both speeds 1, both ionization coefficients 2, zero dead space and zero feedback, with delays [0,0.6] and read gate [0,0.25]. The prompt electron clock requires that all impacts were made by electrons. The hole clock fixes the last impact at time 0.30. At fixed gain each eligible ordered history has constant impact weight because the total integrated hazard is 2aG. All readouts below use the same time unit and induced-charge unit as the inputs.

At gain 3 the first impact time t is uniform on (0,0.30). Set the charge gates to [0,0.15] and [0,0.23]. The two readings are 0.30+2 max(0,0.15−t) and 0.46+2 max(0,0.23−t).

| Component | Target charges | Intensity | Conditional waveform |
| --- | --- | --- | --- |
| Point | (0.30,0.46) | 7/30 | 59/350 |
| Vertical ridge | (0.30,0.54) | 5/3 | 31/150 |
| Diagonal ridge | (0.45,0.77) | 5/(3√2) | 17/60 |

At gain 4 the two free times are uniform over 0<t₁<t₂<0.30. Change the charge gates to [0,0.10] and [0,0.20]. The regular area target (0.31,0.83) selects t₁=0.045 and t₂=0.14. Its intensity is 50/9 and its normalized read is 113/400.

## Interface and scoring

Implement `Model` in `/app/model.py`. `fit(records)` receives dictionaries with `input` and `value`, sets `activity` and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty list.

The starter calculates each individual gate spectrum exactly, then models the two integrators and the waveform read as triggered by independent shots with the same gain and terminal clocks. `single_gate.spectral_law` supplies the ideal single-channel line or density and its gated read moment. `transport.packet_law` supplies the base gain and terminal clock weight and read moment, including the bias feedback and dead flights. Both return arrays indexed by the two terminal heights.

The ten calibration readings require maximum absolute error below 0.00002. The public examples and four held-out groups use absolute error below 0.004 for point mass and waveform, below 0.015 for ridge density and below 0.03 for area density. Each held-out group contains six readings at operating settings distinct from the worked shots.
