# Last-packet timing in an avalanche photodiode

A short optical pulse creates one electron–hole pair at a known depth in a planar multiplication layer. The two contacts resolve groups of electrons and holes collected simultaneously. A second amplifier measures the charge induced while carriers are moving. Predict the mean gated amplifier reading and the coincidence fraction of specified terminal packet heights at fixed gain and recorded contact times.

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

`tag=[u,v,z,s]` accepts a given history with probability exp[−(Q(v)−Q(u)−z)²/(2s²)]. A null tag accepts all histories meeting the clock and gain selections. The trigger leaves the avalanche and its bias undisturbed. `multiplicities=[mₑ,mₕ]` then describes a packet-height match. An integer component requires the stated number of carriers in that final packet. A null component accepts every height at that contact.

For `observable="waveform"`, apply the height match as well and report the mean of [Q(v)−Q(u)]/G for `read=[u,v]`. For `observable="fraction"`, report the fraction matching the two requested heights among the shots already selected by gain, both clock readings and the tag. The read interval has no effect on this fraction. All times are on the laboratory clock from optical injection. The full-time induced charge is G.

Graded shots have depth in [0.25,0.65], vₑ in [0.8,1.35], vₕ in [0.65,1.2], dead spaces in [0.035,0.12], nominal ionization coefficients in [1.5,3], and feedback in [0.35,1.2]. The worked limit allows zero dead space and a clamped bias. With one delayed clock G is 2, 3 or 4; with two delayed clocks it is 3, 4 or 5. Positive delays are at most 1.3. Gate endpoints are in [0,2.5] and tag widths in [0.15,0.4]. Conditioning denominators are positive. Delayed settings avoid discontinuities of the appropriate time density.

For a check, use depth 0.4, both speeds 1, both dead spaces zero, ionization [2,2], feedback zero, gain three, `last_delays=[0.3,0.4]`, `tag=null` and `read=[0.25,0.7]`. The last electron arrives at 0.9 and the last hole at 0.8. For height pairs (2,1), (1,2) and (1,1) the respective normalized charge readings are 0.6, 2/3 and 19/30. The matched fraction is 1/3 for each height pair. For (2,1), a primary hole impact occurs at time 0.15 and position 0.25, followed by an impact of its new electron at time 0.35 and position 0.45. For (1,2), the first impact is a primary-electron impact at time 0.2 and position 0.6. The independent primary impacts give (1,1). These histories have equal joint time density under the stated conditions.

## Interface and scoring

Implement `Model` in `/app/model.py`. `fit(records)` takes calibration dictionaries with `input` and `value`, sets `activity` to a and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty list. The supplied starter samples five positions for each unconstrained impact, weights selected parent hazards, and omits the probability of surviving without other impacts. `current.py` supplies the calibration and the displacement charge of given pairs.

The ten calibration records require maximum absolute error below 0.00002. The worked checks and each of three held-out groups require maximum absolute error below 0.004. Each group contains eight readings. They cover waveforms with one delayed electrode, waveforms with both electrodes delayed, and coincidence fractions of the two terminal packet heights.
