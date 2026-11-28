# Last-packet timing in an avalanche photodiode

A short optical pulse creates one electron–hole pair at a known depth in a planar multiplication layer. A timing electrode resolves groups of electrons collected simultaneously. A second amplifier measures the charge induced while carriers are moving. Predict the mean gated amplifier reading for specified gain, last-packet time and, in some shots, packet multiplicity and an earlier current tag.

## Carrier dynamics

The layer is 0≤x≤1. At time zero the primary pair is at `depth`. The electron travels right at `velocities[0]`=vₑ and the hole left at speed `velocities[1]`=vₕ. The drift speeds are constant in the operating range. A carrier disappears on reaching its collecting contact. Diffusion and trapping are negligible. Distinct charge carriers at the same point keep separate impact clocks.

An impact creates an electron and a hole at the incident carrier's position. The incident carrier continues in the same direction. Its energy and both new carriers' energies are reset, so each of these three must cover its own full `dead_space[c]` before it can impact again. Other carriers keep their energies. The primary pair also starts cold. Carrier type c=0 is an electron and c=1 is a hole.

The amplifier weighting field is uniform. In units of elementary charge its running induced charge Q starts at zero and obeys dQ/dt=vₑ Nₑ(t)+vₕ Nₕ(t), where N counts carriers still in flight. Collection has no extra jump in Q. The bias capacitor discharges by this same total charge during a shot. In the calibrated voltage range the log ionization coefficient drops linearly with Q. After a carrier has crossed its dead space, its instantaneous impact hazard per unit **distance** is

    a × ionization[c] × exp(−feedback × Q(t)).

The factor a is the same for both species; all other quantities are supplied. Conditional on the shared instantaneous field, the active carriers have independent impact hazards. Ineligible carriers can still induce current. The feedback affects impact rates while drift speeds and calibrated dead distances remain fixed. Recovery between shots restores Q=0.

The calibration isolates a single fully energized electron in a voltage-clamped layer with unit nominal ionization coefficient. Its `active_length` is the available distance after the initial dead flight. The recorded value is the probability of no impact, rounded to eight decimals. The unknown a is in [0.8,1.1].

## Selection and reading

`gain` G is the total number of electrons eventually collected at x=1, including the primary electron. Each impact therefore raises G by one. The prompt electron arrival is T₀=(1−depth)/vₑ. Let T be the last electron arrival in a shot, and let M count the electrons arriving exactly at T. The timing counter has ideal resolution. Electrons on the same drift trajectory meet the contact at exactly the same time.

For `last_delay=0` retain the probability atom at T=T₀. For `last_delay` τ>0 condition on T in [T₀+τ,T₀+τ+ε), with ε tending to zero from above. Use the corresponding time density for both numerator and denominator. `multiplicity=null` accepts all last-packet sizes; an integer requires M to equal that size. Always retain the specified final gain.

For any interval [u,v] the analog charge is Y(u,v)=Q(v)−Q(u). If `tag=[u,v,z,s]`, a trigger accepts the trajectory with probability exp[−(Y(u,v)−z)²/(2s²)]. A null tag accepts every trajectory meeting the timing and gain selections. The trigger leaves the carriers and circuit undisturbed. `read=[u,v]` requests E[Y(u,v)/G | all selections]. Both gates run on the laboratory clock from optical injection. The full-time induced charge is G.

Graded shots have depth in [0.25,0.65], vₑ in [0.8,1.35], vₕ in [0.65,1.2], positive dead spaces in [0.035,0.12], nominal ionization coefficients in [1.5,3], and feedback in [0.35,1.2]. The worked limits below also allow a clamped bias and a nonionizing species. Prompt cases have gain 2 or 3. Delayed cases have gain 2, 3 or 4 and τ in (0,1.2]. Gate endpoints are in [0,2.5], tag widths in [0.15,0.4]. Selected masses or time densities are positive. Delayed targets are away from discontinuities of the time density.

For a prompt check use depth 0.2, both speeds 1, dead spaces [0.45,0.12], ionization [1,0], feedback zero, gain two, last-packet multiplicity two, no tag and read [0,0.8]. The electron's unique impact time X is an exponential flight truncated to [0.45,0.8], with the dead time added. Both electrons reach the right electrode at 0.8. The reading is (1+2(0.8−E[X]))/2 with E[X]=0.45+1/a−0.35/(exp(0.35a)−1).

For a delayed check use depth 0.4, both speeds 1, dead spaces [0.05,0.05], ionization [2,2], feedback 0.6, gain two, delay 0.3, multiplicity one, no tag and read [0.22,0.7]. The primary hole creates the second pair at time 0.15 and position 0.25. The last electron arrives at 0.9. The induced charge rises from 0.58 to 1.8 across the read gate, giving 0.61.

## Model and scoring

Implement `Model` in `/app/model.py`. `fit(records)` takes calibration dictionaries with `input` and `value`, sets `activity` to a, and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in input order, including an empty array for an empty list. The supplied starter represents each selected avalanche by one set of birth positions; `current.py` supplies the calibration and the induced charge of a prescribed set of newborn pairs.

The ten calibration records require maximum absolute error below 0.00002. The worked checks and each of three held-out groups require maximum absolute error below 0.003. The groups contain eight prompt-cluster, eight delayed-packet and eight tagged-waveform readings.
