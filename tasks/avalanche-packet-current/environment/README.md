# Induced-charge spectroscopy of an avalanche photodiode

A short optical pulse creates one electron–hole pair at a known depth in a planar multiplication layer. The two contacts resolve groups of electrons and holes collected simultaneously. A second amplifier measures the charge induced while carriers are moving. Determine the spectrum of the induced charge in one time gate and the mean charge in a second gate on the same selected shots.

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

The two entries of `multiplicities` select the exact electron and hole packet heights; JSON null in either position accepts any height on that contact. In all shots here `tag` is null. The base ensemble consists of the specified gain, both terminal clock readings and both height selections.

## Charge spectrum

An ideal gated integrator measures F=Q(v)−Q(u) at `spectrum.window=[u,v]` on the laboratory clock. A group of histories can give exactly the same charge when the same carriers cross the entire gate. Other histories give a continuous set of charges. A line and a continuous density therefore use different units.

With `observable="spectrum"` return the following intensity **within the base ensemble**. The target z is `spectrum.charge`.

- For `spectrum.part="line"`, return the probability that F=z exactly. The supplied target identifies a discrete spectral line. Values rounded to 12 or more decimal places identify the same line.
- For `spectrum.part="density"`, return the probability per unit induced charge at z. This is the limit of Pr(z−h/2<F<z+h/2)/h as h decreases to zero. Density targets are regular interior points away from spectral lines, jumps and support edges. A density can exceed one.

With `observable="waveform"`, use the chosen spectral component to select shots, then return the mean of [Q(B)−Q(A)]/G in those shots, with `read=[A,B]`. For a line, use only its probability mass. For a density, take the conditional mean in the shrinking interval about z. Both integrators measure the same avalanche, including when the two time gates overlap. Apply the terminal clock selection before taking the spectral limit. All conditioning denominators are positive. The base ensemble, line probabilities and densities are independent of the choice of read gate.

For graded shots depth is in [0.25,0.65], electron speed in [0.8,1.35] and hole speed in [0.65,1.2]. Ionization coefficients are in [1.5,3], dead distances in [0.035,0.12] and feedback in [0.35,1.2]. For one delayed contact the gain is 3 or 4. For two it is 4 or 5. Times are in [0,2.5] and positive delays avoid discontinuities of the contact time density. The worked example also allows zero dead space and clamped bias.

## Worked shot

Use depth 0.4, both speeds 1, ionization [2,2], dead spaces [0,0], feedback zero, G=3, delays [0,0.6], any heights and `tag=null`. Put the spectral gate at [0,0.15] and the read gate at [0,0.25]. The electron clock is prompt, so both impacts were caused by electrons. The hole clock fixes the second impact at time 0.3. The first impact time t is uniform on (0,0.3): both eligible electron parents for the second impact have the same constant rate, and the total integrated hazard is 2aG for each atom-selected history at this gain. The early charge is F=0.30+2 max(0,0.15−t).

There is a line at 0.30 with probability 1/2. Its conditioned read is 17/90. At charge 0.45 the continuous density is 5/3 per unit charge and the conditioned read is 17/60. These are four distinct outputs of this shot.

## Interface and scoring

Implement `Model` in `/app/model.py`. `fit(records)` receives dictionaries with `input` and `value`, sets `activity` and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty list.

The starter uses an amplifier with Gaussian noise of fixed width 0.04. It approximates a line by the acceptance of that filter, a density by the normalized filter response and the gated read by the mean among accepted shots. The public `transport.packet_law` integrates the gain and both clock selections and returns two arrays indexed by (electron height, hole height): unnormalized mass and its read-charge moment. It accepts an optional Gaussian `tag=[u,v,z,s]` with acceptance exp(−((Q(v)−Q(u)−z)/s)²/2). The base transport includes the shared feedback and dead flights.

Ten calibration readings require maximum absolute error below 0.00002. The public example and three hidden groups use absolute error below 0.004 for line probabilities and conditioned read charges, and below 0.01 for charge densities. Each group has eight readings at geometries separate from the worked example.
