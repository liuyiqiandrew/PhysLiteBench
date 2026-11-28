# Two photons scattering from a single Λ emitter

Fit the visibility V of a photon-number transfer counter. Then predict its conditional fringe for two shaped photons that pass a single atom in a right-moving, lossless waveguide. The ground state left by Raman emission can affect the next photon. The time slots specify envelopes of traveling waves. Within a slot, radiation emitted at an earlier time has already passed the atom.

## Calibration

In every shot the counter reports V times the mean of cos[λ(q′−q)−φ] over the retained events. `lambda` and `phase` are λ and φ in radians. The unknown V is in [0.65,0.95]. In the twelve calibration records, `source=[[n,0,0],[0,0,0]]` denotes a thermal state of mean n in polarization 0 and vacuum in polarization 1. The entrance bank is polarization 0. `before` contains one lossless mixer `[0,1,t,p]`, and the exit bank counts polarization 0 after that mixer. `prep` and `after` are empty and `gate` is null. The supplied `optics.py` implements this calibration.

All polarization and ground-state mixers use the same convention. For an angle pair `(t,p)`, the columns of R are the outgoing kets for the two incident basis states.

    R(t,p) = [[cos(t), −exp(−ip)sin(t)], [exp(ip)sin(t), cos(t)]].

## Source and atom

A waveguide shot has the field `rates`. There are B consecutive time intervals of positive lengths `durations`. The two rows of `input_modes` each contain B complex pairs `[real,imag]`. A row f has sum |fⱼ|²=1 and represents the continuous envelope f(t)=fⱼ/√durationⱼ on interval j and zero elsewhere. One photon enters in polarization 0 with the first envelope and one enters in polarization 1 with the second. Thus q=1 in the entrance bank. The two photons pass through the same `input_rotation=[t,p]` before reaching the atom. In terms of waveguide creation operators the incident state at the atom is A₀†A₁†|vac⟩, with

    Aᵢ† = ∑c R(input_rotation)_{ci} ∫ fᵢ(t)b_c†(t) dt,
    [b_c(t), b_d†(s)] = δ_cd δ(t−s).

The atom has two degenerate ground states |g₀⟩ and |g₁⟩ and one excited state |e⟩. Both optical transitions share that same excited state. Initially it is in R(`atom`)|g₀⟩, where R acts on the ground doublet. The two positive `rates` are Γ₀,Γ₁, the spontaneous emission rates from e into polarization c and ground state g_c. Coupling is flat across the pulse bandwidth, with no propagation delay within the apparatus. The atom couples to fresh right-moving field increments dB_c. With ℏ=1, the Itô equation in interval j is

    dU = [∑c (L_c dB_c† − L_c† dB_c) − (iHⱼ + ½∑c L_c†L_c)dt] U,
    L_c = √Γ_c |g_c⟩⟨e|,     Hⱼ = detuning[j] |e⟩⟨e|.

The vacuum Itô product is dB_c dB_d† = δ_cd dt. The direct field transmission is identity. The atom, these two polarizations and the applied controls constitute the whole system. At the end of the first k intervals, `control=[k,t,p,s]` applies R(t,p) on the ground doublet and exp(is) on the excited amplitude. It is instantaneous and coherent. No state measurement accompanies the pulse.

## Receiver

`output_modes` gives the two normalized, mutually orthogonal temporal envelopes h₀ and h₁ in the same format as the inputs. The two receivers may overlap in time. An ideal lossless mode sorter separates them after scattering. Each selected mode k has its own polarization mixer `output_rotations[k]`. It detects color r by annihilating

    D_kr = ∑c R(output_rotations[k])_{rc} ∫ h_k*(t) b_c(t) dt

on the outgoing state. Keep exactly one photon in each selected temporal mode. Since only two excitations entered, other outgoing modes are in vacuum on retained events. The atom is in its ground doublet on these events. If `analyzer=[t,p]`, also project the final atom on R(t,p)|g₀⟩. If `analyzer=null`, sum the two orthogonal ground-state probabilities. The exit count q′ is the number of the two selected photons detected with polarization 0. Normalize the fringe by the joint success probability of these selections.

B is 2 through 6, durations are in [0.4,1.8], Γ_c in [0.4,1.8] and |detuning[j]|≤1.1. All angles are in radians with magnitudes ≤π. The boundary is an integer 1≤k<B. Input rows can overlap arbitrarily in time. Output rows are orthogonal to numerical precision. Graded waveguide shots have joint selection probability at least 0.01.

For a worked example take B=2, durations and rates both `[1,1]`, detunings zero, `atom=[0,0]`, analyzer null, and all rotations and Stark phases zero. In both mode arrays use the rows (1,0) and (0,1); the first input photon and first receiver occupy only the early slot. Put λ=π, φ=0. The ground-to-ground amplitude for a Raman flip into the same normalized slot is −x, where x=exp(−1). The accepted probabilities for detector pairs `(0,1)`, `(1,1)`, `(1,0)` are (1−x)², x²(1−x)², x⁴ respectively. The other pair is dark. The joint acceptance is 0.4719688252 and the reading is 0.7708459436 V.

## Interface and scoring

Implement `Model` in `/app/model.py`. `fit(records)` receives dictionaries with `input` and `value`, sets a real `visibility`, and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in input order, including an empty array for an empty list. The supplied calibrated starter scatters the two photons in separate passages. It adds the two assignments to the receivers and applies the RF control between the isolated passages. `apparatus.py` supplies the pulse and receiver conventions and an exact single-photon ground-to-ground transfer matrix for a passage with the control off.

The twelve calibration readings require maximum absolute error below 0.00002. Each of three held-out groups has ten readings and requires maximum absolute error below 0.001. They cover photons arriving in separate time windows with Raman memory, overlapping photons incident on the same emitter, and coherent final ground-state selection with overlapping orthogonal receivers. All three use the dynamics and receiver above.
