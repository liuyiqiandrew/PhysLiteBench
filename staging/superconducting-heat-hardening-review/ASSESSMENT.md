# Superconducting heat hardening review

Do not build the proposed branch-sign revision. It is a defensible wrong readout, but the reviewed evidence does not support it as a harder task. This rejects one concrete proposal, not the superconducting family.

The fresh r1 batch has two physical passes and one clean physical failure. Both passers replace the gap-edge coherence ratio with its energy-dependent value, even though calibration cannot distinguish them. Z3vUZBL retains the original quadrature order; qJoZVQg increases it but also makes the same physical repair. QCiD2Ta only fits the conductance and explicitly endorses the existing kernel. Its three hidden errors are 27.1%, 10.2% and 18.9%; both passing submissions are within 0.037%. No scored outcome is an infrastructure or numerical failure. The attached review verifies all 16 current/frozen files, actual Luna/high/CLI154 inputs and 63 evidence hashes.

The proposed revision retains exact finite-transparency normal and branch-converting transmissions, but interprets their difference as heat transmission. The correct sum is explicit in [Zhao, Löfwander and Sauls, Eqs.16–31](https://arxiv.org/pdf/cond-mat/0308041). For identical electrodes in linear response, put x=E²−Delta² and y=Delta² sin²(phi/2). Then

    D_ee = D*x*(x+y)/(x+D*y)^2
    D_he = D*(1−D)*x*y/(x+D*y)^2
    source = D_ee−D_he = D*x/(x+D*y).

At phi=0 both readouts equal D, so varied-temperature calibration identifies D linearly. The same equality need not hold for unequal gaps: Eq.27 retains branch conversion at phi=0. The proposed equal-gap linear-response preparation therefore matters; this is not a flaw in r1 or an evaluated task.

Own assessment: the difference is positive and bounded, so it can define a completed signed branch-imbalance model. It is not automatically an exact electrical-current model. Its physical error is assigning negative heat to transmitted positive-energy hole excitations. Once both transmissions are supplied in code, correcting that assignment requires only changing their combination. Finite transparency makes the shared scattering calculation richer but adds no unsolved transport constraint. The prior passing trajectories already distinguish energy transport from an approximate coherence prescription. There is no evidence that this sign choice would defeat them, and a larger harness would not resolve that concern.

If reconsidered, the apparatus would need identical clean s-wave leads, a short elastic single-channel contact, common chemical potential, static phase, known gap law, an infinitesimal thermal bias and electronic thermal-conductance readout. Public text should specify these facts without transmission formulas. An independent check would solve BdG scattering and outgoing energy flux. None was built or run here: no new numerical margin, reference convergence, calibration data, source, harness or model evaluation is claimed. Existing canonical sources, outcomes and all raw jobs remain unchanged.
