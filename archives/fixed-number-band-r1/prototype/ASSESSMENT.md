# Fixed-number band calorimetry prototype

This bounded science prototype is complete and pending independent peer review. No task package, calibration dataset, 256-noise study, Docker control or model evaluation was created. Root authorized the prototype after the materials outline; empirical difficulty remains unknown.

## Physical distinction

There are M states at each energy -W/2 and +W/2, occupied by noninteracting spinless fermions. Every observable below is divided by 2M, not N or M. Complete canonical equilibration at fixed N precedes the quasistatic calorimetric measurement. The contact can redistribute particles between bands but cannot exchange them with the sample. The full proposed preparation and measurement are in INPUT-OUTLINE.md.

At finite positive temperature, the thermodynamic-limit equilibrium populations are exactly Fermi occupations f_s, with the chemical potential chosen to make (f_-+f_+)/2=n. Define M_j as the average of (E_s-mu)^j f_s(1-f_s) over the two bands. The completed source keeps this exact state, including its full filling dependence, but returns M_2/T^2. This is the positive fixed-chemical-potential thermal susceptibility T times the temperature derivative of entropy. It is a coherent open-system response at the same equilibrium state, not a wrong chemical-potential solve or a finite-size numerical approximation. It must not be described as the derivative of internal energy at fixed chemical potential: that derivative has a separate chemical-work term.

For the closed calorimeter, differentiation of the fixed-number condition changes the chemical potential during a temperature increment. Its capacity is (M_2-M_1^2/M_0)/T^2. The difference is nonnegative. The two-band identity C_N=W^2 a*b/[2 T^2(a+b)], with a=f_-(1-f_-) and b=f_+(1-f_+), supplies a stable low-temperature evaluation and an independent algebraic normalization check. None of these formulas belongs in the proposed public description.

## Shared calibration

At n=1/2, symmetry gives mu=0 and M_1=0. Both responses equal x^2 sech^2(x), x=W/(4T). Nine temperatures from .10 to .15 have x at least 4/3 across W in [.8,1.2]. Since x*tanh(x)>1 there, dC/dW=2C[1-x*tanh(x)]/W is strictly negative throughout. Thus every calibration curve identifies W globally. This is an analytic injectivity result; the numerical optimizer is not its proof.

All 41 noiseless fits across the width interval recover W within 6.86e-9, including explicit endpoint candidates. Shared calibration is exact to rounding. The smallest derivative magnitude sampled is .17288, so the identification does not depend on a flat response. No private fitted truth or data is present in this prototype.

## Independent references and boundaries

The main reference sums the exact finite canonical partition function over upper-band occupation k. The multiplicity is choose(M,k)*choose(M,N-k), and its temperature weight is exp(-W*k/T). It uses no chemical potential, Fermi occupation, or susceptibility subtraction. The capacity follows from the second energy cumulant of this fixed-N partition sum. A separate fourth-order temperature derivative of its mean energy agrees to 1.61e-8.

At 30 rational-filling cases, sizes M=4000,8000,16000,32000 permit exact fixed n. Quadratic inverse-M extrapolation agrees with the thermodynamic oracle within 4.98e-8 absolute; changing the three-size window changes the result by at most 3.98e-7. The finite-size error maxima decrease from 5.76e-5 to 7.21e-6 over this sequence. These are controlled finite-size checks, not an assertion that finite and thermodynamic ensembles coincide exactly.

Sixty declared-domain corner/interior cases also compare the physical response to a direct temperature derivative of the re-equilibrated mean energy at fixed n, agreeing within 1.90e-12; halving the derivative step changes values by at most 2.20e-12. Their finite canonical size sequences explicitly record the actual rounded filling at each M and compare against the limit at that filling. Integer rounding is not silently treated as physical finite-size bias.

Supplementary limits check positivity, particle-hole symmetry, energy-origin invariance, and the stable two-level identity. The identity error is below 1.53e-16 over 84 controls. The closed capacity tends to zero at low temperature, including partially filled flat bands. At high temperature, C_N*T^2 tends to n(1-n)W^2/4; the source instead tends to n(1-n)logit(n)^2 away from half filling. These outside-domain limits diagnose the ensemble distinction and are not proposed scoring cases.

## Signal separation and preserved screening history

The final bounded diagnostic grid covers W in [.8,1.2], T in [.4,.6], and fillings .2,.25,.3 plus their particle-hole partners. Its 378 controls have relative source excess from 9.46% to 420.3%, with true capacities .0563 to .1963 kB/state. This is an ordinary, positive caloric signal with separation above the unchanged 4% benchmark scale. It is a grid screen, not a proof of the minimum over a continuous diagnostic box and not a final grading design.

The initial screen is preserved unchanged under initial-screen/. A broader temperature range [.25,.55] had accidental additional near-equality at doped filling, so one must not claim all doping discriminates. That initial report also rejected an insufficient M=1000..8000 finite-size extrapolation at the coldest half-filled case. The final reference uses larger systems and narrows only the illustrative diagnostic temperature subset; the complete proposed public domain is still [.1,.6]. No task or model result existed during either change. The final report and supplementary limits are separate, and neither overwrites the initial report.

The analytic example at n=.3 and W/T=log(9) gives occupations (.5,.1), physical capacity 9 log(9)^2/272=.159743 and source 9 log(9)^2/200=.217251. This independently checks the per-state normalization and a healthy 36% discrepancy.

## Overlap and qualifications

Actual AUTHOR files were read for quantum-calorimetry, finite-band-reservoir, finite-heat-reservoir and spin-strain-response; hashes are in archive-overlap-audit.json. Quantum calorimetry changes the measured subsystem-plus-bath energy. Finite-band reservoir changes reachable populations after an isolated quench. Finite heat reservoir fixes total energy instead of a canonical bath. Spin-strain distinguishes quantum static response from equal-time operator variance. Here equilibrium populations are already exact, while the infinitesimal response uses the wrong particle-exchange constraint. These are related ensemble and calorimetric themes. This is not a claim of an unrelated general principle or an exhaustive new archive search; the owner's wider outline audit supplies context.

Primary background: Kuzmenko and Mikhajlov, [The canonical heat capacity of normal mesoscopic fermion systems](https://arxiv.org/abs/0811.2078). Its primary abstract was read and confirms canonical independent-fermion calorimetry as context. No exact two-band formula is attributed to that abstract, and the full paper was not read here. The present model, normalization and partition-function reference are independently derived.

The mathematical repair is small and well known. Strong science separation does not establish model difficulty. Any future package needs a separate authorization, fixed calibration noise, the ordinary .04 gate, complete controls, and source/input peer review before model trials.

Reproduce with the pinned NumPy2.3.3/SciPy1.16.3 environment by running check.py and limits.py in this directory. Both are author science scripts; neither launches models or changes another task.
