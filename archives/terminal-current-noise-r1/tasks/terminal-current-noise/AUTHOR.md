# Terminal current noise, revision 1

This task separates exact island kinetics from the physical current measured in an external voltage-source wire. The completed starter uses the correct finite-temperature bidirectional tunneling rates, stationary charge probabilities, marked current generator, shot term and frequency-dependent resolvent. It reports the left junction's tunneling current. The measured external current also includes redistribution of charge on the electrode's capacitor whenever the island charge changes.

Write `IL` positive for an electron entering through the left junction, `IR` positive for one leaving through the right, and `N` for island occupancy. Charge continuity gives `dN/dt = IL - IR`. With only the two stated capacitors, the left electrode's induced electron number is `-cL N`. Electrode charge balance therefore gives `Iext = IL - cL*dN/dt = (1-cL) IL + cL IR`. An entry through the left/right junction has measured weight `1-cL`/`-cL`, respectively; exit events have opposite signs. All capacitance controls already influence the source's addition energy and rates, so the failure is not caused by an unused control.

The source and oracle use a column probability generator `L` and stationary projector `P=p 1^T`. If `J1` and `J2` mark each jump by its signed charge and squared charge, respectively, then the specified spectrum is `2*1^T J2 p + 4 Re[1^T J1 (i omega I - L + P)^(-1) (I-P) J1 p]`. The shortcut's counting weights are `(1,0)`. The oracle's are `(1-cL,-cL)`. This is a coherent alternative observable with nonnegative noise, not a defective stochastic solver.

The private reference independently reduces the two charge states. For entry rates `aL,aR`, exit rates `bL,bR`, totals `a,b`, and `s=a+b`, the mean is `I=(aL*b-bL*a)/s`. Set `alpha=1-cL`, `beta=cL`, `U=-alpha*bL+beta*bR`, `V=alpha*aL-beta*aR`, and `j2=[(alpha²*aL+beta²*aR)b+(alpha²*bL+beta²*bR)a]/s`. Then `S=2*j2+4*s*(U*V-I²)/(s²+omega²)`. The validator additionally integrates the time correlation, checks the charge-continuity spectrum and compares equilibrium noise with the real admittance obtained by independently perturbing the physical left voltage and solving the driven master equation. The latter verifies the capacitor weighting against the electrostatic voltage coupling, not just another noise formula.

All 216 calibration spectra have zero frequency. The integral of `dN/dt` is bounded, so it cannot change long-time current noise. Correct and shortcut predictions therefore agree for every calibrated bias, capacitance fraction, tunnel asymmetry and temperature. Their dependence on the unknown positive common rate is linear at zero frequency, giving a unique weighted least-squares fit. Both parameter endpoints are recovered exactly. The fixed uncertainty 0.002 is an instrument uncertainty independent of the noiseless signal.

Hidden finite-frequency spectra have signals from 0.446 to 0.704 electron-charge²/ns and shortcut errors of 106–143%. The oracle's nominal error is about 0.010%; across 256 calibration-noise draws its largest error is 0.081%, well below the 4% threshold. Both controls pass zero-frequency anchors. All 256 oracle controls pass and all 256 shortcuts fail the three finite-frequency groups. Local isolated tests give oracle 7/7 and shortcut 4 pass plus 3 intended failures, each under half a second.

Validation covers 768 public-domain corners, positive spectra, direction/frequency symmetry, rate/time scaling, stationarity, high-frequency jump variance, and 216 equilibrium voltage-response cases. The independent closed spectrum agrees at roundoff. The voltage-response fluctuation-dissipation relation `S=4*T*Re(Y)` agrees within 5.1e-11; purely reactive geometric capacitance does not alter its real admittance. The physical approximation is explicitly classical sequential tunneling with instantaneous electrostatics. The stated frequencies and rates are well below thermal energies in frequency units; no unspecified external circuit impedance or third capacitance is needed.

This is related to earlier mesoscopic noise candidates but is distinct from the archived floating voltage-probe cumulant closure: here the missing physics is terminal displacement current and the location of an electrical readout. No difficulty result is claimed before evaluation. A retained junction-current observable is a physical failure; a correct terminal-current derivation followed by a generator sign or normalization bug is an implementation failure.

The neutral instruction is unchanged. The public README defines the circuit and amplifier location without giving the terminal-current weighting or naming the correction. The supplied model includes its fit and complete noise computation. Author-only `hint.md` is excluded from ordinary agent images.

Primary background: A. N. Korotkov, *Intrinsic noise of the single-electron transistor*, Physical Review B 49, 10381–10392 (1994), [publisher](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.49.10381), [author PDF](https://intra.ece.ucr.edu/~korotkov/papers/PRB-49-10381-1994.pdf). That work establishes the master-equation frequency-domain treatment of classical single-electron noise; the specific spin-polarized two-state rates and circuit convention here are fully defined in the task and independently checked.

From the staging root:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_terminal_current_noise.py
```

Use `--generate` only to intentionally regenerate both calibration copies. Reports are `results/terminal-current-noise-r1-validation.json` and `results/terminal-current-noise-r1-local-controls.json`. Model evaluations are pending root-controlled frozen runs.
