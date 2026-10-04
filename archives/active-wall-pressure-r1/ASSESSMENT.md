# Active wall pressure

Status: scientific package complete; unevaluated.

The source retains the exact stationary position/orientation kinetics but applies the torque-free midplane pressure conversion to orienting walls. The source and physical model agree exactly throughout calibration. The unknown common line loading is globally identifiable and starts at None. All wall inputs affect the source's joint-state solution.

All 256 noise checks passed for calibration, parameter recovery and the physical oracle. The shortcut failed all three diagnostic groups in every draw, with minimum relative error 0.3151 at a 0.04 gate. Independent force integration, grid refinement, supported corners, positive normalized densities and the passive thermal limit were checked. Local controls give oracle 7/7 and shortcut 4 passes plus 3 intended hidden failures; both finish under five seconds.

The physical model, calibration rationale, exact moment identity, numerical independence, archive overlap and primary reference are in `tasks/active-wall-pressure/AUTHOR.md`. Full reports and final peer review are in `results/`. Earlier quadratic-wall and final quartic-wall prototypes are preserved in `prototype/`; the original prototype directory remains unchanged.

No agent-model evaluations have been run. Scientific separation does not establish empirical model difficulty. Root owns any evaluation, promotion or archive disposition.
