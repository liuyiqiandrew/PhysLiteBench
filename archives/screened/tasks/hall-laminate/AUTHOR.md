# Hall laminate, revision 2

Revision 1 used arithmetic averaging of local conductivity tensors and scored 3/3 unhinted; its complete sources, controls and validation remain in `archives/hall-laminate-r1`, with all original jobs frozen separately. Revision 2 uses a more informative calibration and a physically more coherent approximate laminate model. Grading and hidden preparations are unchanged.

The shortcut has the exact zero-field series/parallel resistivities for every layer fraction and orientation. It also has the exact effective off-diagonal Hall resistivity. It omits only the extra normal resistivity from spatial variation of the local Hall response. All inputs affect the model; it respects macroscopic Onsager-Casimir symmetry and has positive symmetric resistivity. This is a valid low-field closure, not an unfinished tensor rotation or an algebra error.

In normal/tangential layer coordinates let r_i=1/(nq mu_i) and h_i=rho_i,nt. Write G=<1/r>, H=<h/r>, and K=<h²/r>. The exact resistivity is

    rho_nn = <r> + K - H²/G,
    rho_nt = H/G, rho_tn = -H/G, rho_tt = 1/G.

The shortcut keeps the exact last three entries but uses rho_nn=<r>. The omitted term is G times the conductance-weighted variance of h, hence nonnegative and zero in a pure phase or at zero field. It comes from differing transverse currents and Hall fields in the layers under the stated average-current constraint. Treating the mean Hall response as the complete magnetic correction discards this dissipative effect.

Calibration now includes zero-field mixed specimens with varied orientations, as well as the original pure-phase finite-field probes. Both controls fit all these data exactly at the true mobility. Thus calibration tests geometry and Hall-independent transport without observing the Hall-field variance. Both use a valid linear fit in inverse mobility. The correct oracle still eliminates the full local tensors with normal-current and tangential-field continuity; the independent reference still solves four laboratory interface/current equations. Hidden inputs and the .04 RMS-relative-error limit are unchanged.

Validation command: `uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_hall_laminate.py --noise-trials 256`. Regeneration requires `--generate`. New seed 10003, revision 2. Local and independent controls, noise sensitivity and peer review precede a fresh paired batch.
