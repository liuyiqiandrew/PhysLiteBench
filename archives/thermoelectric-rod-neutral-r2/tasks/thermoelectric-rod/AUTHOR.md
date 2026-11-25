# Thermoelectric rod revision 2

Revision 2 tests distributed Peltier heat in a solid whose Seebeck coefficient has a fixed spatial gradient. The completed shortcut already includes reciprocal Thomson heating, Joule heat, conduction, the full graded terminal-voltage integral, and the supplied current and material grading. It applies the homogeneous-material heat equation locally and omits the heat associated with the spatial part of the Seebeck gradient.

The public reciprocal local transport gives

    q = T*S(x,T)*J - k*T_x,
    E = rho*J + S(x,T)*T_x,
    c*T_t = -q_x + J*E
          = k*T_xx + rho*J² - J*T*(S_T*T_x + S_x|T).

The ordinary terms S*J*T_x cancel. Here S_x|T=seebeck_span/L, so the omitted source is -J*T*seebeck_span/L. The composition is fixed in material coordinates, and fixed-temperature reservoirs absorb any contact heat. No additional mass transport or chemical work is introduced. The apparatus defines an ideal effective local material; it does not require fitting an unstated microscopic transport model.

All calibration preparations have zero grading but nonzero current. They therefore validate the Thomson term and identify conductivity without exposing the spatial Peltier term. Both complete controls give identical calibration predictions and fitted parameters. Graded hidden preparations use both current signs, both grading signs, and both orientations of the imposed temperature difference.

The terminal voltage is

    rho*J*L + S300*(Tr-Tl)
      + S_T/2*((Tr-300)²-(Tl-300)²)
      + seebeck_span*((Tr+Tl)/2 - mean(T)).

Unlike revision 1, voltage now depends on the interior thermal profile. The starter uses this correct expression, so its voltage failure follows from the wrong temperature dynamics. The oracle discretizes the expanded heat equation. The verifier separately forms face Peltier heat flux, its divergence, and electrical work on a refined mesh, then integrates the electric field directly.

The private conductivity is 1.25 W/(m K). Calibration seed 31401 and independent-noise seed 41401 produce two opposite-current preparations with fixed independent temperature uncertainty 0.04 K and voltage uncertainty 2e-5 V. These uncertainties are constant and contain no noiseless-response information. The thermal hidden threshold remains normalized RMSE 0.04. Because voltage now depends on conductivity through the interior temperature, its absolute threshold is relaxed from 1e-5 V to 1e-4 V, five times the stated voltage measurement uncertainty. No threshold was tightened to create difficulty.

The fitted conductivity is 1.25220977 W/(m K), with reduced chi-squared 0.967568. All 256 independent noise realizations pass calibration and parameter checks; the largest relative conductivity error is 0.538%. Nominal normalized hidden error is 0.00135–0.00157 for the oracle and 0.645–0.965 for the completed shortcut. Across the fitted-parameter extrema, oracle error stays below 0.00491 and shortcut error above 0.643. Oracle voltage error stays below 7.77e-5 V at those extrema; shortcut voltage errors are about 0.009–0.012 V.

The independent conservative reference agrees with the expanded heat equation within 0.00263 K. Reference refinement from fourfold to eightfold changes outputs by at most 0.000306 K; oracle twofold to fourfold refinement changes by 0.00210 K. Current reversal together with spatial and grading reversal preserves the temperature solution within 2.28e-13 K and reverses voltage within 5.56e-17 V. An analytic-profile local energy balance and the omitted-term identity agree within 1.4e-9 W/m³. Hidden temperatures range from 280.7 to 359.2 K. Homogeneous nonzero-current and graded zero-current equivalence hold exactly.

Local isolated controls give oracle 7/7 tests passing in 1.76 seconds. The shortcut passes public calibration, private calibration, and parameter recovery, then fails the three intended temperature groups and the consequent voltage check in 1.57 seconds. See [the complete scientific and local-control report](../../results/thermoelectric-rod-r2-validation.json).

The complete prior task, data, controls, validator, science report, and trial reviews are preserved in [the neutral-v1 archive](../../archives/thermoelectric-rod-neutral-v1/README.md). Revision 1 scored 0/3 plain and 3/3 hinted under its original instruction, then 3/3 plain under the neutral instruction. These batches remain separate. Revision 2 preserves that neutral instruction byte for byte. Scientific controls do not establish revision-2 agent difficulty; evaluation is pending.

Reproduce validation with:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_thermoelectric_rod.py
```

Use `--generate` only to intentionally regenerate both calibration copies. The physical mechanism is also treated in [Thiébaut et al., heat transport in graded Peltier devices](https://arxiv.org/abs/1801.05175). The equations here follow directly from the apparatus' reciprocal constitutive laws.
