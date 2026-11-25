# Superconducting ring revision 1

The apparatus is an ideal thin-wire London ring at fixed carrier density, material volume and phase winding. The source correctly fits kinetic inductance and predicts the current for every geometry. It interprets the total radial support force as the electromagnetic Lorentz force, omitting the carrier kinetic stress transferred to the material.

Let L=Lg+kappa*R² and Phi=pi*R²*B. Fluxoid conservation fixes I=(n−Phi)/L. With the external field held by a current source, the controlled-field potential in this winding sector is

    F(R,B,n) = (n-Phi)^2/(2L).

Virtual work at fixed B and n gives the radial force

    force = I*2*pi*R*B + I²*(dLg/dR + 2*kappa*R)/2.

The completed shortcut includes the first term and the geometric-inductance term, but omits kappa*R*I². At fixed material volume, the wire cross section scales as1/R and the flow kinetic inductance asR². The missing term also equals twice the carrier flow kinetic energy divided byR, consistent with the centrifugal mechanical stress of the persistent flow. The physical measurement is total current-dependent support force, with ordinary elastic and condensation backgrounds removed.

Calibration measures current, so both closures agree at every allowed parameter value and geometry. It identifies kappa without testing the interpretation of force. The neutral instruction permits changing any implementation code; no force formula or diagnostic hint is present in the public image.

The oracle evaluates the explicit force balance. The private reference differentiates the constrained London potential through a complex virtual displacement, preserving material volume, winding and controlled field. Additional checks verify fluxoid conservation, fixed wire volume, and zero net controlled-field work around a closed(R,B) cycle. The shortcut fails the energy-cycle check as a physical consequence of the omitted kinetic contribution.

In `results/superconducting-ring-validation.json`, all256 noisy calibrations pass. Oracle hidden normalized RMSE is below0.000504; every shortcut group exceeds0.642, against the0.04 cutoff. Fixed current uncertainty is0.0002, independently of the unknown parameter and response. Calibration seed18017 and noise seed88310 are fixed. Local oracle/shortcut results are in `results/superconducting-ring-local-controls.json`.

Background on fluxoid sectors and magnetic/kinetic ring energies: Brandt and Clem, [Superconducting thin rings with finite penetration depth](https://arxiv.org/abs/cond-mat/0312497). The task uses its own explicitly specified lumped thin-wire constitutive law; it does not ask agents to reproduce that paper's finite-width numerical geometry.

This is a backup candidate, not a demonstrated hard task before its frozen agent screen. Reproduce science with `python scripts/validate_superconducting_ring.py` in the pinned scientific environment. `--generate` intentionally replaces both calibration copies.
