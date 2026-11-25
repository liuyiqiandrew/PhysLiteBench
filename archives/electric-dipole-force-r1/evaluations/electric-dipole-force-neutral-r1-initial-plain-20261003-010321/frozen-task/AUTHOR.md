# Electric-dipole force, revision 1

This candidate tests which local field momentum transfers force to a pure electric dipole. The supplied predictor computes the full coherent Maxwell fields, the electric intensity gradient, and the full Poynting flux. It assigns the latter to radiation pressure. The apparatus instead fixes the force through the electric dipole interaction. The missing distinction survives both single-wave calibration and nontrivial crossed-TE interference calibration.

The exact neutral instruction is retained. The public README specifies an ideal point electric dipole, the passive dressed polarizability, time convention, coherent illumination and force detector. It does not state the force correction, require a solver, or restrict edits. The image contains only the public files. Known calibration uncertainties are fixed instrument noise, independent of the noiseless response. The hint is private unless the runner explicitly adds it.

## Physics

In the stated units and exp(-it) convention, define z_i=sum_j E_j* partial_i E_j. The force on a pure electric point dipole is

    F_i = 0.5 Re[alpha* z_i]
        = 0.5 Re(alpha) Re(z_i) + 0.5 Im(alpha) Im(z_i).

The shortcut keeps the first term and replaces 0.5 Im(z) by the full Poynting vector 0.5 Re(E cross H*). It is a complete, coherent local-pressure approximation, not an incorrect derivative or omitted interference field. A pure electric dipole samples the electric canonical momentum. The additional circulating field momentum does not produce the same dipole force.

The response alpha=s/(1-i*s/(6*pi)) obeys Im(alpha)=|alpha|^2/(6*pi) throughout the allowed s range. There is no material absorption. Radiation reaction is included once through alpha, and the independent stress calculation includes the entire radiated field. There is no magnetic dipole, electric-magnetic recoil, material interface, or source backaction. This differs from the archived dipole-recoil task, whose error was the interference recoil of simultaneous electric and magnetic scatterers.

For a single propagating wave both force laws coincide. The crossed-TE calibration also has a fixed electric polarization, so its electric spin density vanishes and the laws remain identical despite a nonzero intensity-gradient force. For equal-phase crossed TM waves with directions (+sin(theta),0,cos(theta)) and (-sin(theta),0,cos(theta)), the force at the origin along z is 2 Im(alpha) cos(theta)^3. The shortcut gives 2 Im(alpha) cos(theta). Hidden measurements include phase offsets, mixed complex polarizations, displaced detectors and rotated geometry within the same public bounds.

## Independent verification

The oracle evaluates the local force. The verifier independently integrates the time-averaged Maxwell stress of incident plus outgoing retarded dipole fields over a sphere, subtracting the incident-only stress. It does not call the oracle force law. Radius and angular-refinement checks, zero self-force, radiated power, lossless net power, rotations/translations, signed phases and full-domain corners validate the normalization and radiation-reaction treatment. Root peer review additionally computes force as the sum of the electric charge-gradient and oscillating-current magnetic Lorentz terms over 32 general three-dimensional illuminations.

Primary background: Bliokh, Bekshaev and Nori, [Extraordinary momentum and spin in evanescent waves](https://doi.org/10.1038/ncomms4300), Nature Communications 5,3300 (2014), especially Supplementary Note 3 ([author manuscript](https://arxiv.org/abs/1308.0547)). The task uses crossed propagating waves in free space; its reference is derived directly from Maxwell stress rather than an evanescent-interface formula.

## Calibration and controls

There are 144 calibration records, fixed sigma=5e-5, true response strength 1.1, calibration seed 36021 and noise seed 46021. Single-wave data include multiple amplitudes and polarizations; crossed-TE data include phases, positions and detector axes. The one-parameter objective is checked for a single minimum across the full public range at three separated true strengths. All public/private calibration copies are identical.

The unchanged standard checks require reduced calibration chi-square<1.5, parameter error<3%, and hidden group RMS error normalized by the true group RMS<0.04. Fixed measurement noise is below 0.16% of each hidden group's RMS; the 4% prediction tolerance is wider than both noise propagation and numerical error.

Nominal fitted strength is 1.1000094739852382, reduced chi-square 1.246081. Oracle hidden errors are at most 1.72e-5; shortcut errors are 0.531,0.614 and0.822. Across 256 independent noise samples, all parameter/calibration checks pass, all oracle hidden groups pass and every shortcut fails. The worst oracle hidden error is 7.70e-5; the minimum shortcut hidden error is 0.5305. The maximum parameter error is 3.86e-5 relative. Maxwell-stress agreement is 3.54e-15, radius variation 1.38e-14, calibration closure difference 5.56e-17, and full-domain corner error 7.67e-15.

Local isolated pytest: oracle 7/7; shortcut passes both public checks, private calibration and parameter recovery, then fails all three intended hidden groups. Docker controls and Luna screens are scheduled by the root runner after source freeze. These scientific controls do not establish model difficulty; no Luna outcome is claimed here.

Reports: [science](../../results/electric-dipole-force-validation.json), [local controls](../../results/electric-dipole-force-local-controls.json), [source provenance](../../results/electric-dipole-force-source-provenance.json), [root physics review](../../results/root-hardening-physics-review.json), and [public-input review](../../results/neutrality-optical-actuator-input-audit.json).

Reproduce scientific validation with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_electric_dipole_force.py`. Add `--generate` only to intentionally regenerate the frozen public/private data together.
