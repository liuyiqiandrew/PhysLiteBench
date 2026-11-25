# Spin transport on a periodic line

Noninteracting spin-1/2 carriers move on a one-dimensional periodic coordinate of circumference 2*pi mm. Their particle density is uniform. Spin-independent elastic scattering randomizes momentum on a time scale much shorter than any reported time. In the controlled diffusion limit, the positional diffusion coefficient D is constant and unknown between 0.04 and 0.3 mm²/s. There is no drift, localization, interaction, or additional spin relaxation.

A spatially uniform linear spin-orbit coupling acts during free motion. Over a signed unwrapped displacement dx, its spin rotation is about the fixed +y spin axis through angle Q*dx, with Q=1 rad/mm. Reversing displacement reverses this angle. Momentum-scattering events do not themselves rotate spin. The mean free path is negligible compared with 1/Q and all prepared spatial wavelengths. Use this strict diffusion limit, retaining the stated coherent spin rotation during motion. Spin components are laboratory Bloch-vector components; a positive rotation about +y sends +z toward +x. Periodic winding uses the same rotation law.

At preparation, the mean spin texture is

    S(x,0) = cosine*cos(mode*x) + sine*sin(mode*x),

with x in mm and mode an integer 0 through 3. `cosine` and `sine` are three-element real vectors in spin component order x,y,z, chosen so |S|<=1 everywhere. For mode 0, sine is zero. The spin state at each position is prepared with that local Bloch vector; the uniform particle density is independent of it. The spin-orbit coupling and scattering remain unchanged after preparation.

At `time` between 0 and 12 s, the detector reports the cosine or sine Fourier coefficient of one spin component at the prepared mode. Component indices are 0,1,2. For nonzero mode, a coefficient is twice the spatial average of the component times the specified sine/cosine; at mode 0 the cosine coefficient is the ordinary spatial average. Measurements are number-weighted and collect every carrier.

Every experiment includes `time`, `mode`, `cosine`, `sine`, `component`, and `quadrature` (`"cosine"` or `"sine"`). Calibration uses spatially uniform x- or z-polarized preparations and measures their mean spin. Errors are independent Gaussian with listed standard deviations; one D applies throughout.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Store D in `Model.diffusivity`, in mm²/s. Records in `data/calibration.json` contain `input`, `value`, and `sigma`.
