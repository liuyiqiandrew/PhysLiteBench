# Echo attenuation in narrow capillaries

Each sample is an infinite one-dimensional capillary. Three independent,
homogeneous Poisson point processes mark defects on the line. The first species
of defect extinguishes spins of isotope A, the second extinguishes spins of
isotope B, and the third extinguishes either. They have respective intensities
a, b, and c per unit length. The first two are controlled. The third is the
same unknown intensity in every run. The defects remain fixed during all
measurements on one sample. Average over independently fabricated samples.

Each shot releases many noninteracting spins of one isotope at its source.
A spin follows unbiased Brownian motion of diffusivity D (variance 2Dt).
On first hitting an active defect its transverse signal is lost for the
remainder of that shot. Other defects have no effect. Every surviving spin
has unit signal. At time t the coil measures the surviving fraction. Take the
infinite-spin limit in each sample before averaging over samples. Time and
length are numerical values in fixed instrument units.

In single-coil mode release A at 0. Write M_A for that sample's coil reading.
The receiver reports the large-time limit of minus the natural logarithm of
the sample mean ⟨M_A⟩, divided by t^(1/3).

In two-coil mode release A at 0 and B at s t^(1/3) in the same capillary, using
independent spins. At each t, subtract the corresponding sample-ensemble mean
from each coil's reading. The instrument multiplies the two centered readings
and averages over samples. It reports

    lim_{t→∞} -log ⟨(M_A - ⟨M_A⟩)(M_B - ⟨M_B⟩)⟩ / t^(1/3).

The source placement is reset for each observation time. Both coils read at
that same time. Averages ⟨ ⟩ give equal weight to each capillary.

Single inputs have keys `mode="single"`, `a`, and `D_A`. Two-coil inputs
have `mode="pair"`, `a`, `b`, `D_A`, `D_B`, and `s`. Each controlled
intensity is in [0.15, 2.5], each diffusivity in [0.15, 2.5], and s is in
[0, 18]. The common intensity c is in [0.3, 1.4]. Calibration rows contain
`input` and the measured rate `value` rounded to eight decimal places.

Implement `Model.fit(records)` returning self with fitted float `intensity`.
Implement `Model.predict(inputs)` returning the rates in input order as a
finite NumPy array of shape `(len(inputs),)`, including empty input.
