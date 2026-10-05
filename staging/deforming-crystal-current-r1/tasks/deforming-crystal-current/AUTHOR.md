# Deforming crystal current, revision1

The source computes the exact relaxed internal offset and laboratory polarization
of the specified polar insulator. Its directional derivative is also correct.
It interprets that polarization-density derivative through the instantaneous
face as the current in a lead attached to a deforming material electrode. This
is a complete improper polarization susceptibility, used as an approximation
to the measured electrical response. The error is in the physical readout, not
in its nonlinear equilibrium solver or differentiation.

## Apparatus, current and signs

Write x=s−s0, gamma=4, and h=h((FᵀF−I)/2). The internal equation is

    (K+gamma |x|²)x=h,
    D=(K+gamma |x|²)I+2 gamma xxᵀ,
    ds=D^(−1) dh.

D is positive with every eigenvalue at least K. The scalar radial root used by
the two forward models is therefore the unique global internal equilibrium.
The complete homogeneous free energy is an effective zero-macroscopic-field
constitutive input, including internal electrostatics. It is not a bare spring
model with the electrostatic energy of its stated charges omitted. The lattice
is externally held homogeneous. Thermal fluctuations of the macroscopic offset
are negligible, as specified publicly.

In units q0/a0², the polarization is P=−F s/J, where J=detF. The current area
vector of one upper reference cell face is a=J F^(−T)e3. The completed source
returns −a·dP. The amplifier instead measures conventional charge entering that
material electrode. The grounded weighting potential is material z divided by
the slab thickness. Every neutral cell has a positive center followed by the
negative center at offset s; summing the induced top-electrode charge gives
Q_top/q0 per reference lateral cell equal to s3, up to a fixed termination
constant. Thus the measured coefficient is ds3.

For M=F^(−1)G the difference is

    j_source − j_physical = (M s)3 − tr(M)*s3.

Both geometric factors and all internal relaxation are already computed
correctly by the source. Omitting only this conversion in the interpreted
observable is the intended physical approximation. Longitudinal drives have
M proportional to e3e3ᵀ, so their geometric terms cancel exactly.

The ideal electrodes are shorted, periodic lateral boundaries remove fringing,
and the macroscopic-thickness limit precedes the quasistatic response. Bounded
surface relaxation affects only a vanishing fraction of the sample's
charge-displacement response; no additional surface constitutive law is
introduced. The explicit complete neutral-cell termination avoids a hidden
polarization branch. The positive pair position c and the entire negative
position c+s stay inside their material cell. On the full F box, |h|<.223 gives
|x|<.279; c+s0=(.33,.31,.42) then has positive distances from every cell face.
The upper-triangular deformation has positive determinant throughout.

The output normalization uses fixed reference area. The physical current is
I/A_ref=epsilon*omega*(q0/a0²)*j*cos(omega*t) at leading order. No arbitrary
current-density area convention remains to be chosen. The charge and reference
length scales are known; only K in E0 units is fitted.

## Calibration and independent reference

There are18 distinct calibration settings: nine longitudinal stretches in
[.88,1.12] and both signs of longitudinal drive, each repeated16 times for288
readings. All records have fixed instrument sigma=.001, independent of K and
the response. The private true K is1.06. Data/noise seeds are428711 and428713.
Explicit --generate is the only data-regeneration route; public/private data
copies are identical.

For a positive longitudinal drive, x has only an axial component,
h3=.35(l²−1), and the response is .7l/(K+12x3²). The denominator's K derivative
is (K−12x3²)/(K+12x3²)>0: |x3|<=.1113 and12x3²<.149<K. Every positive record
therefore decreases strictly with K, and reversed drives yield the same squared
residuals. The noiseless aggregate objective has a unique minimum. Forty-one
parameter checks, including both endpoints, recover within8.38e−9;201-point
profiles agree with that proof. This does not assert uniqueness for arbitrary
noisy observations. Calibration agreement between the two controls is2.23e−16.

The verifier independently solves the three stationarity equations, rather
than the radial scalar root. It places both charges of every slab cell on a
mesh, solves the grounded Poisson equation, extracts the upper electrode flux,
and differences that charge after four nearby deformations. It does not use
an analytic current conversion or the forward model's internal Hessian.
Planar averaging is exact for the electrode total under lateral periodicity;
nonzero lateral Fourier components integrate to zero. An oblique slab's metric
factor cancels when potential is converted to total electrode charge.

Charge deposition preserves total charge and its first moment, so this
particular integrated Poisson observable is naturally mesh-exact up to
roundoff. Refinement should not be described as a demonstration that the mesh
resolves all microscopic electric fields. The reference shares the specified
constitutive energy, but independently solves equilibrium, the electrical
boundary problem and the response derivative. It is not an ab initio derivation
of the material free energy.

## Scientific and local controls

The three diagnostic groups use ordinary transverse or mixed drives, with
positive nonzero predictions from both controls. At the nominal fit, physical
signals are at least.3768 and source signals at least.1220. Across the full K
interval the minima remain.3377 and.0808. There is no zero-response scored
diagnostic. Four longitudinal anchors test exact shared behavior.

All256 noisy calibrations pass calibration and parameter gates. All256 oracle
controls pass every group; all256 completed shortcuts fail each of the three
diagnostics and pass the anchors. Maximum oracle error is.000242 and minimum
shortcut diagnostic error is.4888, versus the standard.04 normalized-RMS gate.
The nominal shortcut errors are.4947/.4891/.4954. Minimum full-K group gaps are
.3750/.3858/.4107. Maximum parameter error over the noise draws is.000254 and
maximum reduced chi-square1.221. Every noise outcome is preserved separately.

The28 scored reference cases agree within4.62e−12; combined grid/step refinement
changes them by1.57e−11, and nine-to-fifteen-cell slab changes by9.12e−12. Full
validation covers every64-corner deformation, three K values and all12 signed
rate-basis directions (2304 cases), plus64 random interior F/G/K controls.
Maximum reference disagreement is8.62e−12 and refinement3.30e−11. Because the
response is linear in G, basis tests cover the signed rate directions without
restricting an agent's numerical approach. These numerical checks are not a
rigorous interval bound on discretization error.

The two electrode charges sum to zero within1.07e−13. The source matches an
independent finite difference of its own polarization-density observable within
8.62e−13, confirming that it implements that approximation correctly. Signed
rate reversal, rate linearity, internal-state agreement and fixed spatial-frame
covariance pass. Rigid rotation gives zero physical current to roundoff. The
latter author-only check is outside the upper-triangular public schema and is
not a hidden scored control; the prototype preserves its source violation.

Local isolated verification gives oracle7/7 in.625 seconds and shortcut four
passes plus three intended failures in.606 seconds. The complete scientific
validator takes17.76 seconds. No model or Docker evaluation has occurred. The
public starter retains an ordinary unfinished fit and None default; the
completed source changes only fitting. Shared helpers are AST-identical. The
neutral unrestricted instruction,600/60-second limits and public-only image
boundary are unchanged.

## Scope, history and classification

The frozen eight-file feasibility package is copied byte-for-byte, including
its original author manifest, independent peer,305-path archive inventory and
all attempted variants. It preserves transverse-y examples whose source
crosses zero; the final diagnostics use healthy transverse-x/mixed examples.
The new package has one successful scientific execution and no failed local
control attempt. There was no repair to the physical equations after testing.

Six closest author documents were read in full. Piezoelectric-waves concerns a
bulk electrical constraint on acoustic propagation, terminal-current-noise
concerns capacitor-weighted stochastic tunneling, prestrained-solid concerns
initial stress, and actuator-frame-current concerns persistent-force channels.
Nematic boundary work and coherent-alloy compatibility are more distant
continuum analogues. Moving-frame and electrical-readout themes overlap, but
none uses this same moving-electrode polarization susceptibility. Alternative
Shuttleworth and micropolar ideas were not implemented; their rejection is
preserved in the prototype audit.

A failure retaining the polarization-density readout is a physical-model error.
A trajectory that identifies transported electrode charge but makes a sign,
normal, determinant, derivative, index or solver error is an implementation or
mathematical failure. The entire public trajectory must be read before assigning
causality. Scientific separation does not establish empirical model difficulty.

Primary background: Vanderbilt, *Berry-phase theory of proper piezoelectric
response*, Journal of Physics and Chemistry of Solids61,147–151(2000),
[full primary PDF](https://arxiv.org/pdf/cond-mat/9903137),
[DOI](https://doi.org/10.1016/S0022-3697(99)00273-5). The full five-page paper was
read, particularly sectionsIII.A–C. It supports the distinction between
polarization response and electrode current, and their longitudinal equality.
The classical cell potential and parameter choices here are our construction;
this is not a computation of electronic Berry phases or a named real material.

Reproduce from the repository root with:

```sh
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/deforming-crystal-current-r1/scripts/validate_deforming_crystal_current.py
```

Use `--generate` only to intentionally regenerate both calibration copies.
