# Two-isotope capillary transmission

An infinite capillary has impermeable walls at the points of a homogeneous
Poisson process of unknown intensity c in [0.3,1.4] per unit length. Fabricate
a large batch of independent capillaries. Each isotope diffuses in one
dimension with variance 2Dt in the absence of walls. For a given isotope
the faces at the left and right ends of **every pore** can each absorb the
spin on contact (D) or reflect it (N). The coating code lists the left then
right face; allowed codes are DD, DN and ND. All other relaxation is negligible.
Length and time are numerical values in fixed instrument units.

In one sample let K_i(t;x,y) be the surviving number density at y at time t
after releasing unit mass of isotope i at x. Across a wall the transmission
is zero. Detect with point receivers (density per unit length). Take the
infinite-spin limit at fixed walls and average over samples with equal
weight. Write this average as ⟨ ⟩.

For calibration use one isotope with DD walls and measure the log attenuation
of the same-point signal. For `mode="single"` return
lim[-log ⟨K_A(t;0,0)⟩]/t^(1/3) as t tends to infinity.
The diffusivity input is `D_A`.

For `mode="pair"` set both diffusivities and both coating codes from the
input. The instrument uses a reference length for those settings. In a
preliminary scan let L be the full wall-to-wall length of the pore containing
0 and weight each sample by B(t)=K_A(t;0,0) K_B(t;0,0). Define

    w = lim_{t→∞} ⟨L B(t)⟩ / [t^(1/3) ⟨B(t)⟩].

For each subsequent t release isotopes A and B at positions x_i =
left_i × t^(1/6). Their point receivers are at y_i =
w × t^(1/3) + right_i × t^(1/6). Record the product of the two densities on
each sample, then average. Report the limit of the normalized product

    ⟨K_A(t;x_A,y_A) K_B(t;x_B,y_B)⟩
    -----------------------------------      as t → ∞ .
    ⟨K_A(t;0,w t^(1/3)) K_B(t;0,w t^(1/3))⟩

The four offsets are signed. Use the same w, coatings and diffusivities in
the denominator. The wall distribution is stationary in space. Diffusion of
the two isotopes is independent on a given sample.

Pair inputs contain `D_A`, `D_B` (each in [0.2,2]), `coat` (two
coating codes), `left` and `right` (each a two-element list of offsets in
[-2,2]). All pairs are in A,B order. Calibration rows contain `input` and
`value` rounded to eight decimal places. Implement `Model.fit(records)`
returning self with the fitted `intensity`. Implement `Model.predict(inputs)`
returning a finite NumPy array of shape `(len(inputs),)` in input order,
including an empty array for empty input.
