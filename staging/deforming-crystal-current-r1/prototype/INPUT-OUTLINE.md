# Possible apparatus, not a task package

A polar insulating material has one neutral pair of effective charges, +q0 and
−q0, per reference cubic cell of side a0. Write fractional cell coordinates in
its fixed material basis. The positive center is at c=(.25,.25,.20), and the
negative center is at c+s. The material has no mobile carriers. The homogeneous
internal offset s is an equilibrium constitutive variable. The imposed affine
lattice deformation is r=a0 F R. All reference cells and their termination remain
attached to the same material throughout a measurement.

At zero macroscopic electric field the total cell free energy, in E0 units, has
the following dependence on x=s−s0, with s0=(.08,.06,.22):

    w(F,x) = K |x|²/2 + |x|⁴ − h(E)·x + w0(F),
    E = (FᵀF−I)/2,
    h(E) = (.60 E13+.20 E12,
            .50 E23−.10 E12,
            .45 E11+.30 E22+.70 E33+.20 E12).

Here w0 is the known elastic contribution independent of s; it does not affect
the equilibrium internal offset. This is the total homogeneous zero-field
material free energy, including internal electrostatic effects, rather than an
additional bare spring energy to which microscopic Coulomb energy is to be
added. External mechanical control holds the lattice deformation homogeneous.
Internal relaxation is complete at each instantaneous F. The positive unknown
K is measured in E0 and lies in [.8,1.2].

Consider a macroscopic slab with periodic lateral boundaries, N cells thick in
the reference third direction. Its lower and upper material faces are coated by
ideal conducting electrodes maintained at the same electric potential. There
is no applied voltage. The upper lead measures conventional electric current
entering the upper electrode. Ideal electrostatic screening and insulating
bulk transport are assumed. Take the macroscopic-thickness limit before the
quasistatic response; bounded surface relaxation layers do not change the bulk
constitutive state. The selected termination has the complete neutral cells
specified above, with no separate surface charge or surface constitutive law.

Around a fixed F, impose F(t)=F+epsilon G sin(omega t). G is a known dimensionless
matrix. In the small-amplitude limit followed by the quasistatic frequency
limit, report the signed cosine coefficient j defined by

    I_upper(t)/A_ref = epsilon*omega*(q0/a0²)*j*cos(omega*t)
                      + o(epsilon*omega),

where A_ref is the fixed reference electrode area. The electrodes follow the
material faces. The amplifier reads total lead current; strain and electric
current sign conventions are fixed by this definition.

The proposed F domain is upper triangular, diagonal entries [.88,1.12] and
three upper off-diagonal entries [−.12,.12]. G is upper triangular with entries
[−1,1]. For the derivative, epsilon is taken sufficiently small that the
instantaneous state stays on the same material branch. A future implementation
may choose an equivalent compact schema, but must publish all entries and units.
This outline specifies the physical experiment without a current conversion or
numerical solver recipe.
