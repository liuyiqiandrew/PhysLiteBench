# Understanding the physics tasks

## The common problem: a good fit can hide a wrong assumption

Imagine calibrating an instrument under one set of conditions, obtaining an excellent fit, and then using that model for a new experiment. What justifies trusting the prediction? Recovering the right parameter is part of the answer. The equations must also describe the physical constraints of the new experiment.

The tasks explore this distinction. Each gives an agent an apparatus description, noisy calibration measurements, and a partly implemented model. The agent must infer the unknown apparatus parameters and predict measurements for other allowed preparations of the same apparatus.

The supplied predictor contains a simplifying physical assumption. The calibration experiments happen to be a special case in which that assumption gives the correct answer. Completing the fit can therefore produce a model that passes calibration while retaining the physical mistake.

| Task | What is measured? | What is fitted? | What the shortcut misses |
|---|---|---|---|
| Two spin probes | Probability of a joint measurement outcome | Magnetic-noise intensity | Correlations produced by a shared field |
| Two gas chambers | Temperatures as functions of time | Conductance to a heat bath | Work and pressure changes caused by a movable piston |
| Two dissolved salts | Concentration profiles as functions of time | An ionic diffusion coefficient | The electric field shared by all the ions |
| Hall bar | Longitudinal current density | Carrier mobility | The transverse field imposed by insulating sidewalls |
| Moving inductor | Current at a requested time | Coil resistance | Voltage from motion of the magnetic core |
| Effusive beam | Velocity-component cumulative probabilities | Reservoir temperature | Faster particles cross the aperture more often |
| Real-gas expansion | Heating energy or final temperature | Molar heat capacity | Interaction energy changes with volume |
| Induced dipole | Dipole moment or force | Polarizability | The internal work needed to induce polarization |
| Radiative plates | Net radiative heat flux | Surface emissivity | Repeated reflections between gray surfaces |
| Magnetic equilibrium | Magnetization per spin | Ferromagnetic coupling | Equilibrium selects a free-energy minimum |

The **calibration loophole** is this restricted choice of experiments. Even noiseless measurements of the same preparations would not distinguish the shortcut from the correct model. New preparations reveal the difference.

The explanations below assume undergraduate physics. Each starts with a physical picture, develops the equations, and works through an experiment that exposes the mistake. Model results and trajectory analysis are recorded separately in [DASHBOARD.md](DASHBOARD.md).

## 1. Two spin probes in the same fluctuating magnetic field

### Start with two arrows that turn together

A spin-1/2 state can be represented by a Bloch vector. A magnetic field along the $z$ direction rotates its transverse component in the $xy$ plane. If the field fluctuates, the accumulated rotation angle fluctuates too.

Now place two otherwise noninteracting probes in the **same spatially uniform field**. In any one experimental repetition, both experience the same field history and acquire the same angle. Across repetitions, that angle varies randomly.

For two spins initially pointing along $+x$, the picture is two arrows turning together. Eventually their common direction becomes unpredictable, but their relative direction remains fixed. A model that gives each arrow an independent random rotation loses that relationship.

The experiment prepares each spin using an ideal rotation, waits, applies readout rotations, and measures whether both spins point along $+z$. Readout rotations allow this detector to measure other spin directions as well. The unknown parameter is the field's noise intensity, $\gamma$, in inverse seconds.

### From field fluctuations to a random phase

In angular-frequency units, the longitudinal field is $\xi(t)$. Its Hamiltonian and noise covariance are

$$
H(t)=\frac{\hbar\xi(t)}{2}
\left(\sigma_z^{(1)}+\sigma_z^{(2)}\right),
\qquad
\langle\xi(t)\xi(s)\rangle=2\gamma\delta(t-s).
$$

Here $\sigma_z^{(i)}$ acts on probe $i$, and the brackets denote an average over repetitions. The noise is Gaussian with zero mean. Over a wait of duration $t$, both probes acquire the phase

$$
\phi=\int_0^t\xi(s)\,ds,
\qquad
\operatorname{Var}\phi=2\gamma t.
$$

The Gaussian characteristic function gives

$$
\langle e^{ik\phi}\rangle=e^{-k^2\gamma t}.
$$

Thus a single spin's transverse signal decays as $e^{-\gamma t}$. This loss of a reproducible transverse direction is dephasing. It does not require a change in the spin's $z$ population.

### Where the shortcut enters

For a particular phase $\phi$, let $p_i(\phi)$ be the probability that probe $i$ produces the desired readout. Conditional on this phase, the joint probability is $p_1(\phi)p_2(\phi)$. The observed probability requires averaging over the shared phase:

$$
P_{++}=\langle p_1(\phi)p_2(\phi)\rangle.
$$

The supplied predictor instead multiplies the separately averaged probabilities:

$$
P_{++}^{\mathrm{shortcut}}
=\langle p_1(\phi)\rangle\langle p_2(\phi)\rangle.
$$

Each marginal probability can be correct while their product is wrong. For matched preparations and readouts, some field histories make both desired outcomes more likely; other histories make both less likely. Averaging each response separately removes that common variation.

This is a classical correlation caused by a shared environment. Every field realization applies local rotations to a product state, and averaging produces a separable mixture. No entanglement or direct spin–spin interaction is needed. The two measurement outcomes also need not be identical in a given repetition.

### Why calibration cannot catch it

Every calibration preparation leaves one probe along the $z$ axis. A rotation about $z$ leaves that spin's physical state unchanged. Its readout probability is therefore independent of $\phi$, even if a later readout rotation makes that probability different from zero or one.

If this constant probability is $c$, then

$$
\langle c\,p_2(\phi)\rangle
=c\langle p_2(\phi)\rangle.
$$

The correct joint average and the shortcut coincide. The other probe still dephases, so its time dependence identifies $\gamma$. Calibration measures the noise strength while leaving the distinction between common and independent noise unresolved.

### A worked experiment: prepare and measure both along x

Prepare both spins along $+x$, let them evolve, and measure both along $x$. Operationally, the final measurement uses rotations that map the $x$ basis onto the detector's $z$ basis.

For a fixed phase, each return probability is

$$
p(\phi)=\frac{1+\cos\phi}{2}.
$$

The correct prediction is $\langle p^2\rangle$. Using $\cos^2\phi=(1+\cos2\phi)/2$ gives

$$
P_{++}
=\frac38+\frac12e^{-\gamma t}+\frac18e^{-4\gamma t}.
$$

The shortcut predicts

$$
P_{++}^{\mathrm{shortcut}}
=\left(\frac{1+e^{-\gamma t}}{2}\right)^2.
$$

Both start at one. At long times, each spin individually returns with probability $1/2$, but the joint probabilities approach different limits:

$$
P_{++}\longrightarrow\frac38,
\qquad
P_{++}^{\mathrm{shortcut}}\longrightarrow\frac14.
$$

The difference is exactly the variance of the conditional return probability:

$$
P_{++}-P_{++}^{\mathrm{shortcut}}
=\operatorname{Var}[p(\phi)]
=\frac18\left(1-e^{-2\gamma t}\right)^2.
$$

The missing quantity is the correlation between the probes. Refitting $\gamma$ cannot repair the incorrect long-time limit.

See [the benchmark guide](docs/BENCHMARK.md) for task interfaces and grading, and [the reference solution](tasks/qubit-control/solution/model.py) for implementation details.

## 2. Two gas chambers separated by a free piston

### A movable divider inside a rigid container

A rigid vessel contains two chambers, each holding $n$ moles of the same monatomic ideal gas. A freely moving, frictionless piston separates them. The piston is thermally insulating, but it can transfer mechanical energy through work.

Each chamber exchanges heat with a bath at temperature $T_b$ through an identical thermal link of unknown conductance $G$. A separate link with known conductance $K$ connects the gases, bypassing the piston. The measured quantities are the two temperatures, $T_1(t)$ and $T_2(t)$.

The piston rapidly establishes equal pressures. That requirement determines how the fixed total volume is divided: the hotter gas occupies more space. As the temperatures change, the piston can move, and one gas can do work on the other.

The key distinction is between **equal pressures at a given instant** and **a pressure that remains constant over time**. The first follows from mechanical equilibrium. The rigid vessel has no external pressure reservoir to impose the second.

### Begin with heat flow and the first law

Define $q_i$ as the rate of heat entering chamber $i$. For the other chamber $j$,

$$
q_i=-G(T_i-T_b)-K(T_i-T_j),\qquad j\ne i.
$$

Both conductances have units of W/K. A gas hotter than the bath loses heat through its bath link; a gas hotter than its neighbor also loses heat through the connecting link.

Let $V_i$ be chamber $i$'s volume and $P$ the common pressure. The first law is

$$
nC_v\dot T_i=q_i-P\dot V_i,
\qquad C_v=\frac32R.
$$

Expansion costs internal energy; compression supplies it. The heat capacities here are molar, so the factor $n$ is required.

Differentiating $PV_i=nRT_i$ yields

$$
P\dot V_i+V_i\dot P=nR\dot T_i.
$$

Substitution into the first law gives an equivalent and useful form:

$$
nC_p\dot T_i=q_i+V_i\dot P,
\qquad C_p=C_v+R=\frac52R.
$$

This equation explains precisely when the familiar constant-pressure heat capacity applies. The supplied predictor uses $nC_p\dot T_i=q_i$ throughout. Its omitted term is $V_i\dot P$.

### What determines the shared pressure?

Write the fixed vessel volume as $V=V_1+V_2$ and define the temperature sum $S=T_1+T_2$. The two ideal-gas equations give

$$
P=\frac{nRS}{V},
\qquad
V_i=V\frac{T_i}{S}.
$$

The pressure follows the temperature sum. Adding the two first-law equations makes the internal piston work cancel, since $\dot V_1+\dot V_2=0$. Heat exchanged between the gases cancels as well:

$$
nC_v\dot S=q_1+q_2=-G(S-2T_b).
$$

Consequently,

$$
S(t)=2T_b+(S_0-2T_b)e^{-Gt/(nC_v)}.
$$

The total gas energy changes only through the bath links. Moving the piston redistributes energy between the chambers without doing work on the outside world.

### Why the calibration trajectories really are isobaric

Calibration always starts with

$$
T_1(0)+T_2(0)=2T_b.
$$

For example, with $T_b=293\,\mathrm K$, one calibration starts at $298\,\mathrm K$ and $288\,\mathrm K$. The formula for $S(t)$ then gives $S(t)=2T_b$ at all times. The common pressure is constant, and the term dropped by the shortcut is exactly zero.

The piston can still move: as the hotter gas cools and the colder gas warms, their volumes approach equality. The associated work is exactly why $C_p$ is appropriate on these trajectories.

To see what the data identify, define the temperature difference $\Delta=T_1-T_2$. At constant $S$,

$$
nC_p\dot\Delta=-(G+2K)\Delta.
$$

The measured difference decays exponentially with rate $(G+2K)/(nC_p)$. Since $K$ is known, this determines $G$. Every calibration curve can be fitted correctly without checking how the model behaves when the total gas energy changes.

### A worked experiment: both chambers start warm

Now start both chambers at $313\,\mathrm K$, with the bath still at $293\,\mathrm K$. Symmetry keeps their temperatures equal and the piston at the midpoint. Each chamber has constant volume throughout cooling. The connecting link carries no heat because the temperatures are equal.

The first law immediately gives

$$
T_1(t)=T_2(t)
=293\,\mathrm K+20\,\mathrm K\,e^{-Gt/(nC_v)}.
$$

The shortcut puts $C_p$ in this exponential. It therefore predicts a relaxation time that is too long:

$$
\frac{\tau_{\mathrm{shortcut}}}{\tau_{\mathrm{correct}}}
=\frac{nC_p/G}{nC_v/G}=\frac53.
$$

There is no piston work in this preparation, so all the heat lost reduces internal energy. The shortcut assigns the gas the larger heat capacity associated with a different process and predicts cooling too slowly.

### Extending the solution to arbitrary starting temperatures

When the temperature sum changes, the pressure change also affects the temperature difference. Subtracting the two chamber equations gives

$$
nC_p\dot\Delta
=-(G+2K)\Delta+nR\frac{\Delta}{S}\dot S.
$$

Together with the solution for $S(t)$, this integrates to

$$
\Delta(t)=\Delta_0
e^{-(G+2K)t/(nC_p)}
\left(\frac{S(t)}{S_0}\right)^{2/5},
\qquad
T_{1,2}(t)=\frac{S(t)\pm\Delta(t)}{2}.
$$

The extra factor describes the effect of changing shared pressure. It becomes one for every calibration preparation, which is why those data hide the missing physics.

See [the benchmark guide](docs/BENCHMARK.md) for task interfaces and grading, and [the independent sum-and-difference solution](tasks/thermal-bodies/tests/reference.py) for implementation details.

## 3. Two salts sharing one anion

### Diffusion creates an electrical response

Consider an isothermal, dilute solution in a one-dimensional cell. It contains two fully dissociated salts, AC and BC: the mobile species are $A^+$, $B^+$, and their shared anion $C^-$. There are no chemical reactions or fluid flow, and the ends block all ions.

Their diffusion coefficients are

$$
D_A=D,\qquad D_B=4D,\qquad D_C=2D,
$$

where the scale $D$ is unknown. The measurements give the two cation concentration profiles over time.

Begin with only AC present. The anion diffuses faster than the cation. If each followed its own concentration gradient without electrical forces, charge would separate. A small charge redistribution creates an electric field that slows the faster ion and helps the slower one. The two species then spread together at an effective rate called **ambipolar diffusion**.

Adding BC changes this balance. All three species respond to one electric potential, and the anions have no memory of which salt supplied them. A binary-salt diffusion law must therefore be reexamined when both salts are present.

### The bulk constraints

The task resolves transport on scales where the bulk is approximately electroneutral. Fast charge-relaxation processes and thin boundary charge layers are outside the model. Define

$$
a=c_A,\qquad b=c_B,\qquad c_C=a+b.
$$

The last equality is the bulk electroneutrality constraint. In this approximation an internal electric field still enforces the coupled motion; setting the resolved net charge to zero does not justify discarding that field.

Let $\Phi$ be the electric potential and define its dimensionless form as $u=e\Phi/(k_BT)$, with $e>0$. The physical electric field is $E=-\partial_x\Phi$. The Nernst–Planck flux of species $i$ is

$$
J_i=-D_i\left(\partial_x c_i+z_i c_i\,\partial_xu\right),
$$

where $z_A=z_B=+1$ and $z_C=-1$. The first term describes diffusion; the second describes electrical drift. These are molar fluxes when the concentrations are in molar units.

In the stated bulk limit, blocking, insulating ends imply zero electric current:

$$
J_A+J_B-J_C=0.
$$

Individual ion fluxes can be nonzero while their charge-weighted sum vanishes.

### Deriving the field shared by the three species

Substitute the diffusivities and $c_C=a+b$ into the zero-current condition:

$$
0=D\left[\partial_xa-2\partial_xb
-(3a+6b)\partial_xu\right].
$$

The common potential gradient is therefore

$$
\partial_xu=
\frac{\partial_xa-2\partial_xb}{3a+6b}.
$$

Both concentration profiles determine the field. In turn, that field enters both cation fluxes:

$$
J_A=-D\left(\partial_xa+a\partial_xu\right),
\qquad
J_B=-4D\left(\partial_xb+b\partial_xu\right).
$$

The concentrations evolve through conservation, $\partial_ta=-\partial_xJ_A$ and $\partial_tb=-\partial_xJ_B$, with zero flux at the cell ends. Thus each species can influence the other's evolution even without a chemical reaction.

### Why pure-salt calibration supports the shortcut

For pure AC, set $b=0$. The field becomes $\partial_xu=(\partial_xa)/(3a)$, and substitution into $J_A$ gives

$$
J_A=-\frac43D\,\partial_xa.
$$

The cation obeys an ordinary scalar diffusion equation with effective coefficient $D_{\mathrm{AC}}=4D/3$.

For pure BC, set $a=0$. The field becomes $\partial_xu=-(\partial_xb)/(3b)$, giving

$$
J_B=-\frac83D\,\partial_xb,
\qquad D_{\mathrm{BC}}=\frac83D.
$$

These are the usual binary ambipolar coefficients $2D_+D_-/(D_++D_-)$. They are correct descriptions of their respective pure salts.

Calibration contains only pure AC or pure BC preparations. The supplied predictor applies these two scalar diffusion equations independently. It can fit both sets of measurements and recover the same correct $D$. The unsupported step is continuing to use the two independent equations when both salts occupy the cell.

### A worked experiment: an initially uniform ion starts moving

Prepare a varying concentration of A while keeping B uniform:

$$
a(x,0)=a_0+\epsilon\cos(\pi x/L),
\qquad b(x,0)=b_0>0,
\qquad 0<\epsilon<a_0.
$$

Here $L$ is the cell length. The cosine has zero slope at the ends, consistent with the blocking boundaries. Electroneutrality sets the initial anion concentration to $a(x,0)+b_0$.

The shortcut predicts that B remains uniform forever: ordinary diffusion has nothing to act on when $\partial_xb=0$.

The coupled model instead gives, at the initial instant,

$$
\partial_xu=\frac{\partial_xa}{3a+6b_0},
\qquad
J_B=-\frac{4Db_0}{3a+6b_0}\,\partial_xa.
$$

B has no concentration gradient, but it feels the electric field generated by the other ions' tendency to separate. Initially, it flows from the high-A region toward the low-A region. Because this flux varies with position, it changes B's concentration.

For a small modulation, $\epsilon\ll a_0$, linearizing the denominator makes the initial response explicit:

$$
\left.\partial_tb\right|_{t=0}
\simeq
-\frac{4Db_0}{3a_0+6b_0}\,
\epsilon\left(\frac{\pi}{L}\right)^2\cos(\pi x/L).
$$

B starts decreasing where A is high and increasing where A is low. Its total amount remains constant because no ions cross the boundaries. This nonuniformity is transient; both species eventually become uniform in the closed cell.

The smooth, small-modulation example makes the mechanism easy to calculate. The hidden tests use sampled cosine profiles, interpreted as linear between mesh points, and include larger modulations that the reference model evolves numerically.

The failure comes from combining two individually valid binary reductions without enforcing their shared electrical constraint. Changing the fitted $D$ cannot make independent diffusion move an initially uniform B profile.

See [the benchmark guide](docs/BENCHMARK.md) for task interfaces and grading, and [the independent three-ion reference](tasks/reaction-diffusion/tests/reference.py) for the numerical implementation.

## 4. A Hall bar with insulating sides

A long bar contains one isotropic population of classical positive carriers, of
known density \(n\) and charge \(q\). Its unknown mobility is \(\mu\), in
\(\mathrm{m^2/(V\,s)}\). A source fixes the longitudinal field \(E_x\), in V/m,
while a magnetic field \(B\), in tesla, is applied perpendicular to the bar.
Measurements concern the uniform central section after charge redistribution
has reached steady state. The sides are electrically insulating.

For this Drude material, the conductivity tensor is

$$
\boldsymbol{\sigma}
=\frac{nq\mu}{1+(\mu B)^2}
\begin{pmatrix}1&\mu B\\-\mu B&1\end{pmatrix}.
$$

The shortcut returns \(\sigma_{xx}E_x\), implicitly setting \(E_y=0\).
Insulating sides instead impose \(j_y=0\). Charge accumulates until
\(E_y=\mu B E_x\), giving \(j_x=nq\mu E_x\). Thus the measured longitudinal
current is independent of \(B\) in the stated single-carrier model.

Calibration uses only \(B=0\), so both expressions coincide and their slope
identifies mobility. Hidden experiments apply fields of both signs. The
independent reference solves the transverse constraint using the full tensor.
[Apparatus and API](tasks/hall-bar/environment/README.md).

## 5. An inductor with a prescribed moving core

A coil has known, reversible flux linkage \(\lambda=L(x)I\), where
\(L(x)=0.6+0.4x\) henry and \(x\) is a dimensionless core coordinate.
Resistance \(R\), in ohms, is unknown. The actuator prescribes
\(x(t)=x_0+A\sin(\omega t)\), and a source maintains constant terminal voltage
\(V\). Saturation, hysteresis, and parasitic capacitance are excluded.

Faraday's law gives

$$
V=RI+\frac{d(LI)}{dt}
  =RI+L\dot I+I\dot L.
$$

The shortcut solves \(V=RI+L(x(t))\dot I\). Merely substituting the changing
inductance into a fixed-core equation misses voltage generated by motion.
Calibration holds the core fixed, making \(\dot L=0\); ordinary current
transients identify \(R\) exactly under either model. Hidden experiments move
the core, including a source-free decay from nonzero initial current.

The solution integrates flux, \(\dot\lambda=V-R\lambda/L\); the independent
reference integrates current with the \(I\dot L\) term. With both \(R\) and
\(V\) set to zero in an author check, motion conserves \(LI\), not \(I\).
[Apparatus and API](tasks/moving-inductor/environment/README.md).

## 6. An effusive beam counts aperture crossings

An equilibrium argon reservoir supplies atoms through an aperture smaller than
their mean free path into vacuum. The outward normal is \(+z\). The detector
counts every crossing once, with no speed, angle, or residence-time weighting.
The unknown parameter is reservoir temperature \(T\), in kelvin.

Inside the reservoir, Cartesian velocities have independent Gaussian densities
with variance \(s^2=k_BT/m\). A velocity class contributes crossing events in
proportion to its positive \(v_z\). The normalized outward normal density and
CDF are therefore

$$
f_{\mathrm{cross}}(v_z)=
\frac{v_z}{s^2}e^{-v_z^2/(2s^2)},\qquad
F_z(v)=1-e^{-v^2/(2s^2)}\quad(v\geq0).
$$

The CDF is zero for negative thresholds. The transverse components retain their
Gaussian distributions because the extra \(v_z\) factor does not depend on
\(v_x\) or \(v_y\).

Calibration measures only transverse CDFs and identifies \(T\). The shortcut
uses these correctly but treats the outward normal velocity as a half-normal,
with CDF \(2\Phi(v/s)-1\). Hidden experiments measure normal-velocity CDFs.
Independent quadrature of the flux-weighted density checks the analytic answer.
[Apparatus and API](tasks/effusive-beam/environment/README.md).

## 7. Free expansion of a thermodynamically consistent real gas

One mole obeys the supplied van der Waals equation
\(P=nRT/(V-nb)-an^2/V^2\), with known \(a\) and \(b\). Its unknown molar
heat capacity \(C_v\), in J/(mol K), is constant. Thermodynamic consistency
requires

$$
\left(\frac{\partial U}{\partial V}\right)_T
=T\left(\frac{\partial P}{\partial T}\right)_V-P
=\frac{an^2}{V^2},
\qquad
U=nC_vT-\frac{an^2}{V}+\text{constant}.
$$

Calibration supplies fixed-volume heating measurements. Their heat increments
are \(nC_v(T_1-T_0)\), so the shortcut \(U=nC_vT\) identifies the same heat
capacity while omitting all volume dependence.

Hidden experiments remove a partition between the gas and vacuum in an
insulated rigid vessel. There is no external heat or work, so \(U\) is conserved:

$$
T_1=T_0+\frac{an}{C_v}\left(\frac1{V_1}-\frac1{V_0}\right).
$$

Expansion cools the gas; the shortcut predicts unchanged temperature. All
allowed endpoints lie above this material's critical temperature and outside
its excluded volume. An independent reference integrates the volume derivative
of energy. Grading normalizes by the cooling magnitude rather than absolute
temperature. [Apparatus and API](tasks/real-gas-expansion/environment/README.md).

## 8. The internal energy of an induced dipole

A small isotropic neutral particle has no permanent dipole and responds
reversibly as \(p=\alpha E\), with unknown polarizability \(\alpha\).
External sources maintain a static field whose on-axis profile is
\(E(x)=E_0+gx\). Particle interactions and disturbance of the source are ignored.

The energy includes both polarization work and interaction with the field:

$$
U(p,x)=\frac{p^2}{2\alpha}-pE(x).
$$

Minimizing over \(p\) gives \(p=\alpha E\),
\(U_{\mathrm{eq}}=-\alpha E^2/2\), and
\(F_x=-dU_{\mathrm{eq}}/dx=\alpha E g\). The shortcut substitutes the induced
moment into the permanent-dipole energy \(-pE\) alone, producing twice the force.

Uniform-field dipole measurements identify \(\alpha\), while both models predict
zero force there. Hidden experiments use nonzero gradients and measure force.
The laboratory units are kV/m for field, mm for position, \(10^{-21}\) C m for
dipole, \(10^{-24}\) C m²/V for polarizability, and \(10^{-15}\) N for force;
these make \(p=\alpha E\) and \(F_x=\alpha Eg\) numerically consistent.
The reference differentiates the minimized full energy.
[Apparatus and API](tasks/induced-dipole/environment/README.md).

## 9. Radiation reflected between two gray plates

Two opaque diffuse gray plates face each other across a far-field vacuum gap
with unit mutual view factor. Thermostats maintain temperatures \(T_1,T_2\), in
kelvin. Plate 1 has unknown emissivity \(\epsilon_1\); each opposing panel has
known emissivity \(\epsilon_2\).

The outgoing radiation \(J_i\) includes emission and reflected irradiation:

$$
J_1=\epsilon_1\sigma T_1^4+(1-\epsilon_1)J_2,\qquad
J_2=\epsilon_2\sigma T_2^4+(1-\epsilon_2)J_1.
$$

Solving gives the net flux leaving plate 1, in W/m²,

$$
q=J_1-J_2=
\frac{\sigma(T_1^4-T_2^4)}
{1/\epsilon_1+1/\epsilon_2-1}.
$$

The shortcut uses single-pass emission and absorption,
\(\epsilon_1\epsilon_2\sigma(T_1^4-T_2^4)\).
Calibration uses black opposing panels, \(\epsilon_2=1\), where both expressions
coincide and determine \(\epsilon_1\). Hidden panels reflect some radiation,
allowing repeated encounters before absorption. The independent reference
solves the two radiosity equations directly.
[Apparatus and API](tasks/radiative-plates/environment/README.md).

## 10. Magnetic equilibrium is more than a root

An infinite-range ferromagnet has classical spins \(s_i=\pm1\) and Hamiltonian
\(H=-J(\sum_i s_i)^2/(2N)-h\sum_i s_i\). The unknown positive coupling \(J\),
field \(h\), and temperature \(T\) share one energy unit, with \(k_B=1\).
Take \(N\to\infty\) and full canonical equilibrium at every nonzero field;
metastable preparation and hysteresis are excluded.

Stationary magnetizations satisfy

$$
m=\tanh\left(\frac{Jm+h}{T}\right).
$$

Calibration temperatures exceed every allowed \(J\), giving one root and
identifying the coupling. The shortcut correctly solves this equation from an
initial guess of zero. At lower temperatures, however, several roots can exist.
Equilibrium minimizes the free energy per spin,

$$
f(m)=-\frac J2m^2-hm+
T\left[
\frac{1+m}{2}\log\frac{1+m}{2}
+\frac{1-m}{2}\log\frac{1-m}{2}
\right].
$$

Hidden experiments use low temperatures and small fields of both signs.
The shortcut returns unstable central roots with excellent equation residuals.
The physical solution selects the stable branch aligned with the field;
the independent reference minimizes \(f(m)\) globally. These controls isolate
thermodynamic state selection from numerical root convergence.
[Apparatus and API](tasks/magnetic-equilibrium/environment/README.md).

## Optional additional candidates

The following six revision-1 tasks provide alternatives if the first seven
additions do not show the desired agent difficulty. Their local scientific
controls and [independent review](docs/candidates/backup-physics-review.md) pass;
that does not establish their unhinted or hinted agent success rates.

### 11. Position-dependent drag in an isothermal channel

Brownian particles move around a periodic channel of length \(L=10\) micrometers.
Its accessible volume is constant, temperature is uniform, and there is no
potential. A coating changes drag so that
\(D(x)=D_0[1+a\cos(2\pi x/L)]\), with unknown \(D_0\) in micrometer²/s.
Local thermal noise obeys the same drag's fluctuation-dissipation relation.

Eliminating rapid momentum relaxation gives probability flux
\(J=-D(x)\partial_xp\), or the Ito position equation

$$
dx=D'(x)\,dt+\sqrt{2D(x)}\,dW.
$$

The shortcut drops the drift, giving
\(\partial_tp=\partial_x^2(Dp)\) and a stationary density proportional to \(1/D\).
The physical stationary density is uniform. Calibration uses \(a=0\), where both
processes give identical decay of the first two cosine moments and identify
\(D_0\). Hidden nonzero contrasts test both transient moment mixing and
preservation of an initially uniform ensemble. A Fourier oracle is checked
against a conservative periodic finite-volume reference.
[Apparatus and API](tasks/spatial-diffusion/environment/README.md).

### 12. Equilibrium independence does not imply independent bead motion

Two harmonically trapped spheres move longitudinally in an unbounded viscous
fluid. The specified approximation freezes their mobility at the nominal
separation \(r\), with Stokes self mobility and leading Oseen cross mobility:

$$
M_{11}=M_{22}=\frac1{6\pi\eta a},\qquad
M_{12}=M_{21}=\frac1{4\pi\eta r}.
$$

The unknown base trap stiffness \(k\), in pN/micrometer, sets both stiffnesses
through known factors. With consistent thermal noise, equilibrium covariance is
\(k_BT K^{-1}\), where \(K\) is the diagonal stiffness matrix. Calibration
variances therefore identify \(k\) without measuring the mobility coupling.

After a trap-center shift \(\delta\), mean displacement evolves as
\(\bar x(t)=\delta-e^{-MKt}\delta\). Hidden experiments shift one trap and
observe the other bead's transient excursion. The independent-drag shortcut
predicts zero response there. A matrix-exponential solution is checked against
an SI-unit force-balance integration and equal-trap normal modes.
[Apparatus and API](tasks/hydrodynamic-beads/environment/README.md).

### 13. Trapped pore fluid changes compression but not shear

A uniform, sealed, fluid-saturated solid obeys linear Biot poroelasticity.
The unknown drained Young's modulus \(E\), in MPa, combines with known
Poisson ratio \(\nu\), Biot coefficient \(\alpha\), and Biot modulus \(M\).
Total stress and fluid-content increment are

$$
\sigma=2G\epsilon+\lambda_d\,\operatorname{tr}(\epsilon)I-\alpha pI,
\qquad
\zeta=\alpha\operatorname{tr}(\epsilon)+p/M.
$$

No fluid enters or leaves, so \(\zeta=0\). Eliminating pressure adds
\(\alpha^2M\) to the drained bulk modulus while leaving \(G\) unchanged.
Calibration uses only volume-preserving strains, where \(p=0\), and determines
\(E\) identically under drained and sealed models. Hidden volumetric and mixed
strains expose the shortcut's missing pore pressure. Strains are dimensionless
tensor components, with tensile stress positive and pressure positive in
compression. A seven-variable stress-pressure solve independently checks the
oracle's undrained modulus.
[Apparatus and API](tasks/poroelastic-solid/environment/README.md).

### 14. Adsorbates compete for one population of sites

A surface has equivalent, noninteracting sites, each empty or occupied by one
molecule of A or B. A well-stirred reservoir fixes concentrations \(c_A,c_B\),
in mmol/L. The two unknown binding constants \(K_A,K_B\) have units L/mmol.
Mass action and conservation of sites give

$$
\theta_A=K_Ac_A\theta_0,\qquad
\theta_B=K_Bc_B\theta_0,\qquad
\theta_0+\theta_A+\theta_B=1.
$$

Thus \(\theta_A=K_Ac_A/(1+K_Ac_A+K_Bc_B)\), with an analogous expression for B.
Pure-species calibration identifies both constants while making separate
Langmuir curves exact. Hidden mixtures measure occupancy and the loss of
previously adsorbed material when the competitor is introduced at fixed original
concentration. The independent-curves shortcut misses this displacement.
Its total occupancy stays below one in all hidden cases, so a simple bounds
check does not expose the error. The reference solves the site-balance equations.
[Apparatus and API](tasks/competitive-adsorption/environment/README.md).

### 15. Charge compensation hides a shared Hall field

A semiconductor has positive and negative carriers with known densities
\(n_+,n_-\) and one common unknown mobility \(\mu\), in m²/(V s). Pair generation
and recombination maintain bulk populations while conserving electrical charge;
doping fixes their difference. The specified model measures a wide bar's central
bulk and neglects density gradients and recombination edge layers.

Each species has a Drude tensor with an opposite Hall sign. Write
\(n=n_++n_-\), \(d=(n_+-n_-)/n\), and \(\beta=\mu B\). Adding the tensors and
imposing zero total transverse current gives

$$
E_y=d\beta E_x,\qquad
j_x=en\mu\,\frac{1+d^2\beta^2}{1+\beta^2}E_x.
$$

The shortcut adds only their longitudinal conductivities, setting \(E_y=0\).
Calibration uses equal densities, so \(d=0\) and the shortcut remains exact
even at nonzero magnetic field. Zero-field data identify mobility. Hidden
preparations change the density balance, making the shared Hall field matter.
The independent reference solves both species' steady Lorentz-force equations
and the total-current constraint together. The model does not require either
species' transverse particle flux to vanish separately.
[Apparatus and API](tasks/compensated-conductor/environment/README.md).

### 16. A homogeneous pressure law misses fluid coexistence

An idealized fluid has homogeneous molar free energy
\(f(v,T)=-T\log(v-1)-a/v\), with unknown attraction \(a\). Reduced units set
the gas constant and excluded molar volume to one. The homogeneous pressure is

$$
P(v,T)=-\partial_vf=\frac{T}{v-1}-\frac{a}{v^2}.
$$

The vessel fixes mean molar volume and temperature, but permits macroscopic
liquid and vapor regions to exchange material. Global equilibrium minimizes
free energy over those preparations, with no interface or nucleation penalty.
Below \(T_c=8a/27\), a common tangent to \(f\) selects coexistence volumes
\(v_\ell,v_g\), satisfying

$$
P(v_\ell,T)=P(v_g,T)=P_*,
\qquad
\int_{v_\ell}^{v_g}[P(v,T)-P_*]\,dv=0.
$$

Mean volumes between these contacts change phase fractions while pressure stays
at \(P_*\). The shortcut instead evaluates the homogeneous law at the mean
volume. Calibration is entirely above every allowed critical temperature and
identifies \(a\), so it cannot distinguish the preparations. Hidden subcritical
isotherms probe the pressure plateau. The solution uses Maxwell equal area;
the independent reference constructs a convex envelope and refines its common
tangent. [Apparatus and API](tasks/fluid-coexistence/environment/README.md).

### 17. A locally stable magnetic phase can still be metastable

An infinite-range magnet has known four-spin interaction \(Q=1.2\), unknown
two-spin coupling \(J\), and Hamiltonian per spin
\(H/N=-Jm^2/2-Qm^4/4-hm\). Units set \(k_B=1\), so \(J,Q,h,T\) have the
same energy unit and magnetization \(m\) is dimensionless. Its free energy is

$$
f(m)=-\frac{Jm^2}{2}-\frac{Qm^4}{4}-hm+
T\left[\frac{1+m}{2}\log\frac{1+m}{2}
+\frac{1-m}{2}\log\frac{1-m}{2}\right].
$$

Stationarity gives \(m=\tanh[(Jm+Qm^3+h)/T]\). Calibration temperatures make
free energy strictly convex for every allowed \(J\), so the starter's root
from zero is exactly the equilibrium state and identifies \(J\). In the
hidden low-temperature experiments, that same root remains field aligned and
locally stable: \(f''(m)=T/(1-m^2)-J-3Qm^2>0\). Nevertheless, an ordered
state has lower free energy. Bounds, root residuals, field sign, and local
stability therefore cannot detect this shortcut.

The apparatus specifies fresh canonical equilibrium, with no retained phase
history, and excludes exact phase-transition lines. The oracle finds stationary
points between analytic curvature turning points and compares their free
energies. The independent reference instead minimizes free energy directly.
Every hidden case has a positive gap to all competing minima, including under
the tested calibration noise.
[Apparatus and API](tasks/first-order-magnet/environment/README.md).

## What these examples teach about calibration

In each task, calibration varies something useful while leaving a crucial physical distinction untested:

- **Spin probes:** one probe responds to the fluctuating phase, so the data determine dephasing strength without measuring shared-noise correlations.
- **Gas chambers:** heat is redistributed at fixed temperature sum, so the data determine conductance without testing changing pressure.
- **Dissolved salts:** each pure salt spreads correctly, so the data determine diffusivity without testing a mixture's common electric field.
- **Hall bar and moving inductor:** zero magnetic field or a fixed core hides the electrical response to a physical constraint or motion.
- **Effusive beam:** transverse measurements determine temperature without testing the population sampled along the aperture normal.
- **Real gas and induced dipole:** restricted measurements determine a parameter without probing the complete energy law.
- **Radiative plates:** a black opposing surface hides repeated reflections.
- **Magnetic equilibrium:** the unique high-temperature root hides the need to select a stable thermodynamic state.
- **First-order magnet:** the unique high-temperature root also hides the distinction between local stability and global equilibrium.

A field expert can resolve these ambiguities from the apparatus description by applying its constraints, energy laws, sampling rule, and equilibrium preparation. The calibration data then determine the remaining parameter within that physical model.

These are model-selection failures the benchmarks are designed to expose. An actual agent run also needs inspection: a wrong formula, a numerical conservation error, an unfinished fit, or a timeout is a different failure. The [dashboard](DASHBOARD.md) records those distinctions alongside the observed results.
