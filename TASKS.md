# Understanding the three physics tasks

## The common problem: a good fit can hide a wrong assumption

Imagine calibrating an instrument under one set of conditions, obtaining an excellent fit, and then using that model for a new experiment. What justifies trusting the prediction? Recovering the right parameter is part of the answer. The equations must also describe the physical constraints of the new experiment.

The three current tasks explore this distinction. Each gives an agent an apparatus description, noisy calibration measurements, and a partly implemented model. The agent must infer one unknown parameter and predict measurements for other allowed preparations of the same apparatus.

The supplied predictor contains a simplifying physical assumption. The calibration experiments happen to be a special case in which that assumption gives the correct answer. Completing the fit can therefore produce a model that passes calibration while retaining the physical mistake.

| Task | What is measured? | What is fitted? | What the shortcut misses |
|---|---|---|---|
| Two spin probes | Probability of a joint measurement outcome | Magnetic-noise intensity | Correlations produced by a shared field |
| Two gas chambers | Temperatures as functions of time | Conductance to a heat bath | Work and pressure changes caused by a movable piston |
| Two dissolved salts | Concentration profiles as functions of time | An ionic diffusion coefficient | The electric field shared by all the ions |

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

See [the benchmark guide](docs/BENCHMARK.md) for task interfaces and grading, and [the reference solution](qubit-control/solution/model.py) for implementation details.

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

See [the benchmark guide](docs/BENCHMARK.md) for task interfaces and grading, and [the independent sum-and-difference solution](thermal-bodies/tests/reference.py) for implementation details.

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

See [the benchmark guide](docs/BENCHMARK.md) for task interfaces and grading, and [the independent three-ion reference](reaction-diffusion/tests/reference.py) for the numerical implementation.

## What these examples teach about calibration

In each task, calibration varies something useful while leaving a crucial physical distinction untested:

- **Spin probes:** one probe responds to the fluctuating phase, so the data determine dephasing strength without measuring shared-noise correlations.
- **Gas chambers:** heat is redistributed at fixed temperature sum, so the data determine conductance without testing changing pressure.
- **Dissolved salts:** each pure salt spreads correctly, so the data determine diffusivity without testing a mixture's common electric field.

A field expert can resolve these ambiguities from the apparatus description: follow the same noise realization through both spins, include piston work in each first law, or enforce one zero-current condition for all ions. The calibration data then determine the remaining parameter within that physical model.

These are model-selection failures the benchmarks are designed to expose. An actual agent run also needs inspection: a wrong formula, a numerical conservation error, an unfinished fit, or a timeout is a different failure. The [dashboard](DASHBOARD.md) records those distinctions alongside the observed results.
