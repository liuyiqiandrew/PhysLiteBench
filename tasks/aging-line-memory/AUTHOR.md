# Burning a line in a glass

Spectral selection and persistent photochemistry mark the same ordinary basin across two launches. The age-dependent motion of the glass follows the complete-graph trap model. [Fontes and Mathieu, K-processes, scaling limit and aging for the trap model in the complete graph](https://doi.org/10.1214/07-AOP360). The pump, tagged basin and residence gates define this experiment.

## Single launch

Put α=T/T_g and v=N exp(−E/T_g). The ordinary v are uniform on [0,N] before the limit and form a unit-intensity Poisson process in the limit. The shutter wins against escape at ordinary site j with probability h_j=s/(s+v_j^(1/α)). Let H be their sum. After a reporter departure, sum the geometric series over intervening ordinary landings. The next observed outcome is each specified reporter with probability m=1/(2+H), or the shutter on ordinary site j with probability h_j m. This holds on a finite graph with its actual finite H.

Write r_i=exp(−δ_i/T). Starting on A, switches from A to B and back per unit reporter time have rates a=r_A m and b=r_B m. A self return does not interrupt the residence clock. Put d_i=s+r_i(1−m). For positive clock settings x,y, let t=√(abxy) and K=exp(−d_A x−d_B y). Summing the alternating visit histories gives occupation weights W_A=K abx I₁(2t)/t and W_B=K a I₀(2t). At y=0 there can be no visit to B and W_A=exp(−d_A x), W_B=0. Direct shutter weights are sW_A and sW_B. Define F(H)=m(r_A W_A+r_B W_B). A terminal ordinary site contributes h_j F(H). The total weight is D(H)=s(W_A+W_B)+HF(H). The physical public starter implements this single-launch marginal.

## Two launches

The pump selects a site v in the band. Removing that point leaves a background with the original Poisson law. Its two return sums H₁ and H₂ use the same site list at their respective temperatures and shutter rates. Reinsert the selected site with probabilities h₁(v) and h′₂(v), where h′₂ uses the modified escape rate exp(−shift/T₂)v^(1/α₂). The denominator is

    ∫band h₁(v) E[F₁(H₁+h₁(v)) D₂(H₂+h′₂(v))] dv.

For a tagged read replace D₂ by h′₂(v)F₂. Other reads use their respective terminal weights. Both the pre-shutter trajectories and the final basin therefore respond to burning. The joint Laplace transform of H₁,H₂ is exp{−∫[1−exp(−uh₁(z)−wh₂(z))]dz}. The oracle expands the occupation weights in powers of m₁ and m₂ and uses a double Laguerre integral for those inverse powers. It integrates over the selected depth separately.

The reference inverts the joint characteristic function on a two-dimensional Fourier grid and evaluates the Bessel functions directly at the shifted background values. Independent checks explicitly generate the deepest 96 ordinary basins of each of 16,384 specimens and sum the selected marked sites one at a time. Three independent scrambles give a tagged probability between 0.127359 and 0.127366 for the public example. Reversing the shift gives 0.067203 to 0.067210 against the oracle's 0.067199. These specimen calculations use a conditional mean for only the residual deep-rank tail.

The completed shortcut retains the common specimen, both clocks and the pump selection but leaves the tagged basin's lifetime unchanged. It misses the three held-out groups by 0.0935, 0.0169 and 0.0278. The starter independently redraws the ordinary background while keeping the modified tagged site, giving errors 0.00897, 0.0246 and 0.0313. The tolerance is 0.004. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_aging_line_memory.py` to validate all 24 predictions.
