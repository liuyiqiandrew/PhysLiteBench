# Two magnetic films on reversible seams

Two isolated Ising films share a horizontal seam switch ξₓ=±1 and a vertical seam switch ξᵧ=±1. Each film is a square nearest-neighbor lattice with N columns and N×aspect rows. Both directions wrap periodically. All sites carry classical spins ±1. A bond crossing from the last column to column zero has exchange ξₓ. A bond from the last row to row zero has exchange ξᵧ. All other bonds have exchange +1. Each film is at the square-lattice critical coupling K=ln(1+√2)/2. There is no magnetic field.

The films have independent spin configurations at fixed switches. The switches themselves reach thermal equilibrium with both films. For one spin configuration on each film and one switch setting the statistical weight is

    exp[K Σ(films,bonds) Jᵢⱼ Sᵢ Sⱼ + g(hₓ ξₓ + hᵧ ξᵧ + c ξₓ ξᵧ)].

Here `bias=[hₓ,hᵧ]` supplies the two dials, `coupling` gives c and the common control gain g lies in [0.75,1.15]. The calibration disconnects the films, releases a single switch with weight exp(g h ξ), and measures its mean polarity. These calibration inputs contain just a scalar `bias`. Values are rounded to eight decimal places.

Each film supplies `aspect` and a list of two, four or six `marks`. A mark [x,y] addresses site (Nx,N×aspect×y), using fractional coordinates along each respective period. The detector on film i measures the product Oᵢ of the marked spins in that film's current configuration. With the same sites and the same lattice size, measure Cᵢ, the mean of this product on film i when both seams are clamped positive. This reference is nonzero. The two normalizations are measured separately.

* For `read="overlap"`, return the joint mean of O₁O₂ divided by C₁C₂.
* For `read="connected"`, subtract the product of the two single-film means from the joint mean, then divide by C₁C₂. Each single-film mean here is from the same coupled two-film experiment.
* For `read="response"`, return the derivative of the normalized joint mean with respect to the selected bias, holding all other settings fixed. The integer `dial` is 0 for horizontal or 1 for vertical. The normalization uses the clamped instrument and is independent of both dials.

Take N to infinity through multiples of 32. The aspect is 0.75, 1 or 1.25. Each fractional coordinate lies in [0,1) on a grid of eighths. Marks within a film are distinct and may be listed in any order. The two bias values and c lie in [−1.4,1.4].

For a check at g=1, let the first film have aspect 0.75 and marks [[0.25,0.25],[0.625,0.75]]. Give the second film aspect 1 and marks [[0.125,0.25],[0.25,0.5],[0.625,0.25],[0.75,0.75]]. With bias [0.35,−0.2] and coupling 0.4 the overlap is 0.86751282, the connected signal is 0.09834891 and the vertical dial response is 0.17310032. Giving each film a separate pair of switches would make their connected signal zero.

Implement Model in /app/model.py. Its fit(records) sets `gain` to g and returns self. Its predict(inputs) returns a flat finite real NumPy array in input order, including an empty array for an empty list. Calibration errors must be below 0.00002 and other absolute errors below 0.004. The starter models a reversed seam as one straight domain wall with uniform transverse position, and assigns equal thermal partition weights to the four seam conditions. It provides the two-film measurement and bias response once single-film weights are supplied.
