# Antipodal color echo

Each frozen sample at size N has floor(exp(N r v_F)) families. A family has one centered Gaussian energy X of variance N gain² v_F. Within each family there are two banks of sites, one of each color j=0,1. Each bank contains floor(exp(N r v_L)) sites. Every site has a centered Gaussian energy Y of variance N gain² v_L. All the site energies and family energies are independent, apart from the sharing of X by a family's sites. Bank and site labels are the same in the two settings.

Use r from `ratio`, [v_F,v_L] from `variances`, c=sqrt(2r)/gain, a=gain*sqrt(2r), V=v_F+v_L and signs s_0=+1, s_1=−1. Setting j has inverse temperature c+`windows[j]`/sqrt(N). Its energy for a site of color k is

    s_j X + Y + 1_{k≠j}*(sqrt(N)*leakage[j]+offsets[j]).

The two Gibbs draws, one in each setting, are independent conditional on the same frozen sample. Each is normalized over the entire two-color sample using its own energies. In the primary event both draws land in the same family, the draw in setting j has color j, and its base displacement s_j X+Y+NaV is inside `line_windows[j]`. The selected sites thus have different colors. If `outside_setting` is 0 or 1, require an additional independent draw in that setting to land in a different family from the first two. A value of −1 requests no additional draw.

For `slab_echo`, at least one entry of `leakage` is strictly positive. In the primary event additionally require X/sqrt(N) to lie in `root_band`. Return the limit of N exp(N K) times the full event probability, where K=r v_F(3+2v_F/v_L).

For `balanced_echo`, both entries of `leakage` are exactly zero. In the primary event additionally require X itself to lie in `root_band`. Return the limit of N^(3/2) exp(N K) times the full event probability. Form the events at finite N and average the normalized probabilities over equally weighted samples before taking either limit.

Both event observables use exactly the keys named above and `observable`. Valid ranges are r∈[.21,.47], v_F,v_L∈[.47,1.32], the two temperature windows in [−1.25,1.25], the nonnegative leakages in [0,2.1], offsets in [0,1.8], and all line and root window endpoints in [−1.7,1.7] with strictly increasing endpoints.

For `coil` calibration the only additional keys are `current` and `turns`. Return the signed coil reading gain*current*turns. Gain lies in [.72,1.32]. Each calibration record has `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to estimate and store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order, including an empty array for empty input.
