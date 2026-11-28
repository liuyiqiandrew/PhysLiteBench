# Current coexistence dwell

A device has a slow channel A or B and a signed counter. Start in A with probability `initial_a`, otherwise in B. In apparatus size N it switches A→B at rate gain*`switch_rates`[0]/N and B→A at rate gain*switch_rates[1]/N. Run for physical time N. Conditional on the channel path, the counter has independent positive and negative Poisson increments with the instantaneous rates below. These increments do not change the channel.

In A the two rates are gain*u and gain*d, using `base_rates`=[u,d]. In B they are gain*(u+du) and gain*(d+dd), using `increments`=[du,dd]. A positive event adds one and a negative event subtracts one. Set z=dd/du and define

    j_A=gain*(u*z−d/z),
    j_B=gain*((u+du)*z−(d+dd)/z).

The apparatus uses the actual gain. Condition on the final counter equalling floor(N*((1−`target_fraction`)*j_A+target_fraction*j_B)). The counter starts at zero.

For `dwell_record`, require exactly `switches` channel switches and final channel `terminal`, where 0 means A, 1 means B and −1 allows both. If `first_window` is an interval and at least one switch occurs, divide the first switch time by N times the planned fraction of the initial channel. This fraction is 1−target_fraction for A and target_fraction for B. Require the quotient in first_window. A missing switch or a zero planned fraction fails this interval requirement. If first_window is null omit it. Return the limiting conditional probability of all these requirements over the original trajectories. Apply the exact counter and interval requirements at finite N before taking the limit.

Inputs are exactly the named keys and `observable`. Initial_a is in [.13,.87], base_rates entries in [.42,1.29], du in [.33,.63], dd in [.74,1.14], and switch_rates entries in [.21,2.43]. Target_fraction is either 0, 1, or in [.17,.83]. Switches is an integer from 0 to 4, and first_window has increasing endpoints in [0,1.45].

For `wire_mean` calibration, use only observable, `rate` in [.29,1.22] and `time` in [.15,1.39]. The process makes only positive increments at rate gain*rate. Return the mean counter at time. Fit gain in [.68,1.32]. Calibration rows contain input, value and positive sigma. Implement `Model.fit(records)` storing `self.gain` and returning self, and `Model.predict(experiments)` returning one-dimensional NumPy predictions in input order including an empty array for empty input.
