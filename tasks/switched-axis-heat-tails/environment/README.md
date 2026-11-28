# Switched axis heat tails

A planar Brownian particle with unit mobility stays in one sample through a sequence of stages. Write X=(x,y), and let J rotate counterclockwise by 90 degrees. Each stage has drift

    A*X = (-K + gain*drive*J)*X.

The symmetric trap K equals R*diag(stiffness)*R^T, with R the counterclockwise orthogonal rotation by the supplied `orientation`. The two independent laboratory white noises have covariance 2*diag(`temperature`) per unit time. Thus each stage has a complete constant-coefficient overdamped dynamics with no cross-axis bath noise. The gain is the same unknown in all experiments and lies in [.57,1.36].

During a stage two calorimeters measure the heat entering the two laboratory-axis baths. Their heat increments are (A*X)_x circ dx and (A*X)_y circ dy, with the Stratonovich product and fixed lab directions even when the trap axes rotate. The recorded scalar Y is the sum of these increments weighted by the two entries of `weights` in that stage. Each stage is a dictionary containing `stiffness`, `orientation`, `temperature`, `drive`, `weights` and `fraction`. The input `stages` lists the stages in order. The fractions are positive and sum to one. In a record of duration T each stage lasts its fraction times T. Position remains continuous at boundaries, and the trap-adjustment work there adds no calorimeter increment. The initial position has the stationary density of the first stage. It is sampled anew in every record.

For `mean_heat`, there is one stage. Return the mean of Y/T as T tends to infinity. For `tail_rate`, there are one to three stages. Return minus log P(abs(Y/T-`target`)<epsilon) divided by T, taking T to infinity before epsilon to zero.

Stiffness entries lie in [.46,2.35], temperatures in [.44,1.74], drive magnitude in [.29,1.33], weight entries in [.35,1.65], orientations are in radians, fractions are at least .16, and target is in [-15,16]. All physical units are fixed across experiments. Implement `Model.fit(records)`, storing the estimate as `self.gain` and returning self. Each calibration row has `input`, `value`, and positive Gaussian standard deviation `sigma`. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order.
