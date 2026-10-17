Implement the model in `/app/model.py` for the apparatus described in `/app/README.md`, using the calibration data and the documented prediction interface.

You may modify or replace any part of the implementation, including existing prediction functions and helpers. Preserve the documented API and required outputs.

Additional guidance:
The repeated fixed-position preparation does not make the tagged trajectory a process with stationary increments. In the joint long-time limit, relate the marked particle displacement to the net number of independent Brownian paths crossing its original position. Two-time crossing indicators on the same path remain correlated. The calibration measures only equal-time covariance.
