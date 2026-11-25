Implement the model in `/app/model.py` for the apparatus described in `/app/README.md`, using the calibration data and the documented prediction interface.

You may modify or replace any part of the implementation, including existing prediction functions and helpers. Preserve the documented API and required outputs.

Additional guidance:
For the directed valve, solve the conditional population Poisson equation using its full forward generator. Equal stationary valve probabilities do not imply equal position-conditioned residence at the first finite-mass correction. With state-dependent drag, derive effective positional mobility and diffusion from the velocity cell equations.
