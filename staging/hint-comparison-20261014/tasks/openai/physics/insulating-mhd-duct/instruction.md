Implement the model in `/app/model.py` for the apparatus described in `/app/README.md`, using the calibration data and the documented prediction interface.

You may modify or replace any part of the implementation, including existing prediction functions and helpers. Preserve the documented API and required outputs.

Additional guidance:
For an oblique field, the two motional-current contributions belong to one
current field and one electrostatic potential. Enforce zero normal current
and charge conservation for their sum, then use that same total current in
the full Lorentz force. Separate axis responses need not add as independent
magnetic damping operators.
