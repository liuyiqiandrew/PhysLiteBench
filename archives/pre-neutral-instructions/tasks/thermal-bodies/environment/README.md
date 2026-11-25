# Gas temperature records

A rigid sealed vessel of volume 5e-4 m³ is divided by a freely moving, adiabatic, frictionless, unloaded piston. Each chamber contains 0.01 mol of the same monatomic ideal gas. Both gases are internally uniform. Mechanical pressure equilibration is fast on the recorded timescales; piston kinetic energy and fluctuations are negligible. No gas crosses the piston.

Each gas connects to a maintained 293 K bath through a thermal link of the same unknown conductance. A separate thermal link of conductance 1.6e-3 W/K joins the gases, bypassing the piston. Conductances are constant and independent of piston position. The vessel walls and links have negligible heat capacity; there are no other heat paths or external work. Each preparation begins in mechanical equilibrium at the specified gas temperatures.

`metadata.json` lists exact initial temperatures and CSV files. Columns are time (s), gas temperatures (K), and their independent Gaussian measurement uncertainties (K).

Complete `ThermalModel`:
- `fit(runs)` returns self and stores the bath-link conductance in `conductance` (W/K).
- `predict(t, initial_temperature)` returns gas temperatures as an `(N, 2)` NumPy array.

Each run has `t`, `temperature`, `sigma`, and `initial_temperature`; measurements and uncertainties have shape `(N, 2)`. Prediction times are nonnegative, strictly increasing, measured from preparation, and need not include zero. Initial temperatures have shape `(2,)`.
