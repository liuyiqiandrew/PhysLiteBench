# Burning a line in a glass

A fluorescent particle occupies one basin at a time. A specimen contains N ordinary basins and two labeled reporter basins, A and B. Each ordinary basin has a distinct spectral line and an independent binding depth E drawn from an exponential law of mean T_g in [0.9,1.1]. A fresh-basin calibration measures the probability that E exceeds the input threshold in [0.2,2], rounded to eight decimals. The ordinary depths in an individual specimen stay fixed except for the single light-induced change described below.

An experiment makes two successive exposures on the same specimen. The input `exposures=[first,second]` specifies both. At the start of each, empty the apparatus and put a fresh particle in A. At temperature T the escape rate at depth E is exp(−E/T). After every escape a short flight lands uniformly in any of the N+2 basins, including the one just left. Flight times vanish. Attempt time and Boltzmann constant are one. External controls set the A and B depths for this exposure to T_g ln N plus their respective energy offsets. Each exposure has its own independent exponential shutter of microscopic rate s/N^(T_g/T). At its first click, read the occupied line and end that launch.

Two noninvasive clocks measure the total times spent in A and B, including all returns, until the click. Divide each clock reading by N^(T_g/T) at that exposure's temperature. The shutter runs in every basin. Select both clock readings in each exposure. A positive setting t selects a bin [t,t+ε). A zero B setting selects exactly zero time spent in B. The residence clocks reset between exposures.

The first shutter acts as a pump. Keep its shot if it ends on an ordinary basin whose original depth rank v=N exp(−E/T_g) lies in the closed interval `pump_band=[lo,hi]`. That basin's line is now tagged. Before the second launch, the pump changes the binding depth of just that basin from E to E+`shift`. A positive shift strengthens binding. Its spectral identity is preserved and its modified depth applies at every visit in the second launch. All ordinary basins other than the tagged one retain their original depths. The two launches draw independent flights, escape times and shutters on this common specimen.

Among all shots satisfying the pump selection and the four residence readings, report the probability of the second terminal line requested by `read`. Values `"A"` and `"B"` select that reporter. Value `"host"` selects any ordinary line, including the tagged one. Value `"tagged"` selects the same ordinary line found at the first shutter. Each repetition starts with a new specimen. Pool launches before conditioning, take N to infinity, then let ε go to zero.

Each exposure supplies `temperature` T in [0.53,0.71], `clock` s in [0.55,1.45], `offsets=[δ_A,δ_B]` in [−0.72,0.35], and `levels=[x,y]` with x in [0.25,1.65] and y either zero or in [0.18,1.10]. Both pump-band endpoints lie in [0.1,3] and its width is at least 0.35. The shift lies in [−1.3,1.3]. A zero shift still tags the observed line.

For a worked input with T_g=1, the tagged probability is 0.12735679. Changing only shift to zero gives 0.10110947. At shift 0.7 the terminal A probability is 0.16936132 and the ordinary probability is 0.44809463.

```json
{"exposures":[{"temperature":0.55,"clock":0.85,"offsets":[-0.48,0.11],"levels":[0.75,0.4]},{"temperature":0.67,"clock":1.2,"offsets":[0.2,-0.49],"levels":[1.05,0.6]}],"pump_band":[0.24,1.72],"shift":0.7,"read":"tagged"}
```

Implement Model in /app/model.py. Its `fit(records)` receives input/value dictionaries, sets `depth_scale` to the fitted T_g and returns self. Its `predict(inputs)` returns a flat finite real NumPy array in input order, including an empty array for an empty list. The maximum absolute error is below 0.00001 on calibration and below 0.004 on all other readings. The starter retains the selected pump line and its change in binding energy, but samples the rest of the ordinary medium independently for the probe.
