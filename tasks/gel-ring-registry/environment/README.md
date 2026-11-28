# Ring polymers in a two-pore gel

Two narrow channels in a gel form a figure eight. Each full circuit returns to their common junction. A growing ideal filament is confined to the channels and can retrace or cross itself. At the junction it can travel around either pore in either direction. Label these directed circuits 0, 1, 2 and 3. Circuits 0 and 1 go around the first pore in opposite directions. Circuits 2 and 3 go around the second pore in opposite directions. The pore walls cannot be crossed.

In each preparation a filament tip starts at the junction in conformation 1 with probability `blue`, otherwise in conformation 0. At every return to the junction, including the initial visit, it closes into a ring with probability 1/(1+z feed[k]) in its current conformation k. Otherwise it draws its next circuit j with probability proportional to `routes[k][j]`. On returning from j the tip may change conformation. The two entries `switch[j]=[f,g]` give the respective probabilities of changing 0 to 1 and 1 to 0. This change takes place before the next closing decision. All draws are independent given the current state and chosen circuit. A calibration at dose d measures the probability of immediate closure when both feed entries are d. The activity z in [0.75,1.15] is shared by both preparations and all records.

A camera watches each oriented ring relax under tension. The original closing junction remains marked during the first part of the relaxation. With the mark held, erase successive circuits that immediately reverse each other, repeating until there is no adjacent reverse pair. Now release the mark, draw slack through it, and erase the first and last circuits together whenever they are reverses. Repeat until they no longer cancel. The camera preserves the resulting seam, the gap between the last and the first surviving circuit. A ring that collapses completely has length zero. Retain only rings of the specified positive tightened length. This procedure describes relaxation of a phantom filament on the channel graph.

Each observation uses an independent pair from the two `rings` preparations. Retain the pair if the first ring's first surviving circuit belongs to `gate` and the two taut circuits coincide after sliding one ring around its contour. The pore labels and traversal directions must all agree in some cyclic registration. The rings cannot be turned over or relabeled. A pair of images is counted once even when more than one shift aligns them. The two seams remain independently marked for the measurements.

* When `probe` is present, report the fraction of retained pairs in which the second ring's last circuit before its own seam belongs to that list.
* Otherwise report the mean winding signal [1+cos(phase+flux[0] W₀+flux[1] W₁)]/2 on the retained pairs. W₀ is the number of 0 circuits minus 1 circuits. W₁ is the number of 2 circuits minus 3 circuits. The camera computes this signal from the oriented images.

The two preparations may use different feed, route, switching and initial conditions. The positive length is an integer from 2 to 11. Feed entries lie in [1.5,13], route weights in [0.1,13], and probabilities in [0.02,0.98]. Each direction list contains one or more distinct labels. Both flux components and phase are in [−π,π]. Calibration doses range from 0.3 to 5. Values in the calibration file are rounded to eight decimals.

For a check, take identical preparations with feed [4,6], all eight route weights equal, switch [0.2,0.3] on every circuit, and blue=0.45. At length 2, gate [0] and probe [0], the result is 3/5 at every activity. The first ring can have sequence 00, 02 or 03. There are five equally weighted ordered pairs after free registration. Three have second-ring last circuit 0. With flux [π/2,0] and phase zero in place of probe, the mean is 2/5.

An asymmetric example tests the effect of erased slack. At z=1 the input below gives 0.32046907. Treating each raw filament as just the visible three circuits gives 0.36186172.

```json
{"length":3,"rings":[{"feed":[3,9],"blue":0.4,"routes":[[6,1,1,2],[1,2,7,1]],"switch":[[0.7,0.2],[0.15,0.65],[0.1,0.8],[0.8,0.1]]},{"feed":[7,3],"blue":0.7,"routes":[[1,6,2,1],[4,1,1,5]],"switch":[[0.8,0.1],[0.2,0.5],[0.65,0.3],[0.3,0.65]]}],"gate":[3],"probe":[1]}
```

Implement Model in /app/model.py. Its `fit(records)` sets `activity` to the fitted z and returns self. Its `predict(inputs)` returns a flat finite real NumPy array in input order, also for an empty list. Calibration errors must be below 0.00002 and each other absolute error below 0.004. The public starter implements length sorting, image registration and both readouts. It assumes that the raw filament has exactly the observed circuit sequence and no erased slack.
