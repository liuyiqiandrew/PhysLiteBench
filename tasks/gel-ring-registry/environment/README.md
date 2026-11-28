# Two filaments grown along a common route

A gel has two narrow circuit-shaped pores meeting at one junction. A complete traversal of either pore ends back at that junction. Directions 0 and 1 traverse the first pore in opposite directions. Directions 2 and 3 traverse the second pore in opposite directions. A flexible ideal filament can retrace or cross itself but cannot pass through the pore walls.

A feed lays two strands together from a marked clasp at the junction. Start with an empty path. On every visit to the junction, including the initial one, stop laying this paired section with probability 1/(1+z f), where f is its feed setting. Otherwise select direction j with probability proportional to its four positive route weights, traverse the corresponding pore and repeat. The resulting entire sequence is the common trunk of both strands. The input `trunk` contains its `feed` and four `routes` in direction order.

Separate the two growing tips at the junction after the common trunk is complete. Each daughter continues independently according to its own entry in `arms=[first,second]`, using the same stopping and routing rule as the trunk. On stopping, close its filament at the starting clasp. Both complete rings include the very same sequence of trunk traversals, followed by their respective daughter sequences. The gel has no material marker at the fork and each finished strand can later slide through it.

Hold the original clasp of each ring and tighten its contour. Erase any adjacent direction and its immediate reverse, repeating until none remain. A cancellation can straddle the growth fork. The camera records the directed sequence starting at the clasp. Sort for shots where both held sequences have the specified positive `length`. Also require in both rings that the first and last circuits are not reverses, so slack cannot pull the clasp away from the taut circuit when released.

Retain the pair if the first direction of the first sequence belongs to `gate` and the second sequence agrees with the first after some cyclic shift. Compare directions in their stated orientation, with fixed pore labels. Count each pair once even when several cyclic shifts align it. If `shifted` is true also require that the two directed sequences as originally rooted at their respective clasps differ. The unrooted images must still match. If it is false, identical rooted sequences are allowed. All gates refer to the same two growth histories.

With `probe` present, report the conditional probability that the last direction before the second clasp belongs to that list. Otherwise compute W₀=(number of 0 circuits)−(number of 1 circuits) and W₁=(number of 2 circuits)−(number of 3 circuits) on the second retained ring. Report the conditional mean of [1+cos(phase+flux[0] W₀+flux[1] W₁)]/2. Each repetition starts with a fresh paired lay.

Every feed is in [2,12]; route weights are in [0.2,5]. Length is an integer from 2 through 6. Each direction list is a nonempty collection of distinct labels. Both flux components and phase lie in [−π,π]. Preparations have positive acceptance. Calibration at dose d in [0.3,5] measures the probability of stopping before the first traversal at feed d. The material activity z lies in [0.75,1.15], and calibration values are rounded to eight decimals.

For a check at z=1 use the input below. The probability is 0.15036470. Setting shifted=false gives 0.10805815. If the two strands instead received independently drawn trunks, the respective probabilities would be 0.05749654 and 0.06638035. The difference remains even though each individual ring has the same marginal distribution.

```json
{"length":3,"trunk":{"routes":[4,1,2,1],"feed":6},"arms":[{"routes":[5,2,1,1],"feed":4},{"routes":[1,2,5,2],"feed":7}],"gate":[0,2],"probe":[1,3],"shifted":true}
```

Implement Model in /app/model.py. Its fit(records) sets `activity` to z and returns self. Its predict(inputs) returns a flat finite real NumPy array in input order, including empty inputs. Calibration errors must be below 0.00002 and other absolute errors below 0.004. The starter supplies the single-feed first-passage factors and image registration. It computes pair weights as if both daughters had started with empty contours.
