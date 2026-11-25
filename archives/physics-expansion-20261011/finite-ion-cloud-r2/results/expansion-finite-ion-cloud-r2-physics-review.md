# Finite ion cloud, revision 2: source and physics review

**Approve for the frozen evaluation batch. No blocking issue.** Reviewed all 17 task files, both scripts, the validation report, and both fresh Docker verifier outputs. This approves the apparatus and controls, not a claim of model difficulty. No task files were changed or model trials run by this reviewer.

The public specification completely defines the finite canonical two-dimensional logarithmic gas: fixed neutralizing disk, one or two point counterions, homogeneous dielectric without images, both hard walls, and all pair interactions. The temperature convention, common coupling, parameter ranges, finite-radius preparation, pressure derivative at fixed disk radius and coupling, and annular-area normalization are explicit. Absolute-radius independence at fixed ratio is a physical scale invariance. The repulsive coincidence singularity is integrable. There is no missing reservoir, large-N assumption, or dilution limit.

The oracle implements the correct angular Gibbs factor and ordered radial integral from the concept review. Its factor `(1-exp(-2L))/(2N)` is necessary for the specified accessible-area pressure. `exprel` treats Gamma=2 for one ion and Gamma=4/3 for two ions continuously. The control's matrix exponential exactly integrates the angular-mean potential, retaining correlated radii and finite particle counting. It is the Gibbs variational model with independent uniform angles conditional on the radii. Its omitted relative-angle correlation is a physical approximation; rotational symmetry alone does not justify it. The public starter has the same completed response but deliberately leaves fitting unfinished; classification must use completed submissions and the completed control.

One-ion pressure is strictly monotone in the coupling at fixed positive temperature multiplier and radius ratio, so calibration identifies the single global parameter. Public/private calibration copies are identical: 240 records from 24 preparations, sigma 0.001, seed 621001. Both completed fits give g=1.11973766763 and reduced chi-square 0.7962137227. The 4% relative RMS gate has ample separation: oracle prediction errors are at most 0.05153%; the three two-ion control errors are 13.1832%, 23.3268%, and 37.5409%. The 256 noise draws retain an oracle maximum of 0.16912% and a smallest diagnostic control error of 13.0391%. One-ion hidden anchors cover stronger coupling and both radius extremes. Noiseless recovery includes both parameter endpoints; its maximum absolute error, 3.40e-8, is numerical bounded-optimizer accuracy, negligible for grading.

The reference integrates the microscopic positive Gibbs weight over common logarithmic radius, gap, and relative angle, then obtains pressure from outer contact. It uses neither the hypergeometric angular formula nor the analytic common-radius integral. This is a different numerical route through the fully specified equilibrium, with 96/160-node refinement. Reported domain/convergence errors (128 cases; maximum 3.41e-10, falling to 3.24e-11 under refinement) and hidden-value refinement below 1.73e-15 are negligible. The ideal-gas check and 18 inner/outer contact checks also pass. I inspected the contact-integral algebra; the identity is the correct finite-cell scaling relation.

Additional independent checks against the packaged code: the exact angular polynomials at Gamma=2 and 4 reduce the partition function to products of elementary original-radius moments. Twelve such checks over six radius ratios agree with the oracle within 5.56e-17. Forty-two direct H=1 radial integrals agree with the matrix-exponential control within 1.14e-13; 42 one-ion checks across Gamma=0.07 through 4.2 agree between controls within 7.33e-15. The earlier concept review also records original-radius/angle quadrature and independently integrated contact checks, without the logarithmic-radius reduction.

The Docker image copies only public apparatus, starter, tests, and calibration. Author notes, hint, solution, parameter truth, reference, and hidden preparations remain outside it. Tests check finite output shape, calibration, parameter recovery, and each frozen prediction group; the verifier records metrics and a numeric reward on prediction failure. Dependencies and the existing time limits are suitable.

Fresh controls inspected:

- `finite-ion-cloud-expansion-r2-oracle-20261011-1015/finite-ion-cloud__sSyuV4y`: 7/7 passed in 0.78 s, reward 1, zero infrastructure exceptions.
- `finite-ion-cloud-expansion-r2-shortcut-20261011-1015/finite-ion-cloud__pm2RHEF`: four passed and only `moderate_coupling`, `correlated_pair`, `strong_coupling` failed in 0.68 s, reward 0, zero infrastructure exceptions. Its fitted parameter, calibration and shared anchors pass.

SHA-256 of reviewed sources and validation evidence (20 files):

```text
6b67ee6a04cdeb06051d3579b6cb6320f55ddd778fe72634252be030e3f200d0  tasks/finite-ion-cloud/AUTHOR.md
375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70  tasks/finite-ion-cloud/environment/Dockerfile
9d269306fe7f34fd23605fb58cf4b2d2809126021dc4d2b988fd23a048392427  tasks/finite-ion-cloud/environment/README.md
11e11822163f4f5e6e27028aec2bfa938f6ed20b32d39ee70d4c9b30e7c68802  tasks/finite-ion-cloud/environment/data/calibration.json
fab79d2d25cc7ac40460e85e159bd5914245040fdda7c2d485e208ddc0fff099  tasks/finite-ion-cloud/environment/model.py
9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9  tasks/finite-ion-cloud/environment/test_public.py
a13c21749f5d8ee374478214f270eceefcd9497f50502d320eac7b52262b5747  tasks/finite-ion-cloud/hint.md
d45ee4a47bc9a6eac698f1025bbad6e995b7d73b56e5e3bbf7055599642aa94d  tasks/finite-ion-cloud/instruction.md
0fb7504d96acf0b9c793ffc7ac143a98e63bff97a8380a16c7ffc8aa6ee4dcb3  tasks/finite-ion-cloud/solution/model.py
bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644  tasks/finite-ion-cloud/solution/solve.sh
cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af  tasks/finite-ion-cloud/task.toml
11e11822163f4f5e6e27028aec2bfa938f6ed20b32d39ee70d4c9b30e7c68802  tasks/finite-ion-cloud/tests/data/calibration.json
b91c4cd12046511dcda2e45cca440d2a0ff31902b3e3848bce7d3293ee49dfea  tasks/finite-ion-cloud/tests/data/hidden_truth.json
e7e73504277a03622823d3a03a52315f7383d161d0b94df65dcbb2a1f0c00a89  tasks/finite-ion-cloud/tests/metadata.json
d85ee389c105322add51ebc45f2b9e2622b031b8c0c8192d1984d7dd4afd5b2b  tasks/finite-ion-cloud/tests/reference.py
74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c  tasks/finite-ion-cloud/tests/test.sh
663c0feb7fe5898f6ffbcdc6ffe12b85b795fc72b728ca970c7f9fd1a03b949d  tasks/finite-ion-cloud/tests/test_hidden.py
fa5e83e3dcb179420a19f101e2d3d997e2cb051fda2443be494431ed5174ae7d  scripts/finite_ion_cloud_baseline.py
fec903d959e23d98d5d01bcd85e9671f1b422abd6b919eebf5c511b269d4b939  scripts/validate_finite_ion_cloud.py
894a11b01f42c75be1390fd5d1e5e97f8bf0c5e43dcdf4fa9425c34bcc64f080  results/expansion-finite-ion-cloud-r2-validation.json
```
