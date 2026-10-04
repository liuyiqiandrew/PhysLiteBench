# Nonreciprocal cavity revision 3

The uncertainty-repaired neutral revision scored2/3. Its full task, controls, data and job identities are preserved in `archives/nonreciprocal-cavity-neutral-r2`; the older0/1 score belongs to different calibration data. This revision keeps the passive biased resonator circuit and changes the reservoir preparation: each resonator has the same internal damping rate but its own known local thermal occupation. The neutral instruction, loss-rate bounds, fixed-control instrument uncertainty and0.025 prediction cutoff are unchanged.

The starter now has the correct full coherent resolvent and external input propagation. It also has the correct outgoing internal emission at a common internal temperature. Its approximation is to treat Hamiltonian normal modes as independent thermal noise sources when the local reservoir temperatures differ.

Let U diagonalize the Hermitian cavity Hamiltonian, N=diag(n_1,n_2,n_3), W=diag(sqrt(kappa)), and

    G = [i(H-frequency*I)+(K+gamma*I)/2]^-1,
    S = I-W G W,
    B = sqrt(gamma)*W G U.

The correct internal output covariance is B*(U† N U)*B†. The shortcut replaces U† N U by its diagonal. It retains the correct weighted occupation of every normal mode, all coherent response functions, both magnetic-bias signs, and every external reservoir input. Both covariance choices are positive. Independent local baths need not remain statistically independent after transforming their amplitudes to Hamiltonian normal modes.

At common internal occupation n, the noise matrix is nI in either basis. The two closures agree for every allowed loss rate and all calibration settings. Varied total-output spectra identify gamma. Nonequilibrium port measurements retain the off-diagonal modal noise covariance, which is visible even though the diagonal mode occupations are correct. External damping is retained in the full resolvent, including its mode mixing.

The independent reference propagates each local internal and external Langevin input separately in the physical resonator basis and adds its output covariance. It never diagonalizes H or projects noise in a mode basis. Checks establish the matrix emission identity, equilibrium output covariance, lossless unitarity, magnetic-bias reversal and a unique calibration minimum over the entire allowed parameter interval. Peer checks also establish invariance under arbitrary eigenvector phases and positive emission covariance.

Calibration sigma is0.006 times the mean of the known internal occupations. It contains no fitted or noiseless-response information. With the retained fixed seeds, all256 calibration noise draws pass. Oracle hidden RMSE is at most0.000646; shortcut RMSE stays above0.0613 in every hidden group, against0.025. The nominal shortcut errors are0.0998,0.0856 and0.0614. Local controls give oracle7/7 and shortcut4passes with3hidden failures.

Evidence: `results/nonreciprocal-cavity-r3-validation.json`, `results/nonreciprocal-cavity-r3-local-controls.json`, and `results/brownian-cross-review-cavity-r3.json`. These validate the physical task; its difficulty must be measured with the new frozen neutral-r3 screen. The earlier revisions' outcomes remain historical.

Reproduce scientific checks with `python scripts/validate_nonreciprocal_cavity.py` in the pinned scientific environment. Use `--generate` only for intentional regeneration of both calibration copies.
