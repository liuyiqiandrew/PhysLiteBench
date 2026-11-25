# Controlled resonator bath revision 3

The neutral single-wait quantum-bath task passed 3/3 Luna high trials. Its full sources, scientific controls and job identities are preserved in `archives/qubit-control-neutral-r2`. This revision adds an intermediate spin rotation between two waits. The apparatus Hamiltonian, parameter interval, fixed instrument noise, neutral instruction and grading cutoffs are unchanged. Calibration intentionally includes both individual and paired probe couplings, with a single nonzero wait.

The starter includes exact shared colored Gaussian fluctuations across both waits and the exact coherent quantum self phase within each wait. It describes correlated classical noise supplemented by the correct single-interval coherent interaction. That model is completely positive. It omits the retarded quantum bath response between distinct intervals. The task therefore does not rely on missing common noise, a missing static mediated interaction, or an independent-noise/reset approximation.

For one resonator, write c=sqrt(gamma*a)/omega. A spin path has weighted eigenvalues s1 and s2 during waits t1 and t2. Its conditional oscillator unitary is a displacement followed by free rotation. Define

    alpha1 = c*s1*(exp(-i*omega*t1)-1),
    alpha2 = c*s2*(exp(-i*omega*t2)-1),
    beta1 = alpha1*exp(-i*omega*t2),
    alpha = alpha2+beta1,
    phi_self = c^2*[s1^2*(omega*t1-sin(omega*t1))
                    +s2^2*(omega*t2-sin(omega*t2))],
    phi = phi_self + Im(alpha2*conj(beta1)).

For ket and bra paths p and q, the exact thermal influence factor is

    exp[-coth(omega/(2T))*abs(alpha_p-alpha_q)^2/2
        +i*(phi_p-phi_q+Im(alpha_p*conj(alpha_q)))].

The zero-temperature coth factor is one. Independent resonators multiply these factors. The intermediate spin rotation supplies amplitudes for all paths. The shortcut keeps the full real exponent and the phi_self differences. It omits both the within-path displacement-composition phase and the cross-path Weyl phase. In a single interval all displacements are collinear, so the omitted phase is exactly zero even when both probes are coupled. The supplied calibration thus exercises the coherent single-interval interaction but cannot expose the temporal quantum-response error.

The private reference multiplies conditional finite-Fock Hamiltonian unitaries along each path and directly traces their products against a thermal state. It does not use the displacement influence formula. Spin pulses use a matrix exponential and measurement projectors use Bloch rotations. The validator compares Fock cutoffs 80 and 104 over full parameter corners, checks reduced-state trace, Hermiticity and positivity, and confirms analytic probabilities to better than 7e-14.

In `results/qubit-neutral-r3-validation.json`, all 256 calibration noise draws pass. Oracle probability RMSE stays below 0.000453. The shortcut fails the echo-sequence group on every draw, with its minimum RMSE 0.03604 against the unchanged 0.03 gate. Its other two groups are below the gate; they are not reported as failures. The fixed sigma remains 0.003, independently of the true parameter and noiseless output. The parameter cutoff remains 0.05.

Local harness results are in `results/qubit-neutral-r3-local-controls.json`; the oracle passes seven tests and the shortcut fails the echo prediction test while passing the other six. This establishes the intended physical separation, not an agent failure rate. Agent outcomes must be taken from the new frozen conditional screen.
