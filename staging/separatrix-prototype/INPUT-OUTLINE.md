# Proposed physical inputs

This is an apparatus outline, not a task or evaluation instruction.

A classical particle has dimensionless canonical position q, momentum p and unit mass. The controlled Hamiltonian, including its energy zero, is

    H(q,p;s,delta)=p²/2+q⁴/4-s*q²/2+delta*s^(3/2)*q.

The asymmetry delta is known and lies in [.15,.25]. Before each shot, s=.05. The particle is prepared on a connected periodic orbit whose action J=(1/(2*pi))*integral p dq is uniform on [J0,1.2*J0]. Its orbital phase is independently uniform in time over one initial period. The unknown positive J0 in [.12,.16] is common to all experiments. Each shot is freshly prepared.

After preparation the particle is isolated: no friction, bath, noise, collisions or measurement feedback acts during the motion. Over duration T, s changes from .05 to the known endpoint s_final according to s=.05+(s_final-.05)*(3*(t/T)²-2*(t/T)³), then remains fixed. Candidate endpoints span [.1,2.2]. All potential coefficients, the path, position and momentum units are known.

The readout is the ensemble mean final total energy H(q,p;s_final,delta), reconstructed from final position and momentum. Average over the stated preparation at each finite T, then take T to infinity. Do not change the energy zero as the control varies. The physical idealization is classical throughout.

Only the proposed apparatus is specified here. No calibration file, API, noise level, grading cases or benchmark prompt exists yet.
