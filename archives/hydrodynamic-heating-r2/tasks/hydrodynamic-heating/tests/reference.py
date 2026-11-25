"""Independent real first-order Maxwell/momentum boundary-value solution."""
import numpy as np
from scipy.integrate import solve_bvp
from numpy.polynomial.legendre import leggauss
TRUE_PARAMETER = 1.03
GAMMA = .06
VISCOSITY = .04


def fields(experiment, plasma_frequency, tolerance=1e-8):
    w, d, angle = (experiment[key] for key in ('frequency', 'thickness', 'angle'))
    k, normal = w*np.sin(angle), w*np.cos(angle)
    matrix = np.array([[0,1,0,0],[-(w*w-k*k),0,-1j*w,0],[0,0,0,1],
        [-plasma_frequency**2/VISCOSITY,0,(GAMMA-1j*w+VISCOSITY*k*k)/VISCOSITY,0]],complex)
    system = np.block([[matrix.real,-matrix.imag],[matrix.imag,matrix.real]])
    left = np.array([[1j*normal,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]],complex)
    right = np.array([[0,0,0,0],[-1j*normal,1,0,0],[0,0,0,0],[0,0,1,0]],complex)
    a = np.block([[left.real,-left.imag],[left.imag,left.real]])
    b = np.block([[right.real,-right.imag],[right.imag,right.real]])
    rhs = np.array([0,0,0,0,2*normal,0,0,0])
    grid = np.linspace(0,d,25)
    solution = solve_bvp(lambda z,y:system@y, lambda y0,y1:a@y0+b@y1-rhs,
        grid,np.zeros((8,len(grid))),tol=tolerance,max_nodes=4000,
        fun_jac=lambda z,y:np.repeat(system[:,:,None],len(z),axis=2),
        bc_jac=lambda y0,y1:(a,b))
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def response(experiment, plasma_frequency, tolerance=1e-8, order=96):
    sol = fields(experiment,plasma_frequency,tolerance)
    x, weights = leggauss(order)
    lo, hi = np.asarray(experiment['window'])*experiment['thickness']
    z = (lo+hi)/2+(hi-lo)*x/2
    real = sol.sol(z)
    electric,electric_z,current,current_z = real[:4]+1j*real[4:]
    k = experiment['frequency']*np.sin(experiment['angle'])
    # Sum the local symmetric-strain and friction entropy-production terms.
    loss = GAMMA*abs(current)**2+VISCOSITY*(abs(current_z)**2+k*k*abs(current)**2)
    return float((hi-lo)/2*np.dot(weights,loss)/(plasma_frequency**2*np.cos(experiment['angle'])))


def predict(experiments, plasma_frequency=TRUE_PARAMETER):
    return np.array([response(e,plasma_frequency) for e in experiments])


def calibration_inputs():
    unique = [dict(frequency=.75+.2*(i%4),thickness=.4+.2*(i//4),
                   angle=.15+.2*(i%3),window=[0.,1.]) for i in range(16)]
    return [dict(e) for _ in range(16) for e in unique]


def hidden_inputs():
    groups = {'front_window':[], 'interior_window':[], 'rear_window':[], 'whole_slab':[]}
    for i in range(12):
        e = dict(frequency=.75+.055*i,thickness=.65+.035*i,angle=(-1)**i*(.3+.035*i))
        for name,window in [('front_window',[0.,.25]),('interior_window',[.3,.7]),
                            ('rear_window',[.75,1.]),('whole_slab',[0.,1.])]:
            groups[name].append(dict(e,window=window))
    return groups
