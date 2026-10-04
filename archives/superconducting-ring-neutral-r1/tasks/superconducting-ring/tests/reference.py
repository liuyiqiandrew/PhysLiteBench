import numpy as np

PARAMETER='kinetic_scale'
TRUE_PARAMETER=9.


def experiment(radius,field,winding,observable='current'):
    return dict(radius=float(radius),magnetic_field=float(field),winding=int(winding),observable=observable)


def inductance(radius,kinetic_scale):
    cross_radius=.012/np.sqrt(radius)
    return radius*(np.log(8*radius/cross_radius)-2)+kinetic_scale*radius**2


def free_energy(radius,field,winding,kinetic_scale):
    # Controlled-field London energy in a fixed winding sector.
    return (winding-np.pi*radius**2*field)**2/(2*inductance(radius,kinetic_scale))


def predict(experiments,kinetic_scale):
    values=[]
    for e in experiments:
        r,b,n=e['radius'],e['magnetic_field'],e['winding']
        if e['observable']=='current':
            value=(n-np.pi*r*r*b)/inductance(r,kinetic_scale)
        else:
            # Complex virtual displacement preserves winding, material volume,
            # density and the controlled external field.
            value=-np.imag(free_energy(r+1e-20j,b,n,kinetic_scale))/1e-20
        values.append(float(value))
    return np.array(values)


def calibration_inputs():
    return [experiment(r,b,n) for r in [.75,.9,1.1,1.3,1.55]
            for b in [-.3,-.18,-.05,.05,.18,.3] for n in [-2,-1,1,2]]


def hidden_inputs():
    return {
        'zero_field_stress':[experiment(r,0,n,'radial_force') for r in [.72,.85,1.05,1.25,1.55] for n in [-3,-1,1,3]],
        'biased_stress':[experiment(r,b,n,'radial_force') for r in [.8,1.2,1.5] for b in [-.28,.12,.32] for n in [-2,1,3]],
        'screening_branch':[experiment(r,b,0,'radial_force') for r in [.72,1.,1.35,1.58] for b in [-.33,-.15,.2,.34]],
    }
