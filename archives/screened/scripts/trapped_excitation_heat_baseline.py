import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

ENERGIES=np.array([1.,3.])
DEGENERACIES=np.array([1.,9.])
HOP_RATIOS=np.array([1.,.4])
HEAT_CAPACITY=.15
THERMAL_CONDUCTIVITY=.08


def internal_state(temperature):
    temperature=np.asarray(temperature)
    shape=(2,)+(1,)*temperature.ndim
    energy=ENERGIES.reshape(shape);weight=DEGENERACIES.reshape(shape)*np.exp(-energy/temperature)
    p=weight/(1+np.sum(weight,axis=0));u=np.sum(energy*p,axis=0)
    derivative=p*(energy-u)/temperature**2
    mobility=np.sum(HOP_RATIOS.reshape(shape)*p,axis=0)
    carried=np.sum(energy*HOP_RATIOS.reshape(shape)*p,axis=0)
    capacity=np.sum(energy*derivative,axis=0)
    return p,u,capacity,mobility,carried


def excited_fraction(temperature):
    return internal_state(temperature)[3]


def profile(experiment,times,diffusivity,points=64):
    x=np.arange(points)*2*np.pi/points
    k=np.fft.rfftfreq(points,1/points)
    wave=np.cos(experiment['mode']*x)
    c=1+experiment['amplitude']*wave
    temperature=experiment['temperature']+experiment['thermal_amplitude']*wave
    def lap(a):
        return np.fft.irfft(-k*k*np.fft.rfft(a),n=points)
    def derivative(a):
        return np.fft.irfft(1j*k*np.fft.rfft(a),n=points)
    def rhs(t,state):
        c,temperature=state.reshape(2,points)
        p,u,du,mobility,carried=internal_state(temperature)
        particle_flux=-diffusivity*derivative(mobility*c)
        dc=-derivative(particle_flux)
        energy_flux=carried/mobility*particle_flux
        dt=(THERMAL_CONDUCTIVITY*lap(temperature)-derivative(energy_flux)-u*dc)/(HEAT_CAPACITY+c*du)
        return np.r_[dc,dt]
    initial=np.r_[c,temperature]
    if max(times)==0:
        return np.repeat(initial[:,None],len(times),axis=1).reshape(2,points,-1)
    solution=solve_ivp(rhs,(0,max(times)),initial,t_eval=times,method='BDF',rtol=2e-8,atol=2e-10)
    if not solution.success: raise RuntimeError(solution.message)
    return solution.y.reshape(2,points,-1)


class Model:
    def __init__(self):
        self.diffusivity=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        rate=np.array([excited_fraction(e['temperature'])*e['mode']**2*e['time'] for e in inputs])
        amplitude=np.array([e['amplitude'] for e in inputs])
        value=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.diffusivity=float(minimize_scalar(lambda d: np.sum(((amplitude*np.exp(-d*rate)-value)/sigma)**2),bounds=(.5,3.),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):
        result=np.empty(len(experiments));groups={}
        for i,e in enumerate(experiments):
            if e['clamped']:
                result[i]=e['amplitude']*np.exp(-self.diffusivity*excited_fraction(e['temperature'])*e['mode']**2*e['time']) if e['observable']=='concentration' and e['harmonic']==1 else 0.
            else:
                key=tuple(e[k] for k in ['temperature','amplitude','thermal_amplitude','mode'])
                groups.setdefault(key,[]).append((i,e))
        for rows in groups.values():
            times=sorted(set(e['time'] for _,e in rows));p=profile(rows[0][1],times,self.diffusivity)
            x=np.arange(p.shape[1])*2*np.pi/p.shape[1]
            for i,e in rows:
                values=p[0 if e['observable']=='concentration' else 1,:,times.index(e['time'])]
                result[i]=2*np.mean(values*np.cos(e['mode']*e['harmonic']*x))
        return result
