import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

GAP=2.
HEAT_CAPACITY=.15
THERMAL_CONDUCTIVITY=.08


def excited_fraction(temperature):
    return 1/(1+np.exp(GAP/temperature))


def profile(experiment,times,diffusivity,points=64):
    x=np.arange(points)*2*np.pi/points
    k=np.fft.rfftfreq(points,1/points)
    wave=np.cos(experiment['mode']*x)
    c=1+experiment['amplitude']*wave
    temperature=experiment['temperature']+experiment['thermal_amplitude']*wave
    def lap(a):
        return np.fft.irfft(-k*k*np.fft.rfft(a),n=points)
    def rhs(t,state):
        c,temperature=state.reshape(2,points)
        p=excited_fraction(temperature)
        dc=diffusivity*lap(p*c)
        capacity=HEAT_CAPACITY+GAP*c*p*(1-p)*GAP/temperature**2
        dt=(THERMAL_CONDUCTIVITY*lap(temperature)+GAP*(1-p)*dc)/capacity
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
