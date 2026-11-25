"""Conservative finite-volume transport of particle number and total energy."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

PARAMETER='diffusivity'
TRUE_PARAMETER=1.2


def experiment(temperature=1.,amplitude=.3,thermal_amplitude=0.,mode=1,time=1.,observable='temperature',harmonic=1,clamped=False):
    return dict(temperature=float(temperature),amplitude=float(amplitude),thermal_amplitude=float(thermal_amplitude),mode=int(mode),time=float(time),observable=observable,harmonic=int(harmonic),clamped=bool(clamped))


def calibration_inputs():
    return [experiment(temperature=t,amplitude=a,mode=k,time=s,observable='concentration',clamped=True) for t in [.85,1.15] for a in [.2,-.3] for k in [1,2] for s in np.linspace(.1,7.,15)]


def hidden_inputs():
    return {'thermal_counterflow':[experiment(temperature=t,amplitude=-.09,thermal_amplitude=.075,mode=2,time=s) for t in [.85,1.1] for s in np.linspace(.1,8.,18)],
            'thermal_wave':[experiment(temperature=.9,amplitude=0.,thermal_amplitude=a,mode=3,time=t) for a in [.07,-.075] for t in np.linspace(.1,6.,18)],
            'concentration_heat':[experiment(temperature=1.,amplitude=.34,thermal_amplitude=0.,mode=2,time=t) for t in np.linspace(.2,8.,24)]}


def weights(temperature):
    w1=np.exp(-1/temperature);w2=9*np.exp(-3/temperature);z=1+w1+w2
    return w1/z,w2/z


def invert_energy(c,u):
    temperature=np.ones_like(c)
    for _ in range(12):
        p1,p2=weights(temperature);mean=p1+3*p2
        residual=.15*temperature+c*mean-u
        derivative=.15+c*(p1+9*p2-mean**2)/temperature**2
        step=np.clip(residual/derivative,-.3,.3)
        temperature-=step
        if np.max(abs(step))<2e-13: break
    return temperature


def profile(e,times,diffusivity,points=256):
    x=np.arange(points)*2*np.pi/points;dx=2*np.pi/points
    c=1+e['amplitude']*np.cos(e['mode']*x)
    temperature=e['temperature']+e['thermal_amplitude']*np.cos(e['mode']*x)
    p1,p2=weights(temperature);u=.15*temperature+c*(p1+3*p2)
    def rhs(t,state):
        state=state.reshape(points,2);c,u=state.T
        temperature=invert_energy(c,u);p1,p2=weights(temperature)
        f1=c*p1;f2=c*p2
        j1=-diffusivity*(np.roll(f1,-1)-f1)/dx
        j2=-.4*diffusivity*(np.roll(f2,-1)-f2)/dx
        particle=j1+j2
        energy=-.08*(np.roll(temperature,-1)-temperature)/dx+j1+3*j2
        dc=(np.roll(particle,1)-particle)/dx
        du=(np.roll(energy,1)-energy)/dx
        return np.column_stack([dc,du]).ravel()
    sparsity=lil_matrix((2*points,2*points))
    for i in range(points):
        for j in [(i-1)%points,i,(i+1)%points]: sparsity[2*i:2*i+2,2*j:2*j+2]=1
    initial=np.column_stack([c,u]).ravel()
    if max(times)==0:
        fields=np.repeat(initial[:,None],len(times),axis=1)
    else:
        result=solve_ivp(rhs,(0,max(times)),initial,t_eval=times,method='BDF',jac_sparsity=sparsity.tocsr(),rtol=2e-9,atol=2e-11)
        if not result.success: raise RuntimeError(result.message)
        fields=result.y
    fields=fields.reshape(points,2,-1);c=fields[:,0,:];u=fields[:,1,:]
    return np.array([c,invert_energy(c,u)]),u


def predict(experiments,diffusivity,points=256):
    result=np.empty(len(experiments));groups={}
    for i,e in enumerate(experiments):
        if e['clamped']:
            p1,p2=weights(e['temperature']);p=p1+.4*p2
            result[i]=e['amplitude']*np.exp(-diffusivity*p*e['mode']**2*e['time']) if e['observable']=='concentration' and e['harmonic']==1 else 0.
        else:
            key=tuple(e[k] for k in ['temperature','amplitude','thermal_amplitude','mode'])
            groups.setdefault(key,[]).append((i,e))
    for rows in groups.values():
        times=sorted(set(e['time'] for _,e in rows));fields,_=profile(rows[0][1],times,diffusivity,points)
        x=np.arange(points)*2*np.pi/points
        for i,e in rows:
            value=fields[0 if e['observable']=='concentration' else 1,:,times.index(e['time'])]
            result[i]=2*np.mean(value*np.cos(e['mode']*e['harmonic']*x))
    return result
