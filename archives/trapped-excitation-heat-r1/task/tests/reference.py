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
    return {'concentration_to_heat':[experiment(time=t,amplitude=a) for a in [.22,-.32] for t in np.linspace(.15,7.,16)],
            'thermal_feedback':[experiment(temperature=.9,amplitude=.28,thermal_amplitude=-.065,mode=2,time=t,observable=o) for o in ['temperature','concentration'] for t in np.linspace(.1,4.,16)],
            'generated_harmonic':[experiment(temperature=1.1,amplitude=.34,thermal_amplitude=.07,time=t,observable='temperature',harmonic=2) for t in np.linspace(.2,8.,20)]}


def invert_energy(c,u):
    temperature=np.ones_like(c)
    for _ in range(12):
        free=1/(1+np.exp(2/temperature))
        residual=.15*temperature+2*c*free-u
        derivative=.15+4*c*free*(1-free)/temperature**2
        step=np.clip(residual/derivative,-.3,.3)
        temperature-=step
        if np.max(abs(step))<2e-13: break
    return temperature


def profile(e,times,diffusivity,points=256):
    x=np.arange(points)*2*np.pi/points;dx=2*np.pi/points
    c=1+e['amplitude']*np.cos(e['mode']*x)
    temperature=e['temperature']+e['thermal_amplitude']*np.cos(e['mode']*x)
    u=.15*temperature+2*c/(1+np.exp(2/temperature))
    def rhs(t,state):
        state=state.reshape(points,2);c,u=state.T
        temperature=invert_energy(c,u);free=c/(1+np.exp(2/temperature))
        # Face currents enter cells from the left and leave to the right.
        particle=-diffusivity*(np.roll(free,-1)-free)/dx
        energy=-.08*(np.roll(temperature,-1)-temperature)/dx+2*particle
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
            p=1/(1+np.exp(2/e['temperature']))
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
