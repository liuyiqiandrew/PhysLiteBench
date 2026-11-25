"""Full finite-mass six-dimensional OU covariance, measured before extrapolation."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov

TRUE_PARAMETER=.8


@lru_cache(4096)
def finite_mass(friction,field,temperature,kx,ky,time,cxx,cxy,cyy,memory_drag,memory_time,memory_temperature,mass):
    a=np.zeros((6,6));a[:2,2:4]=np.eye(2)
    a[2:4,:2]=-np.diag([kx,ky])/mass
    a[2:4,2:4]=np.array([[-friction,field],[-field,-friction]])/mass
    a[2:4,4:]=np.eye(2)/mass
    a[4:,2:4]=-memory_drag/memory_time*np.eye(2)
    a[4:,4:]=-np.eye(2)/memory_time
    q=np.zeros((6,6));q[2:4,2:4]=2*friction*temperature/mass**2*np.eye(2)
    q[4:,4:]=2*memory_drag*memory_temperature/memory_time**2*np.eye(2)
    equilibrium=solve_continuous_lyapunov(a,-q)
    initial=np.zeros((6,6));initial[:2,:2]=[[cxx,cxy],[cxy,cyy]]
    initial[2:4,2:4]=temperature/mass*np.eye(2)
    initial[4:,4:]=memory_drag*memory_temperature/memory_time*np.eye(2)
    delta=initial-equilibrium;propagator=expm(time*a);evolved=propagator@delta@propagator.T
    covariance=equilibrium+evolved
    measure=np.zeros((6,6));measure[0,3]=measure[3,0]=.5;measure[1,2]=measure[2,1]=-.5
    primitive=solve_continuous_lyapunov(a.T,-measure)
    area=time*np.trace(measure@equilibrium)+np.trace(primitive@(delta-evolved))
    return np.array([covariance[0,0],covariance[0,1],covariance[1,1],area])


def predict(experiments,friction=TRUE_PARAMETER,mass_scale=.001):
    output=[];index={'xx':0,'xy':1,'yy':2,'area':3}
    for e in experiments:
        c=e['initial_covariance'];tau=e.get('memory_time',1.)
        mass=mass_scale*min(friction**2/max(e['stiffness']),friction*tau)
        args=(float(friction),e['field'],e['temperature'],*e['stiffness'],e['time'],c[0][0],c[0][1],c[1][1],e.get('memory_drag',0.),tau,e.get('memory_temperature',e['temperature']))
        vals=[]
        for factor in [1,2,4]:
            m=mass/factor
            actual_args=list(args)
            if 'memory_mass_ratio' in e:actual_args[-2]=m*e['memory_mass_ratio']
            vals.append(finite_mass(*actual_args,m))
        limit=(vals[0]-6*vals[1]+8*vals[2])/3
        output.append(limit[index[e['readout']]])
    return np.array(output)


def experiment(time,field=.9,temperature=.8,stiffness=(.8,1.6),initial=((1.5,.3),(.3,.5)),readout='area',memory_drag=.8,memory_time=.8,memory_temperature=1.1):
    return dict(time=float(time),field=float(field),temperature=float(temperature),stiffness=list(stiffness),initial_covariance=[list(r) for r in initial],readout=readout,memory_drag=memory_drag,memory_time=memory_time,memory_temperature=memory_temperature)


def hidden_inputs():
    scenarios={
      'retained_contact':dict(memory_drag=.9,memory_time=.8,memory_temperature=1.1),
      'cold_contact':dict(field=-1.4,temperature=1.1,memory_drag=1.3,memory_time=.3,memory_temperature=.4,stiffness=(1.5,.7)),
      'slow_contact':dict(field=1.8,temperature=.5,memory_drag=.6,memory_time=1.6,memory_temperature=1.3,initial=((1.2,-.4),(-.4,1.1))),
      'zero_field_memory':dict(field=0.,temperature=.9,memory_drag=1.2,memory_time=1.2,memory_temperature=.5),
      'memory_free_anchor':dict(field=-1.5,temperature=.8,memory_drag=0.,memory_time=1.,memory_temperature=.8),
    }
    return {name:[experiment(t,readout=r,**settings) for t in [.15,.4,.9,1.8,3.5] for r in ['xx','xy','yy','area']] for name,settings in scenarios.items()}

previous_hidden_inputs=hidden_inputs
def hidden_inputs():
    scenarios={
        'hot_fast_contact':dict(field=1.2,temperature=.5,memory_drag=1.2,memory_temperature=1.4),
        'cold_fast_contact':dict(field=-1.7,temperature=1.2,memory_drag=1.1,memory_temperature=.4),
        'slow_kinetic_contact':dict(field=1.8,temperature=.5,memory_drag=.7,memory_temperature=1.3),
    }
    result={}
    for name,settings in scenarios.items():
        ratios=[.12,.4,.9] if name!='slow_kinetic_contact' else [1.2,2.,3.]
        result[name]=[dict(experiment(t,**settings),memory_mass_ratio=ratio)
                      for ratio in ratios for t in [.3,1.2,3.5]]
    result['position_covariance_anchor']=[dict(experiment(1.5,**settings),memory_mass_ratio=.7,readout=readout)
                                         for settings in scenarios.values() for readout in ['xx','xy','yy']]
    result['fixed_contact_anchor']=previous_hidden_inputs()['retained_contact']
    result['memory_free_anchor']=previous_hidden_inputs()['memory_free_anchor']
    return result
