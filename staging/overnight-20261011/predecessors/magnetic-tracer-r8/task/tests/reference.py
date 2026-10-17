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

import itertools,time
from scipy.sparse import csc_matrix,diags,eye,kron,lil_matrix
from scipy.sparse.linalg import spsolve

@lru_cache(512)
def finite_channel(mass,gamma=.8,zeta=1.,kappa=.7,T=.5,Tb=1.4,B0=.8,B1=.9,phase=.7,V=.8,Fy=0.,degree=6,n=33):
    t=time.monotonic();states=[k for k in itertools.product(range(degree+1),repeat=4) if sum(k)<=degree];states.sort(key=lambda x:(sum(x),x));idx={a:i for i,a in enumerate(states)};h=len(states);ann=[]
    for axis in range(4):
        rows=[];cols=[];vals=[]
        for j,a in enumerate(states):
            if a[axis]:
                b=list(a);b[axis]-=1;rows.append(idx[tuple(b)]);cols.append(j);vals.append(np.sqrt(a[axis]))
        ann.append(csc_matrix((vals,(rows,cols)),shape=(h,h)))
    create=[a.T for a in ann];I=eye(h,format='csc');Ix=eye(n,format='csc');scale=(gamma*T+zeta*Tb)/(gamma+zeta)
    A=np.array([[-gamma,B0,np.sqrt(zeta/kappa),0],[-B0,-gamma,0,np.sqrt(zeta/kappa)],[-np.sqrt(zeta/kappa),0,-1/kappa,0],[0,-np.sqrt(zeta/kappa),0,-1/kappa]])
    fast=csc_matrix((h,h))
    for i in range(4):
        for j in range(4):
            if A[i,j]:fast+=A[i,j]*create[i]@(create[j]+ann[j])
    for i,D in enumerate([gamma*T,gamma*T,Tb/kappa,Tb/kappa]):fast+=D/scale*create[i]@create[i]
    # Fourier coefficients in k=-K,...,K. Truncate modes outside this band.
    wave=np.arange(-(n//2),n//2+1);plus=diags(np.ones(n-1),-1,format='csc');minus=plus.T
    field=B1/2*(np.exp(1j*phase)*plus+np.exp(-1j*phase)*minus)
    force=V/(2j)*(plus-minus)
    Lorentz=create[0]@(create[1]+ann[1])-create[1]@(create[0]+ann[0])
    L=kron(Ix,fast/mass)+kron(field,Lorentz/mass)-kron(diags(1j*wave),np.sqrt(scale/mass)*(create[0]+ann[0]))+kron(force,create[0]/np.sqrt(mass*scale))+kron(Ix,Fy*create[1]/np.sqrt(mass*scale))
    norm=(n//2)*h;L=L.tolil();L[norm,:]=0;L[norm,norm]=1;rhs=np.zeros(n*h,complex);rhs[norm]=1
    p=spsolve(L.tocsc(),rhs).reshape(n,h);density=p[:,0]
    return np.array([value for k in [1,2,3]
        for value in [density[n//2+k].real,-density[n//2+k].imag]])

previous_predict=predict
def predict(experiments,friction=TRUE_PARAMETER,mass_scale=.01,degree=6,n=33):
    result=[]
    for e in experiments:
        if e.get('geometry','harmonic')!='periodic_channel':
            result.append(previous_predict([e],friction)[0]);continue
        args=(friction,e['memory_drag'],e['memory_mass_ratio'],e['temperature'],e['memory_temperature'],
              e['field'],e['field_modulation'],e['field_phase'],e['potential_height'])
        vals=[finite_channel(mass_scale/k,*args,degree=degree,n=n) for k in [1,2,4]]
        limiting=(vals[0]-6*vals[1]+8*vals[2])/3
        result.append(limiting[2*(e['density_harmonic']-1)+(e['quadrature']=='sin')])
    return np.array(result)

def channel_experiment(field=.6,modulation=.9,phase=.8,height=.8,T=.5,Tb=1.4,zeta=1.,kappa=.7,harmonic=1,quadrature='sin'):
    return dict(geometry='periodic_channel',field=field,field_modulation=modulation,field_phase=phase,
                potential_height=height,temperature=T,memory_temperature=Tb,memory_drag=zeta,
                memory_mass_ratio=kappa,density_harmonic=harmonic,quadrature=quadrature)

uniform_hidden_inputs=hidden_inputs
def hidden_inputs():
    cases={
      'hot_contact_density':dict(field=.6,T=.5,Tb=1.4,zeta=1.,phase=.8),
      'cold_contact_density':dict(field=-.7,T=1.2,Tb=.4,zeta=1.2,phase=-1.1),
      'zero_potential_density':dict(field=.8,T=.5,Tb=1.3,zeta=.6,phase=1.,height=0.),
    }
    groups={name:[channel_experiment(**kw,kappa=k,harmonic=h) for k in [.15,.8,2.5] for h in [1,2]] for name,kw in cases.items()}
    groups['density_cosine_anchor']=[channel_experiment(**kw,kappa=.8,harmonic=h,quadrature='cos') for kw in cases.values() for h in [1,2,3]]
    groups['equal_temperature_anchor']=[channel_experiment(T=t,Tb=t,phase=p,harmonic=h,quadrature='cos') for t in [.5,1.] for p in [-.7,1.] for h in [1,2]]
    groups['harmonic_anchor']=uniform_hidden_inputs()['position_covariance_anchor']+uniform_hidden_inputs()['fixed_contact_anchor']
    return groups

import itertools, time
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron, block_diag
from scipy.sparse.linalg import spsolve

@lru_cache(512)
def finite_valve(mass,gamma=.8,zeta=1.,kappa=.7,T=.5,Tb=1.4,B0=.6,B1=.9,phase=.8,V=.8,rate=.7,drag_contrast=.6,temperature_contrast=.5,degree=4,n=25,transverse_drive=0.,drive_phase=0.):
    states=[a for a in itertools.product(range(degree+1),repeat=4) if sum(a)<=degree]
    states.sort(key=lambda a:(sum(a),a));idx={a:i for i,a in enumerate(states)};h=len(states);ann=[]
    for axis in range(4):
        rows=[];cols=[];vals=[]
        for j,a in enumerate(states):
            if a[axis]:
                b=list(a);b[axis]-=1;rows.append(idx[tuple(b)]);cols.append(j);vals.append(np.sqrt(a[axis]))
        ann.append(csc_matrix((vals,(rows,cols)),shape=(h,h)))
    create=[a.T for a in ann];Ix=eye(n,format='csc');scale=(gamma*T*(1+drag_contrast*temperature_contrast)+zeta*Tb)/(gamma+zeta)
    wave=np.arange(-(n//2),n//2+1);plus=diags(np.ones(n-1),-1,format='csc');minus=plus.T
    field=B1/2*(np.exp(1j*phase)*plus+np.exp(-1j*phase)*minus)
    force=V/(2j)*(plus-minus)
    Lorentz=create[0]@(create[1]+ann[1])-create[1]@(create[0]+ann[0])
    transport=kron(field,Lorentz/mass)-kron(diags(1j*wave),np.sqrt(scale/mass)*(create[0]+ann[0]))+kron(force,create[0]/np.sqrt(mass*scale))
    drive=transverse_drive/2*(np.exp(1j*drive_phase)*plus+np.exp(-1j*drive_phase)*minus)
    transport+=kron(drive,create[1]/np.sqrt(mass*scale))
    operators=[]
    for sign in [-1,1]:
        g=gamma*(1+sign*drag_contrast);temp=T*(1+sign*temperature_contrast)
        A=np.array([[-g,B0,np.sqrt(zeta/kappa),0],[-B0,-g,0,np.sqrt(zeta/kappa)],[-np.sqrt(zeta/kappa),0,-1/kappa,0],[0,-np.sqrt(zeta/kappa),0,-1/kappa]])
        fast=csc_matrix((h,h))
        for i in range(4):
            for j in range(4):
                if A[i,j]:fast+=A[i,j]*create[i]@(create[j]+ann[j])
        for i,D in enumerate([g*temp,g*temp,Tb/kappa,Tb/kappa]):fast+=D/scale*create[i]@create[i]
        operators.append(kron(Ix,fast/mass)+transport)
    L=block_diag(operators)+kron(csc_matrix([[-rate,rate],[rate,-rate]]),eye(n*h,format='csc'))/mass
    norm=n//2*h;L=L.tolil();L[norm,:]=0;L[norm,norm]=1;L[norm,n*h+norm]=1
    rhs=np.zeros(2*n*h,complex);rhs[norm]=1
    solution=spsolve(L.tocsc(),rhs).reshape(2,n,h);density=solution[:,:,0].sum(axis=0)
    return np.array([v for k in [1,2,3] for v in [density[n//2+k].real,-density[n//2+k].imag]])


unvalved_predict=predict
def predict(experiments,friction=TRUE_PARAMETER,mass_scale=.01,degree=4,n=25):
    result=[]
    for e in experiments:
        if e.get('geometry','harmonic')!='periodic_channel' or not (e.get('drag_contrast',0.) or e.get('temperature_contrast',0.)):
            result.append(unvalved_predict([e],friction)[0]);continue
        args=(friction,e['memory_drag'],e['memory_mass_ratio'],e['temperature'],e['memory_temperature'],
              e['field'],e['field_modulation'],e['field_phase'],e['potential_height'],
              e.get('valve_rate',1.),e.get('drag_contrast',0.),e.get('temperature_contrast',0.))
        vals=[finite_valve(mass_scale/k,*args,degree=degree,n=n) for k in [1,2,4]]
        limiting=(vals[0]-6*vals[1]+8*vals[2])/3
        result.append(limiting[2*(e['density_harmonic']-1)+(e['quadrature']=='sin')])
    return np.array(result)

unvalved_hidden_inputs=hidden_inputs
def valve_experiment(rate=.4,drag=.65,contrast=.55,**kwargs):
    return dict(channel_experiment(**kwargs),valve_rate=rate,drag_contrast=drag,temperature_contrast=contrast)

def hidden_inputs():
    groups=unvalved_hidden_inputs()
    settings={
      'valve_hot_memory':dict(T=.5,Tb=1.4,field=.6,phase=.8,drag=.65,contrast=.55),
      'valve_cold_memory':dict(T=1.2,Tb=.4,field=-.7,phase=-1.1,drag=.65,contrast=-.55),
      'valve_zero_potential':dict(T=.5,Tb=1.3,field=.8,phase=1.,drag=-.65,contrast=.55,height=0.),
    }
    for name,kw in settings.items():
        groups[name]=[valve_experiment(**kw,rate=r,kappa=k,harmonic=h) for r,k in [(.15,.2),(.6,.8),(2.5,2.5)] for h in [1,2]]
    groups['valve_cosine_anchor']=[valve_experiment(**kw,rate=.3,kappa=.8,harmonic=h,quadrature='cos') for kw in settings.values() for h in [1,2,3]]
    groups['valve_equilibrium_anchor']=[valve_experiment(T=t,Tb=t,contrast=0.,drag=d,rate=r,harmonic=h,quadrature='cos') for t,d,r in [(.5,.7,.1),(1.2,-.7,3.)] for h in [1,2,3]]
    return groups

import itertools, time
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron, block_diag
from scipy.sparse.linalg import spsolve

@lru_cache(512)
def finite_circulation(mass,gamma=.8,zeta=1.,kappa=.7,T=.5,Tb=1.4,B0=.6,B1=.9,phase=.8,V=.8,rate=.7,drag_contrast=.6,temperature_contrast=.5,degree=4,n=25,harmonic=1,quadrature='cos',transverse_drive=0.,drive_phase=0.):
    states=[a for a in itertools.product(range(degree+1),repeat=4) if sum(a)<=degree]
    states.sort(key=lambda a:(sum(a),a));idx={a:i for i,a in enumerate(states)};h=len(states);ann=[]
    for axis in range(4):
        rows=[];cols=[];vals=[]
        for j,a in enumerate(states):
            if a[axis]:
                b=list(a);b[axis]-=1;rows.append(idx[tuple(b)]);cols.append(j);vals.append(np.sqrt(a[axis]))
        ann.append(csc_matrix((vals,(rows,cols)),shape=(h,h)))
    create=[a.T for a in ann];Ix=eye(n,format='csc');scale=(gamma*T*(1+drag_contrast*temperature_contrast)+zeta*Tb)/(gamma+zeta)
    wave=np.arange(-(n//2),n//2+1);plus=diags(np.ones(n-1),-1,format='csc');minus=plus.T
    field=B1/2*(np.exp(1j*phase)*plus+np.exp(-1j*phase)*minus)
    force=V/(2j)*(plus-minus)
    Lorentz=create[0]@(create[1]+ann[1])-create[1]@(create[0]+ann[0])
    transport=kron(field,Lorentz/mass)-kron(diags(1j*wave),np.sqrt(scale/mass)*(create[0]+ann[0]))+kron(force,create[0]/np.sqrt(mass*scale))
    drive=transverse_drive/2*(np.exp(1j*drive_phase)*plus+np.exp(-1j*drive_phase)*minus)
    transport+=kron(drive,create[1]/np.sqrt(mass*scale))
    operators=[]
    for sign in [-1,1]:
        g=gamma*(1+sign*drag_contrast);temp=T*(1+sign*temperature_contrast)
        A=np.array([[-g,B0,np.sqrt(zeta/kappa),0],[-B0,-g,0,np.sqrt(zeta/kappa)],[-np.sqrt(zeta/kappa),0,-1/kappa,0],[0,-np.sqrt(zeta/kappa),0,-1/kappa]])
        fast=csc_matrix((h,h))
        for i in range(4):
            for j in range(4):
                if A[i,j]:fast+=A[i,j]*create[i]@(create[j]+ann[j])
        for i,D in enumerate([g*temp,g*temp,Tb/kappa,Tb/kappa]):fast+=D/scale*create[i]@create[i]
        operators.append(kron(Ix,fast/mass)+transport)
    L=block_diag(operators)+kron(csc_matrix([[-rate,rate],[rate,-rate]]),eye(n*h,format='csc'))/mass

    plus_k=diags(np.ones(n-harmonic),-harmonic,format='csc');minus_k=plus_k.T
    weight=(plus_k+minus_k)/2 if quadrature=='cos' else (plus_k-minus_k)/(2j)
    A1=kron(eye(2),kron(weight,np.sqrt(scale/mass)*(create[1]+ann[1]))).tocsc()
    size=2*n*h;norm=n//2*h;normalization=np.zeros(size);normalization[norm]=1;normalization[n*h+norm]=1
    system=L.tolil();system[norm,:]=normalization;system=system.tocsc();rhs=np.zeros(size,complex);rhs[norm]=1
    p=spsolve(system,rhs);c1=normalization@(A1@p)
    rhs=c1*p-A1@p;rhs[norm]=0;r1=spsolve(system,rhs);c2=2*normalization@(A1@r1)
    rhs=c2*p+2*c1*r1-2*A1@r1;rhs[norm]=0;r2=spsolve(system,rhs);c3=3*normalization@(A1@r2)
    result=np.array([c1,c2,c3]);assert max(abs(result.imag))<1e-8
    return result.real


density_predict=predict
def predict(experiments,friction=TRUE_PARAMETER,mass_scale=.0025,degree=4,n=25):
    result=[]
    for e in experiments:
        observable=e.get('observable','density')
        if observable=='density':
            result.append(density_predict([e],friction,mass_scale,degree,n)[0]);continue
        args=(friction,e['memory_drag'],e['memory_mass_ratio'],e['temperature'],e['memory_temperature'],
              e['field'],e['field_modulation'],e['field_phase'],e['potential_height'],
              e.get('valve_rate',1.),e.get('drag_contrast',0.),e.get('temperature_contrast',0.))
        vals=[finite_circulation(mass_scale/k,*args,degree=degree,n=n,harmonic=e['density_harmonic'],quadrature=e['quadrature']) for k in [1,2,4]]
        limiting=(vals[0]-6*vals[1]+8*vals[2])/3
        result.append(limiting[{'circulation_mean':0,'circulation_noise':1,'circulation_third':2}[observable]])
    return np.array(result)

density_hidden_inputs=hidden_inputs
def circulation_experiment(observable='circulation_third',**kwargs):
    return dict(valve_experiment(**kwargs),observable=observable)

def hidden_inputs():
    groups=density_hidden_inputs()
    settings=[dict(T=.5,Tb=1.4,field=.6,phase=.8,drag=.65,contrast=.55,rate=.2,kappa=.3),
              dict(T=1.2,Tb=.4,field=-.7,phase=-1.1,drag=.65,contrast=-.55,rate=.7,kappa=1.2),
              dict(T=.5,Tb=1.3,field=.8,phase=1.,drag=-.65,contrast=.55,rate=2.,kappa=2.5,height=0.)]
    for observable in ['circulation_mean','circulation_noise','circulation_third']:
        groups[observable]=[circulation_experiment(observable,**kw,harmonic=h,quadrature=q) for kw in settings for h,q in [(1,'cos'),(2,'sin')]]
    groups['equilibrium_circulation_noise']=[circulation_experiment('circulation_noise',T=t,Tb=t,contrast=0.,drag=d,rate=r,harmonic=h,quadrature=q) for t,d,r in [(.5,.7,.1),(1.2,-.7,3.)] for h,q in [(1,'cos'),(2,'sin')]]
    return groups


undriven_predict=predict
def predict(experiments,friction=TRUE_PARAMETER,mass_scale=.0025,degree=4,n=25):
    result=[]
    for e in experiments:
        drive=e.get('transverse_drive',0.)
        if not drive:
            result.append(undriven_predict([e],friction,mass_scale,degree,n)[0]);continue
        args=(friction,e['memory_drag'],e['memory_mass_ratio'],e['temperature'],e['memory_temperature'],
              e['field'],e['field_modulation'],e['field_phase'],e['potential_height'],
              e.get('valve_rate',1.),e.get('drag_contrast',0.),e.get('temperature_contrast',0.))
        kw=dict(degree=degree,n=n,transverse_drive=drive,drive_phase=e.get('drive_phase',0.))
        observable=e.get('observable','density')
        if observable=='density':
            values=[finite_valve(mass_scale/k,*args,**kw) for k in [1,2,4]]
            limiting=(values[0]-6*values[1]+8*values[2])/3
            result.append(limiting[2*(e['density_harmonic']-1)+(e['quadrature']=='sin')])
        else:
            values=[finite_circulation(mass_scale/k,*args,harmonic=e['density_harmonic'],quadrature=e['quadrature'],**kw) for k in [1,2,4]]
            limiting=(values[0]-6*values[1]+8*values[2])/3
            result.append(limiting[{'circulation_mean':0,'circulation_noise':1,'circulation_third':2}[observable]])
    return np.array(result)

undriven_hidden_inputs=hidden_inputs
def hidden_inputs():
    groups=undriven_hidden_inputs()
    settings=[dict(T=.5,Tb=1.4,field=.6,phase=.8,drag=.65,contrast=.55,rate=.2,kappa=.3),
              dict(T=1.2,Tb=.4,field=-.7,phase=-1.1,drag=.65,contrast=-.55,rate=.7,kappa=1.2),
              dict(T=.5,Tb=1.3,field=.8,phase=1.,drag=-.65,contrast=.55,rate=2.,kappa=2.5,height=0.)]
    for obs in ['density','circulation_mean','circulation_noise','circulation_third']:
        groups['driven_'+obs]=[dict(circulation_experiment(obs,**kw,harmonic=h,quadrature=q),transverse_drive=f,drive_phase=ph)
            for kw,f,ph in zip(settings,[.65,-.6,.7],[.4,-.7,1.2]) for h,q in [(1,'cos'),(2,'sin')]]
    return groups
