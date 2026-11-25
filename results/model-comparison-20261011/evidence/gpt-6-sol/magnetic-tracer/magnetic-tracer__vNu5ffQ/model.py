"""Small-mass charged tracer with a retained linear thermal contact."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

KINDS = {'xx': 0, 'xy': 1, 'yy': 2, 'area': 3}
ROTATION = np.array([[0., 1.], [-1., 0.]])


@lru_cache(4096)
def instantaneous(friction, field, temperature, kx, ky, time, cxx, cxy, cyy):
    mobility = np.array([[friction, field], [-field, friction]])/(friction**2+field**2)
    drift = -mobility@np.diag([kx, ky])
    stationary = np.diag([temperature/kx, temperature/ky])
    delta = np.array([[cxx, cxy], [cxy, cyy]])-stationary
    propagator = expm(time*drift)
    evolved = propagator@delta@propagator.T
    covariance = stationary+evolved
    weight = (ROTATION@drift+(ROTATION@drift).T)/2
    primitive = solve_continuous_lyapunov(drift.T, -weight)
    area = np.trace(primitive@(delta-evolved))
    return np.array([covariance[0,0],covariance[0,1],covariance[1,1],area])


@lru_cache(4096)
def retained(friction, field, temperature, kx, ky, time, cxx, cxy, cyy,
             memory_drag, memory_time, memory_temperature):
    if memory_drag == 0:
        return instantaneous(friction,field,temperature,kx,ky,time,cxx,cxy,cyy)
    mobility = np.array([[friction,field],[-field,friction]])/(friction**2+field**2)
    stiffness = np.diag([kx,ky])
    ratio = memory_drag/memory_time
    drift = np.block([[-mobility@stiffness,mobility],
                      [ratio*mobility@stiffness,-np.eye(2)/memory_time-ratio*mobility]])
    shared = np.vstack([mobility,-ratio*mobility])
    noise = 2*friction*temperature*(shared@shared.T)
    noise[2:,2:] += 2*memory_drag*memory_temperature/memory_time**2*np.eye(2)
    stationary = solve_continuous_lyapunov(drift,-noise)
    initial = np.zeros((4,4))
    initial[:2,:2] = [[cxx,cxy],[cxy,cyy]]
    initial[2:,2:] = memory_drag*memory_temperature/memory_time*np.eye(2)
    delta = initial-stationary
    propagator = expm(time*drift)
    evolved = propagator@delta@propagator.T
    covariance = stationary+evolved
    projector = np.c_[np.eye(2),np.zeros((2,2))]
    raw = projector.T@ROTATION@drift[:2,:]
    weight = (raw+raw.T)/2
    primitive = solve_continuous_lyapunov(drift.T,-weight)
    fast_velocity_area = temperature*np.trace(ROTATION@mobility)
    area = time*(np.trace(weight@stationary)+fast_velocity_area)+np.trace(primitive@(delta-evolved))
    return np.array([covariance[0,0],covariance[0,1],covariance[1,1],area])



@lru_cache(4096)
def kinetic_contact(friction,field,temperature,kx,ky,time,cxx,cxy,cyy,
                    memory_drag,memory_mass_ratio,memory_temperature):
    gamma=friction;zeta=memory_drag;ratio=memory_mass_ratio
    mobility=np.linalg.inv((gamma+zeta)*np.eye(2)-field*ROTATION)
    drift=-mobility@np.diag([kx,ky])
    diffusion=2*(gamma*temperature+zeta*memory_temperature)*(mobility@mobility.T)
    stationary=solve_continuous_lyapunov(drift,-diffusion)
    initial=np.array([[cxx,cxy],[cxy,cyy]])
    propagator=expm(time*drift);delta=initial-stationary
    evolved=propagator@delta@propagator.T;covariance=stationary+evolved
    weight=(ROTATION@drift+(ROTATION@drift).T)/2
    primitive=solve_continuous_lyapunov(drift.T,-weight)
    area=time*np.trace(weight@stationary)+np.trace(primitive@(delta-evolved))
    # Scale the fast contact coordinate by sqrt(ratio) to balance its covariance.
    fast=np.block([[-gamma*np.eye(2)+field*ROTATION,np.eye(2)/np.sqrt(ratio)],
                   [-zeta/np.sqrt(ratio)*np.eye(2),-np.eye(2)/ratio]])
    noise=np.diag([2*gamma*temperature]*2+[2*zeta*memory_temperature/ratio]*2)
    fast_covariance=solve_continuous_lyapunov(fast,-noise)
    integrated=-np.linalg.solve(fast,fast_covariance.T).T[:2,:2]
    area+=time*(integrated[0,1]-integrated[1,0])
    return np.array([covariance[0,0],covariance[0,1],covariance[1,1],area])


def zero_field_calibration(experiments, friction):
    t=np.array([e['time'] for e in experiments]);kx=np.array([e['stiffness'][0] for e in experiments]);ky=np.array([e['stiffness'][1] for e in experiments])
    temp=np.array([e['temperature'] for e in experiments]);c=np.array([e['initial_covariance'] for e in experiments])
    xx=temp/kx+(c[:,0,0]-temp/kx)*np.exp(-2*kx*t/friction)
    xy=c[:,0,1]*np.exp(-(kx+ky)*t/friction)
    yy=temp/ky+(c[:,1,1]-temp/ky)*np.exp(-2*ky*t/friction)
    area=(kx-ky)/(kx+ky)*c[:,0,1]*(1-np.exp(-(kx+ky)*t/friction))
    return np.array([np.array([xx[i],xy[i],yy[i],area[i]])[KINDS[e['readout']]] for i,e in enumerate(experiments)])


def fit_friction(records):
    records=list(records)
    inputs=[r['input'] for r in records]
    assert all(e['field']==0 and e.get('memory_drag',0)==0 for e in inputs)
    values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
    def objective(friction):
        residual=(zero_field_calibration(inputs,friction)-values)/sigma
        return residual@residual
    result=minimize_scalar(objective,bounds=(.5,1.5),method='bounded',options={'xatol':1e-12})
    return float(result.x)

@lru_cache(512)
def channel_density(gamma=.8,zeta=1.,kappa=.7,T=.5,Tb=1.4,B0=.8,B1=.9,phase=.7,V=.8,Fy=0.,n=65):
    x=np.arange(n)*2*np.pi/n;B=B0+B1*np.cos(x+phase);prime=-B1*np.sin(x+phase)
    forces=np.array([V*np.sin(x),np.full(n,Fy)]).T;drift=[];diff=[]
    for b,db,F in zip(B,prime,forces):
        M=np.linalg.inv((gamma+zeta)*np.eye(2)-b*ROTATION);dM=M@ROTATION@M*db
        A=np.block([[-gamma*np.eye(2)+b*ROTATION,np.sqrt(zeta/kappa)*np.eye(2)],[-np.sqrt(zeta/kappa)*np.eye(2),-np.eye(2)/kappa]])
        Q=np.diag([2*gamma*T]*2+[2*Tb/kappa]*2)
        S=solve_continuous_lyapunov(A,-Q)
        C=np.c_[M,np.sqrt(zeta*kappa)*M];D=C@S[:,:2]
        expected=(gamma*T+zeta*Tb)/( (gamma+zeta)**2+b*b )
        assert np.max(abs(D+D.T-2*expected*np.eye(2)))<1e-10
        response=dM@(S[:2,0]+np.sqrt(zeta*kappa)*S[2:,0])
        drift.append(M@F+response);diff.append(expected)
    drift=np.array(drift);diff=np.array(diff);wave=np.fft.fftfreq(n,1/n)
    dx=np.fft.ifft(1j*wave[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
    dxx=np.fft.ifft(-wave[:,None]**2*np.fft.fft(np.eye(n),axis=0),axis=0).real
    L=-dx@np.diag(drift[:,0])+dxx@np.diag(diff);rhs=np.zeros(n);rhs[0]=1;L[0]=1/n
    p=np.linalg.solve(L,rhs)
    assert p.min()>0
    return x,p,drift,diff

@lru_cache(512)
def channel_moments(gamma,zeta,kappa,T,Tb,B0,B1,phase,V):
    x,p,_,_=channel_density(gamma,zeta,kappa,T,Tb,B0,B1,phase,V)
    return np.array([value for k in [1,2,3]
                     for value in [np.mean(p*np.cos(k*x)),np.mean(p*np.sin(k*x))]])

P=np.c_[np.eye(2),np.zeros((2,2))]
def valve_cell(gamma,zeta,kappa,T,Tb,field,rate,drag_contrast,temperature_contrast,fluid_anisotropy=0.,fluid_axis=0.):
    matrices=[];noises=[];gammas=[];temperatures=[]
    a=fluid_anisotropy;phi=fluid_axis;theta=np.array([[1+a*np.cos(2*phi),a*np.sin(2*phi)],[a*np.sin(2*phi),1-a*np.cos(2*phi)]])
    for sign in [-1,1]:
        g=gamma*(1+sign*drag_contrast);temp=T*(1+sign*temperature_contrast);gammas.append(g);temperatures.append(temp)
        matrices.append(np.block([[-g*np.eye(2)+field*ROTATION,np.sqrt(zeta/kappa)*np.eye(2)],[-np.sqrt(zeta/kappa)*np.eye(2),-np.eye(2)/kappa]]))
        noise=np.diag([2*g*temp]*2+[2*Tb/kappa]*2);noise[:2,:2]=2*g*temp*theta;noises.append(noise)
    q=np.eye(4);H=np.block([[matrices[0].T-rate*q,rate*q],[rate*q,matrices[1].T-rate*q]])
    coef=np.linalg.solve(H,np.vstack([-P.T,-P.T]));change=np.zeros((8,8));change[:2,:2]=ROTATION.T;change[4:6,4:6]=ROTATION.T
    derivative=-np.linalg.solve(H,change@coef);C=[coef[:4].T,coef[4:].T];dC=[derivative[:4].T,derivative[4:].T]
    K=[np.kron(q,a)+np.kron(a,q) for a in matrices];joint=np.block([[K[0]-rate*np.eye(16),rate*np.eye(16)],[rate*np.eye(16),K[1]-rate*np.eye(16)]])
    rhs=-.5*np.concatenate([q.flatten(order='F') for q in noises]);cov=np.linalg.solve(joint,rhs);S=[cov[:16].reshape(4,4,order='F'),cov[16:].reshape(4,4,order='F')]
    M=.5*(C[0][:,:2]+C[1][:,:2]);K=sum(C[i]@S[i][:,:2] for i in range(2));D=(K+K.T)/2
    drift=sum(dC[i]@S[i][:,0] for i in range(2))
    energy=sum(gammas[i]*(np.trace(S[i][:2,:2])-temperatures[i])+(np.trace(S[i][2:,2:])-Tb)/kappa for i in range(2))
    entropy=sum(gammas[i]*(np.trace(np.linalg.solve(theta,S[i][:2,:2]))-temperatures[i])/temperatures[i]+(np.trace(S[i][2:,2:])-Tb)/(kappa*Tb) for i in range(2))
    assert abs(energy)<1e-10 and entropy>-1e-10 and np.linalg.eigvalsh(D).min()>0
    return M,D,drift,S,K

@lru_cache(512)
def valve_moments(gamma,zeta=1.,kappa=.7,T=.5,Tb=1.4,B0=.6,B1=.9,phase=.8,V=.8,rate=.7,drag_contrast=.6,temperature_contrast=.5,n=65,transverse_drive=0.,drive_phase=0.,fluid_anisotropy=0.,fluid_axis=0.):
    x=np.arange(n)*2*np.pi/n;B=B0+B1*np.cos(x+phase);Bp=-B1*np.sin(x+phase);drift=[];diff=[]
    for b,bp,angle in zip(B,Bp,x):
        M,D,c,_,_=valve_cell(gamma,zeta,kappa,T,Tb,b,rate,drag_contrast,temperature_contrast,fluid_anisotropy,fluid_axis);drift.append((M@np.array([V*np.sin(angle),transverse_drive*np.cos(angle+drive_phase)])+bp*c)[0]);diff.append(D[0,0])
    wave=np.fft.fftfreq(n,1/n);dx=np.fft.ifft(1j*wave[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real;dxx=np.fft.ifft(-wave[:,None]**2*np.fft.fft(np.eye(n),axis=0),axis=0).real
    L=-dx@np.diag(drift)+dxx@np.diag(diff);rhs=np.zeros(n);rhs[0]=1;L[0]=1/n;p=np.linalg.solve(L,rhs)
    assert p.min()>0
    return np.array([v for k in [1,2,3] for v in [np.mean(p*np.cos(k*x)),np.mean(p*np.sin(k*x))]])

@lru_cache(512)
def circulation(gamma,zeta=1.,kappa=.7,T=.5,Tb=1.4,B0=.6,B1=.9,phase=.8,V=.8,rate=.7,drag_contrast=.6,temperature_contrast=.5,n=65,harmonic=1,quadrature='cos',transverse_drive=0.,drive_phase=0.,fluid_anisotropy=0.,fluid_axis=0.):
    x=np.arange(n)*2*np.pi/n;B=B0+B1*np.cos(x+phase);Bp=-B1*np.sin(x+phase);drift=[];diff=[];g=[];h=[];cross=[]
    weight=np.cos(harmonic*x) if quadrature=='cos' else np.sin(harmonic*x)
    derivative=-harmonic*np.sin(harmonic*x) if quadrature=='cos' else harmonic*np.cos(harmonic*x)
    for j,(b,bp,angle) in enumerate(zip(B,Bp,x)):
        M,D,c,S,K=valve_cell(gamma,zeta,kappa,T,Tb,b,rate,drag_contrast,temperature_contrast,fluid_anisotropy,fluid_axis)
        velocity=M@np.array([V*np.sin(angle),transverse_drive*np.cos(angle+drive_phase)])+bp*c
        drift.append(velocity[0]);diff.append(D[0,0]);g.append(weight[j]*velocity[1]+derivative[j]*K[1,0]);h.append(weight[j]**2*D[1,1]);cross.append(2*weight[j]*D[0,1])
    wave=np.fft.fftfreq(n,1/n);dx=np.fft.ifft(1j*wave[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real;dxx=np.fft.ifft(-wave[:,None]**2*np.fft.fft(np.eye(n),axis=0),axis=0).real
    L=-dx@np.diag(drift)+dxx@np.diag(diff);system=L.copy();system[0]=1;rhs=np.zeros(n);rhs[0]=1;p=np.linalg.solve(system,rhs)
    # The weighted transverse record and x use the same microscopic velocity.
    # Their limiting Brownian increments therefore have cross diffusivity
    # a(x) * D_xy.  In the tilted forward generator this contributes
    # -lambda * d/dx [2*a(x)*D_xy(x)*p].  It is needed for both the variance
    # and the third cumulant, including when the fluid bath is anisotropic.
    A1=np.diag(g)-dx@np.diag(cross);A2=2*np.diag(h);c1=float(np.sum(A1@p))
    rhs=(c1*np.eye(n)-A1)@p;rhs[0]=0.;r1=np.linalg.solve(system,rhs);c2=float(np.sum(A2@p+2*A1@r1))
    rhs=c2*p+2*c1*r1-2*A1@r1-A2@p;rhs[0]=0.;r2=np.linalg.solve(system,rhs);c3=float(np.sum(3*A1@r2+3*A2@r1))
    return np.array([c1,c2,c3])

def channel_prediction(e,friction):
    observable=e.get('observable','density')
    if observable!='density':
        values=circulation(float(friction),float(e['memory_drag']),float(e['memory_mass_ratio']),
            float(e['temperature']),float(e['memory_temperature']),float(e['field']),float(e['field_modulation']),
            float(e['field_phase']),float(e['potential_height']),float(e.get('valve_rate',1.)),
            float(e.get('drag_contrast',0.)),float(e.get('temperature_contrast',0.)),
            harmonic=int(e['density_harmonic']),quadrature=e['quadrature'],transverse_drive=float(e.get('transverse_drive',0.)),drive_phase=float(e.get('drive_phase',0.)),fluid_anisotropy=float(e.get('fluid_anisotropy',0.)),fluid_axis=float(e.get('fluid_axis',0.)))
        return values[{'circulation_mean':0,'circulation_noise':1,'circulation_third':2}[observable]]
    moments=valve_moments(float(friction),float(e['memory_drag']),float(e['memory_mass_ratio']),
        float(e['temperature']),float(e['memory_temperature']),float(e['field']),
        float(e['field_modulation']),float(e['field_phase']),float(e['potential_height']),
        float(e.get('valve_rate',1.)),float(e.get('drag_contrast',0.)),float(e.get('temperature_contrast',0.)),transverse_drive=float(e.get('transverse_drive',0.)),drive_phase=float(e.get('drive_phase',0.)),fluid_anisotropy=float(e.get('fluid_anisotropy',0.)),fluid_axis=float(e.get('fluid_axis',0.)))
    return moments[2*(e['density_harmonic']-1)+(e['quadrature']=='sin')]


def predict_at(experiments,friction):
    if not experiments:return np.empty(0)
    out=[]
    for e in experiments:
        if e.get('geometry','harmonic')=='periodic_channel':
            out.append(channel_prediction(e,friction));continue
        c=e['initial_covariance']
        function=kinetic_contact if 'memory_mass_ratio' in e else retained
        memory_scale=e['memory_mass_ratio'] if 'memory_mass_ratio' in e else e.get('memory_time',1)
        values=function(float(friction),float(e['field']),float(e['temperature']),
                        *map(float,e['stiffness']),float(e['time']),float(c[0][0]),float(c[0][1]),float(c[1][1]),
                        float(e.get('memory_drag',0)),float(memory_scale),
                        float(e.get('memory_temperature',e['temperature'])))
        out.append(values[KINDS[e['readout']]])
    return np.array(out)


class Model:
    def __init__(self):self.friction=None
    def fit(self,records):
        self.friction=fit_friction(records)
        return self
    def predict(self,experiments):return predict_at(experiments,self.friction)
