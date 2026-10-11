"""Independent finite-mass contact calorimetry with measured weighted residence."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

PARAMETER='friction'
TRUE_PARAMETER=1.1
INPUT_FIELDS=('force_x','force_y','drag_ratio','magnetic','temperature','contrast','wavenumber','bath_contrast','switch_rate')


def experiment(force_x=.12,force_y=.1,drag_ratio=1.,magnetic=1.5,temperature=1.,contrast=.3,wavenumber=1,bath_contrast=.3,switch_rate=.8):
    return dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))


def port(parameters):
    e=experiment(*parameters[:9])
    e['detector_phase']=parameters[9]
    return e


def hidden_inputs():
    return {
        'isotropic_drive': [port(p) for p in [
            (.8,0.,1.,0.,1.,.6,1,.6,.2,-np.pi/2),
            (-.8,0.,1.,0.,1.,.6,1,.6,.2,np.pi/2),
            (1.2,.2,1.,0.,1.,.65,2,.65,.3,-np.pi/2),
            (-1.2,-.2,1.,0.,1.,.65,2,.65,.3,np.pi/2)]],
        'anisotropic_contacts': [port(p) for p in [
            (.8,-.5,.4,0.,1.,.65,3,.65,.2,0.),
            (-.8,.5,.4,0.,1.,.65,3,.65,.2,0.),
            (1.,.2,4.,0.,1.,.65,2,.65,.2,-np.pi/2),
            (-1.,-.2,4.,0.,1.,.65,2,.65,.2,np.pi/2)]],
        'magnetic_drive': [port(p) for p in [
            (.6,1.2,8.,5.,1.1,.65,2,.65,.2,-np.pi/2),
            (-.6,-1.2,8.,5.,1.1,.65,2,.65,.2,np.pi/2),
            (-1.3,1.,8.,-5.,1.1,.65,3,.65,.2,np.pi/2),
            (1.3,-1.,8.,-5.,1.1,.65,3,.65,.2,-np.pi/2)]],
        'bulk_anchor': [experiment(.12,.1,ratio,field,1.,.55,2,delta,rate)
            for field,ratio,delta,rate in [(-3.,.4,.65,.2),(2.,1.,.6,.8),(3.5,5.,0.,2.)]],
        'uniform_port_anchor': [port(p) for p in [
            (.6,-.8,.25,5.,1.,0.,1,.65,.2,0.),
            (.3,.4,1.,0.,1.2,0.,2,.6,.3,-np.pi/2),
            (-.4,.5,8.,-3.,.9,0.,3,.55,2.,np.pi/2)]],
    }


def kinetic_state(e,gamma,mass,degree=12,spatial=41):
    modes=np.arange(-(spatial//2),spatial//2+1)
    basis=[(p,n-p) for n in range(degree+1) for p in range(n,-1,-1)]
    lookup={pair:i for i,pair in enumerate(basis)}
    block=len(basis)*spatial;rows=[];cols=[];data=[]
    gx=gamma;gy=gamma*e['drag_ratio'];b=e['magnetic'];t=e['temperature']*(1+abs(e['bath_contrast']))
    a=e['contrast'];k=e['wavenumber'];fx=e['force_x'];fy=e['force_y'];switch=e['switch_rate']
    def add(state,row,source,values,shift=0,source_state=None):
        if source not in lookup:return
        if source_state is None:source_state=state
        r=np.arange(spatial);c=r-shift;keep=(c>=0)&(c<spatial);r=r[keep];c=c[keep]
        v=np.broadcast_to(values,(spatial,))[keep]
        rows.extend((state*block+row*spatial+r).tolist())
        cols.extend((source_state*block+lookup[source]*spatial+c).tolist())
        data.extend(v.tolist())
    for state,sign in enumerate([1,-1]):
        temp_ratio=e['temperature']*(1+sign*e['bath_contrast'])/t
        for i,(p,q) in enumerate(basis):
            add(state,i,(p,q),-gx*p-gy*q-switch)
            add(state,i,(p,q),switch,source_state=1-state)
            if p:add(state,i,(p-1,q),np.sqrt(p*mass/t)*fx-np.sqrt(p*mass*t)*1j*modes*k)
            add(state,i,(p+1,q),-np.sqrt((p+1)*mass*t)*1j*modes*k)
            if q:add(state,i,(p,q-1),np.sqrt(q*mass/t)*fy)
            if p>=2:
                add(state,i,(p-2,q),gx*np.sqrt(p*(p-1))*(temp_ratio-1))
                for shift in [-1,1]:add(state,i,(p-2,q),gx*np.sqrt(p*(p-1))*temp_ratio*a/2,shift)
            if q>=2:
                add(state,i,(p,q-2),gy*np.sqrt(q*(q-1))*(temp_ratio-1))
                for shift in [-1,1]:add(state,i,(p,q-2),gy*np.sqrt(q*(q-1))*temp_ratio*a/2,shift)
            if p:add(state,i,(p-1,q+1),b*np.sqrt(p*(q+1)))
            if q:add(state,i,(p+1,q-1),-b*np.sqrt(q*(p+1)))
    gen=coo_matrix((data,(rows,cols)),shape=(2*block,2*block)).tolil()
    zero=int(np.flatnonzero(modes==0)[0]);gen[zero,:]=0;gen[zero,zero]=1;gen[zero,block+zero]=1
    rhs=np.zeros(2*block,complex);rhs[zero]=1/(2*np.pi)
    coeff=spsolve(gen.tocsc(),rhs).reshape(2,len(basis),spatial)
    x=np.arange(513)*2*np.pi/513
    waves=np.exp(1j*modes[:,None]*x)
    temperatures=np.array([e['temperature']*(1+sign*e['bath_contrast'])*(1+a*np.cos(x)) for sign in [1,-1]])
    fields=[(coeff[state]@waves).real for state in range(2)]
    moments=[{pair:values[i] for i,pair in enumerate(basis)} for values in fields]
    return x,temperatures,moments,t


def finite_mass_entropy(e,gamma,mass,degree=12,spatial=41):
    _,temperatures,states,basis_temperature=kinetic_state(e,gamma,mass,degree,spatial)
    result=0.
    for temperature,c in zip(temperatures,states):
        for drag,pair in [(gamma,(2,0)),(gamma*e['drag_ratio'],(0,2))]:
            result+=2*np.pi*drag/mass*np.mean(basis_temperature*(c[0,0]+np.sqrt(2)*c[pair])/temperature-c[0,0])
    return float(result)


def background_experiment(e):
    return dict(e, force_x=0., force_y=0., contrast=0.)


def finite_mass_excess(e, gamma, mass, degree=12, spatial=41):
    # Actual port entropy and residence are measured at the same positive
    # mass. The separate matched background is measured over all positions.
    if e.get('detector_phase') is None:
        actual=finite_mass_entropy(e,gamma,mass,degree,spatial)
        residence=1.
    else:
        phase,temperatures,states,basis_temperature=kinetic_state(e,gamma,mass,degree,spatial)
        weight=(1+np.cos(phase-e['detector_phase']))/2
        actual=0.
        for temperature,c in zip(temperatures,states):
            for drag,pair in [(gamma,(2,0)),(gamma*e['drag_ratio'],(0,2))]:
                actual+=2*np.pi*drag/mass*np.mean(weight*(basis_temperature*(c[0,0]+np.sqrt(2)*c[pair])/temperature-c[0,0]))
        residence=2*np.pi*np.mean(weight*(states[0][0,0]+states[1][0,0]))
    background=finite_mass_entropy(background_experiment(e),gamma,mass,degree,spatial)
    return float(actual-background*residence)


@lru_cache(2048)
def limiting_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction,degree,spatial,epsilon,detector_phase=None):
    e=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))
    if detector_phase is not None:e['detector_phase']=detector_phase
    mass=epsilon*(friction*min(1.,drag_ratio))**2/(temperature*wavenumber**2)
    values=[finite_mass_excess(e,friction,mass/factor,degree,spatial) for factor in [1,2,4]]
    return values[0]/3-2*values[1]+8*values[2]/3


def predict(experiments,friction=TRUE_PARAMETER,degree=12,spatial=41,epsilon=.006):
    return np.array([limiting_rate(*(e[field] for field in INPUT_FIELDS),float(friction),degree,spatial,epsilon,e.get('detector_phase')) for e in experiments])
