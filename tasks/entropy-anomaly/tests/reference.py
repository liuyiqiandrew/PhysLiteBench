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
    b=e['magnetic'];t=e['temperature']*(1+abs(e['bath_contrast']))
    if 'valve_cycle' in e:
        labels=[-1,0,1];count=3;valve=np.zeros((3,3))
        for s in range(3):
            valve[(s+1)%3,s]=e['switch_rate']*(1+e['valve_cycle'])
            valve[(s-1)%3,s]=e['switch_rate']*(1-e['valve_cycle'])
            valve[s,s]=-2*e['switch_rate']
        scales=1+e.get('contact_drag_contrast',0.)*np.array(labels)
    else:
        labels=[1,-1];count=2;scales=np.ones(2)
        valve=e['switch_rate']*np.array([[-1,1],[1,-1]])
    a=e['contrast'];k=e['wavenumber'];fx=e['force_x'];fy=e['force_y'];switch=e['switch_rate']
    def add(state,row,source,values,shift=0,source_state=None):
        if source not in lookup:return
        if source_state is None:source_state=state
        r=np.arange(spatial);c=r-shift;keep=(c>=0)&(c<spatial);r=r[keep];c=c[keep]
        v=np.broadcast_to(values,(spatial,))[keep]
        rows.extend((state*block+row*spatial+r).tolist())
        cols.extend((source_state*block+lookup[source]*spatial+c).tolist())
        data.extend(v.tolist())
    for state,sign in enumerate(labels):
        gx=gamma*scales[state];gy=gx*e['drag_ratio']
        temp_ratio=e['temperature']*(1+sign*e['bath_contrast'])/t
        for i,(p,q) in enumerate(basis):
            add(state,i,(p,q),-gx*p-gy*q)
            for source_state in range(count):add(state,i,(p,q),valve[state,source_state],source_state=source_state)
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
    gen=coo_matrix((data,(rows,cols)),shape=(count*block,count*block)).tolil()
    zero=int(np.flatnonzero(modes==0)[0]);gen[zero,:]=0;gen[zero,np.arange(count)*block+zero]=1
    rhs=np.zeros(count*block,complex);rhs[zero]=1/(2*np.pi)
    coeff=spsolve(gen.tocsc(),rhs).reshape(count,len(basis),spatial)
    x=np.arange(513)*2*np.pi/513
    waves=np.exp(1j*modes[:,None]*x)
    temperatures=np.array([e['temperature']*(1+sign*e['bath_contrast'])*(1+a*np.cos(x)) for sign in labels])
    fields=[(coeff[state]@waves).real for state in range(count)]
    moments=[{pair:values[i] for i,pair in enumerate(basis)} for values in fields]
    return x,temperatures,moments,t


def finite_mass_entropy(e,gamma,mass,degree=12,spatial=41):
    _,temperatures,states,basis_temperature=kinetic_state(e,gamma,mass,degree,spatial)
    result=0.
    for state,(temperature,c) in enumerate(zip(temperatures,states)):
        scale=1+e.get('contact_drag_contrast',0.)*[-1,0,1][state] if 'valve_cycle' in e else 1.
        for drag,pair in [(gamma*scale,(2,0)),(gamma*scale*e['drag_ratio'],(0,2))]:
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
        for state,(temperature,c) in enumerate(zip(temperatures,states)):
            scale=1+e.get('contact_drag_contrast',0.)*[-1,0,1][state] if 'valve_cycle' in e else 1.
            for drag,pair in [(gamma*scale,(2,0)),(gamma*scale*e['drag_ratio'],(0,2))]:
                actual+=2*np.pi*drag/mass*np.mean(weight*(basis_temperature*(c[0,0]+np.sqrt(2)*c[pair])/temperature-c[0,0]))
        residence=2*np.pi*np.mean(weight*sum(state[0,0] for state in states))
    background=finite_mass_entropy(background_experiment(e),gamma,mass,degree=2,spatial=1)
    return float(actual-background*residence)


@lru_cache(2048)
def limiting_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction,degree,spatial,epsilon,detector_phase=None,valve_cycle=None,contact_drag_contrast=0.):
    e=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))
    if detector_phase is not None:e['detector_phase']=detector_phase
    if valve_cycle is not None:e.update(valve_cycle=valve_cycle,contact_drag_contrast=contact_drag_contrast)
    mass=epsilon*(friction*min(1.,drag_ratio))**2/(temperature*wavenumber**2)
    values=[finite_mass_excess(e,friction,mass/factor,degree,spatial) for factor in [1,2,4]]
    return values[0]/3-2*values[1]+8*values[2]/3


def predict(experiments,friction=TRUE_PARAMETER,degree=12,spatial=41,epsilon=.006):
    return np.array([limiting_rate(*(e[field] for field in INPUT_FIELDS),float(friction),degree,spatial,epsilon,e.get('detector_phase'),e.get('valve_cycle'),e.get('contact_drag_contrast',0.)) for e in experiments])

old_hidden_inputs=hidden_inputs
def hidden_inputs():
    old=old_hidden_inputs();result={}
    for name,bias,drag in [('isotropic_drive',.8,.6),('anisotropic_contacts',-.7,.65),('magnetic_drive',.9,.7)]:
        result['cyclic_'+name]=[dict(e,valve_cycle=bias,contact_drag_contrast=drag) for e in old[name]]
    result['two_contact_anchor']=old['bulk_anchor']+old['uniform_port_anchor']
    return result
