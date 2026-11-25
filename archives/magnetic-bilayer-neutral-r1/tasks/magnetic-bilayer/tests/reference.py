"""Piecewise Poisson interface solve and implicit Gilbert dynamics."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

TRUE_PARAMETER = .93


def experiment(kx, ky, gap, time, layer=0, quadrature='real', initial_real=(.01, .006), initial_imag=(0., 0.)):
    return dict(wavevector=[float(kx), float(ky)], gap=float(gap), time=float(time), layer=layer,
                quadrature=quadrature, initial_real=list(initial_real), initial_imag=list(initial_imag))


def calibration_inputs():
    return [experiment(k, 0., gap, time, layer, initial_real=initial)
            for k in [.4, .8, 1.2, 1.7] for gap in [.1, .5]
            for initial in [(.01, 0.), (.006, .01)] for time in [.3, .7, 1.2, 2., 3., 4.]
            for layer in [0, 1]]


def hidden_inputs():
    cases = {'lower_film':[(.2,.9,.1),(.4,1.25,.2),(-.3,1.5,.15)],
             'upper_film':[(.2,-.9,.1),(-.4,1.25,.2),(.3,-1.5,.15)],
             'phased_preparation':[(.3,.85,.15),(.5,-1.1,.2),(-.4,-1.4,.1)]}
    result={}
    for name,preparations in cases.items():
        initial_real=(.014,.008) if name!='phased_preparation' else (.011,-.006)
        initial_imag=(0.,0.) if name!='phased_preparation' else (.004,.009)
        layer=1 if name=='upper_film' else 0
        result[name]=[experiment(kx,ky,gap,t,layer,quadrature,initial_real,initial_imag)
                      for kx,ky,gap in preparations for t in [1.3,3.7,6.1,9.4]
                      for quadrature in ['real','imag']]
    return result


def poisson_field(kx, ky, gap, magnetization):
    k=np.hypot(kx,ky);top=2+gap
    def basis(region,z,derivative=False):
        values=np.zeros(8,dtype=complex)
        if region==0:
            values[0]=np.exp(k*z)*(k if derivative else 1)
        elif region==4:
            values[7]=np.exp(-k*(z-top))*(-k if derivative else 1)
        else:
            index={1:1,2:3,3:5}[region]
            center={1:.5,2:1+gap/2,3:1.5+gap}[region]
            x=k*(z-center)
            values[index:index+2]=k*np.array([np.sinh(x),np.cosh(x)]) if derivative else [np.cosh(x),np.sinh(x)]
        return values
    my=magnetization[[0,2]];mz=magnetization[[1,3]]
    particular=np.array([0.,-1j*ky*my[0]/k**2,0.,-1j*ky*my[1]/k**2,0.])
    normal=np.array([0.,mz[0],0.,mz[1],0.])
    rows=[];rhs=[]
    for left,z in enumerate([0.,1.,1+gap,top]):
        right=left+1
        rows.append(basis(left,z)-basis(right,z));rhs.append(particular[right]-particular[left])
        rows.append(basis(right,z,True)-basis(left,z,True));rhs.append(normal[right]-normal[left])
    coefficients=np.linalg.solve(np.array(rows),np.array(rhs))
    response=[]
    for region,index in [(1,1),(3,5)]:
        average=particular[region]+coefficients[index]*2*np.sinh(k/2)/k
        field_y=-1j*ky*average
        field_z=-2*coefficients[index+1]*np.sinh(k/2)
        response.extend([field_y,field_z])
    return np.array(response)


@lru_cache(maxsize=512)
def field_matrix(kx,ky,gap):
    return np.column_stack([poisson_field(kx,ky,gap,np.eye(4,dtype=complex)[:,j]) for j in range(4)])


@lru_cache(maxsize=512)
def dynamics(kx,ky,gap):
    field=field_matrix(kx,ky,gap)
    restoring=np.diag([.45,.45,.70,.70])+.08*(kx*kx+ky*ky)*np.eye(4)-field
    torque=np.array([-restoring[1],restoring[0],-restoring[3],restoring[2]])
    gilbert=np.kron(np.eye(2),np.array([[1.,.04],[-.04,1.]]))
    return np.linalg.solve(gilbert,torque)


def predict(experiments,gyro_rate=TRUE_PARAMETER):
    values=[]
    for e in experiments:
        start=np.zeros(4,dtype=complex)
        start[[0,2]]=np.array(e['initial_real'])+1j*np.array(e['initial_imag'])
        state=expm(gyro_rate*e['time']*dynamics(*e['wavevector'],e['gap']))@start
        value=state[2*e['layer']+1]
        values.append(float(value.real if e['quadrature']=='real' else value.imag))
    return np.array(values)
