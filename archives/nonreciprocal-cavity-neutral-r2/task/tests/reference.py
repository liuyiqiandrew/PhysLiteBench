"""Propagate each independent Langevin reservoir into each measured output."""
import numpy as np

TRUE_PARAMETER=.27


def experiment(frequency=0.,phase=1.5,body=1.4,external=(0.,0.,0.),port=0):
    return {'frequency':float(frequency),'flux_phase':float(phase),'body_occupation':float(body),
            'external_occupation':list(external),'output_port':int(port)}


def bath_transfer(e,loss_rate):
    h=np.array([[-.3,.8,.8*np.exp(-1j*e['flux_phase'])],
                [.8,.2,.8],[.8*np.exp(1j*e['flux_phase']),.8,.6]])
    port=np.array([1.,1.5,.8]);internal=loss_rate*np.array([1.,.4,1.8])
    dynamics=1j*h+np.diag(.5*(port+internal)-1j*e['frequency'])
    incoming=[];emitted=[]
    for j in range(3):
        drive=np.zeros(3);drive[j]=np.sqrt(port[j])
        mode=np.linalg.solve(dynamics,drive)
        out=-np.sqrt(port)*mode;out[j]+=1
        incoming.append(out)
        drive=np.zeros(3);drive[j]=np.sqrt(internal[j])
        mode=np.linalg.solve(dynamics,drive)
        emitted.append(-np.sqrt(port)*mode)
    return np.array(incoming).T,np.array(emitted).T


def covariance(e,loss_rate):
    incoming,emitted=bath_transfer(e,loss_rate)
    result=np.zeros((3,3),complex)
    for j in range(3):
        result+=e['external_occupation'][j]*np.outer(incoming[:,j],incoming[:,j].conj())
        result+=e['body_occupation']*np.outer(emitted[:,j],emitted[:,j].conj())
    return result


def predict(experiments,loss_rate):
    out=[]
    for e in experiments:
        photons=np.diag(covariance(e,loss_rate)).real
        out.append(float(np.sum(photons) if e['output_port']==-1 else photons[e['output_port']]))
    return np.array(out)


def calibration_inputs():
    return [experiment(frequency=w,phase=p,body=n,port=-1)
            for w in np.linspace(-1.8,1.8,12) for p in [-1.3,.4,1.8,2.7] for n in [.5,1.5,2.5]]


def hidden_inputs():
    return {'clockwise_ports':[experiment(w,1.5,1.4,port=p) for w in np.linspace(-1.4,1.4,10) for p in range(3)],
            'reversed_bias':[experiment(w,-1.2,2.1,port=p) for w in np.linspace(-1.5,1.2,10) for p in range(3)],
            'thermal_background':[experiment(w,2.2,1.8,(.4,.2,.6),p) for w in np.linspace(-1.2,1.3,10) for p in range(3)]}
