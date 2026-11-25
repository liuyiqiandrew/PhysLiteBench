import numpy as np
from scipy.linalg import eigh


def response(e, coupling, quantum=True):
    omega, t = e['frequency'], e['time']
    weights = np.asarray([e['weight_1']+e['weight_2'], e['weight_1']-e['weight_2'],
                          -e['weight_1']+e['weight_2'], -e['weight_1']-e['weight_2']])
    scale = (coupling/omega)**2
    noise = scale*(1-np.cos(omega*t))*(2*e['occupation']+1)
    phase = scale*(omega*t-np.sin(omega*t)) if quantum else 0.
    return float(np.mean(np.exp(-noise*(weights[:,None]-weights)**2)
                         *np.cos(phase*(weights[:,None]**2-weights**2))))


def fock(e, coupling, cutoff=96):
    omega, t = e['frequency'], e['time']
    n = np.arange(cutoff)
    a = np.diag(np.sqrt(np.arange(1,cutoff)),1)
    h0 = np.diag(omega*n)
    weights = [e['weight_1']+e['weight_2'], e['weight_1']-e['weight_2'],
               -e['weight_1']+e['weight_2'], -e['weight_1']-e['weight_2']]
    propagator = np.zeros((cutoff,cutoff),complex)
    for s in weights:
        values,vectors = eigh(h0+coupling*s*(a+a.T))
        propagator += (vectors*np.exp(-1j*t*values))@vectors.T/4
    occupation = e['occupation']
    thermal = (occupation/(1+occupation))**n/(1+occupation)
    return float(np.sum(abs(propagator)**2*thermal))


if __name__ == '__main__':
    for w in [.8,1.6]:
      for t in [0,1,2*np.pi/w,16]:
       e=dict(frequency=w,time=t,occupation=1.5,weight_1=1.2,weight_2=1.2)
       exact=response(e,.4)
       print(w,t,exact,response(e,.4,False),[fock(e,.4,n)-exact for n in [64,96,128]])
