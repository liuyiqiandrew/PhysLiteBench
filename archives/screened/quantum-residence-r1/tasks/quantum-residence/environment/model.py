import numpy as np
from scipy.linalg import expm_frechet


def scattering(energy,potential,width):
    generator=np.array([[0.,1.],[-2*(energy-potential),0.]])*width
    derivative=np.array([[0.,0.],[-2.,0.]])*width
    transfer,change=expm_frechet(generator,derivative)
    a,b,c,d=transfer.ravel();da,db,dc,dd=change.ravel()
    k=np.sqrt(2*energy);dk=1/k
    denominator=k*k*b-c+1j*k*(a+d)
    change_denominator=2*b+k*k*db-dc+1j*dk*(a+d)+1j*k*(da+dd)
    numerator=c+k*k*b+1j*k*(d-a)
    change_numerator=dc+2*b+k*k*db+1j*dk*(d-a)+1j*k*(dd-da)
    reflection=numerator/denominator
    transmission=2j*k/denominator
    dr=(change_numerator*denominator-numerator*change_denominator)/denominator**2
    dt=(2j*dk*denominator-2j*k*change_denominator)/denominator**2
    return np.array([[reflection,transmission],[transmission,reflection]]),np.array([[dr,dt],[dt,dr]])


def residence(experiment,width):
    energy,potential=experiment['energy'],experiment['potential']
    if potential==0:return width/np.sqrt(2*energy)
    matrix,derivative=scattering(energy,potential,width)
    value=np.imag(np.vdot(matrix[:,0],derivative[:,0]))

    return float(value)


class Model:
    def __init__(self):self.width=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):return np.array([residence(e,self.width) for e in experiments])
