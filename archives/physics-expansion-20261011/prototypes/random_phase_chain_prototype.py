import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from scipy.stats import qmc


def coherent_mean(transmission, count, grid=1200, angles=96):
    if count == 1:
        return transmission
    if count == 2:
        return transmission/(2-transmission)
    jump=np.arccosh(2/transmission-1)
    nodes,weights=roots_legendre(angles)
    cosine=np.cos(np.pi*(nodes+1)/2)
    x=np.linspace(0,count*jump,grid+1)
    values=1/np.cosh(x/2)**2
    for i in range(1,count):
        spline=CubicSpline(x,values,bc_type=((1,0.),'not-a-knot'))
        x=np.linspace(0,(count-i)*jump,grid+1)
        argument=np.cosh(x[:,None])*np.cosh(jump)+np.sinh(x[:,None])*np.sinh(jump)*cosine
        following=np.arccosh(np.maximum(argument,1))
        values=spline(following)@(weights/2)
    return float(values[-1])


def transfer_mean(transmission,count,power=18,seed=1):
    phases=qmc.Sobol(count-1,scramble=True,seed=seed).random_base2(power)*2*np.pi
    # SU(1,1) transfer matrix: each barrier has |alpha|=1/sqrt(T),
    # |beta|=sqrt((1-T)/T). Uniform phase absorbs its fixed scattering phase.
    a=np.full(len(phases),1/np.sqrt(transmission),complex)
    b=np.full(len(phases),np.sqrt((1-transmission)/transmission),complex)
    for phase in phases.T:
        c=np.exp(1j*phase)
        a,b=(a*c+np.sqrt(1-transmission)*b.conj()/c)/np.sqrt(transmission), (b*c+np.sqrt(1-transmission)*a.conj()/c)/np.sqrt(transmission)
    return float(np.mean(1/abs(a)**2))


if __name__=='__main__':
 for transmission in [.39,.5,.7,.88]:
  for count in [2,4,8,12]:
   base=coherent_mean(transmission,count)
   refined=coherent_mean(transmission,count,2400,192)
   numeric=transfer_mean(transmission,count)
   approx=transmission/(count-(count-1)*transmission)
   print(transmission,count,base,refined-base,numeric-base,approx,flush=True)


def spectral_mean(transmission,count,radial=160,angular=192,upper=14.):
    a=np.arccosh(2/transmission-1)
    k,kw=roots_legendre(radial);k=(k+1)*upper/2;kw=kw*upper/2
    theta,tw=roots_legendre(angular);theta=(theta+1)*np.pi/2
    b=np.cosh(a)+np.sinh(a)*np.cos(theta)
    legendre=(np.cos(k[:,None]*np.log(b))/np.sqrt(b))@(tw/2)
    return float(np.dot(kw,2*np.pi*k*np.tanh(np.pi*k)/np.cosh(np.pi*k)*legendre**count))
