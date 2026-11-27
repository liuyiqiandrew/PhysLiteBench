"""Retarded Volterra propagation on a Fourier-velocity grid."""
import numpy as np
from scipy.signal import fftconvolve

TRUE_PARAMETER = .947


def density_transform(z, mass, u, sigma):
    return mass*np.exp(-1j*u*z - .5*(sigma*z)**2)


def derivative_transform(z, mass, mean, sigma, band):
    # Complete the square in W(v) f(v) before transforming W times f'.
    if band is None:
        return 1j*z*density_transform(z,mass,mean,sigma)
    center, width = band
    variance = 1/(1/sigma**2 + 1/width**2)
    shifted = variance*(mean/sigma**2 + center/width**2)
    mass *= np.sqrt(variance)/sigma*np.exp(-.5*(mean-center)**2/(sigma**2 + width**2))
    return ((mean-shifted+1j*variance*z)/sigma**2
            * density_transform(z,mass,shifted,np.sqrt(variance)))


def signal(q, sigma=TRUE_PARAMETER, spacing=.035, extent=32.):
    mass = [.62,.38]
    widths = [sigma,1.25*sigma]
    u = q["drift"]
    mu = q["strength"]
    a,b = q["k"]
    k = b-a
    s = "AB".index(q["population"])
    m = int(np.ceil(extent/spacing))
    x = np.arange(m+1)*spacing
    z = np.arange(-2*m,2*m+1)*spacing

    def pulse(harmonic,pattern):
        # Coefficients are integrated acceleration per cell of |harmonic|*time.
        kernels = [mu * spacing*x/harmonic**2
                   * density_transform(np.sign(harmonic)*x,mass[i],u[i],widths[i])
                   for i in range(2)]
        kernel = kernels[0]+kernels[1]
        wake = pattern[0]*kernels[0]+pattern[1]*kernels[1]
        for n in range(1,len(wake)):
            wake[n] += kernel[1:n+1] @ wake[n-1::-1]
        pulses = np.tile(wake,(2,1))
        pulses[:,0] += pattern
        return pulses

    first = pulse(-a,q["first"])
    second = pulse(b,q["second"])
    def bare(o,band):
        g = derivative_transform(z,mass[o],u[o],widths[o],band)
        # First harmonic runs backward in this coordinate, second runs forward.
        y = fftconvolve(g,first[o,::-1])[m:m+len(z)]
        return (1j*a/4)*fftconvolve(y,second[o])[:len(z)]

    total = bare(0,None) + bare(1,None)
    # Response of both populations to the mixed density, irrespective of band.
    total = fftconvolve(total,pulse(k,[1.,1.])[0])[:len(z)]
    receiver = (-1j*mu*spacing/k**2
                * derivative_transform(x,mass[s],u[s],widths[s],q["band"]))
    receiver[0] *= .5
    out = bare(s,q["band"]) + fftconvolve(total,receiver)[:len(z)]
    offset = k*q["delay"]
    rotated = (out*np.exp(-1j*q["angle"])).real
    return float(np.interp(offset,z,rotated))
