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
    def first_filter(g,o,weighted=False):
        impulse = first[o] * (-1j*x if weighted else 1)
        return fftconvolve(g,impulse[::-1])[m:m+len(z)]

    def second_filter(g,o,weighted=False):
        impulse = second[o] * (1j*x if weighted else 1)
        return fftconvolve(g,impulse)[:len(z)]

    def primary(o,band):
        g = derivative_transform(z,mass[o],u[o],widths[o],band)
        return (1j*a/4)*second_filter(first_filter(g,o),o)

    def dressed_total(bare,harmonic):
        return fftconvolve(bare(0,None)+bare(1,None),pulse(harmonic,[1.,1.])[0])[:len(z)]

    channel = q.get("channel","primary")
    if channel == "intercept":
        def bare(o,band):
            # Different retarded acceleration moments survive after the T terms cancel.
            f1 = derivative_transform(z,mass[o],u[o],widths[o],band)
            if band is None:
                f2 = -z**2 * density_transform(z,mass[o],u[o],widths[o])
            else:
                center,win = band
                var = 1/(1/widths[o]**2+1/win**2)
                mean = var*(u[o]/widths[o]**2+center/win**2)
                weight = np.sqrt(var)/widths[o]*np.exp(-(u[o]-center)**2/(2*(widths[o]**2+win**2)))
                f2 = density_transform(z,mass[o]*weight,mean,np.sqrt(var)) * (
                    (var + (mean-u[o]-1j*var*z)**2)/widths[o]**4-1/widths[o]**2)
            return (second_filter(first_filter(f2,o)+first_filter(f1,o,True),o)
                    + (a/b)*second_filter(first_filter(f1,o),o,True))/4
    elif channel == "relay":
        first_echo = dressed_total(primary,k)
        coefficient = -a*mu*b/k**3
        def bare(o,band):
            g = derivative_transform(z,mass[o],u[o],widths[o],band)
            # Convolve the complete waveform of the first echo with the surviving filament.
            return coefficient*spacing*fftconvolve(first_filter(g,o),first_echo,mode="same")
        k = b-2*a
    else:
        bare = primary
    total = dressed_total(bare,k)
    receiver = (-1j*mu*spacing/k**2
                * derivative_transform(x,mass[s],u[s],widths[s],q["band"]))
    receiver[0] *= .5
    out = bare(s,q["band"]) + fftconvolve(total,receiver)[:len(z)]
    offset = k*q["delay"]
    rotated = (out*np.exp(-1j*q["angle"])).real
    return float(np.interp(offset,z,rotated))


def finite_delay(q,sigma,T,dt=.05,n=1024):
    """Split streaming and self-consistent kicks through mixed cubic order."""
    a,b=q['k'];k=b-a;h=b-2*a
    modes=np.array([-a,b,-2*a,k,h])[:,None,None]
    dv=20/n;v=np.arange(n)*dv-10
    xi=2j*np.pi*np.fft.fftfreq(n,dv)
    d=np.array(q['drift'])[:,None];sig=np.array([sigma,1.25*sigma])[:,None]
    yo=v-d
    f=np.array([.62,.38])[:,None]*np.exp(-.5*(yo/sig)**2)/(np.sqrt(2*np.pi)*sig)
    f1=-yo/sig**2*f;f2=(yo**2/sig**4-1/sig**2)*f;f3=(3*yo/sig**4-yo**3/sig**6)*f
    def deriv(x,order=1):return np.fft.ifft(xi**order*np.fft.fft(x,axis=-1),axis=-1)
    u=np.zeros((5,2,n),complex)
    d1=.5*np.array(q['first'])[:,None]
    u[0]=-d1*f1;u[2]=d1**2*f2
    def segment(elapsed,full):
        steps=int(np.ceil(elapsed/dt));ds=elapsed/steps
        phase=np.exp(-.5j*v*modes*ds)
        for _ in range(steps):
            u[:]*=phase
            z=1j*q['strength']*dv*u.sum(axis=(1,2))[:,None,None]/modes*ds
            zz,ww,zz2,zw,zzw=z
            if full:
                du=deriv(u[:4]);ddu=deriv(u[:2],2)
                u[4]-=zzw*f1 + 2*zz*du[3] + ww*du[2]+zz2*du[1]+2*zw*du[0]
                u[4]+=(2*zz*zw+ww*zz2)*f2+zz**2*ddu[1]+2*zz*ww*ddu[0]-zz**2*ww*f3
                u[3]-=zw*f1+ww*du[0]+zz*du[1]-zz*ww*f2
                u[1]-=ww*f1
            else:du=deriv(u[:1])
            u[2]-=zz2*f1+2*zz*du[0]-zz**2*f2
            u[0]-=zz*f1
            u[:]*=phase
    segment(T,False)
    d2=.5*np.array(q['second'])[:,None]
    u[1]=-d2*f1;u[3]=-d2*deriv(u[0]);u[4]=-d2*deriv(u[2])
    relay = q.get('channel') == 'relay'
    segment((2*a/h if relay else a/k)*T+q['delay'],True)
    s='AB'.index(q['population']);g=q['band'];W=1 if g is None else np.exp(-.5*((v-g[0])/g[1])**2)
    return float(np.real(np.exp(-1j*q['angle'])*dv*sum(W*u[4 if relay else 3,s]))/(T**2 if relay else 1))
