import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import wofz


def population_data(q, sigma):
    return zip([.62,.38], q["drift"], [sigma,1.25*sigma])


def filtered(mean, width, band):
    if band is None:
        return 1., mean, width
    center, window = band
    variance = width**2 + window**2
    return (window/np.sqrt(variance)*np.exp(-(mean-center)**2/(2*variance)),
            (mean*window**2 + center*width**2)/variance, width*window/np.sqrt(variance))


def ballistic(q, sigma):
    s = "AB".index(q["population"])
    mass, mean, width = list(population_data(q,sigma))[s]
    weight, shift, spread = filtered(mean,width,q["band"])
    a,b = q["k"]
    z = (b-a)*q["delay"]
    fourier = -mass*weight/width**2 * (shift-mean-1j*z*spread**2)
    fourier *= np.exp(-1j*z*shift - .5*(z*spread)**2)
    drive = q["first"][s]*q["second"][s]/4
    if q.get("channel") == "relay":
        return 0.
    if q.get("channel") == "intercept":
        transformed = mass*weight*np.exp(-1j*z*shift - .5*(z*spread)**2)
        second = transformed * (((shift-mean-1j*z*spread**2)**2+spread**2)/width**4-1/width**2)
        return (drive * second * np.exp(-1j*q["angle"])).real
    return (1j*a*drive*fourier*np.exp(-1j*q["angle"])).real


def response(q, sigma):
    mu = q["strength"]
    if mu == 0:
        return ballistic(q,sigma)
    a,b = q["k"]
    k = b-a
    v = np.linspace(-14,14,1025)
    data = list(population_data(q,sigma))

    def susceptibility(harmonic, band, derivative=False):
        values = []
        for mass, mean, width in data:
            weight, shift, spread = filtered(mean,width,band)
            dv = v-shift
            z = np.sign(harmonic)*dv/(np.sqrt(2)*spread)
            i0 = np.sqrt(np.pi/2)*wofz(z)/(abs(harmonic)*spread)
            i1 = (1 + 1j*harmonic*dv*i0)/(harmonic*spread)**2
            if derivative:
                # Only the unfiltered susceptibility is differentiated.
                values.append(mu*mass*(1j*harmonic*i0-harmonic**2*dv*i1)/(harmonic*spread)**2)
                continue
            values.append(mu*mass*weight/width**2 * (
                spread**2*i1 + 1j*(shift-mean)/harmonic*i0))
        return np.array(values)

    def impulse(harmonic, pattern):
        chi = susceptibility(harmonic,None)
        drive = np.asarray(pattern)[:,None]
        pole = 1-chi.sum(axis=0)
        common = np.sum(chi*drive,axis=0)/pole
        dressed = drive + common
        slope = np.sum(susceptibility(harmonic,None,True)*dressed,axis=0)/pole
        return dressed, slope

    first, first_slope = impulse(-a,q["first"])
    second, second_slope = impulse(b,q["second"])
    distribution = np.array([mass*np.exp(-.5*((v-mean)/width)**2)/(np.sqrt(2*np.pi)*width)
                             for mass,mean,width in data])
    slope = np.array([-(v-mean)/width**2 for _,mean,width in data])
    df = distribution * slope
    source = 1j*a/4 * df * first * second
    channel = q.get("channel","primary")
    if channel == "intercept":
        ddf = distribution * (slope**2 - np.array([1/w**2 for _,_,w in data])[:,None])
        source = ((ddf*first + df*first_slope)*second + (a/b)*df*first*second_slope)/4
    if channel == "relay":
        intermediate = source.sum(axis=0)/(1-susceptibility(k,None).sum(axis=0))
        source = -2*np.pi*a*mu*b/k**3 * df*first*intermediate
        k = b-2*a
    s = "AB".index(q["population"])
    band = q["band"]
    window = 1. if band is None else np.exp(-.5*((v-band[0])/band[1])**2)
    denominator = 1-susceptibility(k,None).sum(axis=0)
    signal = window*source[s] + susceptibility(k,band)[s]*source.sum(axis=0)/denominator
    phase = np.exp(-1j*(q["angle"]+k*q["delay"]*v))
    return trapezoid((phase*signal).real,v)


class Model:
    def fit(self, records):
        records = list(records)
        def cost(sigma):
            return sum((ballistic(r["input"],sigma)-r["value"])**2 for r in records)
        self.dispersion = float(minimize_scalar(cost,bounds=(.8,1.2),method="bounded",
                                                options={"xatol":1e-12}).x)
        return self

    def predict(self, inputs):
        return np.array([response(row,self.dispersion) for row in inputs])
