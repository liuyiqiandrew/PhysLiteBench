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
    return (1j*a/4*q["first"][s]*q["second"][s]*fourier*np.exp(-1j*q["angle"])).real


def response(q, sigma):
    mu = q["strength"]
    if mu == 0:
        return ballistic(q,sigma)
    a,b = q["k"]
    k = b-a
    v = np.linspace(-14,14,1025)
    data = list(population_data(q,sigma))

    def susceptibility(harmonic, band):
        values = []
        for mass, mean, width in data:
            weight, shift, spread = filtered(mean,width,band)
            dv = v-shift
            z = np.sign(harmonic)*dv/(np.sqrt(2)*spread)
            i0 = np.sqrt(np.pi/2)*wofz(z)/(abs(harmonic)*spread)
            i1 = (1 + 1j*harmonic*dv*i0)/(harmonic*spread)**2
            values.append(mu*mass*weight/width**2 * (
                spread**2*i1 + 1j*(shift-mean)/harmonic*i0))
        return np.array(values)

    def impulse(harmonic, pattern):
        chi = susceptibility(harmonic,None)
        drive = np.asarray(pattern)[:,None]
        return drive + np.sum(chi*drive,axis=0)/(1-chi.sum(axis=0))

    derivative = np.array([-mass*(v-mean)*np.exp(-.5*((v-mean)/width)**2)
                           /(np.sqrt(2*np.pi)*width**3) for mass,mean,width in data])
    source = 1j*a/4 * derivative * impulse(-a,q["first"]) * impulse(b,q["second"])
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
