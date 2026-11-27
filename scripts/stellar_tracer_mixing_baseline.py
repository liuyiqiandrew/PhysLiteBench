import numpy as np
from scipy.optimize import minimize_scalar


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


class Model:
    def fit(self, records):
        records = list(records)
        def cost(sigma):
            return sum((ballistic(r["input"],sigma)-r["value"])**2 for r in records)
        self.dispersion = float(minimize_scalar(cost,bounds=(.8,1.2),method="bounded",
                                                options={"xatol":1e-12}).x)
        return self

    def predict(self, inputs):
        return np.array([ballistic(row,self.dispersion) for row in inputs])
