import numpy as np


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


class Model:
    def fit(self, records):
        raise NotImplementedError("Infer the dispersion from the free-streaming records.")

    def predict(self, inputs):
        return np.array([ballistic(row,self.dispersion) for row in inputs])
