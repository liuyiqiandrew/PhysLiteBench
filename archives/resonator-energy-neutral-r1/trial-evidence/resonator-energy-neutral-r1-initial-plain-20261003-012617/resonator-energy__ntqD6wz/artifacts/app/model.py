import numpy as np


LEAD_IMPEDANCE = np.sqrt(2.)


def wave_data(experiment, host_stiffness):
    """Return the effective density and the two displacement integrals.

    The resonators can be eliminated from the equation for the host.  If
    ``U`` is the complex host amplitude, their amplitude is

        V = resonance_frequency**2 / (resonance_frequency**2-frequency**2) U.

    Consequently the host sees the (frequency-dependent) density returned by
    this function.  ``displacement`` is integral(abs(U)**2 dx), and ``strain``
    is integral(abs(U')**2 dx), both over the slab.
    """
    length = experiment['length']
    frequency = experiment['frequency']
    mass = experiment['resonator_mass']
    resonance = experiment['resonance_frequency']
    amplitude = experiment['incident_amplitude']
    density = (1 + mass*resonance**2/(resonance**2-frequency**2)
               if mass else 1.)
    wave_number = frequency*np.sqrt(density/host_stiffness)
    cosine, sine = np.cos(wave_number*length), np.sin(wave_number*length)
    impedance = 1j*frequency*LEAD_IMPEDANCE
    matrix = np.array([[impedance, host_stiffness*wave_number],
        [-host_stiffness*wave_number*sine-impedance*cosine,
         host_stiffness*wave_number*cosine-impedance*sine]])
    a,b = np.linalg.solve(matrix, [2*impedance*amplitude,0.])
    first = length/2+np.sin(2*wave_number*length)/(4*wave_number)
    second = length-first
    mixed = np.real(a.conjugate()*b)*sine**2/wave_number
    displacement = abs(a)**2*first+abs(b)**2*second+mixed
    strain = wave_number**2*(abs(a)**2*second+abs(b)**2*first-mixed)
    return density, displacement, strain


def predict_at(experiments, host_stiffness):
    result = []
    for experiment in experiments:
        density, displacement, strain = wave_data(experiment, host_stiffness)
        frequency = experiment['frequency']
        mass = experiment['resonator_mass']

        # The host equation uses the dynamic density above, but the measured
        # energy contains the resonator kinetic and spring energies separately.
        # With V = alpha U, their contribution to the coefficient multiplying
        # integral(abs(U)**2 dx) is
        #
        #   m * (frequency**2 * alpha**2
        #        + resonance**2 * (alpha - 1)**2).
        if mass:
            resonance = experiment['resonance_frequency']
            alpha = resonance**2/(resonance**2-frequency**2)
            energy_coefficient = (frequency**2 + mass*(
                frequency**2*alpha**2 + resonance**2*(alpha-1)**2))
        else:
            energy_coefficient = frequency**2

        result.append((energy_coefficient*displacement
                       + host_stiffness*strain)/4)
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.host_stiffness = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records],
                              dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)
        if (not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("Calibration values and sigmas must be finite, "
                             "with positive sigmas.")

        def objective(stiffness):
            residual = (predict_at(experiments, stiffness)-observed)/sigma
            return float(residual @ residual)

        # The physical prior specifies the complete admissible interval.  A
        # bounded golden-section search avoids an optional optimizer
        # dependency while providing more than enough precision for the data.
        lower, upper = 0.7, 1.6
        golden = (np.sqrt(5.)-1.)/2.
        left = upper-golden*(upper-lower)
        right = lower+golden*(upper-lower)
        f_left, f_right = objective(left), objective(right)
        for _ in range(100):
            if f_left <= f_right:
                upper, right, f_right = right, left, f_left
                left = upper-golden*(upper-lower)
                f_left = objective(left)
            else:
                lower, left, f_left = left, right, f_right
                right = lower+golden*(upper-lower)
                f_right = objective(right)

        candidates = (lower, upper, (lower+upper)/2, .7, 1.6)
        self.host_stiffness = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.host_stiffness)
