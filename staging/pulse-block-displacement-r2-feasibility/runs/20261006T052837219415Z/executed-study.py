"""Execute the single, predeclared electric-sheet feasibility study."""
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import shutil
import time
import traceback

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import beta, spherical_jn


STAGE = Path(__file__).resolve().parent
EXPECTED_PLAN = "3a322005ff2013b08166710e8a8771de8886ca1fdf910d469ef7db9a74acef21"
PLAN = json.loads((STAGE / "plan.json").read_text())
assert sha256((STAGE / "plan.json").read_bytes()).hexdigest() == EXPECTED_PLAN
RUN = STAGE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RUN.mkdir(parents=True)
shutil.copy2(__file__, RUN / "executed-study.py")
shutil.copy2(STAGE / "plan.json", RUN / "executed-plan.json")
START = time.perf_counter()
REPORT = {"plan_sha256": EXPECTED_PLAN, "run": str(RUN),
          "started_utc": datetime.now(timezone.utc).isoformat(),
          "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "status": "running", "sections": {}}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def checkpoint(name, value):
    REPORT["sections"][name] = value
    REPORT["elapsed_seconds"] = time.perf_counter() - START
    save("report.partial.json", REPORT)
    print(json.dumps({"section": name, "elapsed_seconds": REPORT["elapsed_seconds"]},
                     default=json_default), flush=True)


def incident(t, center, width, amplitude=1.0):
    """Real, zero-phase pulse, independently evaluated with a Bessel identity."""
    v = width * t
    av = abs(v)
    if av < 0.1:
        h = dh = 0.0
        factorial = 1.0
        for n in range(7):
            if n:
                factorial *= (2*n-1)*(2*n)
            coefficient = (-1.0)**n * beta(n+0.5, 5) / factorial
            h += coefficient * v**(2*n)
            if n:
                dh += 2*n*coefficient*v**(2*n-1)
    else:
        j = spherical_jn(4, av)
        dj = spherical_jn(4, av, derivative=True)
        h = 768*j/av**4
        dh = np.sign(v)*768*(dj/av**4 - 4*j/av**5)
    norm = amplitude*np.sqrt(width/(np.pi*beta(0.5, 9)))
    co, si = np.cos(center*t), np.sin(center*t)
    return norm*h*co, norm*(width*dh*co-center*h*si)


@lru_cache(maxsize=12)
def optical_fields(strength, center, width, dt=0.125, half_window=32.0):
    """Frequency-domain oracle; independent of the time-domain jump solver."""
    n = 1 << int(np.ceil(np.log2(2*half_window/(width*dt))))
    period = n*dt
    omega = 2*np.pi*np.fft.fftfreq(n, dt)
    u = (np.abs(omega)-center)/width
    spectrum = np.zeros(n)
    mask = np.abs(u) < 1
    norm = np.sqrt(width/(np.pi*beta(0.5, 9)))
    spectrum[mask] = norm*np.pi/width*(1-u[mask]**2)**4
    chi = strength/(1-omega**2-0.5j*strength*omega)

    def transform(a):
        return np.fft.fftshift(np.fft.fft(a).real/period)

    e = transform(spectrum)
    edot = transform(-1j*omega*spectrum)
    p = transform(chi*spectrum)
    pdot = transform(-1j*omega*chi*spectrum)
    t = (np.arange(n)-n/2)*dt
    return {"t": t, "e": e, "edot": edot, "p": p, "pdot": pdot,
            "force": pdot*e, "source_force": -p*edot,
            "energy": float(simpson(e*e, x=t)), "n": n, "period": period}


def kernel(tau, frequency):
    if frequency == 0:
        return np.zeros_like(tau)
    x = frequency*tau
    result = (x-np.sin(x))/frequency
    small = np.abs(x) < 0.01
    result[small] = (x[small]**3/6-x[small]**5/120+x[small]**7/5040)/frequency
    return result


def predict(case, dt=0.125, half_window=32.0):
    f, center, width, ratio, clock = map(float, case)
    field = optical_fields(f, center, width, dt, half_window)
    tstar, mechanical = clock/width, ratio*width
    mask = field["t"] <= tstar
    t = field["t"][mask]
    response = kernel(tstar-t, mechanical)
    physical = float(simpson(response*field["force"][mask], x=t))
    source = float(simpson(response*field["source_force"][mask], x=t))
    # E=1 analytically. Do not normalize away finite spectral/tail energy error.
    return physical, source


def reference(case, half_window=32.0, refined=False, amplitude=1.0):
    """Time-domain sheet jumps, exterior Maxwell stresses and Newton equations."""
    before = time.perf_counter()
    f, center, width, ratio, clock = map(float, case)
    om = ratio*width
    k = om*om/4

    def rhs(t, state):
        p, current, x1, v1, x2, v2 = state[:6]
        incoming, incoming_dot = incident(t, center, width, amplitude)
        # E continuity and B_right-B_left=-current solve the sheet scattering.
        reflected = -current/2
        transmitted = incoming+reflected
        e_left, b_left = incoming+reflected, incoming-reflected
        e_right = b_right = transmitted
        e_sheet = (e_left+e_right)/2
        radiation_load = ((e_left*e_left+b_left*b_left)
                          -(e_right*e_right+b_right*b_right))/2
        net_energy_flux = incoming**2-reflected**2-transmitted**2
        spring_force = k*(x1-x2)
        return [current, f*e_sheet-p, v1, 2*(radiation_load-spring_force),
                v2, 2*spring_force, net_energy_flux, radiation_load,
                reflected**2, -p*incoming_dot, radiation_load*v1, incoming**2]

    tolerance = (2e-11, 2e-13, 0.2) if refined else (2e-9, 2e-11, 0.4)
    start, end = -half_window/width, half_window/width
    solution = solve_ivp(rhs, (start, end), np.zeros(12), method="DOP853",
                         rtol=tolerance[0], atol=tolerance[1],
                         max_step=tolerance[2], dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    readout = solution.sol(clock/width)
    final = solution.y[:, -1]
    sample = solution.sol(np.linspace(start, end, 2001))
    oscillator_energy = (sample[0]**2+sample[1]**2)/(2*f) if f else np.zeros(2001)
    mech_energy = (sample[3]**2+sample[5]**2)/4 + k*(sample[2]-sample[4])**2/2
    return {"case": list(case), "half_window": half_window, "refined": refined,
            "amplitude": amplitude, "z": float(readout[4]/amplitude**2),
            "readout_state": readout[:6], "optical_energy_min": oscillator_energy.min(),
            "optical_energy_balance_max_abs": np.max(np.abs(oscillator_energy-sample[6])),
            "mechanical_energy_balance_max_abs": np.max(np.abs(mech_energy-sample[10])),
            "impulse": final[7], "twice_reflected_energy": 2*final[8],
            "source_impulse": final[9], "incident_energy": final[11],
            "impulse_balance_abs": abs(final[7]-2*final[8]),
            "source_impulse_difference_abs": abs(final[7]-final[9]),
            "residual_oscillator_energy": (final[0]**2+final[1]**2)/(2*f) if f else 0,
            "nfev": solution.nfev, "seconds": time.perf_counter()-before}


def cal_physical(f, frequencies, intensities):
    w = np.asarray(frequencies)
    refl = f*f*w*w/(4*(1-w*w)**2+f*f*w*w)
    return 2*np.asarray(intensities)*refl


def cal_source(f, frequencies, intensities):
    w = np.asarray(frequencies)
    chi = f/(1-w*w-0.5j*f*w)
    return np.asarray(intensities)*w*chi.imag


CAL_W = np.repeat(np.repeat([1.5, 1.65, 1.8], 3), 16)
CAL_I = np.repeat(np.tile([0.5, 1.0, 2.0], 3), 16)
SIGMA = 0.0005


def fit(values):
    def loss(f):
        residual = (cal_physical(f, CAL_W, CAL_I)-values)/SIGMA
        return float(residual@residual)
    fitted = minimize_scalar(loss, bounds=(0.2, 0.4), method="bounded",
                             options={"xatol": 1e-12})
    f = min([0.2, float(fitted.x), 0.4], key=loss)
    return f, loss(f)/143


GROUPS = []
for center in [1.5, 1.65, 1.8]:
    rows = []
    for i, width in enumerate([0.04, 0.07, 0.1]):
        for j, ratio in enumerate([1.0, 2.0, 4.0]):
            rows.append([center, width, ratio, [1.0, 2.5, 4.0][(i+j)%3]])
    GROUPS.append(rows)


def group_predictions(f):
    values = np.asarray([[predict([f]+r) for r in group] for group in GROUPS])
    return values[:, :, 0], values[:, :, 1]


def rms(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x)**2, axis=axis))


def run():
    checkpoint("preflight", {"python_source_frozen": True,
                             "new_model_docker_git_operations": 0})
    # Stationary observations determine f and agree for both physical laws.
    static_rows, noiseless = [], []
    for f in np.linspace(0.2, 0.4, 41):
        physical = cal_physical(f, CAL_W, CAL_I)
        source = cal_source(f, CAL_W, CAL_I)
        estimate, chi2 = fit(physical)
        noiseless.append({"truth": f, "fit": estimate, "absolute_error": abs(f-estimate),
                          "chi2": chi2})
        for w, intensity in product([1.5, 1.65, 1.8], [0.5, 1, 2]):
            alpha = f/(1-w*w-0.5j*f*w)
            reflection = 0.5j*w*alpha
            transmission = 1+reflection
            phase = 2*np.pi*np.arange(512)/512
            e0 = np.sqrt(2*intensity)
            e = e0*np.cos(phase)
            edot = -w*e0*np.sin(phase)
            p = np.real(alpha*e0*np.exp(-1j*phase))
            pdot = np.real(-1j*w*alpha*e0*np.exp(-1j*phase))
            static_rows.append({"f": f, "omega": w, "intensity": intensity,
                                "mean_lorentz": np.mean(pdot*e),
                                "mean_gradient": np.mean(-p*edot),
                                "calibration": float(cal_physical(f, w, intensity)),
                                "unitarity_error": abs(abs(reflection)**2+abs(transmission)**2-1)})
        if not np.allclose(physical, source, rtol=1e-13, atol=1e-14):
            raise AssertionError("Stationary source and physical maps disagree")
    save("stationary-checks.json", static_rows)
    save("noiseless-fits.json", noiseless)
    checkpoint("calibration", {"settings": 9, "records_per_dataset": 144,
               "stationary_cases": len(static_rows),
               "max_force_difference": max(abs(x["mean_lorentz"]-x["mean_gradient"]) for x in static_rows),
               "max_mean_calibration_difference": max(abs(x["mean_lorentz"]-x["calibration"]) for x in static_rows),
               "max_unitarity_error": max(x["unitarity_error"] for x in static_rows),
               "max_noiseless_parameter_error": max(x["absolute_error"] for x in noiseless)})

    bounds = np.asarray([[.2, .4], [1.5, 1.8], [.04, .1], [.5, 4], [1, 4]])
    interior = np.random.default_rng(614021).uniform(bounds[:, 0], bounds[:, 1], size=(128, 5))
    tensor = list(product([.2, .25, .3, .35, .4], [1.5, 1.6, 1.7, 1.8],
                          [.04, .07, .1], [.5, 1.5, 2.5, 4], [1, 2, 3, 4]))
    survey = []
    for index, case in enumerate(tensor+interior.tolist()):
        physical, source = predict(case)
        gap = abs(source-physical)/max(abs(physical), 1e-300)
        flags = []
        if abs(physical) < .05:
            flags.append("small_physical")
        if abs(source) < .05:
            flags.append("small_source")
        if gap < .04:
            flags.append("gap_below_gate")
        if physical*source < 0:
            flags.append("opposite_sign")
        survey.append({"index": index, "kind": "tensor" if index < 960 else "interior",
                       "case": case, "physical": physical, "source": source,
                       "absolute_gap": abs(source-physical), "relative_gap": gap, "flags": flags})
    save("domain-survey.json", survey)
    curve = []
    for f in np.linspace(.2, .4, 41):
        truth, source = group_predictions(f)
        norms = rms(truth, axis=1)
        curve.append({"strength": f, "physical": truth, "source": source,
                      "true_group_rms": norms, "source_group_error": rms(source-truth, axis=1)/norms})
    save("diagnostic-strength-curve.json", curve)
    save("diagnostic-rows.json", GROUPS)
    checkpoint("survey", {"cases": len(survey),
               "flag_counts": {flag: sum(flag in row["flags"] for row in survey)
                               for flag in ["small_physical", "small_source", "gap_below_gate", "opposite_sign"]},
               "min_absolute_physical": min(abs(row["physical"]) for row in survey),
               "min_absolute_source": min(abs(row["source"]) for row in survey),
               "min_relative_gap": min(row["relative_gap"] for row in survey),
               "min_group_rms_41_strengths": min(float(np.min(row["true_group_rms"])) for row in curve),
               "min_source_group_error_41_strengths": min(float(np.min(row["source_group_error"])) for row in curve)})

    corners = list(product(*bounds))
    reference_cases = corners+interior[:8].tolist()
    refined_indices = list(range(8))+list(range(32, 36))
    references, refinements = [], []
    for index, case in enumerate(reference_cases):
        base = reference(case)
        physical, source = predict(case)
        base.update({"index": index, "oracle": physical, "source": source,
                     "scaled_agreement_error": abs(base["z"]-physical)/max(.05, abs(base["z"]))})
        references.append(base)
        save("reference-cases.json", references)
        if index in refined_indices:
            time_ref = reference(case, refined=True)
            tail_ref = reference(case, half_window=64)
            oracle_time = predict(case, dt=.0625)[0]
            oracle_tail = predict(case, half_window=64)[0]
            scale = max(.05, abs(time_ref["z"]))
            refinements.append({"index": index, "case": case, "time_reference": time_ref,
                                "tail_reference": tail_ref,
                                "oracle_time": oracle_time, "oracle_tail_spectral": oracle_tail,
                                "scaled_reference_time_change": abs(time_ref["z"]-base["z"])/scale,
                                "scaled_reference_tail_change": abs(tail_ref["z"]-base["z"])/scale,
                                "scaled_oracle_time_change": abs(oracle_time-physical)/scale,
                                "scaled_oracle_tail_spectral_change": abs(oracle_tail-physical)/scale})
            save("refinements.json", refinements)
        print(json.dumps({"reference_case": index, "z": base["z"],
                          "scaled_error": base["scaled_agreement_error"],
                          "elapsed_seconds": time.perf_counter()-START}), flush=True)
    checkpoint("independent_reference", {"base_cases": len(references), "refined_cases": len(refinements),
               "max_scaled_agreement_error": max(x["scaled_agreement_error"] for x in references),
               "max_refinement_changes": {key: max(x[key] for x in refinements) for key in refinements[0] if key.startswith("scaled_")},
               "max_optical_energy_balance_abs": max(x["optical_energy_balance_max_abs"] for x in references),
               "max_mechanical_energy_balance_abs": max(x["mechanical_energy_balance_max_abs"] for x in references),
               "max_impulse_balance_abs": max(x["impulse_balance_abs"] for x in references),
               "max_source_impulse_difference_abs": max(x["source_impulse_difference_abs"] for x in references),
               "max_incident_energy_error": max(abs(x["incident_energy"]-1) for x in references),
               "maximum_reference_seconds": max(x["seconds"] for x in references),
               "total_reference_seconds": sum(x["seconds"] for x in references) + sum(x["time_reference"]["seconds"]+x["tail_reference"]["seconds"] for x in refinements)})

    noise_rows, datasets = [], []
    strengths = [0.31]*128 + np.linspace(.2, .4, 128).tolist()
    generator = np.random.default_rng(614027)
    fixed_values = cal_physical(.31, CAL_W, CAL_I)+np.random.default_rng(614023).normal(0, SIGMA, 144)
    save("calibration-fixed.json", [{"omega": w, "intensity": intensity, "value": value, "sigma": SIGMA}
                                    for w, intensity, value in zip(CAL_W, CAL_I, fixed_values)])
    all_strengths = [.31]+strengths
    for index, truth_f in enumerate(all_strengths):
        values = fixed_values if index == 0 else cal_physical(truth_f, CAL_W, CAL_I)+generator.normal(0, SIGMA, 144)
        fitted, chi2 = fit(values)
        true, unused = group_predictions(truth_f)
        oracle, source = group_predictions(fitted)
        norms = rms(true, axis=1)
        oracle_error = rms(oracle-true, axis=1)/norms
        source_error = rms(source-true, axis=1)/norms
        row = {"index": index-1, "kind": "fixed" if index == 0 else "independent_noise",
               "true_strength": truth_f, "fitted_strength": fitted,
               "parameter_relative_error": abs(fitted-truth_f)/truth_f, "reduced_chi2": chi2,
               "oracle_group_error": oracle_error, "source_group_error": source_error,
               "true_group_rms": norms,
               "calibration_pass": bool(chi2 < 1.5 and abs(fitted-truth_f)/truth_f < .03),
               "oracle_prediction_pass": bool(np.all(oracle_error < .04)),
               "source_all_groups_fail": bool(np.all(source_error >= .04))}
        noise_rows.append(row)
        datasets.append({"index": index-1, "true_strength": truth_f, "values": values})
        if index % 32 == 0:
            save("noise-results.json", noise_rows)
            save("noise-datasets.json", datasets)
            print(json.dumps({"noise_dataset": index-1, "elapsed_seconds": time.perf_counter()-START}), flush=True)
    save("noise-results.json", noise_rows)
    save("noise-datasets.json", datasets)
    random_rows = noise_rows[1:]
    checkpoint("noise", {"fixed": noise_rows[0], "independent_datasets": len(random_rows),
               "calibration_passes": sum(x["calibration_pass"] for x in random_rows),
               "oracle_prediction_passes": sum(x["oracle_prediction_pass"] for x in random_rows),
               "source_all_groups_fail_count": sum(x["source_all_groups_fail"] for x in random_rows),
               "max_parameter_error": max(x["parameter_relative_error"] for x in random_rows),
               "max_chi2": max(x["reduced_chi2"] for x in random_rows),
               "max_oracle_prediction_error": max(float(np.max(x["oracle_group_error"])) for x in random_rows),
               "min_source_prediction_error": min(float(np.min(x["source_group_error"])) for x in random_rows)})

    nominal = [.31, 1.65, .07, 2, 2.5]
    scale_refs = [reference(nominal, amplitude=a) for a in [.5, 1, 2]]
    zero_ref = reference([.31, 1.65, .07, 0, 2.5])
    zero_field = optical_fields(0., 1.65, .07)
    fields = optical_fields(.31, 1.65, .07)
    mask = fields["t"] <= 2.5/.07
    t = fields["t"][mask]
    center_displacement = simpson((2.5/.07-t)*fields["force"][mask], x=t)
    absolute_impulse = simpson(np.abs(fields["force"][mask]), x=t)
    limits = []
    for ratio in [.01, .02, 40, 80]:
        physical, source = predict([.31, 1.65, .07, ratio, 2.5])
        limits.append({"ratio": ratio, "physical": physical, "source": source,
                       "center_displacement": center_displacement,
                       "rigid_difference": abs(physical-center_displacement),
                       "rigid_analytic_bound": absolute_impulse/(ratio*.07)})
    pulse_checks = []
    for center, width in product([1.5, 1.65, 1.8], [.04, .07, .1]):
        field = optical_fields(.31, center, width)
        indices = np.linspace(0, len(field["t"])-1, 257).astype(int)
        analytic = np.asarray([incident(float(field["t"][i]), center, width) for i in indices])
        energy = field["energy"]
        centroid = simpson(field["t"]*field["e"]**2, x=field["t"])/energy
        pulse_checks.append({"center": center, "width": width, "energy": energy,
                             "centroid": centroid,
                             "max_incident_field_difference": np.max(np.abs(analytic[:, 0]-field["e"][indices])),
                             "max_incident_derivative_difference": np.max(np.abs(analytic[:, 1]-field["edot"][indices]))})
    save("limit-and-scaling-checks.json", {"scaling_references": scale_refs,
         "zero_mechanical_reference": zero_ref, "mechanical_limits": limits,
         "pulse_reconstruction": pulse_checks})
    checkpoint("limits_and_scaling", {
        "weak_equation_scaling_max_difference": max(abs(x["z"]-scale_refs[1]["z"]) for x in scale_refs),
        "zero_mechanical_inert_displacement": zero_ref["z"],
        "zero_strength_max_force": float(np.max(np.abs(zero_field["force"]))),
        "small_frequency_quadratic_ratio": limits[1]["physical"]/limits[0]["physical"],
        "rigid_bounds_hold": all(x["rigid_difference"] <= x["rigid_analytic_bound"] for x in limits[2:]),
        "max_pulse_energy_error": max(abs(x["energy"]-1) for x in pulse_checks),
        "max_pulse_centroid_abs": max(abs(x["centroid"]) for x in pulse_checks),
        "max_analytic_fft_field_difference": max(x["max_incident_field_difference"] for x in pulse_checks)})

    ref_section = REPORT["sections"]["independent_reference"]
    conditions = {
        "independent_numerical_agreement": ref_section["max_scaled_agreement_error"] < .0002,
        "separate_refinements": max(ref_section["max_refinement_changes"].values()) < .0002,
        "ordinary_group_norms": REPORT["sections"]["survey"]["min_group_rms_41_strengths"] > .05,
        "all_strength_groups_separated": REPORT["sections"]["survey"]["min_source_group_error_41_strengths"] >= .04,
        "all_noise_calibrations": all(x["calibration_pass"] for x in noise_rows),
        "all_noise_oracles": all(x["oracle_prediction_pass"] for x in noise_rows),
        "all_noise_source_groups_fail": all(x["source_all_groups_fail"] for x in noise_rows),
    }
    REPORT["status"] = "complete"
    REPORT["bounded_feasibility_checks"] = conditions
    REPORT["all_predeclared_feasibility_checks_pass"] = all(conditions.values())
    REPORT["model_difficulty_or_population_pass_rate"] = None
    REPORT["qualification"] = False
    REPORT["elapsed_seconds"] = time.perf_counter()-START
    REPORT["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save("report.json", REPORT)
    shutil.copy2(RUN / "report.json", STAGE / "report.json")
    print(json.dumps({"status": "complete", "seconds": REPORT["elapsed_seconds"],
                      "feasibility_checks": conditions}), flush=True)


try:
    run()
except BaseException as exc:
    REPORT["status"] = "failed_execution"
    REPORT["exception_type"] = type(exc).__name__
    REPORT["exception"] = str(exc)
    REPORT["traceback"] = traceback.format_exc()
    REPORT["elapsed_seconds"] = time.perf_counter()-START
    save("report.failed.json", REPORT)
    raise
