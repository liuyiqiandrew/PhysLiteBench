"""Independent sum/difference solution of the coupled heat/work balances."""
import json
import numpy as np

TRUE_CONDUCTANCE = 8e-4
R = 8.31446261815324
AMOUNT = .01
CV = 1.5*R
CP = 2.5*R
LINK = 1.6e-3
TB = 293.
VOLUME = 5e-4


def load_runs(directory):
    runs = []
    for entry in json.loads((directory / "metadata.json").read_text())["runs"]:
        data = np.genfromtxt(directory / entry["file"], delimiter=",", names=True)
        runs.append({"t": data["time"],
                     "temperature": np.column_stack((data["temperature_1"], data["temperature_2"])),
                     "sigma": np.column_stack((data["sigma_1"], data["sigma_2"])),
                     "initial_temperature": np.array(entry["initial_temperature"])})
    return runs


def trajectory(t, initial, g=TRUE_CONDUCTANCE):
    t = np.asarray(t)
    initial = np.asarray(initial)
    total = 2*TB+(initial.sum()-2*TB)*np.exp(-g*t/(AMOUNT*CV))
    difference = (initial[0]-initial[1])*np.exp(-(g+2*LINK)*t/(AMOUNT*CP))*(total/initial.sum())**(R/CP)
    return np.column_stack((total+difference, total-difference))/2


def hidden_cases():
    return [{"t": np.arange(0, 241, 2, dtype=float), "initial_temperature": np.array(initial)}
            for initial in ([313., 313.], [313., 283.], [298., 308.])]


def metrics(model, runs):
    assert np.isfinite(model.conductance) and model.conductance > 0
    loss = 0.
    for run in runs:
        value = model.predict(run["t"], run["initial_temperature"])
        assert isinstance(value, np.ndarray) and value.shape == run["temperature"].shape
        assert np.isfinite(value).all()
        loss += float(np.sum(((value-run["temperature"])/run["sigma"])**2))
    errors = []
    for case in hidden_cases():
        value = model.predict(case["t"], case["initial_temperature"])
        assert isinstance(value, np.ndarray) and value.shape == (len(case["t"]), 2)
        assert np.isfinite(value).all()
        truth = trajectory(case["t"], case["initial_temperature"])
        errors.append(float(np.linalg.norm(value-truth)/np.linalg.norm(truth-TB)))
    return {"conductance_W_per_K": float(model.conductance),
            "relative_conductance_error": float(abs(model.conductance/TRUE_CONDUCTANCE-1)),
            "calibration_reduced_chi2": loss/(sum(r["temperature"].size for r in runs)-1),
            "hidden_nrmse": errors}
