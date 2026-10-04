import json
from pathlib import Path

import numpy as np
import pytest

from model import ThermalModel


def load_runs(directory):
    runs = []
    for entry in json.loads((directory / "metadata.json").read_text())["runs"]:
        data = np.genfromtxt(directory / entry["file"], delimiter=",", names=True)
        runs.append({"t": data["time"],
                     "temperature": np.column_stack((data["temperature_1"], data["temperature_2"])),
                     "sigma": np.column_stack((data["sigma_1"], data["sigma_2"])),
                     "initial_temperature": np.array(entry["initial_temperature"])})
    return runs


@pytest.fixture(scope="module")
def runs():
    return load_runs(Path(__file__).parent)


@pytest.fixture(scope="module")
def model(runs):
    result = ThermalModel()
    assert result.fit(runs) is result
    return result


def test_parameter(model):
    assert np.isfinite(model.conductance) and model.conductance > 0


def test_predictions(model, runs):
    for run in runs:
        y = model.predict(run["t"], run["initial_temperature"])
        assert isinstance(y, np.ndarray) and y.shape == run["temperature"].shape
        assert np.isfinite(y).all()


def test_calibration(model, runs):
    chi2 = sum(np.sum(((model.predict(r["t"], r["initial_temperature"])-r["temperature"])/r["sigma"])**2) for r in runs)
    assert chi2/(sum(r["temperature"].size for r in runs)-1) < 1.5
