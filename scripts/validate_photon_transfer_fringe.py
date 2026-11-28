"""Validate the two-photon Λ waveguide with a path expansion and continuous-field Fock amplitudes."""
import argparse
import copy
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/photon-transfer-fringe"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from reference import reading
from waveguide import waveguide


def pairs(rows):
    return np.stack((rows.real, rows.imag), axis=-1).tolist()


def simple():
    waves = pairs(np.eye(2, dtype=complex))
    return dict(
        rates=[1.0, 1.0],
        durations=[1.0, 1.0],
        detuning=[0.0, 0.0],
        atom=[0.0, 0.0],
        analyzer=None,
        control=[1, 0.0, 0.0, 0.0],
        input_modes=waves,
        output_modes=waves,
        input_rotation=[0.0, 0.0],
        output_rotations=[[0.0, 0.0], [0.0, 0.0]],
        phase=0.0,
        **{"lambda": np.pi}
    )


def shots():
    rng = np.random.default_rng(842511)

    def uniform(lo, hi, n=None):
        return np.round(rng.uniform(lo, hi, n), 5).tolist()

    def angle():
        return [uniform(0.14, 1.42), uniform(-2.8, 2.8)]

    sets = {}
    for family in ("raman_memory", "overlapping_pairs", "atomic_eraser"):
        out = []
        for k in range(10):
            n = 4 + k % 3
            boundary = (n + (k % 2)) // 2

            def normalized(size):
                a = rng.uniform(0.6, 1.5, size) * np.exp(
                    1j * rng.uniform(-1.5, 1.5, size)
                )
                return a / np.linalg.norm(a)

            src = np.zeros((2, n), complex)
            rec = np.zeros((2, n), complex)
            if family == "raman_memory":
                order = np.array([0, 1]) if k % 2 else np.array([1, 0])
                src[order[0], :boundary] = normalized(boundary)
                src[order[1], boundary:] = normalized(n - boundary)
            else:
                src = np.array([normalized(n), normalized(n)])
            if family == "atomic_eraser":
                cand = np.array([normalized(n), normalized(n)]).T
                rec = np.linalg.qr(cand)[0].T
            else:
                rec[0, :boundary] = normalized(boundary)
                rec[1, boundary:] = normalized(n - boundary)
            pulse = [boundary, *angle(), uniform(-2, 2)]
            if family in ("raman_memory", "overlapping_pairs") and k % 2 == 0:
                pulse = [boundary, 0.0, 0.0, 0.0]
            q = dict(
                rates=uniform(0.4, 1.8, 2),
                durations=uniform(0.4, 1.8, n),
                detuning=uniform(-1.1, 1.1, n),
                atom=angle(),
                analyzer=angle() if family == "atomic_eraser" or k % 3 == 0 else None,
                control=pulse,
                input_modes=pairs(src),
                output_modes=pairs(rec),
                input_rotation=angle(),
                output_rotations=[angle(), angle()],
                phase=uniform(-0.9, 0.9),
                **{"lambda": uniform(1.55, 3.1)}
            )
            out.append(q)
        sets[family] = out
    return sets


def validate(generate=False):
    vis = 0.8436
    calibration = json.loads((TASK / "environment/data/calibration.json").read_text())
    assert len(calibration) == 12 and calibration == json.loads(
        (TASK / "tests/data/calibration.json").read_text()
    )
    expected = {}
    for family, queries in shots().items():
        rows = []
        for q in queries:
            result = reading(q)
            np.testing.assert_allclose(waveguide(q), result, atol=2e-12, rtol=0)
            assert 0.01 <= result[1] <= 1
            h = np.array([[complex(*z) for z in row] for row in q["output_modes"]])
            np.testing.assert_allclose(h @ h.conj().T, np.eye(2), atol=1e-13)
            rows.append(dict(input=q, value=round(vis * result[0], 8)))
        expected[family] = rows
    path = TASK / "tests/data/predictions.json"
    if generate:
        path.write_text(json.dumps(expected, indent=1) + "\n")
    frozen = json.loads(path.read_text())
    assert expected.keys() == frozen.keys()
    for family, rows in expected.items():
        for row, item in zip(rows, frozen[family], strict=True):
            assert (
                row["input"] == item["input"]
                and abs(row["value"] - item["value"]) < 1e-7
            )
    for name, model_file in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("which-state", ROOT / "scripts/photon_transfer_fringe_baseline.py"),
    ]:
        model = runpy.run_path(str(model_file))["Model"]().fit(calibration)
        assert abs(model.visibility - vis) < 1e-8
        errors = {}
        for family, rows in frozen.items():
            predictions = model.predict([r["input"] for r in rows])
            assert np.isfinite(predictions).all()
            errors[family] = float(
                np.max(abs(predictions - [r["value"] for r in rows]))
            )
        assert (
            max(errors.values()) < 1e-7
            if name == "oracle"
            else min(errors.values()) > 0.1
        )
        print(name, errors)
    q = simple()
    x = np.exp(-1)
    norm = (1 - x) ** 2 * (1 + x * x) + x**4
    known = 1 - 2 * x * x * (1 - x) ** 2 / norm, norm
    for method in (reading, waveguide):
        np.testing.assert_allclose(method(q), known, atol=1e-13)
        q0 = frozen["atomic_eraser"][0]["input"]
        split = copy.deepcopy(q0)
        split["durations"][:1] = [split["durations"][0] / 2] * 2
        split["detuning"][:1] *= 2
        split["control"][0] += 1
        for key in ("input_modes", "output_modes"):
            for row in split[key]:
                row[:1] = [(np.array(row[0]) / np.sqrt(2)).tolist()] * 2
        np.testing.assert_allclose(method(split), method(q0), atol=2e-12)
        q1 = copy.deepcopy(q0)
        q1["input_modes"][1] = [[-b, a] for a, b in q1["input_modes"][1]]
        np.testing.assert_allclose(method(q1), method(q0), atol=2e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
