"""Validate heralded photon conversion against independent frequency-resolved wavefunctions."""
import argparse, json, runpy, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/photon-transfer-fringe"
sys.path[:0] = [str(TASK / "environment"), str(TASK / "solution"), str(TASK / "tests")]
from optics import calibration_signal
from spectral import spectral_signal
from herald import conversion_signal
from frequency_reference import reading as linear_ref
from reference import reading as nonlinear_ref


def calibration_inputs():
    rng = np.random.default_rng(911247)

    def draw(lo, hi):
        return round(float(rng.uniform(lo, hi)), 4)

    calibration = []
    for i in range(12):
        q = dict(
            source=[[draw(0.12, 0.85), 0.0, 0.0], [0.0, 0.0, 0.0]],
            prep=[],
            after=[],
            before=[[0, 1, draw(0.2, 1.2), draw(-3.1, 3.1)]],
            bank=[0],
            gate=None,
            phase=draw(-0.8, 0.8),
        )
        q["lambda"] = draw(-2.8, 2.8)
        calibration.append(q)
    return calibration


def spectral_preparations():
    rng = np.random.default_rng(5867231)
    result = {}

    def state(n):
        a = np.linalg.qr(rng.normal(size=(n, n)))[0]
        b = np.linalg.qr(rng.normal(size=(n, n)))[0]
        p = rng.uniform(0.55, 1.5, n)
        p[0] = rng.uniform(0.53, 0.7)
        p[-1] = rng.uniform(1.32, 1.51)
        return dict(
            mean=np.round(rng.uniform(-0.6, 0.6, n), 3).tolist(),
            cov=np.round((a * p) @ a.T, 4).tolist(),
            chirp=np.round((b * rng.uniform(-0.76, 0.76, n)) @ b.T, 4).tolist(),
            delay=np.round(rng.uniform(-1.29, 1.29, n), 3).tolist(),
        )

    def mesh(n, K):
        return [
            [
                *map(int, rng.choice(n, 2, replace=False)),
                round(float(rng.uniform(0.37, 1.23)), 3),
                round(float(rng.uniform(-3.0, 3.0)), 3),
            ]
            for _ in range(K)
        ]

    for name in ("joint_spectra", "dispersive_paths", "spectral_eraser"):
        shots = []
        for i in range(8):
            n = 2 if i in (0, 4) else 3
            before = mesh(n, n + 1)
            after = mesh(n, n + 2)
            s = state(n)
            spec = dict(preparations=[s], plate=[i % n, 0.0, 0.0])
            gate = None if i % 3 == 0 else dict(rails=[(i + 1) % n], parity=(-1) ** i)
            angle = float(rng.uniform(1.3, 3.1))
            phase = float(rng.uniform(-1.1, 1.1))
            bank = [i % n]
            if name == "joint_spectra" and i == 1:
                s = dict(
                    mean=[0.0, 0.0, 0.0],
                    cov=[[1.0, -0.43, 0.0], [-0.43, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    chirp=np.zeros((3, 3)).tolist(),
                    delay=[1.17, 0.0, 0.0],
                )
                spec["preparations"] = [s]
                before = [[0, 1, 0.71, 0.11]]
                after = []
                bank = [0]
                gate = None
                angle = 2.89
                phase = 0.03
            if name != "joint_spectra":
                spec["plate"] = [
                    i % n,
                    round((-1) ** i * float(rng.uniform(0.78, 1.57)), 3),
                    round(float(rng.uniform(-0.67, 0.67)), 3),
                ]
            if name == "spectral_eraser":
                spec["preparations"].append(state(n))
                spec["weight"] = np.round(
                    rng.uniform(0.53, 1.18)
                    * np.array([np.cos(i * 0.8), np.sin(i * 0.8)]),
                    3,
                ).tolist()
                spec["marker"] = np.round(
                    rng.uniform(0.6, 0.98)
                    * np.array([np.cos(1.6 + i * 0.53), np.sin(1.6 + i * 0.53)]),
                    3,
                ).tolist()
            shots.append(
                dict(
                    spectral=spec,
                    before=before,
                    after=after,
                    bank=bank,
                    gate=gate,
                    phase=phase,
                    **{"lambda": angle}
                )
            )
        result[name] = shots
    return result


def conversion_preparations():
    rng = np.random.default_rng(823501)
    out = {}

    def source():
        h, g = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in (0, 1)]
        c = (h * rng.uniform(0.6, 1.42, 3)) @ h.T
        l = (g * rng.uniform(-0.7, 0.7, 3)) @ g.T
        return dict(
            mean=np.round(rng.uniform(-0.5, 0.5, 3), 3).tolist(),
            cov=np.round(c, 3).tolist(),
            chirp=np.round(l, 3).tolist(),
            delay=np.round(rng.uniform(-1.2, 1.2, 3), 3).tolist(),
        )

    def circ():
        channel = rng.permutation(3)
        return [
            [
                int(channel[i % 3]),
                int(channel[(i + 1) % 3]),
                round(float(rng.uniform(0.34, 1.13)), 3),
                round(float(rng.uniform(-2.8, 2.8)), 3),
            ]
            for i in range(4)
        ]

    for family in ["conversion_ports", "sum_spectrum", "arrival_herald"]:
        rows = []
        for i in range(8):
            spec = dict(
                preparations=[source()],
                plate=[
                    i % 3,
                    round(float(rng.uniform(-1.2, 1.2)), 3),
                    round(float(rng.uniform(-0.6, 0.6)), 3),
                ],
            )
            if i % 2:
                spec["preparations"].append(source())
                spec.update(weight=[0.57, 0.36], marker=[-0.23, 0.59])
            coeff = np.round(
                rng.uniform(0.55, 1.08) * np.array([np.cos(0.7 + i), np.sin(0.7 + i)]),
                3,
            ).tolist()
            paths = []
            for j, (r, s) in enumerate([(0, 1), (1, 2)]):
                paths.append(
                    [
                        r,
                        s,
                        *([1, 0] if j == 0 else coeff),
                        round(float(rng.uniform(-0.6, 0.6)), 3),
                        round(float(rng.uniform(0.75, 1.3)), 3),
                        round(float(rng.uniform(-0.85, 0.85)), 3),
                        round(float(rng.uniform(-1.1, 1.1)), 3),
                    ]
                )
            det = (
                None
                if family == "conversion_ports"
                else {"band": [-0.5, 1.24]}
                if family == "sum_spectrum" and i % 2 == 0
                else {
                    "mode": [
                        round(float(rng.uniform(-0.7, 0.7)), 3),
                        0.97,
                        round(float(rng.uniform(-1.0, 1.0)), 3),
                        0.22,
                    ]
                }
                if family == "sum_spectrum"
                else {"time": [round(-0.85 + 0.09 * i, 3), round(0.44 + 0.13 * i, 3)]}
            )
            q = dict(
                spectral=spec,
                conversion=dict(paths=paths, detector=det),
                before=circ(),
                after=circ(),
                bank=[i % 3],
                gate=None if i % 3 == 0 else dict(rails=[i % 3], parity=(-1) ** i),
                phase=round(float(rng.uniform(-1, 1)), 3),
                **{"lambda": round(float(rng.uniform(1.7, 3.1)), 3)}
            )
            rows.append(q)
        out[family] = rows
    return out


def sample():
    p = dict(
        mean=[0] * 3,
        cov=np.eye(3).tolist(),
        chirp=np.zeros((3, 3)).tolist(),
        delay=[0] * 3,
    )
    return dict(
        bank=[0],
        before=[],
        after=[[0, 2, np.pi / 4, 0]],
        phase=0.0,
        gate=None,
        spectral=dict(preparations=[p], plate=[0, 0.0, 0.0]),
        conversion=dict(
            paths=[[0, 1, 1, 0, 0, 1, 0, 0], [1, 2, 1, 0, 0, 1, 0, 1]], detector=None
        ),
        **{"lambda": np.pi}
    )


def main(generate):
    vis = 0.8436
    calibration = [
        dict(input=q, value=round(vis * calibration_signal(q), 8))
        for q in calibration_inputs()
    ]
    groups = spectral_preparations()
    groups.update(conversion_preparations())
    expected = {}
    for family, qs in groups.items():
        rows = []
        for q in qs:
            if "conversion" in q:
                y, p = nonlinear_ref(q, 40)
                y0, p0 = nonlinear_ref(q, 29)
                physical = conversion_signal(q)
                assert p > 0.01 and max(abs(y0 - y), abs(p0 - p)) < 3e-6
            else:
                y, p = linear_ref(q, 46)
                physical = spectral_signal(q)
                assert p >= 0.05
            np.testing.assert_allclose(physical, (y, p), atol=2e-7, rtol=0)
            rows.append(dict(input=q, value=round(vis * y, 8)))
        expected[family] = rows
    for path, data in [
        (TASK / "tests/data/calibration.json", calibration),
        (TASK / "environment/data/calibration.json", calibration),
        (TASK / "tests/data/predictions.json", expected),
    ]:
        if generate:
            path.write_text(json.dumps(data, indent=1) + "\n")
        frozen = json.loads(path.read_text())
        if isinstance(data, list):
            assert frozen == data
        else:
            assert frozen.keys() == data.keys()
            for k, rows in data.items():
                assert [r["input"] for r in rows] == [r["input"] for r in frozen[k]]
                np.testing.assert_allclose(
                    [r["value"] for r in rows],
                    [r["value"] for r in frozen[k]],
                    atol=1e-7,
                    rtol=0,
                )
    for name, path in [
        ("oracle", TASK / "solution/model.py"),
        ("starter", TASK / "environment/model.py"),
        ("shortcut", ROOT / "scripts/photon_transfer_fringe_baseline.py"),
    ]:
        model = runpy.run_path(str(path))["Model"]().fit(calibration)
        assert abs(model.visibility - vis) < 1e-8
        errors = {}
        for k, rows in expected.items():
            y = model.predict([r["input"] for r in rows])
            assert np.isfinite(y).all()
            errors[k] = float(max(abs(y - [r["value"] for r in rows])))
            assert (
                errors[k] < 1e-6
                if k in ("joint_spectra", "dispersive_paths", "spectral_eraser")
                or name == "oracle"
                else errors[k] > 0.001
            )
        print(name, errors)
    q = sample()
    for detector, truth in [
        (None, -np.exp(-1)),
        ({"mode": [0.0, np.sqrt(2), 0.0, 0.0]}, -1 / np.cosh(1)),
    ]:
        q["conversion"]["detector"] = detector
        assert abs(conversion_signal(q)[0] - truth) < 1e-12
        assert abs(nonlinear_ref(q, 27)[0] - truth) < 1e-8
    # Splitting a frequency acceptance window only partitions intensity.
    q = conversion_preparations()["sum_spectrum"][0]

    def band(lo, hi):
        return conversion_signal(
            dict(q, conversion=dict(q["conversion"], detector={"band": [lo, hi]}))
        )

    y, p = band(-1.4, 1.5)
    a, pa = band(-1.4, 0.1)
    b, pb = band(0.1, 1.5)
    np.testing.assert_allclose(
        [p, y * p], [pa + pb, a * pa + b * pb], atol=2e-11, rtol=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
