"""Check the two controls, frozen data, symmetries and impulse convergence."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/slab-flux-pump"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query(bias, low, high, readout):
    return dict(bias=bias, low=low, high=high, readout=readout)


def records(inputs, reference, n, increments=1):
    return [{"input": q, "value": reference.signal(q, n=n, increments=increments)}
            for q in inputs]


def validate(generate):
    ref = load(TASK / "tests/reference.py")
    oracle = load(TASK / "solution/model.py")
    baseline = load(ROOT / "scripts/slab_flux_pump_baseline.py")
    cal_inputs = [query(0., lo, hi, r)
                  for lo, hi in [(-1.45, 1.32), (-1.20, .42), (-.65, 1.4),
                                 (.13, 1.28), (-1.54, -.25), (-.3, .24)]
                  for r in ["crest", "trough"]]
    calibration = records(cal_inputs, ref, 16384)
    for row in calibration:
        row["value"] = round(row["value"], 8)
    settings = [(.26, -1.40, 1.30), (-.26, -1.40, 1.30),
                (.23, -.65, 1.50), (-.21, -1.45, .81),
                (.25, .10, 1.55), (-.27, -1.55, -.30),
                (.18, -.82, .77), (-.12, -.66, .88), (.24, -.2, .3),
                (0., -1.26, 1.4)]
    truth = {}
    for r in ["crest", "trough", "impulse"]:
        inputs = [query(*params, r) for params in settings]
        truth[r] = records(inputs, ref, 4800 if r == "impulse" else 16384, 1280)
    for mode in ("induction", "electric"):
        rows = []
        for ratio, h, x in [(.48,.31,.14), (2.4,.29,.19), (.66,.21,.08), (1.9,.30,.27),
                            (1.,.25,.11), (2.45,.15,.67)]:
            for sense, axis in [(1, "y"),(-1,"z")]:
                q = dict(readout=mode, amplitude=h, cutting=ratio, depth=x, sense=sense, axis=axis)
                value = ref.rotary(q, ref.TRUE_PARAMETER)
                assert abs(value - ref.rotary(q, ref.TRUE_PARAMETER, seed=1e-6)) < 2e-8
                mirror = ref.rotary(dict(q, sense=-sense), ref.TRUE_PARAMETER)
                sign = (1 if axis == "y" else -1) * (1 if mode == "induction" else -1)
                assert abs(value - sign * mirror) < 1e-9
                rows.append(dict(input=q, value=float(value)))
        truth[mode] = rows
    setups = [
        (.215,.20,.50,.049,1.65,.60,1),
        (.219,-.19,2.5,.062,1.70,-.62,-1),
        (.211,.30,2.3,.051,1.57,-.55,1),
        (.201,-.27,.62,.084,1.36,.61,1),
        (.217,-.20,.48,.070,1.63,.57,-1),
        (.211,.31,2.46,.048,1.68,.60,-1),
        (.185,.23,1.,.085,1.50,.33,1),
        (.176,-.28,1.7,.093,0.,-.50,-1),
    ]
    for group, readout in [("laminate_induction","induction"), ("laminate_electric","electric"),
                           ("normal_voltage","normal_voltage"), ("sheet_power","sheet_power")]:
        rows = []
        for h, n, ratio, coating, g, hall, sense in setups:
            common = dict(readout=readout, amplitude=h, normal=n, cutting=ratio,
                          coating=coating, conductance=g, hall=hall, sense=sense)
            if readout == "sheet_power":
                cases = [common]
            elif readout == "normal_voltage":
                cases = [dict(common, depth=x) for x in (.04, .19, .76)]
            else:
                cases = [dict(common, depth=x, axis=axis)
                         for x, axis in [(coating, "y"), (coating+.0005, "z"), (.18, "y")]]
            for q in cases:
                value = ref.rotary(q, ref.TRUE_PARAMETER)
                # Reversing both rotation and threading conjugates B, negates
                # conjugate E and leaves E_x and power unchanged.
                reverse = ref.rotary(dict(q, normal=-n, sense=-sense), ref.TRUE_PARAMETER)
                parity = (-1 if q.get("axis") == "z" else 1) * (-1 if readout == "electric" else 1)
                assert abs(value-parity*reverse) < 2e-10
                if readout == "sheet_power":
                    assert value >= 0.
                if readout == "normal_voltage" and ratio == 1.:
                    assert abs(value) < 1e-12
                rows.append(dict(input=q, value=float(value)))
        truth[group] = rows
    # Verify both sheet jump components and tangential E continuity directly.
    for h, n, ratio, coating, g, hall, sense in setups:
        q = dict(amplitude=h, normal=n, cutting=ratio, coating=coating,
                 conductance=g, hall=hall, sense=sense)
        def field(readout, x):
            return sum(mult*oracle.rotary(dict(q, readout=readout, depth=x, axis=a),
                                          ref.TRUE_PARAMETER)
                       for a, mult in [("y", 1), ("z", 1j)])
        e = field("electric", coating)
        jump = field("induction", coating) - field("induction", coating+1e-8)
        assert abs(jump - 1j*g*(1+1j*hall*np.sign(n))*e) < 2e-8
        assert abs(e - field("electric", coating+1e-8)) < 2e-8
    film_cases = [
        (.215,.20,.38,.049,1.65,.60,1,.05), (.219,-.19,.42,.062,1.70,-.62,-1,.13),
        (.211,.30,.54,.051,1.57,-.55,1,.19), (.201,-.27,.35,.084,1.36,.61,1,.24),
        (.217,-.20,.74,.070,1.63,.57,-1,.065), (.211,.31,1.32,.048,1.68,.60,-1,.11),
        (.185,.23,.55,.085,.15,.33,1,.255), (.176,-.28,.72,.093,0.,-.50,-1,.075),
    ]
    for group,readout in [("film_induction","induction"), ("film_electric","electric"),
                          ("film_voltage","normal_voltage"), ("film_power","sheet_power")]:
        rows=[]
        for h,n,c,t,g,hall,sense,k in film_cases:
            common=dict(readout=readout,amplitude=h,normal=n,cutting=c,coating=t,
                        conductance=g,hall=hall,sense=sense,film=k)
            if readout=="sheet_power":
                cases=[common]
            elif readout=="normal_voltage":
                cases=[dict(common,depth=x) for x in (.036,.15,.8)]
            else:
                cases=[dict(common,depth=x,axis=a) for x,a in [(t,"y"),(t+.0003,"z"),(.13,"y")]]
            for q in cases:
                value=ref.film_signal(q,ref.TRUE_PARAMETER)
                refined=ref.film_signal(q,ref.TRUE_PARAMETER,seed=9e-8)
                assert abs(value-refined)<2e-7
                rows.append(dict(input=q,value=float(value)))
        truth[group]=rows
    assert sum(r["value"]==0. for r in truth["film_power"]) in (2,3,4,5)
    # With no conducting shunt, a moving critical film still dissipates.
    assert truth["film_power"][-1]["value"]>1e-6
    for path in [TASK / "environment/data/calibration.json", TASK / "tests/data/calibration.json"]:
        if generate:
            path.write_text(json.dumps(calibration, indent=2) + "\n")
        assert json.loads(path.read_text()) == calibration
    path = TASK / "tests/data/predictions.json"
    if generate:
        path.write_text(json.dumps(truth, indent=2) + "\n")
    shipped = json.loads(path.read_text())
    assert set(shipped) == set(truth)
    for key, rows in truth.items():
        assert [r["input"] for r in rows] == [r["input"] for r in shipped[key]]
        np.testing.assert_allclose([r["value"] for r in rows], [r["value"] for r in shipped[key]],
                                   atol=1e-10, rtol=0)

    # Separate space and time refinements on all ten impulse preparations.
    convergence = {}
    for label, n, increments in [("halve spatial resolution", 2400, 1280),
                                  ("halve temporal resolution", 4800, 640)]:
        convergence[label] = max(abs(ref.signal(r["input"], n=n, increments=increments) - r["value"])
                                 for r in truth["impulse"])
    print("finite-volume refinement differences", convergence)
    assert max(convergence.values()) < 0.004
    # Closed cycles with a flux-free core have zero integrated voltage.
    # Reflection reverses current; the cycle voltages at both faces are equal.
    impulse = [r["value"] for r in truth["impulse"]]
    assert max(abs(impulse[-2]), abs(impulse[-1]), abs(impulse[0] + impulse[1])) < 0.002
    # Fully penetrated crest at zero bias: average sqrt((1+h)^2 - 2*j*d) - 1 over 0<d<1.
    h, j = 1.5, ref.TRUE_PARAMETER
    exact = ((1 + h)**3 - ((1 + h)**2 - 2*j)**1.5) / (3*j) - 1
    q = query(0., -1.5, h, "crest")
    assert abs(ref.signal(q, n=4096) - exact) < 1e-7
    assert abs(oracle.signal(q, j) - exact) < 1e-7

    for name, module in [("oracle", oracle), ("shortcut", baseline)]:
        model = module.Model().fit(calibration)
        errors = {}
        for key, rows in [("calibration", calibration), *truth.items()]:
            calculated = model.predict([r["input"] for r in rows])
            errors[key] = float(np.max(abs(calculated - [r["value"] for r in rows])))
        print(name, "pinning", model.pinning, "maximum absolute errors", errors)
        assert abs(model.pinning - j) < 1e-4 and errors["calibration"] < 1e-5
        limits = dict(crest=.003,trough=.003,impulse=.008,induction=.001,electric=.0006,
                      laminate_induction=.001,laminate_electric=.0006,normal_voltage=.00025,
                      sheet_power=.00015,film_induction=.001,film_electric=.0006,
                      film_voltage=.00025,film_power=.00015)
        for key,limit in limits.items():
            if name=="oracle" or not key.startswith("film_"):
                assert errors[key]<limit,(key,errors[key])
            else:
                assert errors[key]>limit,(key,errors[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
