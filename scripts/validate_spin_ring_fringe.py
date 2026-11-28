"""Check the exchange fringe controls against direct spin traces."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/spin-ring-fringe"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load(TASK/"tests/reference.py")
oracle = load(TASK/"solution/model.py")
shortcut = load(ROOT/"scripts/spin_ring_fringe_baseline.py")


def calibration():
    rng = np.random.default_rng(92711)
    rows = []
    for n in [7,8,19,28]:
        for m in [1,2,3,2,4,3]:
            sites = list(range(1,1+m))
            a = [1.25,.41,.87,.18] if n<10 else [2.08,0.,.95,0.]
            b = [1.64,-.17,.69,.37] if n<10 else [1.75,0.,1.17,0.]
            q = dict(n=n,first=a,second=b,sites=sites)
            signal = ref.TRUE_PARAMETER*ref.fringe(q)
            rows.append(dict(input=q,value=signal+rng.normal(0,.0015),sigma=.0015))
    return rows


def families():
    groups = {}
    for name in ["anisotropic","frustrated","commuting"]:
        rows = []
        for i in range(6):
            if name == "anisotropic":
                n = 7 + i%2
                a,b = [2.16,.54,.98,.21],[2.42,-.31,1.16,-.12]
            elif name == "frustrated":
                n = 7 + 2*(i%2)
                a,b = [2.74,-.29,-1.27,-.09],[2.25,.59,-1.12,.27]
            else:
                n = [17,22,27,28,31,32][i]
                a,b = ([3.1,0.,1.24,0.],[2.42,0.,.98,0.]) if i%2 else ([2.83,0.,0.,.89],[3.08,0.,0.,1.16])
            sites = [[0,2,n-2], [3,0,n-1], [1,2,n-3,n-2]][i%3]
            q = dict(n=n,first=a,second=b,sites=sites)
            rows.append(dict(input=q,value=ref.TRUE_PARAMETER*ref.fringe(q)))
        groups[name] = rows
    large = [
        dict(n=10, first=[2.8,.41,1.21,.16], second=[2.45,-.23,1.17,.31], sites=[8,1,4]),
        dict(n=11, first=[2.3,-.38,-1.12,.22], second=[3.,.17,-1.2,-.15], sites=[9,0,6,2]),
        dict(n=19, first=[2.,1.45,.34,.22], second=[3.1,1.18,.38,.26], sites=[14]),
        dict(n=32, first=[2.4,1.23,.4,-.13], second=[2.8,-1.37,.31,.16], sites=[31]),
    ]
    groups["large_xy"] = [dict(input=q, value=ref.TRUE_PARAMETER*ref.fringe(q)) for q in large]
    return groups


def validate(regenerate):
    cal, groups = calibration(), families()
    fixtures = [(TASK/"tests/data/calibration.json",cal),
                (TASK/"environment/data/calibration.json",cal),
                (TASK/"tests/data/hidden.json",groups)]
    for path, obj in fixtures:
        if regenerate:
            path.write_text(json.dumps(obj,indent=1)+"\n")
        other = json.loads(path.read_text())
        if isinstance(obj,list):
            obj,other = {"cal":obj},{"cal":other}
        assert obj.keys()==other.keys()
        for key in obj:
            rows, expected = other[key],obj[key]
            assert [{k:v for k,v in r.items() if k!="value"} for r in rows] == [
                {k:v for k,v in r.items() if k!="value"} for r in expected]
            np.testing.assert_allclose([r["value"] for r in rows],[r["value"] for r in expected],atol=1e-12,rtol=0)

    # At high temperature both reductions are identity divided by 2^m.
    inf_temp = dict(n=31,first=[0.,1.,-.3,.98],second=[2.1,-.45,-.6,1.02],sites=[20,6,7,1])
    np.testing.assert_allclose(oracle.fringe(inf_temp),1/16,atol=2e-14)
    public_example = dict(n=12, sites=[0,3], first=[2.,0.,1.,0.], second=[2.,0.,1.,0.])
    for model in (ref, oracle):
        np.testing.assert_allclose(model.fringe(public_example), .49118946, atol=5e-9, rtol=0)
    # Compare the partition-function derivative to a one-spin Gibbs trace
    # on even and odd rings before using it to label larger rings.
    for n in (8, 9):
        for row in groups["large_xy"]:
            setting = row["input"]["first"]
            one_site = ref.reduce_state(n, setting, [0])
            np.testing.assert_allclose(ref.site_magnetization(n, setting),
                                       one_site[0,0] - one_site[1,1], atol=1e-12, rtol=0)
    # Check the transfer expression against the spin matrix in both Ising bases.
    for f,s in [([2.,0,.7,0],[1.1,0,-.5,0]),([2.,0,0,.7],[1.1,0,0,-.5])]:
        sites = [0,2,3,6]
        a,p = ref.ising_marginal(8,f,sites)
        b,v = ref.ising_marginal(8,s,sites)
        np.testing.assert_allclose(p@v,ref.fringe(dict(n=8,first=f,second=s,sites=sites)),atol=1e-12)

    for label,mod in [("oracle",oracle),("shortcut",shortcut)]:
        model = mod.Model().fit(cal)
        x = np.array([mod.fringe(r["input"]) for r in cal])
        np.testing.assert_allclose(x,[ref.fringe(r["input"]) for r in cal],atol=1e-12)
        measurements = np.array([r["value"] for r in cal])
        sigma = cal[0]["sigma"]
        chi2 = np.sum(((measurements-model.visibility*x)/sigma)**2)/(len(x)-1)
        rng = np.random.default_rng(92712)
        repeats = ref.TRUE_PARAMETER*x+rng.normal(0,sigma,(256,len(x)))
        assert max(abs((repeats@x)/(x@x)-ref.TRUE_PARAMETER)) < .005
        assert chi2 < 2.5 and abs(model.visibility-ref.TRUE_PARAMETER) < .005
        errors = {}
        for name, rows in groups.items():
            inputs = [r["input"] for r in rows]
            truth = np.array([r["value"] for r in rows])
            if label=="oracle":
                np.testing.assert_allclose([ref.TRUE_PARAMETER*mod.fringe(q) for q in inputs],truth,atol=1e-12)
            errors[name] = float(np.sqrt(np.mean((model.predict(inputs)-truth)**2)))
        assert max(errors.values()) < .008 if label=="oracle" else min(errors.values()) > .008
        print(label,model.visibility,chi2,errors)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    validate(parser.parse_args().generate)
