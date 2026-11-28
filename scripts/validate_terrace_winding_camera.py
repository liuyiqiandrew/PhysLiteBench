"""Validate two-height step readings against positive, winding-resolved Markov propagation."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/terrace-winding-camera"


def load(relative):
    spec = importlib.util.spec_from_file_location("camera", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load(TASK / "tests/reference.py")
oracle = load(TASK / "solution/model.py")
shortcut = load("scripts/terrace_winding_camera_baseline.py")


def pins(n, length, rng):
    while True:
        gaps = rng.integers(2, 6, n)
        if sum(gaps) == length:
            shift = rng.integers(length)
            return sorted(int(x) for x in (shift + np.r_[0, gaps[:-1].cumsum()]) % length)


def preparations():
    rng = np.random.default_rng(11467)
    groups = {}
    for mode in ("population", "dyed", "joint"):
        rows = []
        for i, (n, length) in enumerate([(2,7),(3,9),(4,12),(5,15),(2,8),(3,11),(4,16),(5,20)]):
            bottom, top = pins(n, length, rng), pins(n, length, rng)
            w = [-1,1,0,1,0,-1,1,-1][i]
            low = int(rng.integers(0, length//2+1))
            high = int(rng.integers(low+2, length+1))
            expected = n*(high-low)/length
            count = min(n, max(0, int(np.rint(expected)) + (i%3 == 0)))
            tag = (n-1-i) % n
            a, b = (round(z, 3) for z in (rng.uniform(.18,.42), rng.uniform(.58,.82)))
            winding, final_index = divmod(tag+w, n)
            center = ((1-a)*bottom[tag]+a*(top[final_index]+length*winding)) % length
            lo = max(0, min(length-2, int(center)-int(rng.integers(0,3))))
            hi = min(length, lo+int(rng.integers(2, max(4,length//2))))
            q = dict(mode="pinned", circumference=length, bottom=bottom, top=top,
                     height=round(rng.uniform(2.5,6.5),3), winding=w, levels=[a,b], tag=tag,
                     window=[0,length] if mode=="population" else [lo,hi],
                     count_window=[0,length] if mode=="dyed" else [low,high],
                     count=n if mode=="dyed" else count)
            rows.append(q)
        groups[mode] = rows
    return groups


def public_example():
    return dict(mode="pinned",circumference=9,bottom=[0,3,6],top=[1,4,7],height=4.,
                winding=1,levels=[.3,.7],tag=2,window=[6,9],count_window=[2,6],count=1)


def main(generate):
    cal = [dict(input=dict(mode="wandering",height=h),value=round(2*ref.TRUE_PARAMETER*h,8))
           for h in [.35,.707,.92,1.643,2.21,3.735,4.996,6.207]]
    root = TASK / "tests/data"
    groups = preparations()
    expected = {k: [dict(input=q,value=round(ref.image(q),10)) for q in qs] for k,qs in groups.items()}
    for path, records in [(root/"calibration.json",cal),
                          (TASK/"environment/data/calibration.json",cal),
                          (root/"predictions.json",expected)]:
        if generate:
            path.write_text(json.dumps(records, indent=1) + "\n")
        assert json.loads(path.read_text()) == records
    for module in [oracle, shortcut]:
        model = module.Model().fit(cal)
        assert abs(model.rate-ref.TRUE_PARAMETER)<1e-7
        assert max(abs(model.predict([r["input"] for r in cal])-[r["value"] for r in cal])) < 2e-5
        for k, rows in expected.items():
            err = max(abs(model.predict([r["input"] for r in rows])-[r["value"] for r in rows]))
            print(module.__file__, k, float(err))
            assert err < 1e-7 if module is oracle else err > .003
    example = public_example()
    assert abs(ref.image(example, cutoff=13)-.53847829) < 5e-9
    assert abs(oracle.image(example,ref.TRUE_PARAMETER)-.53847829) < 5e-9
    # Summing over every second-camera count and partitioning a tagged window gives unity.
    n, length = len(example["bottom"]), example["circumference"]
    total = sum(oracle.image(dict(example,count=k,window=win),ref.TRUE_PARAMETER)
                for k in range(n+1) for win in ([0,4],[4,length]))
    assert abs(total-1) < 1e-10
    # Time reversal swaps the pins, reverses the winding and changes the tag's pin index.
    tagged = dict(example, count_window=[0,length],count=n)
    reverse = dict(tagged, bottom=tagged["top"],top=tagged["bottom"],winding=-tagged["winding"],
                   tag=(tagged["tag"]+tagged["winding"])%n,levels=[1-tagged["levels"][0],.82])
    assert abs(oracle.image(tagged,ref.TRUE_PARAMETER)-oracle.image(reverse,ref.TRUE_PARAMETER)) < 1e-11
    # Large time and maximal line density stress the winding cutoff.
    q = dict(groups["joint"][3],height=6.5)
    assert abs(ref.image(q,1.15,9)-ref.image(q,1.15,13)) < 5e-11


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    main(parser.parse_args().generate)
