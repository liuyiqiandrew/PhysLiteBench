"""Check the conformal response against physical spins and extrapolated finite rings."""
from pathlib import Path
import json, sys, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/order-disorder-echo"
sys.path[:0] = [str(TASK / "solution"), str(TASK / "tests"), str(TASK / "environment")]
from blocks import response
from lattice import value
from spins import direct
from model import Model

VIS = 0.8134


def shots():
    rng = np.random.default_rng(32719)
    cal = [
        {"input": {"phase": float(x)}, "value": round(VIS * np.cos(x), 9)}
        for x in np.round(np.linspace(-1.37, 1.49, 10), 3)
    ]
    groups = {}
    for group in ("vacuum", "magnetic", "echo"):
        rows = []
        for j in range(6):
            m = [2, 3, 4, 3, 4, 2][j]
            if group == "echo":
                m = 3 + j % 2
            pulses = []
            kinds = ["spin", "string", *rng.choice(["spin", "string"], size=m - 2)]
            rng.shuffle(kinds)
            for kind in kinds:
                span = int(rng.integers(3, 9))
                start = int(rng.integers(0, 16 - span))
                pulses.append([kind, start / 16, (start + span) / 16])
            gaps = []
            for k in range(m - 1):
                reg = round(rng.uniform(0.095, 0.16), 3)
                t = (
                    0.0
                    if group != "echo" and j < 2
                    else round(rng.uniform(0.12, 0.62), 3)
                )
                if group == "echo" and k % 2:
                    t = -t
                gaps.append([reg, t])
            q = dict(
                pulses=pulses,
                segments=gaps,
                sector="even"
                if group == "vacuum" or (group == "echo" and j % 2 == 0)
                else "odd",
                phase=float(np.round(rng.uniform(-2.5, 2.5), 3)),
            )
            rows.append(q)
        groups[group] = rows
    return cal, groups


def proj(q, z):
    return VIS * np.real(z * np.exp(-1j * q["phase"]))


def lattice_centered(q, N):
    # The string's two dual endpoints sit half a lattice unit left of integer
    # labels. Averaging the two nearest placements cancels the linear error.
    return (value(q, N, 0) + value(q, N, 1)) / 2


def dump(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def checks(write=False, full=False):
    cal, qs = shots()
    if write:
        dump(TASK / "environment/data/calibration.json", cal)
        dump(TASK / "tests/data/calibration.json", cal)
    assert cal == json.loads((TASK / "environment/data/calibration.json").read_text())
    file = TASK / "tests/data/hidden.json"
    truth = {} if write else json.loads(file.read_text())
    m = Model().fit(cal)
    assert abs(m.visibility - VIS) < 1e-8
    for g, items in qs.items():
        ed = []
        app = []
        vac = []
        sheets = []
        err = []
        if write:
            truth[g] = []
        for i, q in enumerate(items):
            exact = response(q, steps=256)
            assert abs(exact - response(q, steps=64)) < 1e-8
            if write or full:
                a, b = (lattice_centered(q, N) for N in (64, 128))
                c = lattice_centered(q, 192)
                lo = (4 * b - a) / 3
                hi = (9 * c - 4 * b) / 5
                ed.append((abs(hi - exact), abs(lo - hi)))
                print(g, i, "finite extrapolation", ed[-1], flush=True)
                assert abs(hi - exact) < 0.00045 and abs(lo - hi) < 0.0018
                if write:
                    truth[g].append(dict(input=q, value=round(float(proj(q, hi)), 8)))
            row = truth[g][i]
            assert row["input"] == q
            err.append(abs(m.predict([q])[0] - row["value"]))
            app.append(abs(proj(q, response(q, ignore_disorder=True)) - row["value"]))
            vac.append(abs(proj(q, response(dict(q, sector="even"))) - row["value"]))
            sheets.append(abs(proj(q, response(q, reset_sheet=True)) - row["value"]))
        print(
            g,
            "oracle",
            max(err),
            "order-only control",
            max(app),
            "vacuum",
            max(vac),
            "principal-root",
            max(sheets),
            flush=True,
        )
        assert max(err) < 0.00045 and max(app) > 0.08
    if write:
        dump(file, truth)
    test = dict(
        pulses=[
            ["spin", 0.125, 0.375],
            ["string", 0.25, 0.625],
            ["spin", 0.375, 0.875],
        ],
        segments=[[0.13, 0.31], [0.11, -0.43]],
        sector="odd",
    )
    for sector in ["even", "odd"]:
        q = dict(test, sector=sector)
        microscopic = direct(q, 8)
        print("literal spins", sector, microscopic, flush=True)
        assert abs(value(q, 8) - microscopic) < 1.0e-10
    w = dict(
        pulses=[["spin", 0.125, 0.375], ["string", 0.25, 0.8125]],
        segments=[[0.12, 0.30]],
    )
    for d in [dict(w, sector="even"), dict(w, sector="odd"), dict(test, sector="odd")]:
        print("worked", d, response(d), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--write", action="store_true")
    cli.add_argument("--full", action="store_true")
    opt = cli.parse_args()
    checks(opt.write, opt.full)
