"""Verify the molecular film speed probabilities and the two completed controls."""
import argparse
from itertools import product
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/surfactant-film-response'


def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preparations():
    rng=np.random.default_rng(94923)
    rows=[]
    for i in range(15):
        rows.append(dict(mode='bands',lo=rng.uniform(-.68,-.02),hi=rng.uniform(.18,.68),
                         q=rng.uniform(0,.5),word='LMH'[i%3]))
    patterns={'bands':['LM','ML','HML','L*H','MMH','LM*H','H*L*L','LLH','MH*M','LM**H**'],
              'locked':['MM','LHM','M*M','HM','M**M','HL*M','LHM*M','LLM','*H*HM','LH**M']}
    groups={'calibration':rows}
    for label, words in patterns.items():
        for rev in [False,True]:
            records=[]
            for i in range(20):
                lo=rng.uniform(-.68,.18); hi=rng.uniform(lo+.15,.68)
                records.append(dict(mode='bands' if label=='bands' else 'lock',lo=lo,hi=hi,
                                    q=rng.uniform(.22,.5) if rev else 0.,word=words[i%len(words)]))
            groups[('reversible_' if rev else '') + (label if label=='bands' or not rev else 'lock')]=records
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    oracle=load(TASK/'solution/model.py')
    ref=load(TASK/'tests/reference.py')
    baseline=load(ROOT/'scripts/surfactant_film_response_baseline.py')
    groups=preparations()
    truth={g:[{'input':e,'value':round(ref.response(e),8)} for e in es] for g,es in groups.items()}
    cal=truth.pop('calibration')
    files={TASK/'tests/data/calibration.json':cal,TASK/'environment/data/calibration.json':cal,
           TASK/'tests/data/probabilities.json':truth}
    for path,value in files.items():
        if args.generate:
            path.write_text(json.dumps(value,indent=2)+'\n')
        assert json.loads(path.read_text())==value
    for name,module in [('oracle',oracle),('shortcut',baseline)]:
        model=module.Model().fit(cal)
        assert abs(model.rate-ref.TRUE_PARAMETER)<1e-6
        for g,rows in [('calibration',cal),*truth.items()]:
            error=max(abs(model.predict([r['input'] for r in rows])-[r['value'] for r in rows]))
            print(name,g,'max_abs',error)
            assert (error<.001) == (name=='oracle' or not g.startswith('reversible'))
    # A single threshold gives independent Bernoulli labels for any q.
    for q in (0., 1e-100, .5):
        experiment = dict(mode='bands', lo=-.2, hi=.2, q=q, word='LLLLLLL')
        low = (1 + experiment['lo']/ref.TRUE_PARAMETER)/2
        assert abs(ref.response(experiment) - low**7) < 1e-13
    # A full partition, scatter order, and the point mass of adjacent equal speeds.
    for q in (0.,.18,.5):
        a,b=.3,.67
        settings=dict(lo=2*a-1,hi=2*b-1,q=q)
        partition=sum(oracle.response(dict(settings,word=''.join(w),mode='bands'),1.)
                      for w in product('LMH', repeat=4))
        assert abs(partition-1)<1e-11
        adjacent=oracle.response(dict(settings,word='MM',mode='lock'),1.)
        mean=(b+a)/2
        mean_square=(a*a+a*b+b*b)/3
        assert abs(adjacent-(1-q)*(mean-mean_square))<1e-11
    # The known two-tag density is constant on the strict decreasing triangle at q=0.
    anchor=dict(lo=2*a-1,hi=2*b-1,q=0.,mode='bands',word='ML')
    assert abs(oracle.response(anchor,1.) - (b-a)*a)<1e-12


if __name__=='__main__':
    main()
