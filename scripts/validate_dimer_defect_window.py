"""Check the positive equilibrium measure, optical parity and the two controls."""
import argparse
import json
from pathlib import Path
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/dimer-defect-window'
ref = runpy.run_path(str(TASK/'tests/reference.py'))
oracle = runpy.run_path(str(TASK/'solution/model.py'))
starter = runpy.run_path(str(ROOT/'scripts/dimer_defect_window_baseline.py'))


def calibration():
    rows = []
    for d in [.41,.53,.72,.85,1.,1.19,1.42,1.75,2.02,2.31]:
        q = dict(w=3,h=2,v=1.1,d=d,vacancies=[[2,0],[0,1]],probe=[[0,0,'D']])
        x = d*ref['TRUE_PARAMETER']
        rows.append(dict(input=q,value=round(x*x/(1+x*x),8)))
    return rows


def design():
    rng = np.random.default_rng(43115)
    groups = {}
    for group in ('single_bond','correlated','vacancy_edges','wide_sheet'):
        rows = []
        for i in range(10):
            if group=='wide_sheet':
                w,h = [(19,20),(24,26),(31,18),(32,32),(20,23)][i%5]
            elif i<4:
                w,h = [(5,6),(6,5),(4,6),(6,6)][i]
            else:
                w,h = [(12,10),(13,8),(9,14)][i%3]
            holes = set()
            # Start from a covering and split selected dimers by sliding their
            # endpoints along alternating paths. Feasibility is constructive.
            vertical = w%2==1
            partner = {}
            for x in range(w):
                for y in range(h):
                    p = (x,y)
                    partner[p] = (x,y^1) if vertical else (x^1,y)
            if group!='single_bond' or i%2:
                location = [(x,y) for x in range(1,w-1) for y in range(1,h-1)]
                for _ in range(1+(i%3)):
                    choices = [p for p in location if p in partner]
                    p = choices[int(rng.integers(len(choices)))]
                    other = partner.pop(p)
                    del partner[other]
                    holes.update([p,other])
                    for _ in range(4):
                        adj = [(p[0]+dx,p[1]+dy) for dx,dy in ((1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1))]
                        choices = [a for a in adj if a in partner and a in location]
                        if not choices:
                            break
                        a = choices[int(rng.integers(len(choices)))]
                        b = partner[a]
                        del partner[b]
                        partner[a] = p;partner[p] = a
                        holes.remove(p);holes.add(b);p = b
            avail = [(x,y,label) for (x,y) in sorted(partner) for label,(dx,dy) in ref['DIRECTIONS'].items()
                     if (x+dx,y+dy) in partner]
            if group=='vacancy_edges':
                dist = lambda e: min(abs(e[0]-x)+abs(e[1]-y) for x,y in holes)
                pool = sorted(avail,key=dist)[:max(8,len(holes)*4)]
            else:
                center = (int(rng.integers(1,w-1)),int(rng.integers(1,h-1)))
                pool = sorted(avail,key=lambda e:abs(e[0]-center[0])+abs(e[1]-center[1]))[:12]
            n = 1 if group=='single_bond' else 2+i%4
            probe = [list(pool[j]) for j in rng.choice(len(pool),n,replace=False)]
            q = dict(w=w,h=h,v=round(float(rng.uniform(.36,2.3)),3),
                     d=round(float(rng.uniform(.36,2.3)),3),
                     vacancies=[list(p) for p in sorted(holes)],probe=probe)
            val = ref['probability'](q)
            if w*h<=36:
                assert abs(ref['enumerate_parity'](q)-val)<2e-11,q
            rows.append(dict(input=q,value=round(val,9)))
        groups[group] = rows
    return groups


def validate(write=False):
    cal,groups = calibration(),design()
    for path,data in [(TASK/'environment/data/calibration.json',cal),
                      (TASK/'tests/data/calibration.json',cal),(TASK/'tests/data/predictions.json',groups)]:
        if write:
            path.write_text(json.dumps(data,indent=1)+'\n')
        assert json.loads(path.read_text())==data
    worked = dict(cal[0]['input'],probe=[[0,0,'D'],[1,0,'D']])
    assert ref['enumerate_parity'](worked)==0 and abs(oracle['odd'](worked,.8))<1e-12
    for role,module in [('oracle',oracle),('shortcut',starter)]:
        model = module['Model']().fit(cal)
        def error(rows):
            return float(max(abs(model.predict([r['input'] for r in rows])-[r['value'] for r in rows])))
        err = {g:error(rows) for g,rows in groups.items()}
        assert error(cal)<2e-5 and abs(model.activity-ref['TRUE_PARAMETER'])<2e-7
        assert (max(err.values())<.003)==(role=='oracle'),err
        print(role,model.activity,err)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    validate(parser.parse_args().generate)
