"""Check optical fringe targets against purified overlaps and number-state evolution."""
import argparse
from itertools import product
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/photon-transfer-fringe'
sys.path.insert(0,str(TASK/'environment'))
from optics import calibration_signal

ref = runpy.run_path(str(TASK/'tests/reference.py'))
oracle = runpy.run_path(str(TASK/'solution/model.py'))['Model']
rays = runpy.run_path(str(ROOT/'scripts/photon_transfer_fringe_baseline.py'))['Model']
starter = runpy.run_path(str(TASK/'environment/model.py'))['Model']
VIS = ref['TRUE_PARAMETER']


def preparations():
    rng = np.random.default_rng(911247)
    def draw(lo, hi):
        return round(float(rng.uniform(lo,hi)),4)
    def mesh(size, offset):
        rows = []
        order = rng.permutation(size).tolist()
        for j in range(2*size-1):
            i,k = order[j%size],order[(j+1)%size]
            rows.append([i,k,draw(.25,1.3),draw(-3.1,3.1)+offset])
        return rows

    calibration = []
    for i in range(12):
        q = dict(source=[[draw(.12,.85),0.,0.],[0.,0.,0.]], prep=[], after=[],
                 before=[[0,1,draw(.2,1.2),draw(-3.1,3.1)]], bank=[0], gate=None,
                 phase=draw(-.8,.8))
        q['lambda'] = draw(-2.8,2.8)
        calibration.append(q)
    groups = {}
    for group in ('mixing','squeezed','even_gate','odd_gate'):
        qs = []
        for i,n in enumerate((2,3,4,5,6,4,3,5)):
            source = [[draw(.12,.78),0. if group=='mixing' else draw(.22,.82),
                       draw(-3.1,3.1)] for _ in range(n)]
            bank = rng.choice(n, 1+(i%(n-1)),replace=False).tolist()
            gate = None
            if group.endswith('gate'):
                gate = dict(rails=rng.choice(n,1+(i%(n-1)),replace=False).tolist(),
                            parity=1 if group=='even_gate' else -1)
            q = dict(source=source, prep=mesh(n,0), before=mesh(n,.01), after=mesh(n,-.01),
                     bank=bank,gate=gate,phase=draw(-2.6,2.6))
            q['lambda'] = draw(.72,3.12) * (-1 if i%3==0 else 1)
            qs.append(q)
        groups[group] = qs
    return calibration,groups


def fock_reference(q, limit):
    # Total photon number commutes with both meters and all passive meshes.
    n = len(q['source'])
    cut = 2*limit+5
    a = np.diag(np.sqrt(np.arange(1,cut)),1)
    local = []
    for mean,r,phase in q['source']:
        pair = .5*r*np.exp(-1j*phase)*(a@a)
        squeeze = expm(pair-pair.conj().T)
        thermal = (mean/(1+mean))**np.arange(cut)/(1+mean)
        local.append((squeeze*thermal) @ squeeze.conj().T)
    numerator = 0j
    denominator = 0.
    mass = 0.
    for total in range(limit+1):
        states = [s for s in product(range(total+1),repeat=n) if sum(s)==total]
        lookup = {s:i for i,s in enumerate(states)}
        numbers = np.array(states)
        rho = np.ones((len(states),len(states)),complex)
        for i,source in enumerate(local):
            occ = numbers[:,i]
            rho *= source[np.ix_(occ,occ)]
        mass += np.trace(rho).real
        def unitary(elements):
            out = np.eye(len(states),dtype=complex)
            for i,j,t,p in elements:
                move = np.zeros_like(out)
                for k,s in enumerate(states):
                    if s[i]:
                        dest = list(s); dest[i]-=1; dest[j]+=1
                        move[lookup[tuple(dest)],k] = np.exp(1j*p)*np.sqrt(s[i]*(s[j]+1))
                out = expm(t*(move-move.conj().T)) @ out
            return out
        prep,before,after = [unitary(q[key]) for key in ('prep','before','after')]
        rho = prep @ rho @ prep.conj().T
        count = np.sum(numbers[:,q['bank']],axis=1).astype(int)
        if q['gate']:
            gate = q['gate']
            keep = (-1)**numbers[:,gate['rails']].sum(axis=1).astype(int) == gate['parity']
            before = keep[:,None]*before
        network = after @ before
        for initial in np.unique(count):
            sector = count==initial
            arm = network[:,sector]
            prob = np.diag(arm @ rho[np.ix_(sector,sector)] @ arm.conj().T).real
            numerator += prob @ np.exp(1j*q['lambda']*(count-initial))
            denominator += prob.sum()
    return (np.exp(-1j*q['phase'])*numerator).real/denominator, denominator, mass


def write_or_check(path, obj, generate):
    if generate:
        path.write_text(json.dumps(obj,indent=1,allow_nan=False)+'\n')
    assert json.loads(path.read_text()) == obj, path


def main(generate):
    calibration,groups = preparations()
    cal = [dict(input=q,value=round(VIS*calibration_signal(q),8)) for q in calibration]
    for side in ('environment','tests'):
        write_or_check(TASK/side/'data/calibration.json',cal,generate)
    models = {name:typ().fit(cal) for name,typ in [('oracle',oracle),('rays',rays),('starter',starter)]}
    assert all(abs(m.visibility-VIS)<2e-8 for m in models.values())
    results = {}
    errors = {name:{} for name in models}
    retention = 1.
    convergence = 0.
    for group,qs in groups.items():
        rows = []
        for i,q in enumerate(qs):
            y, success = ref['reference'](q)
            if i in (3,4):
                finer,_ = ref['reference'](q,points=192)
                convergence=max(convergence,abs(finer-y))
            retention = min(retention,success)
            assert abs(y) <= 1.0000001
            rows.append(dict(input=q,value=round(VIS*y,8)))
        results[group]=rows
        for label,model in models.items():
            errors[label][group] = max(abs(model.predict(qs)-[r['value'] for r in rows]))
    stored = TASK/'tests/data/predictions.json'
    write_or_check(stored,results,generate)
    print('max errors',errors,'minimum retention',retention,'quadrature difference',convergence)
    assert max(errors['oracle'].values()) < 1e-6
    assert all(min(errors[k].values())>.001 for k in ('rays','starter'))
    assert retention >= .05 and convergence < 1e-7
    # A swap moves all photons from the pumped rail. Its vacuum input emits pairs.
    sample = dict(source=[[0,.7,0],[0,0,0]],prep=[],before=[[0,1,np.pi/2,0]],after=[],
                  bank=[0],gate=None,phase=0)
    sample['lambda'] = np.pi
    assert abs(ref['reference'](sample)[0]-1)<1e-12
    assert abs(models['oracle'].reading(sample)-1)<1e-12
    # A three-rail bank has degenerate number eigenspaces in each many-body sector.
    sample.update(source=[[.045,.21,.73],[.073,.14,-.62],[.035,.18,.42]],bank=[0,2],
                  prep=[[0,1,.72,.33],[1,2,.45,-.6]],before=[[0,1,.49,.82],[2,0,.76,.24]],
                  after=[[2,1,.7,-.28],[1,0,.4,.46]],phase=.76)
    sample['lambda'] = 1.82
    for gate in (None,dict(rails=[0,1],parity=1),dict(rails=[1],parity=-1)):
        sample['gate'] = gate
        exact, p = ref['reference'](sample)
        number, prob, mass = fock_reference(sample,10)
        print('number-state check',gate,'missing mass',1-mass,'error',abs(exact-number))
        assert 1-mass < 1.5e-6
        assert abs(exact-number) < 2e-5 and abs(p-prob)<1.5e-6


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    main(parser.parse_args().generate)
