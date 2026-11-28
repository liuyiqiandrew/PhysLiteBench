"""Check optical fringe targets against purified overlaps and number-state evolution."""
import argparse
from itertools import product
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/photon-transfer-fringe'
sys.path[:0] = [str(TASK/'solution'),str(TASK/'environment'),str(TASK/'tests')]
from optics import calibration_signal

ref = runpy.run_path(str(TASK/'tests/reference.py'))
oracle = runpy.run_path(str(TASK/'solution/model.py'))['Model']
shortcut = runpy.run_path(str(ROOT/'scripts/photon_transfer_fringe_baseline.py'))['Model']
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
    for j,family in enumerate(('ordered_clicks','erased_order','subtraction_port')):
        qs = []
        for i,n in enumerate((2,3,4,5,6,3)):
            source = [[draw(.12,.78),draw(.12,.82),draw(-3.1,3.1)] for _ in range(n)]
            bank = rng.choice(n,1+i%(n-1),replace=False).tolist()
            a,b = rng.choice(n,2,replace=False).tolist()
            # Overlapping optical modes resolve the commutator in the subtraction port.
            c = b if j==2 else int(rng.choice([k for k in range(n) if k!=a]))
            op_a = [a,b,draw(.15,.9),draw(-2.8,2.8),draw(.15,1.48),draw(-3,3)]
            op_b = [a,c,draw(.15,.9),draw(-2.8,2.8),draw(.15,1.48),draw(-3,3)]
            angle = [draw(-.4,.4),draw(.8,2.3),draw(2.42,3.14)][j]
            switch = dict(clicks=[op_a,op_b],middle=[] if j==2 else mesh(n,.02),angle=angle)
            q = dict(source=source, prep=mesh(n,0), before=mesh(n,.01), after=mesh(n,-.01),
                     bank=bank,gate=None,switch=switch,phase=draw(-2.5,2.5))
            q['lambda'] = draw(.75,3.13)*(-1 if i%2 else 1)
            qs.append(q)
        groups[family] = qs
    for name,seed in [('finite_pulses','ordered_clicks'),('finite_eraser','subtraction_port')]:
        rows = []
        for i,original in enumerate(groups[seed]):
            q = dict(original)
            q['pulse'] = [draw(.57,1.07),draw(.65,1.17)]
            rows.append(q)
        groups[name] = rows
    # The pump number and analyzer settings describe the fabricated sources.
    designs = json.loads((TASK/'tests/data/predictions.json').read_text())
    for label in ('pump_number','pump_erasure'):
        groups[label] = [row['input'] for row in designs[label]]
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


def trigger_fock(occupations, setting, duration, limit):
    """Project the original system plus trigger unitary between |0> and |1>."""
    n = len(occupations[0])
    basis = [tuple(row)+(trigger,) for row in occupations for trigger in range(limit+1-sum(row))]
    index = {occ:i for i,occ in enumerate(basis)}
    def annihilator(mode):
        rows,cols,values = [],[],[]
        for j,occ in enumerate(basis):
            if occ[mode]:
                lowered=list(occ)
                lowered[mode]-=1
                rows.append(index[tuple(lowered)])
                cols.append(j)
                values.append(np.sqrt(occ[mode]))
        return csr_matrix((values,(rows,cols)),shape=(len(basis),len(basis)))
    i,j,t,p,b,f=setting
    mode=np.cos(t)*annihilator(i)+np.exp(1j*p)*np.sin(t)*annihilator(j)
    L=np.cos(b)*mode+np.exp(1j*f)*np.sin(b)*mode.conj().T
    half=annihilator(n).conj().T@L
    initial=np.zeros((len(basis),len(occupations)))
    initial[[index[tuple(row)+(0,)] for row in occupations],np.arange(len(occupations))]=1
    output=expm_multiply(-1j*duration*(half+half.conj().T),initial,traceA=0)
    projected=np.zeros((len(occupations),len(occupations)),complex)
    for i,occ in enumerate(occupations):
        if sum(occ)<limit:
            projected[i]=output[index[tuple(occ)+(1,)]]
    return projected


def switch_fock(q, limit):
    # Full density matrix across different total photon numbers. The weak
    # herald can connect them by adding or subtracting a photon.
    dims = len(q['source'])
    states = [v for v in product(range(limit+1),repeat=dims) if sum(v)<=limit]
    index = {v:i for i,v in enumerate(states)}
    occupations = np.array(states)
    rho = np.ones((len(states),len(states)),complex)
    lower = []
    for mode,(mean,r,phase) in enumerate(q['source']):
        a = np.diag(np.sqrt(np.arange(1,2*limit+4)),1)
        g = .5*r*np.exp(-1j*phase)*(a@a)
        squeeze = expm(g-g.conj().T)
        pop = (mean/(1+mean))**np.arange(len(a))/(1+mean)
        density = (squeeze*pop) @ squeeze.conj().T
        occ = occupations[:,mode]
        rho *= density[np.ix_(occ,occ)]
        a = np.zeros_like(rho)
        for i,v in enumerate(states):
            if not v[mode]:continue
            new = list(v)
            new[mode] -= 1
            a[index[tuple(new)],i] = np.sqrt(v[mode])
        lower.append(a)
    def passive(elements):
        out = np.eye(len(states),dtype=complex)
        for i,j,t,p in elements:
            forward = np.exp(1j*p)*lower[j].conj().T @ lower[i]
            out = expm(t*(forward-forward.conj().T)) @ out
        return out
    prep,before,after = [passive(q[k]) for k in ('prep','before','after')]
    rho = prep @ rho @ prep.conj().T
    count = occupations[:,q['bank']].sum(axis=1)
    ctrl = q['switch']
    if 'pump' in q:
        source=q['pump']
        dim=1+max(x[0] for x in source['state']+(source['detect'] or []))
        def ket(terms):
            psi=np.zeros(dim,complex)
            for k,x,y in terms:psi[k]=x+1j*y
            return psi/np.linalg.norm(psi)
        psi=ket(source['state'])
        out=ket(source['detect']) if source['detect'] is not None else None
        pump=np.diag(np.sqrt(np.arange(1,dim)),1)
        between=np.kron(np.diag(np.exp(-.5j*source['kerr']*np.arange(dim)*(np.arange(dim)-1))),passive(ctrl['middle']))
        ops=[]
        for i,j,t,p,b,f in ctrl['clicks']:
            mode=np.cos(t)*lower[i]+np.exp(1j*p)*np.sin(t)*lower[j]
            ops.append(np.cos(b)*np.kron(np.eye(dim),mode)+np.exp(1j*f)*np.sin(b)*np.kron(pump,mode.conj().T))
        initial=np.kron(psi[:,None],before)
        trigger=(ops[1]@between@(ops[0]@initial) + np.exp(1j*ctrl['angle'])*ops[0]@between@(ops[1]@initial))/2
        optical=after@trigger.reshape(dim,len(states),len(states))
        if out is not None:optical=np.array([np.tensordot(out.conj(),optical,axes=1)])
        rate=moment=0.
        for first in np.unique(count):
            sel=count==first
            for transfer in optical:
                block=transfer[:,sel]
                probabilities=np.diag(block@rho[np.ix_(sel,sel)]@block.conj().T).real
                moment+=probabilities@np.cos(q['lambda']*(count-first)-q['phase'])
                rate+=probabilities.sum()
        return moment/rate,rate
    if 'pulse' in q:
        ops = [trigger_fock(occupations,settings,eps,limit)
               for settings,eps in zip(ctrl['clicks'],q['pulse'])]
    else:
        ops = []
        for i,j,t,p,b,f in ctrl['clicks']:
            op = np.cos(t)*lower[i] + np.exp(1j*p)*np.sin(t)*lower[j]
            ops.append(np.cos(b)*op + np.exp(1j*f)*np.sin(b)*op.conj().T)
    middle = passive(ctrl['middle'])
    first = ops[1] @ middle @ ops[0]
    second = ops[0] @ middle @ ops[1]
    transfer = after @ ((first+np.exp(1j*ctrl['angle'])*second)/2) @ before
    value = 0.
    survival = 0.
    for initial in np.unique(count):
        keep = count==initial
        arm = transfer[:,keep]
        weight = np.diag(arm @ rho[np.ix_(keep,keep)] @ arm.conj().T).real
        value += weight @ np.cos(q['lambda']*(count-initial)-q['phase'])
        survival += weight.sum()
    return value/survival,survival


def write_or_check(path, obj, generate):
    if generate:
        path.write_text(json.dumps(obj,indent=1,allow_nan=False)+'\n')
    assert json.loads(path.read_text()) == obj, path


def main(generate):
    calibration,groups = preparations()
    cal = [dict(input=q,value=round(VIS*calibration_signal(q),8)) for q in calibration]
    for side in ('environment','tests'):
        write_or_check(TASK/side/'data/calibration.json',cal,generate)
    models = {name:typ().fit(cal) for name,typ in [('oracle',oracle),('shortcut',shortcut),('starter',starter)]}
    assert all(abs(m.visibility-VIS)<2e-8 for m in models.values())
    results = {}
    errors = {name:{} for name in models}
    retention = 1.
    convergence = 0.
    for group,qs in groups.items():
        rows = []
        for i,q in enumerate(qs):
            y, success = ref['reference'](q)
            if i in (3,4) and 'pulse' not in q:
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
    new = ('pump_number','pump_erasure')
    assert max(v for k,v in errors['starter'].items() if k not in new)<1e-6
    assert min(errors['starter'][g] for g in new)>.001
    assert max(v for k,v in errors['shortcut'].items() if k not in new)<1e-6
    assert min(errors['shortcut'][k] for k in new)>.001
    assert retention >= .0005 and convergence < 1e-7
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

    switched = groups['erased_order'][0].copy()
    switched['source'] = [[.08,.18,.7],[.1,.22,-.6]]
    target, rate = ref['reference'](switched)
    for cut in (12,18):
        count, probability = switch_fock(switched,cut)
        print('switched number-state check',cut,abs(count-target),abs(probability-rate))
        assert abs(count-target)<(2e-6 if cut==12 else 1e-7)
        assert abs(probability-rate)<(2e-5 if cut==12 else 1e-7)
    commutator = dict(source=[[.5,0,0],[0,0,0]],prep=[],before=[],
                      after=[[0,1,np.pi/2,0]],bank=[0],gate=None,phase=0,
                      switch=dict(clicks=[[0,1,0,0,np.pi/2,0],[0,1,0,0,0,0]],
                                  middle=[],angle=np.pi))
    commutator['lambda'] = np.pi
    assert abs(ref['reference'](commutator)[0]-.5)<1e-12
    assert abs(models['oracle'].reading(commutator)-.5)<1e-12

    for m,answer in [(1,.5),(2,-1/28)]:
        quantum = dict(commutator,pump=dict(state=[[m,1.,0]],detect=None,kerr=np.pi))
        assert abs(ref['reference'](quantum)[0]-answer)<1e-11
        assert abs(models['oracle'].reading(quantum)-answer)<1e-11
    probe = dict(groups['pump_erasure'][0],source=[[.084,.155,.7],[.11,.21,-.2]])
    exact=ref['reference'](probe)
    for limit in (12,20):
        number=switch_fock(probe,limit)
        assert max(abs(np.asarray(number)-exact)) < (2e-6 if limit==12 else 2e-8)
    commutator['pulse']=[.5,.65]
    finite,p=ref['reference'](commutator)
    assert abs(finite-.8922324303158284)<2e-7 and abs(p-.010848247529066407)<2e-9
    assert abs(models['oracle'].reading(commutator)-.8922324303158284)<1e-12
    for name in ('finite_pulses','finite_eraser'):
        probe = dict(groups[name][0],source=[[.08,.18,.7],[.1,.22,-.6]],pulse=[.57,.69])
        target,rate=ref['reference'](probe)
        for cut in (14,18):
            number,p=switch_fock(probe,cut)
            print('finite number-state check',name,cut,abs(number-target),abs(p-rate))
            assert abs(number-target)<(1e-4 if cut==14 else 6e-6) and abs(p-rate)<1e-6


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    main(parser.parse_args().generate)
