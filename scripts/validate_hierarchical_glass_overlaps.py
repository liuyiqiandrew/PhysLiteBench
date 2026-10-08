"""Validate the shipped data and both models; this command never regenerates data."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,warnings
import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(task,baseline,output):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((task/'tests/metadata.json').read_text())
    paths=[task/'environment/data/calibration.json',task/'tests/data/calibration.json']
    for path in paths:assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['calibration_sha256']
    truthpath=task/'tests/data/prediction_truth.json'
    assert hashlib.sha256(truthpath.read_bytes()).hexdigest()==metadata['truth_sha256']
    data=json.loads(paths[0].read_text());truth=json.loads(truthpath.read_text());queries=[r['input'] for r in truth];target=np.array([r['value'] for r in truth]);results={}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');ref=load('reference',task/'tests/reference.py')
        expected,evidence=ref.predict(queries,metadata['true_width'],return_evidence=True)
        assert np.max(np.abs(target-expected))<2e-9
        for name,path in [('oracle',task/'solution/model.py'),('source',baseline)]:
            module=load(name,path);model=module.Model();assert model.fit(data) is model
            pred=model.predict(queries);cal=model.predict_calibration(data)
            groups={g:float(np.linalg.norm((pred-target)[[i for i,r in enumerate(truth) if g in r['groups']]])/np.linalg.norm(target[[i for i,r in enumerate(truth) if g in r['groups']]])) for g in sorted({x for r in truth for x in r['groups']})}
            errors=(cal-np.array([r['value'] for r in data]))/np.array([r['sigma'] for r in data]);chi=float(np.mean(errors**2));assert chi<1.5 and abs(model.width/metadata['true_width']-1)<.03
            if name=='oracle':assert max(groups.values())<.04
            else:assert groups['state_pairs']>.04 and max(v for g,v in groups.items() if g.startswith('joint_'))>.04 and groups['parent_collisions']<.04
            results[name]={'width':model.width,'groups':groups,'chi2_per_record':chi,'predictions':pred.tolist(),'calibration_predictions':cal.tolist()}
        assert not caught,[str(w.message) for w in caught]
    report={'status':'passed','actual_fit_count':2,'reference_values':expected,'reference_evidence':evidence,'models':results,'warnings':[],'seconds':time.perf_counter()-started}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'actual_fit_count':2,'models':{k:{a:b for a,b in v.items() if a not in ['predictions','calibration_predictions']} for k,v in results.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.task,args.baseline,args.output)
