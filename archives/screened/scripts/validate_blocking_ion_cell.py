"""Validate blocking-ion/capacitor constraints, independent fields, and calibration noise."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK='blocking-ion-cell'


def module(path):
    spec=importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def metrics(model,records,true):
    residual=(model.predict([r['input'] for r in records])-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(parameter=model.diffusivity,parameter_relative_error=abs(model.diffusivity/true-1),calibration_chi2=float(residual@residual)/(len(records)-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'jobs'/(TASK+'-validation')/'summary.json')
    args=parser.parse_args();task=ROOT/'tasks'/TASK
    reference=module(task/'tests/reference.py');om=module(task/'solution/model.py');bm=module(ROOT/'scripts/blocking_ion_cell_baseline.py')
    oracle,shortcut=om.Model,bm.Model
    meta=json.loads((task/'tests/metadata.json').read_text());true=meta['true_parameter']
    inputs=reference.calibration_inputs()*2;noiseless=reference.predict(inputs,true)
    sigma=meta['measurement_sigma_fraction']*max(abs(noiseless))
    if args.generate:
        y=noiseless+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        data=json.dumps([dict(input=e,value=float(v),sigma=float(sigma)) for e,v in zip(inputs,y)],indent=2)+'\n'
        for folder in ['environment','tests']:(task/folder/'data/calibration.json').write_text(data)
    public,private=task/'environment/data/calibration.json',task/'tests/data/calibration.json'
    assert public.read_bytes()==private.read_bytes()
    records=json.loads(public.read_text());assert [r['input'] for r in records]==inputs
    hidden=reference.hidden_inputs();truths={k:reference.predict(v,true) for k,v in hidden.items()}
    def scores(model):return {k:float(np.linalg.norm(model.predict(hidden[k])-y)/np.linalg.norm(y)) for k,y in truths.items()}
    report=dict(revision=1,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,measurement_sigma=float(sigma),controls={})
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        exact=cls();exact.diffusivity=true;assert max(abs(exact.predict(inputs)-noiseless))<1e-14
        model=cls().fit(records);result=metrics(model,records,true);result['hidden']=scores(model)
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert (max(result['hidden'].values())<meta['prediction_limit'] if name=='oracle' else min(result['hidden'].values())>meta['prediction_limit'])
        report['controls'][name]=result
    exact=oracle();exact.diffusivity=true;agreement=[];refinement=[];oracle_refinement=[]
    for name,es in hidden.items():
        fine=reference.predict(es,true,cells=320)
        refinement.append(float(np.linalg.norm(fine-truths[name])/np.linalg.norm(fine)))
        coarse=exact.predict(es)
        agreement.append(float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine)))
        om.CELLS=192;om.trajectory.cache_clear();doubled=exact.predict(es)
        oracle_refinement.append(float(np.linalg.norm(coarse-doubled)/np.linalg.norm(doubled)))
        om.CELLS=96;om.trajectory.cache_clear()
    assert max(refinement)<.001 and max(agreement)<.003 and max(oracle_refinement)<.003
    mass_error=[];circuit_error=[];poisson_error=[];positive_min=[];energy_increase=[]
    for source in [om,bm]:
        for es in hidden.values():
            e=es[0];sol=source.trajectory(e['first'],e['second'],e['voltage'],e['series_capacitance'])
            free_energy=[]
            for time in np.r_[0.,np.geomspace(1e-7,.01,80)]:
                c=sol(true*time/source.LENGTH**2).reshape(2,source.CELLS)
                mass_error.append(float(max(abs(c.mean(axis=1)-1))))
                positive_min.append(float(c.min()))
                q,field=source.electric_state(c,e['voltage'],e['series_capacitance'])
                x=(np.arange(source.CELLS)+.5)/source.CELLS;density=c[0]-c[1]
                polarization=(1-x)@density/source.CELLS
                ratio=source.PERMITTIVITY/(e['series_capacitance']*source.LENGTH)
                v=e['voltage']/source.THERMAL_VOLTAGE
                gap=source.COUPLING*(q+polarization)
                circuit_error.append(abs(gap+source.COUPLING*ratio*q-v))
                if source is om:
                    edges=source.COUPLING*(q+np.r_[0.,np.cumsum(density)/source.CELLS])
                    poisson_error.append(float(max(abs(np.diff(edges)*source.CELLS-source.COUPLING*density))))
                    field_energy=np.sum(edges[:-1]**2+edges[:-1]*edges[1:]+edges[1:]**2)/(6*source.COUPLING*source.CELLS)
                    free_energy.append(float(np.sum(c*np.log(c))/source.CELLS+field_energy+.5*source.COUPLING*ratio*q*q-v*q))
            if free_energy:energy_increase.append(float(max(np.diff(free_energy))))
    assert max(mass_error)<1e-10 and max(circuit_error)<1e-12 and max(poisson_error)<1e-10 and min(positive_min)>0
    assert max(energy_increase)<1e-6
    report['physical_checks']=dict(reference_refinement_relative_max=max(refinement),oracle_reference_relative_max=max(agreement),oracle_refinement_relative_max=max(oracle_refinement),both_controls_species_mass_error=max(mass_error),both_controls_circuit_voltage_error=max(circuit_error),oracle_gauss_law_residual=max(poisson_error),minimum_dimensionless_concentration=min(positive_min),maximum_fixed_voltage_free_energy_increase=max(energy_increase))
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi2=[];errors=[]
    for i in range(args.noise_trials):
        y=noiseless+rng.normal(0,sigma,len(inputs));sample=[dict(input=e,value=float(v),sigma=float(sigma)) for e,v in zip(inputs,y)]
        model=oracle().fit(sample);m=metrics(model,sample,true)
        parameters.append(model.diffusivity);chi2.append(m['calibration_chi2']);errors.append(m['parameter_relative_error'])
        if i in [0,args.noise_trials-1]:assert abs(shortcut().fit(sample).diffusivity-model.diffusivity)<1e-17
    sensitivity={}
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        values=[]
        for p in [min(parameters),max(parameters)]:
            model=cls();model.diffusivity=p;values.extend(scores(model).values())
        sensitivity[name]=dict(min=min(values),max=max(values))
    assert max(errors)<.03 and np.mean(np.array(chi2)<1.5)>=.99
    assert sensitivity['oracle']['max']<meta['prediction_limit'] and sensitivity['shortcut']['min']>meta['prediction_limit']
    report['noise']=dict(parameter_min=min(parameters),parameter_max=max(parameters),parameter_relative_error_max=max(errors),calibration_chi2_max=max(chi2),calibration_pass_fraction=float(np.mean(np.array(chi2)<1.5)),hidden_at_parameter_extrema=sensitivity)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(TASK,'PASS',args.output)


if __name__=='__main__':main()
