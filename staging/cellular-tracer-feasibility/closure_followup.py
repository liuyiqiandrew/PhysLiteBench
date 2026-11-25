"""Preserved followup: self-consistent Gaussian sampling, not a task harness."""
from pathlib import Path
import json,time
import numpy as np
from check import fourier,source,reference,finite_time


def self_consistent(field,D):
    a=(field['a']**2+field['c']**2/4)/2
    b=(field['b']**2+field['d']**2/4)/2
    delta=(a-b)/D
    z=delta-D
    root=np.sqrt(z*z+4*a)
    yy=2*a/(root+z) if z>0 else (root-z)/2
    xx=D+a/yy
    return np.diag([xx,yy])


def residual(field,D,K):
    a=(field['a']**2+field['c']**2/4)/2
    b=(field['b']**2+field['d']**2/4)/2
    return max(abs(K[0,0]-D-a/K[1,1]),abs(K[1,1]-D-b/K[0,0]))


def main():
    start=time.monotonic();root=Path(__file__).parent
    old=json.loads((root/'report.json').read_text());original=[]
    for r in old['field_cases']:
        field=r['field'];D=r['diffusivity'];K=np.array(r['physical']);S=self_consistent(field,D)
        original.append({'field':field,'D':D,'physical':K.tolist(),'self_consistent':S.tolist(),'oneway_source':r['source'],'signed_relative_difference_per_axis':((np.diag(S)-np.diag(K))/np.diag(K)).tolist(),'residual':float(residual(field,D,S))})
    sweep=[]
    for D in np.linspace(.8,1.2,21):
        for amp in [4.,5.,6.,8.]:
            for harmonics in [False,True]:
                f=dict(a=amp,b=amp,c=.6 if harmonics else 0.,d=-.4 if harmonics else 0.)
                K,_=fourier(f,D);S=self_consistent(f,D)
                sweep.append({'field':f,'D':float(D),'physical':np.diag(K).tolist(),'source':np.diag(S).tolist(),'relative_difference':((np.diag(S)-np.diag(K))/np.diag(K)).tolist(),'residual':float(residual(f,D,S))})
    anchors=[]
    for D in [.8,1.,1.2]:
        for amp in [0.,.6,1.2,2.,3.]:
            for axis in [0,1]:
                f=dict(a=amp if axis==0 else 0.,b=amp if axis==1 else 0.,c=.25*amp if axis==0 else 0.,d=.25*amp if axis==1 else 0.)
                S=self_consistent(f,D);K,_=fourier(f,D,10)
                anchors.append({'field':f,'D':D,'error':float(np.max(abs(S-K))),'residual':float(residual(f,D,S))})
    summary={'original_cases':len(original),'full_D_sweep_cases':len(sweep),'calibration_and_zero_cases':len(anchors),'max_fixed_point_residual':max([r['residual'] for r in original+sweep+anchors]),'max_calibration_error':max(r['error'] for r in anchors),'minimum_strong_A5_to8_relative_gap':min(min(r['relative_difference']) for r in sweep if r['field']['a']>=5),'minimum_A4_relative_difference':min(min(r['relative_difference']) for r in sweep if r['field']['a']==4),'minimum_strong_physical':min(min(r['physical']) for r in sweep if r['field']['a']>=5),'maximum_strong_physical':max(max(r['physical']) for r in sweep if r['field']['a']>=5),'minimum_strong_source':min(min(r['source']) for r in sweep if r['field']['a']>=5),'maximum_strong_source':max(max(r['source']) for r in sweep if r['field']['a']>=5),'run_seconds':time.monotonic()-start}
    output={'status':'bounded_source_refinement','model_runs':0,'prior_source_and_report_unchanged':True,'closure':'Auxiliary phase diffuses with diag(Kx,Ky); its complete velocity-correlation integral plus independent bare displacement noise yields Kx=D+a/Ky, Ky=D+b/Kx. Use the unique positive solution.','original_cases_including_weak_and_sign_reversed_errors':original,'full_D_sweep':sweep,'calibration_and_zero_cases':anchors,'summary':summary,'limitations':'No new independent continuum reference was run because the physical model did not change. Existing reference evidence covers 12 cases and refinements, not every followup sweep point. No data, fitting or grader exists; no measured agent difficulty.'}
    (root/'closure-followup-report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    assert summary['max_fixed_point_residual']<1e-12
    assert summary['max_calibration_error']<1e-12
    assert summary['minimum_strong_A5_to8_relative_gap']>.08

if __name__=='__main__':main()
