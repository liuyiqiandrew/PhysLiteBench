"""Supplementary analytic limits; no task, calibration data or model run."""
from pathlib import Path
import hashlib
import json
import numpy as np
from check import responses, state


def stable_fixed_number(width, temperature, filling):
    # The weighted variance of two energy values reduces to this expression.
    # This avoids cancellation in the exponentially small low-T response.
    _, _, f = state(width, temperature, filling)
    a, b = f * (1-f)
    return .5 * (width/temperature)**2 * a*b/(a+b)


def run():
    declared=[]
    for w in [.8,1.,1.2]:
        for t in [.1,.2,.4,.6]:
            for n in [.1,.2,.3,.5,.7,.8,.9]:
                physical,source=responses(w,t,n)
                declared.append(abs(physical-stable_fixed_number(w,t,n)))
                assert physical>0 and source>=physical-1e-14
    high=[]
    for n in [.1,.3,.5,.7,.9]:
        for t in [20.,40.,80.]:
            physical,source=responses(1.,t,n)
            high.append({'filling':n,'temperature':t,
                         'physical_coefficient_error':abs(physical*t*t-n*(1-n)/4),
                         'source_limit_error':abs(source-n*(1-n)*np.log(n/(1-n))**2)})
    low=[]
    for n in [.1,.3,.5,.7,.9]:
        vals=[stable_fixed_number(1.,t,n) for t in [.08,.06,.04]]
        assert vals[0]>vals[1]>vals[2]>0
        low.append({'filling':n,'temperatures':[.08,.06,.04],'physical_capacities':vals})
    for n in [.1,.3,.5,.7,.9]:
        rows=[r for r in high if r['filling']==n]
        assert rows[2]['physical_coefficient_error']<rows[1]['physical_coefficient_error']<rows[0]['physical_coefficient_error']
        assert rows[2]['source_limit_error']<rows[1]['source_limit_error']<rows[0]['source_limit_error']
    result={'status':'supplementary_limits_passed','model_evaluations':0,
            'declared_domain_two_level_identity_cases':len(declared),
            'max_two_level_identity_error':max(declared),
            'positive_physical_and_source_not_smaller':True,
            'high_temperature':high,'low_temperature':low,
            'low_temperature_qualification':'Finite positive T first; exponentially small closed heat capacity tends to zero even with partial occupation of a flat band. No low-temperature task grading is proposed.',
            'high_temperature_qualification':'Both formulas are per available one-particle state. Closed C*T^2 tends to n(1-n)W^2/4; the open derivative instead tends to n(1-n)logit(n)^2 away from half filling.',
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(__file__).with_name('limits-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(declared),'max_identity_error':max(declared),'status':result['status']},indent=2))


if __name__=='__main__':run()
