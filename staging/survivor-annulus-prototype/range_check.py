"""Diffusivity/reversal checks of the largest subthreshold annular screen gap."""
from pathlib import Path
import datetime
import hashlib
import json
from localized_check import localized


def main():
    rows=[]
    for d in [.7,1.,1.4]:
        for rotation in [-16.,16.]:
            rows.append({'diffusivity':d,'rotation':rotation,
                         **localized(d,.6,rotation,200.,4.,False,48,128,True)})
    report={'status':'rejected_for_insufficient_separation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'grid':[48,128],'rows':rows,'maximum_fractional_error':max(x['fractional_error'] for x in rows),
            'required_gate':.04,'minimum_drift':min(x['physical_drift'] for x in rows),
            'maximum_tilted_relative_error':max(x['tilted_relative_error'] for x in rows),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p=Path(__file__).parent/'range-report.json';p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
