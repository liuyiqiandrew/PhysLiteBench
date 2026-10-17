from pathlib import Path
import collections, datetime, hashlib, json, re, tomllib, zipfile
import xml.etree.ElementTree as E
from PIL import Image, ImageChops

W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench')
R=Path('/Users/yp9846/Desktop/PhysLiteBench')
B=W/'.presentation-trajectory-20261015'
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-plain-language.pptx'
final=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-trajectory-visual-v3.pptx'
dest=R/'output/presentation-20261014/PhysLiteBench-10-minute-trajectory-visual.pptx'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def root(z,p): return E.fromstring(z.read(p))
def text(r): return [x.text or '' for x in r.findall('.//a:t',ns)]
fonts=collections.Counter(); timing=[]; comparisons=[]
with zipfile.ZipFile(source) as old, zipfile.ZipFile(final) as new:
    assert text(root(new,'ppt/slides/slide1.xml'))==['PhysLiteBench','Yue Pan and Andrew Liu']
    assert 'Pending completion' in text(root(new,'ppt/slides/slide7.xml'))
    for i in range(1,17):
        sp=f'ppt/slides/slide{i}.xml'; np=f'ppt/notesSlides/notesSlide{i}.xml'
        if i!=10:
            assert text(root(old,sp))==text(root(new,sp)), ('changed text',i)
            assert text(root(old,np))==text(root(new,np)), ('changed notes',i)
        nt=' '.join(text(root(new,np)))
        m=re.search(r'Target time: (\d+) seconds',nt)
        timing.append(int(m.group(1)) if m else 0)
        a=Image.open(W/f'.presentation-plain-20261015/final-previews/slide-{i:02}.png').convert('RGB')
        b=Image.open(B/f'final-previews/slide-{i:02}.png').convert('RGB')
        box=ImageChops.difference(a,b).getbbox()
        if i!=10: assert box is None, ('unexpected render change',i)
        comparisons.append({'slide':i,'pixel_identical':box is None,'diff_box':box})
    records=json.loads((B/'workbook-preservation.json').read_text())['records']
    assert len(records)==4
    for x in records: assert hashlib.sha256(new.read(x['workbook'])).hexdigest()==x['sha256']
    for p in new.namelist():
        if p.startswith('ppt/') and p.endswith('.xml'):
            for x in root(new,p).iter():
                f=x.get('typeface')
                if f and not f.startswith('+'): fonts[f]+=1
    assert set(fonts)=={'Times New Roman'}
    assert sum(timing[:11])==600 and timing[11]==180
    main=' '.join(text(root(new,'ppt/slides/slide10.xml')))
    for forbidden in ['stationary increments','quenched','covariance','deterministic preparation','Poisson']:
        assert forbidden.lower() not in main.lower()
    assert '136%' in main and '4%' in main
diag=next(x for x in json.loads((R/'results/trajectory-analysis-20261014/diagnostic-repairs.json').read_text())['cases'] if x['task']=='single-file-memory')
assert round(100*max(diag['before'].values()))==136
assert f"{100*max(diag['after'].values()):.3f}"=='0.003'
audit={'source':str(source),'source_sha256':sha(source),'final':str(final),'final_sha256':sha(final),'changed_slides':[10],'source_text_and_notes_preserved_on_other_slides':True,'render_comparison':comparisons,'all_four_source_workbooks_byte_identical':True,'font_policy':'Times New Roman','font_counts':dict(fonts),'timing_seconds':timing,'hint_results_pending':True,'claims_checked':{'largest_error_percent_rounded':136,'allowed_error_percent':4,'diagnostic_correction_error_percent_rounded':0.003,'saved_agent_reward_unchanged':0},'diagram':'Editable schematic. Identical tube size and bead count in all three snapshots. Orange marks the same bead. Red arrow denotes the statistical connection across repeats, not a force. It overestimates the connection; it does not assume independent positions.','native_powerpoint_tested':False}
(B/'content-audit.json').write_text(json.dumps(audit,indent=2)+'\n')

cp=R/'results/presentation-20261014/update-on-completion.json'
c=json.loads(cp.read_text()); (B/'update-on-completion-before-trajectory.json').write_text(cp.read_text())
assert c['current_deck']==str(source) and c['current_deck_sha256']==sha(source)
c.update(current_deck=str(dest),current_deck_sha256=sha(final),prior_deck=str(source),prior_deck_sha256=sha(source),authoring_script=str(B/'build.mjs'),authoring_backup=str(R/'staging/presentation-20261014/trajectory-visual'),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_filename_suggestion='PhysLiteBench-10-minute-trajectory-visual-final.pptx')
c['user_request']+=' Latest edit: simplify slide 10 to a plain diagram of the bead setup, what the fitted model captures, and what it misses. Keep the approved setup on slide 3 unchanged.'
c['narrative']=c['narrative'].replace('plain saved-bead-failure flowchart 10','plain saved-bead-failure diagram 10')
c['narrative']+=' Slide 10 shows the same starting row and the same bead at two times. The fit matches the amount of motion, but the retained starter rule overstates the link between the two observations. The 136% error remains on the slide. Detailed actions and the diagnostic correction stay in the notes and backup.'
c['final_update_requirements'][0]=c['final_update_requirements'][0].replace('CURRENT plain-language deck','CURRENT trajectory-visual deck')
c['final_update_requirements'][5]+=' Keep slide 10 visually simple: the same bead diagram, the correct amount of motion, and the overly strong link between early and later positions. Do not restore the dense four-step action trace or the diagnostic numbers to the main slide.'
(B/'update-on-completion.json').write_text(json.dumps(c,indent=2)+'\n')

a=tomllib.loads(Path('/Users/yp9846/.codex/automations/physics-hint-comparison-progress/automation.toml').read_text())
prompt=a['prompt'].replace('output/presentation-20261014/PhysLiteBench-10-minute-plain-language.pptx','output/presentation-20261014/PhysLiteBench-10-minute-trajectory-visual.pptx').replace('bead failure flowchart 10','bead failure diagram 10')
marker=' Use update-on-completion.json for the current source hash and timing.'
assert marker in prompt
prompt=prompt.replace(marker,' Preserve the latest plain slide 10: it uses the same bead diagram to show that the agent fits the amount of motion but keeps a rule making early and later positions too strongly linked. It misses how the evenly spaced starting row changes that link. Keep the detailed action trace and diagnostic correction in notes and backup, and leave the approved setup slide 3 unchanged.'+marker)
args=dict(mode='update',id=a['id'],kind=a['kind'],name=a['name'],prompt=prompt,rrule=a['rrule'],status=a['status'],targetThreadId=a['target_thread_id'])
if 'notification_policy' in a: args['notificationPolicy']=a['notification_policy']
(B/'automation-update.json').write_text(json.dumps(args,indent=2)+'\n')
print(json.dumps({'final':str(final),'desktop_destination':str(dest),'sha256':sha(final),'only_slide_10_changed':True,'other_15_slides_pixel_identical':True,'main_seconds':600},indent=2))
