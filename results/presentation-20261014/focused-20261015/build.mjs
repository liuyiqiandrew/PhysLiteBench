import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench',B=W+'/.presentation-focused-20261015';
const R='/Users/yp9846/Desktop/PhysLiteBench';
const current=R+'/output/presentation-20261014/PhysLiteBench-10-minute-unified-study.pptx';
const U=W+'/output/PhysLiteBench-focused-study-figures-20261015',UR=R+'/output/presentation-20261014/PhysLiteBench-focused-study-figures-20261015';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
if(hash(await fs.readFile(current))!==hash(await fs.readFile(B+'/source.pptx')))throw Error('New user save; inspect before editing');
const P=await PresentationFile.importPptx(await FileBlob.load(B+'/source.pptx'));
const changed=[],moved=[];
async function layout(n){return JSON.parse(await(await P.slides.items[n-1].export({format:'layout',detail:'full'})).text());}
async function find(n,start){const e=(await layout(n)).elements.find(e=>e.text?.startsWith(start));if(!e)throw Error('Missing '+n+' '+start);return e;}
async function replace(n,start,value,style){const e=await find(n,start);const q=P.resolve(e.id);q.text=value;if(style)q.text.style={typeface:'Times New Roman',...style};changed.push({slide:n,old:e.text,new:value});return q;}
async function toNotes(n,start){const e=await find(n,start);P.slides.items[n-1].speakerNotes.text+='\n\n'+e.text;P.delete(e.id);moved.push({slide:n,text:e.text});}
// Focus the introduction on reasoning beyond the supplied observations and the hint test.
await replace(2,'Can a coding agent','Can agents identify the physics behind a scenario and predict beyond the observed data?');
await replace(2,'physics tasks that test','physics tasks require predictions beyond the supplied observations.\nGeneral and physics hints test what guidance helps agents succeed.');
await toNotes(2,'Tasks were selected');
P.slides.items[1].speakerNotes.text='Target time: 45 seconds.\n\nOur key question is whether an agent can identify the physics behind a scenario and predict beyond what it observes. A good fit to the supplied data is not enough. We use thirteen tasks with hidden predictions, then test whether general checking instructions and task-specific physics guidance improve performance. The hints probe what the agent misses without guidance.\n\nThe suite was selected using GPT-5.6 Luna. The original ten tasks had zero of three Luna High passes during initial selection. All thirteen are implemented with Harbor. Selection outcomes differ from independent evaluation. Historical controls, runtime differences and unmatched hint length limit causal interpretation. General-hint gains alone cannot identify a specifically physical reasoning defect. The task-specific hints and reviewed failures provide further evidence.\n\nSources:\n'+R+'/results/task-roster.json\n'+UR+'/sources/hint-matrix-summary.md\n'+UR+'/summary.json';
// Keep the approved example and the user's new "correlated" wording. Move asides to notes.
await toNotes(3,'One run shown.');await toNotes(3,'Illustrative setup.');
await toNotes(4,'No-hint instruction shortened.');
await replace(5,'2,068 attempts','2,068 attempts across 48 model, effort and hint settings.');
await replace(5,'Harbor 0.21.0','Harbor with Codex agents. Time limits: 600 s for solving and 60 s for grading.',{fontSize:24,color:'#444444',wrap:'square',insets:0});
// Essential chart interpretation stays in the main subtitle. Detailed accounting is in notes.
await replace(6,'All model versions','19 settings with complete costs. The dashed line marks the observed Pareto frontier.');
await toNotes(6,'19 of 24 settings');
await replace(7,'Three prompt conditions','22 settings with complete costs, compared across the three prompt conditions.');
await toNotes(7,'22 of 36 settings');
// Slide 8 uses the same single 39-attempt panel for each of six effort settings.
const s8=P.slides.items[7],pic=(await layout(8)).elements.find(e=>e.kind==='image');if(!pic)throw Error('Slide 8 image not found');
P.delete(pic.id);s8.images.add({blob:await fs.readFile(U+'/figures/03-luna-effort-hints.png'),contentType:'image/png',alt:'GPT-5.6 Luna, one 39-attempt panel per effort and condition. High: 4/39 no hint, 18/39 general hint, 30/39 physics hint. Reproducible Python figure.',position:pic.position,fit:'contain'});
await replace(8,'GPT-5.6 Luna.','GPT-5.6 Luna. 13 tasks × 3 attempts for each effort and hint condition.');
await replace(8,'Physics hints have','General hints help at higher efforts. Physics hints help at every effort.');
s8.speakerNotes.text='Target time: 55 seconds.\n\nThe model is GPT-5.6 Luna throughout. Every bar uses thirteen tasks with three attempts each, giving thirty-nine attempts. At High effort, the completed passes are four without a hint, eighteen with a general hint, and thirty with a physics hint. General hints increase the observed pass rate from Medium effort upward. Physics hints have the highest observed rate at all six efforts. These results suggest that relevant physics is often available to the agent but is not identified or applied without guidance.\n\nThis figure selects the complete six-effort matrix and its historical no-hint controls. It does not add the later independent High-effort replication or the one fresh baseline. The prior 78-attempt bars pooled independent trials, rather than duplicate trial identities, but this view deliberately keeps one three-repeat panel per cell. Full-study plots and backup tables still preserve the independent replication. Execution errors remain in each denominator. No trial outcomes were changed.\n\nThe hint comparison remains descriptive. Earlier controls differ in runtime, concurrency and CPU quotas. Hint lengths are not matched. A general hint may improve checking, implementation or attention, and alone does not isolate a physics deficit. Task-specific hint gains together with submitted-code failure evidence support the proposed interpretation. Published hinted evidence is aggregate only; raw transcripts were not independently inspected here.\n\nSources:\n'+UR+'/luna-effort-panel.json\n'+UR+'/sources/baseline-luna-trials.csv\n'+UR+'/sources/hint-matrix-summary.csv\n'+UR+'/plot_results.py';
await toNotes(9,'Review subsets use');
await replace(10,'Fits the supplied data.','Fits the supplied data. Covariance errors reach 136% (allowed tolerance: 4%).');
await toNotes(10,'Allowed tolerance:');
// Conclusion: connect the hint result directly to the benchmark's motivating question.
await replace(11,'A good fit can hide','Agents can fit observations and still miss the physics');
await replace(11,'Cost, reasoning effort','General and physics hints improve predictions');
await replace(11,'GPT-6 Sol Low','Luna High: 4/39 without hints, 18/39 with a general hint,\nand 30/39 with a physics hint.');
await replace(11,'Physics hints show','The gains suggest a gap in physical reasoning');
await replace(11,'At High effort:','Agents often need guidance to identify and use relevant physics.');
await toNotes(11,'Historical controls');
P.slides.items[10].speakerNotes.text='Target time: 30 seconds.\n\nThe benchmark tests whether agents can reason about the physics beyond the supplied observations. The bead example shows a good fit with the wrong model for covariance. Both hint types improve Luna at higher efforts, and physics hints improve it at every tested effort. Together with the failure analysis, these gains suggest that agents often miss relevant physics without guidance.\n\nThe quoted Luna High counts are the same single 39-attempt panels shown on slide 8. The full study retains additional independent High repetitions in the other figures and backup tables. General hints can help with implementation or checking as well as reasoning. Historical controls, runtime differences and hint length prevent a causal attribution solely to physics reasoning. These results support a limitation on this selected task suite, rather than a universal claim about agent capability. DeepSeek has many execution errors, which are not diagnoses of physics mistakes.\n\nSources:\n'+UR+'/luna-effort-panel.json\n'+UR+'/sources/trajectory-analysis.json\n'+UR+'/sources/hint-matrix-summary.md';
// Full study counts in backups remain unchanged; make their distinction from slide 8 explicit in notes.
P.slides.items[13].speakerNotes.text+='\n\nSlide 8 now shows one six-effort Luna matrix with 39 attempts per cell. This backup table continues to show all unique attempts, including the independent later High repetition. These are different declared summaries, not duplicated trial identities.';
await fs.mkdir(B+'/draft-previews',{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const n=String(i+1).padStart(2,'0');await fs.writeFile(B+'/draft-previews/slide-'+n+'.png',new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/draft-previews/slide-'+n+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
await(await PresentationFile.exportPptx(P)).save(B+'/candidate.pptx');
await fs.writeFile(B+'/changes.json',JSON.stringify({sourceSha256:hash(await fs.readFile(B+'/source.pptx')),changed,movedToNotes:moved,lunaHigh:{baseline:[4,39],general:[18,39],physics:[30,39]},unchangedDataSlides:[5,6,7,9,13,14,17],timingSeconds:[10,45,85,60,55,75,50,55,55,80,30]},null,2));
console.log('Focused draft saved from latest user source');
