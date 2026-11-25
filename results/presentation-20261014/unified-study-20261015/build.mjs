import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench',B=W+'/.presentation-unified-20261015',R='/Users/yp9846/Desktop/PhysLiteBench';
const U=W+'/output/PhysLiteBench-study-figures-20261015',UR=R+'/output/presentation-20261014/PhysLiteBench-study-figures-20261015';
const current=R+'/output/presentation-20261014/PhysLiteBench-10-minute-collaborator-and-local-hints-final.pptx';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
if(hash(await fs.readFile(current))!==hash(await fs.readFile(B+'/source.pptx')))throw Error('Source changed; reimport new user edits.');
const D=JSON.parse(await fs.readFile(U+'/summary.json','utf8')),rows=D.summary;
const P=await PresentationFile.importPptx(await FileBlob.load(B+'/source.pptx'));
const C={ink:'#111111',muted:'#444444',gray:'#777777',blue:'#1D4E89',gold:'#B45309',teal:'#0F766E',red:'#B00032'};
async function layout(s){return JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());}
function text(s,t,x,y,w,h,size=26,color=C.ink,bold=false){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'Times New Roman',fontSize:size,color,bold,wrap:'square',verticalAlignment:'top',insets:0};return a;}
async function clear(n,title,subtitle){const s=P.slides.items[n-1];P.delete((await layout(s)).elements.map(e=>e.id));text(s,title,48,22,1184,67,48,C.ink,true);if(subtitle)text(s,subtitle,48,99,1184,48,25,C.muted);text(s,String(n),1176,680,56,20,16,C.gray);return s;}
async function replace(n,start,value){const e=(await layout(P.slides.items[n-1])).elements.find(e=>e.text?.startsWith(start));if(!e)throw Error('Missing '+start);P.resolve(e.id).text=value;}
function table(s,vals,x,y,w,h,widths,size=23,pad=7){const t=s.tables.add({rows:vals.length,columns:vals[0].length,left:x,top:y,width:w,height:h,values:vals,columnWidths:widths});t.borders.assign({style:'solid',fill:'#D1D5DB',width:.6});for(let r=0;r<vals.length;r++){t.rows[r].height=h/vals.length;for(let c=0;c<vals[0].length;c++){const q=t.getCell(r,c);q.fill=r===0?'#374151':r%2===0?'#F3F4F6':'#FFFFFF';q.text.style={typeface:'Times New Roman',fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0,insets:pad,verticalAlignment:'middle'};}}return t;}
async function plot(s,name,x=48,y=151,w=1184,h=471){s.images.add({blob:await fs.readFile(U+'/figures/'+name+'.png'),contentType:'image/png',alt:'Python / Matplotlib figure: '+name+'. Data and reproducible script in the accompanying plotting bundle. No smoothing.',position:{left:x,top:y,width:w,height:h},fit:'contain'});}
function foot(s,t,y=650){text(s,t,48,y,1170,36,19,C.gray);}
const models=['gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-luna','gpt-6-sol','gpt-6-astra','deepseek/deepseek-v4.1-flash'];
const names={'gpt-5.6-luna':'GPT-5.6 Luna','gpt-5.6-terra':'GPT-5.6 Terra','gpt-5.6-sol':'GPT-5.6 Sol','gpt-6-luna':'GPT-6 Luna','gpt-6-sol':'GPT-6 Sol','gpt-6-astra':'GPT-6 Astra','deepseek/deepseek-v4.1-flash':'DeepSeek V4.1 Flash'};
const EN={none:'None',low:'Low',medium:'Medium',high:'High',xhigh:'Extra High',max:'Max'},es=['none','low','medium','high','xhigh','max'];
const noteSrc=`\n\nSources:\n${UR}/summary.csv\n${UR}/source-manifest.json\n${UR}/aggregate_results.py\n${UR}/plot_results.py`;
const method=`Results are grouped by exact model version, effort and reported prompt condition. Thirty-eight duplicate baseline records were removed by trial identity. Rates are total completed passes divided by all recorded attempts, including execution errors. Most settings have 39 attempts, repeated settings have 78. Luna High no-hint has 40: four attempts on entropy-anomaly and three on each other task. The three-attempt DeepSeek None pilot is retained outside the primary summary. Five earlier setup records were replaced before the reported 780 model-attempt matrix; the later setup failure and administrative interruptions remain in their original denominators. No outcomes have been rerun or changed.\n\nThe task content and grading match the frozen 13-task suite according to saved hashes and published verification metadata. Runtime varies across batches: Codex CLI 0.154.0 or 0.160.0; concurrency four or up to eight under CPU quotas in earlier batches; later concurrency up to three, then six sharing eight Docker CPUs. Agent and verifier limits remain 600 and 60 seconds. Baselines can predate hint runs, and hint length is not matched. These pooled comparisons are descriptive, not randomized causal estimates. Published aggregate data is included, but the 780 hinted raw transcripts were not independently reviewed on this machine. Their published model/session and source checks are preserved in the source archive.\n\nCosts use saved Standard API-equivalent token rates, including caching and per-request long-context factors. They are not subscription charges and exclude local compute. Incomplete costs are never imputed as zero. Unknown provider charges retain bounds, whose upper reserves are not confirmed spending.`;
// Preserve the latest user wording; correct the unsupported all-thirteen qualification statement.
await replace(2,'Tasks, all achieved','physics tasks that test predictions beyond calibration.\nTen had 0/3 Luna High passes during initial selection.\nImplemented with the Harbor framework.');
const introBody=(await layout(P.slides.items[1])).elements.find(e=>e.text?.startsWith('physics tasks'));P.resolve(introBody.id).position={...introBody.position,height:108};
P.slides.items[1].speakerNotes.text='Target time: 45 seconds.\n\nOur benchmark asks a coding agent to infer a physical model from a description and limited observations, then predict new conditions. It receives starter code, calibration data and public tests. The hidden evaluator tests predictions that a good calibration fit does not guarantee. The thirteen tasks cover several physics families. The original ten had zero of three GPT-5.6 Luna High passes during selection. The roster does not support claiming the same current qualification result for all thirteen. Selection outcomes and independent evaluation results answer different questions.\n\nSources:\n'+R+'/results/task-roster.json\n'+R+'/results/qualified-tasks.json';
await replace(4,'The task stays the same.','The task stays the same. Hints probe why the agent fails.');
// Slide 5: one experiment inventory, defined entirely by model and effort.
let s=await clear(5,'Experiment and evaluation','13 frozen tasks. A pass requires completion within the time limit and all grading checks.');
const coverage={
'gpt-5.6-luna':['None to Max (6 levels)','None to Max (6 levels)'],
'gpt-5.6-terra':['Low, Medium, High, Extra High','Low, Medium, High'],
'gpt-5.6-sol':['Low, Medium','Low, Medium'],
'gpt-6-luna':['Low, High, Extra High','—'],
'gpt-6-sol':['Low, High, Extra High','—'],
'gpt-6-astra':['Low, High, Extra High','—'],
'deepseek/deepseek-v4.1-flash':['Low, High, Max','High']};
const vals=[['Model','No-hint efforts','Hinted efforts','Tasks','Attempts'],...models.map(m=>[names[m],...coverage[m],'13',String(rows.filter(r=>r.model===m).reduce((a,r)=>a+r.attempts,0))])];
table(s,vals,48,158,1184,416,[255,340,295,100,194],23,7);
text(s,'2,068 attempts across 48 settings. Reused baseline trials count once.',48,590,1184,40,27,C.ink);
text(s,'Harbor 0.21.0 and Codex agent. Limits: 600 s for the agent, 60 s for grading.',48,635,1184,35,22,C.muted);
s.speakerNotes.text='Target time: 55 seconds.\n\nThis is one study across seven model versions. Each listed setting covers all thirteen tasks. A batch usually samples three independent attempts per task, without stopping after a pass. We combine independent repetitions of the same model, effort and condition, and count each reused baseline only once. The table shows total attempts over the listed conditions, including exceptions. The exact breakdown is in the backup tables and CSV. No-hint totals are 1,054; general and physics hints each have 507. An incomplete three-attempt DeepSeek None pilot remains outside these totals.\n\n'+method+noteSrc;
// Slide 6: unified no-hint cost frontier.
s=await clear(6,'Cost and success without hints','All model versions on one cost axis. Each point is one model and reasoning effort.');
await plot(s,'01-no-hint-frontier');
foot(s,'19 of 24 settings have complete costs. Dashed line: observed frontier. Five incomplete-cost settings are omitted.');
s.speakerNotes.text='Target time: 75 seconds.\n\nThe horizontal axis uses a logarithmic scale because costs span more than two orders of magnitude. Moving left means lower cost and moving up means more completed passes. Colors identify exact model versions; marker shapes identify reasoning effort. The dashed line joins observed nondominated settings with straight segments, without smoothing or interpolation of new observations. GPT-6 Luna Low and Extra High anchor the inexpensive end. GPT-6 Sol Low and High reach about 95% and 100% completion. The remaining complete-cost settings are shown even when dominated. Five settings have incomplete costs: Luna 5.6 High and Extra High plus DeepSeek Low, High and Max. Their outcomes stay in the backup and data, but their uncertain cost does not enter an exact frontier.\n\n'+method+noteSrc;
// Slide 7: same model/effort coverage, three guidance conditions.
s=await clear(7,'Cost and success with hints','Three prompt conditions for the 12 model–effort combinations tested under every condition.');
await plot(s,'02-hint-frontiers');
foot(s,'22 of 36 settings have complete costs. Lines join observed frontiers. Historical controls and runtime differences remain.');
s.speakerNotes.text='Target time: 50 seconds.\n\nThis view holds the set of model and effort combinations fixed across the three prompt conditions. It includes GPT-5.6 Luna at six efforts, Terra at Low, Medium and High, Sol at Low and Medium, and DeepSeek at High. Color shows the hint condition. Shape shows the model and nearby letters show selected effort labels. We combine the available independent attempts within each model, effort and condition. The log-scaled cost axis exposes the low-cost Luna settings and the more expensive Terra and Sol settings. Physics hints generally reach higher observed success, but fourteen settings have incomplete costs, including all three Luna High cost estimates after pooling, so they are omitted from exact cost frontiers. All outcomes remain in the next slide and backup table. The curves join observed points with straight segments. They do not imply interpolated model configurations or a causal effect.\n\n'+method+noteSrc;
// Slide 8: Luna effort comparison with actual denominators after deduplication.
s=await clear(8,'Luna: reasoning effort and hints','GPT-5.6 Luna. Bar height is the pass rate; labels show completed passes / attempts.');
await plot(s,'03-luna-effort-hints',48,147,1184,471);
text(s,'Physics hints have the highest observed pass rate at every effort setting.',48,619,1184,39,29,C.ink,true);
foot(s,'High has 40 no-hint attempts and 78 per hint condition. Other settings have 39. Execution errors stay in denominators.',663);
s.speakerNotes.text='Target time: 55 seconds.\n\nHere we keep the model fixed and compare effort and guidance. The height of each bar is a rate, so the additional High-effort attempts do not create an artificial height advantage. At High, the combined results are four of forty without a hint, twenty-five of seventy-eight with a general hint, and sixty-two of seventy-eight with a physics hint. Other cells have thirty-nine attempts. Physics hints have the highest observed rate at each effort setting. Extra High and Max still differ, and the small number of repeats limits precision.\n\n'+method+noteSrc;
// Slide 9: aggregate outcomes and explicitly scoped failure diagnoses.
s=await clear(9,'Where the agents fail','Outcome totals combine all tested efforts and hints. Failure diagnoses use reviewed subsets.');
await plot(s,'04-outcomes-by-model',32,149,854,520);
text(s,'186 / 191',917,191,315,64,49,C.gold,true);
text(s,'Reviewed no-hint failures\nfit calibration but failed\non new predictions.',917,263,315,98,27,C.ink);
text(s,'229 / 315',917,402,315,64,49,C.red,true);
text(s,'Reviewed hinted failures\ninvolved physics or modeling.',917,474,315,98,27,C.ink);
foot(s,'Review subsets use different diagnostic categories. Execution errors alone do not establish a physics mistake.',665);
s.speakerNotes.text='Target time: 55 seconds.\n\nThe outcome chart combines all recorded attempts by model. Its different sample sizes are printed beside the names. The study has 899 completed passes, 904 graded failures and 265 execution errors. These model totals combine different effort and hint settings, so this chart describes outcome composition rather than a controlled ranking.\n\nThe failure diagnoses on the right refer to explicitly reviewed subsets. Among 191 normally completed failures in a 585-attempt no-hint subset, 186 passed calibration and parameter checks but failed new predictions. At least 141 of these 186 also show public-test passes. The published hinted failure review classified 229 of 315 graded failures as physics/modeling, 54 as mathematical/numerical and 32 as implementation. These are different review taxonomies and should not be subtracted as a hint-effect estimate. The no-hint analysis used saved tool actions, submitted code and verifier evidence. The hinted diagnoses are published review results; their raw trajectories were not independently reviewed here. None of these counts infer physics failure from a timeout.\n\nSources:\n'+UR+'/sources/trajectory-analysis.json\n'+UR+'/sources/hint-matrix-summary.md'+noteSrc;
// Plain grammar and user-specified covariance term on the approved bead diagram.
await replace(10,'Keeps the starter’s rule','Keeps the starter’s correlation formula.\nIt misses the effect of the initial spacing,\nso the covariance is too large.');
await replace(10,'Allowed error:','Allowed error: 4%. The covariance describes averages across repeated runs. Bead positions are schematic.');
// Conclusions reflect the pooled counts and exact model generations.
await replace(11,'The bead example fits','The bead example fits the data but predicts the covariance incorrectly.');
await replace(11,'Sol 6 Low','GPT-6 Sol Low and High reach the high-success end of the no-hint frontier.');
await replace(11,'Local High study:','At High effort: Luna 62/78 and Terra 36/39 passes with physics hints.');
P.slides.items[10].speakerNotes.text='Target time: 30 seconds.\n\nA good calibration fit can hide the wrong physical model, as the bead covariance example shows. Model choice and reasoning effort trade off cost against success. Across the combined evidence, physics hints have the highest observed success at every tested Luna effort. At High, Luna passes sixty-two of seventy-eight attempts and Terra thirty-six of thirty-nine with physics hints. Historical controls and runtime differences qualify causal claims, and execution errors remain separate from physics failure.'+noteSrc;
// Backup 13: every no-hint setting, split into two readable tables by model version.
s=await clear(13,'Backup: no-hint results by model and effort','Completed passes / attempts. Errors are retained. Mean cost is shown only when complete.');
const base=rows.filter(r=>r.condition==='baseline');
for(let j=0;j<2;j++){const part=base.slice(j*12,(j+1)*12);const values=[['Model and effort','Passes / n','Errors','Mean cost'],...part.map(r=>[names[r.model].replace('GPT-','').replace('DeepSeek V4.1 Flash','DeepSeek')+' '+EN[r.effort],`${r.passes}/${r.attempts}`,String(r.errors),r.cost_complete?'$'+r.mean_cost_usd.toFixed(4):'Incomplete'])];table(s,values,48+j*604,161,580,442,[248,111,85,136],21,5);}
foot(s,'All 24 settings cover 13 tasks. DeepSeek None has only 3 attempts and is excluded. Costs are API-equivalent estimates.');
s.speakerNotes.text='Backup.\n\nAll complete no-hint settings, aggregated by exact model version and effort. This table retains results with missing cost, even though those settings are absent from the exact frontier. Terra Low and High, and DeepSeek High, each have 78 distinct attempts. Luna High has 40 because one fresh baseline was already started before historical baseline reuse. Other settings have 39. Cost bounds and original records are in the plotting bundle.\n\n'+method+noteSrc;
// Backup 14: every setting measured in all three conditions, without worker labels.
s=await clear(14,'Backup: results under all three prompt conditions','Each cell shows completed passes / attempts, followed by execution errors in parentheses.');
const keys=[];for(const m of models)for(const e of es)if(['baseline','general','physics'].every(c=>rows.some(r=>r.model===m&&r.effort===e&&r.condition===c)))keys.push([m,e]);
const tv=[['Model and effort','No hint','General hint','Physics hint'],...keys.map(([m,e])=>[names[m]+' '+EN[e],...['baseline','general','physics'].map(c=>{const r=rows.find(r=>r.model===m&&r.effort===e&&r.condition===c);return `${r.passes}/${r.attempts} (${r.errors})`;})])];
table(s,tv,48,161,1184,455,[464,240,240,240],22,6);
foot(s,'13 tasks per setting. Counts combine distinct attempts; repeated baseline records count once.');
s.speakerNotes.text='Backup.\n\nThese twelve model-effort combinations have all three prompt conditions. The full experiment also includes models and effort settings tested without hints only. Every result in this table belongs to the same frozen thirteen-task benchmark. Parenthesized values are execution errors, not additional passes or independently diagnosed physics failures. The setup replacement policy differs across the archived schedules and remains documented, rather than being changed after the fact.\n\n'+method+noteSrc;
// Keep approved bead illustration; render the actual analytic series in Python.
const b16=P.slides.items[15];P.delete((await layout(b16)).elements.filter(e=>e.kind==='chart').map(e=>e.id));
await plot(b16,'05-bead-covariance',632,198,600,350);
await replace(16,'Schematic and analytic curves','Analytic covariance curves and schematic positions, not measured trajectories. The task asks for the long-time scaling coefficient.');
b16.speakerNotes.text+='\n\nThe Python plot uses the exact 57 coordinate pairs per series from the user-edited source chart. No spline or smoothing is applied. The source series and plotting script are archived in '+UR+'.';
// Backup 17: same metric and unified no-hint summary across all model versions.
s=await clear(17,'Backup: reasoning effort without hints','Each line holds the model fixed. Only the observed effort settings are connected.');
await plot(s,'06-no-hint-efforts');
foot(s,'Rates use each setting’s actual denominator. Model generations remain distinct. Historical and runtime differences apply.');
s.speakerNotes.text='Backup.\n\nAll available model versions and no-hint effort settings appear in one chart. Rates use the pooled distinct-attempt denominators in the preceding table. Points are connected by straight segments only for visual guidance. Untested effort settings are not inferred. More reasoning does not monotonically increase the observed completion rate under the fixed time limit.\n\n'+method+noteSrc;
// Remove obsolete worker-oriented narrative from unrelated notes while keeping source paths intact.
P.slides.items[11].speakerNotes.text='Target time: 180 seconds.\n\nQuestions. Useful scope distinctions are task selection versus evaluation, completed passes versus timeout checkpoints, and descriptive hint comparisons versus causal estimates. Backup slides provide the full model/effort tables, task roster and bead covariance formulas. The plotting bundle contains the archived sources, hash manifest, aggregation script and Python plotting script.'+noteSrc;
P.slides.items[14].speakerNotes.text='Backup.\n\nThe same thirteen task families are used throughout the study. Content and grading match the frozen source snapshot. The original ten have a preserved zero-of-three Luna High qualification record; the three later tasks have separate selection and confirmation evidence. Subsequent independent evaluations can pass.\n\nSources:\n'+R+'/results/task-roster.json';
const timings=[10,45,85,60,55,75,50,55,55,80,30];
for(let i=0;i<11;i++)P.slides.items[i].speakerNotes.text=P.slides.items[i].speakerNotes.text.replace(/^Target time: .*?\./,`Target time: ${timings[i]} seconds.`);
await fs.mkdir(B+'/draft-previews',{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 await fs.writeFile(B+`/draft-previews/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+`/draft-previews/slide-${String(i+1).padStart(2,'0')}.json`,await(await s.export({format:'layout',detail:'full'})).text());
}
await(await PresentationFile.exportPptx(P)).save(B+'/candidate.pptx');
await fs.writeFile(B+'/edit-summary.json',JSON.stringify({sourceSha256:hash(await fs.readFile(B+'/source.pptx')),slideCount:17,timings,talkSeconds:600,qaSeconds:180,charts:'Python-rendered PNGs at explicit user request. PDF/SVG and reproducible data/scripts supplied.',unchangedVisibleSlides:[1,3,12,15],noNewBenchmarkCalls:true},null,2));
console.log('Unified study draft built from current user-edited source.');
