import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const B=W+'/.presentation-friend-hints-20261015';
const S='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const INPUT=B+'/source.pptx';
const D=JSON.parse(await fs.readFile(B+'/data.json','utf8'));
const P=await PresentationFile.importPptx(await FileBlob.load(INPUT));
const FONT='Times New Roman';
const C={ink:'#111111',muted:'#444444',gray:'#777777',blue:'#1D4E89',gold:'#B45309',teal:'#0F766E',grid:'#D1D5DB',white:'#FFFFFF',light:'#F3F4F6',header:'#374151'};
const E=['none','low','medium','high','xhigh','max'];
const effort={none:'None',low:'Low',medium:'Medium',high:'High',xhigh:'Extra High',max:'Max'};
const conds=['baseline','general','physics'];
const labels={baseline:'No hint',general:'General hint',physics:'Physics hint'};
const colors={baseline:C.blue,general:C.gold,physics:C.teal};
function text(s,txt,x,y,w,h,size=26,color=C.ink,bold=false){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=txt;sh.text.style={typeface:FONT,fontSize:size,color,bold,wrap:'square',verticalAlignment:'top',insets:0};return sh;
}
async function clear(s){const l=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());P.delete(l.elements.map(e=>e.id));}
function head(s,n,title,sub){s.background.fill=C.white;text(s,title,48,22,1184,67,48,C.ink,true);if(sub)text(s,sub,48,99,1184,48,25,C.muted);text(s,String(n),1176,680,56,20,16,C.gray);}
function table(s,values,x,y,w,h,widths,size=24){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:C.grid,width:.7,style:'solid'});
 for(let row=0;row<values.length;row++)for(let col=0;col<values[row].length;col++){
  const cell=t.getCell(row,col);cell.fill=row===0?C.header:row%2?C.white:C.light;
  cell.text.style={typeface:FONT,fontSize:size,color:row===0?C.white:C.ink,bold:row===0,insets:10};
 }
 return t;
}
function chart(s,type,cfg){const ch=s.charts.add(type,{chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},...cfg});applyPresentationChartFont(ch,{fontFamily:FONT});return ch;}
function notes(s,seconds,body,extra=''){s.speakerNotes.text=`Target time: ${seconds ? seconds+' seconds' : 'skip in current talk'}.\n\n${body}\n\nSources:\n${D.source_url}\n${D.source_url.replace('summary.md','summary.csv')}\n${D.source_url.replace('summary.md','verification.json')}\n${extra}\n\nProvenance: published aggregates and verification metadata checked locally. Raw collaborator trajectories were not available for independent local review. Historical no-hint runs used four workers without the new per-container CPU quotas. Hint runs used up to eight workers with CPU quotas. These are independent batches, not matched-seed pairs. Costs are frozen API-equivalent estimates, not subscription charges.`;}

// Preserve the original no-hint effort comparison as a backup.
const oldEffort=P.slides.items[7].duplicate();oldEffort.moveTo(16);
oldEffort.shapes.items.find(s=>s.text?.toString()==='Fixed model, different reasoning effort')?.text.replace('Fixed model, different reasoning effort','Backup: effort in the original no-hint sweep');
for(const sh of oldEffort.shapes.items){if(sh.text?.toString()==='8')sh.text='17';}
oldEffort.speakerNotes.text=oldEffort.speakerNotes.text.replace('Target time: 55 seconds.','Target time: skip in current talk.');

// Experiment: identify all three cohorts rather than pooling them.
{
 const s=P.slides.items[4];await clear(s);head(s,5,'Experiment and evaluation','13 frozen tasks, 3 trials per setting. A pass requires timely completion and all grading gates.');
 table(s,[['Study','Models and reasoning settings','Observations'],
  ['Original no-hint sweep','GPT-6 Luna, Sol, Astra; GPT-5.6 Terra\nDeepSeek V4.1 Flash, multiple efforts','585 completed'],
  ['Completed collaborator study','GPT-5.6 Luna: None through Max\nGPT-5.6 Terra and Sol: Low, Medium','780 hinted trials\n390 historical controls'],
  ['Local hint study','GPT-5.6 Luna, Terra; DeepSeek V4.1 Flash\nHigh effort, all 3 prompt conditions','351 planned\nStill running']],48,167,1184,350,[235,640,309],24);
 text(s,'Harbor 0.21.0 + Codex agent     600 s agent     60 s verifier',48,550,1184,38,27);
 text(s,'Costs are API-equivalent estimates. Setup errors and timeouts stay separate from physics failures.',48,600,1184,42,23,C.muted);
 text(s,'Collaborator controls are historical. Worker concurrency and CPU limits differ between batches.',48,650,1184,26,19,C.gray);
 notes(s,55,'We now have three experiment cohorts. The original no-hint sweep contains 585 attempts and includes GPT-6 models. The collaborator has completed 780 hinted trials, with 390 historical no-hint controls. Its Luna model is GPT-5.6, with six reasoning settings. It also includes Terra and Sol at Low and Medium. The next two slides use the completed collaborator Luna results. Our separate three-model High-effort study is still running. These cohorts stay separate. We use the same 600-second agent and 60-second verifier limits. A timeout cannot be called a physical modeling failure. Historical control batches differ in concurrency and CPU quotas, which matters for timing-sensitive comparisons.','/Users/yp9846/Desktop/PhysLiteBench/results/reasoning-sweep-20261011/README.md\n/Users/yp9846/Desktop/PhysLiteBench/results/hint-comparison-20261014/plan.json');
}

// Cost-success frontiers: only complete costs can define these curves.
{
 const s=P.slides.items[6];await clear(s);head(s,7,'Pareto frontiers with hints: GPT-5.6 Luna','Completed collaborator study. Each setting contains 13 tasks and 3 trials per task.');
 const series=[];
 for(const c of conds){
  const eligible=D.luna.filter(r=>r.condition===c && r.cost_complete).sort((a,b)=>a.mean_cost-b.mean_cost);
  const frontier=eligible.filter(a=>!eligible.some(b=>b!==a&&b.mean_cost<=a.mean_cost&&b.passes>=a.passes&&(b.mean_cost<a.mean_cost||b.passes>a.passes)));
  series.push({name:labels[c],xValues:frontier.map(x=>Number(x.mean_cost.toFixed(9))),values:frontier.map(x=>x.passes),line:{fill:colors[c],width:2.5,style:'solid'},marker:{symbol:c==='baseline'?'circle':c==='general'?'square':'triangle',size:8,fill:colors[c]},dataLabels:{showValue:false}});
 }
 chart(s,'scatter',{position:{left:48,top:164,width:850,height:425},series,hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:22}},scatterOptions:{style:'lineWithMarkers'},xAxis:{title:'Estimated API cost per attempt (USD)',min:0,max:.06,majorUnit:.02,numberFormatCode:'$0.00',textStyle:{fontSize:21},line:{fill:C.grid,width:1}},yAxis:{title:'Completed passes out of 39',min:0,max:39,majorUnit:10,textStyle:{fontSize:21},majorGridlines:{fill:C.grid,width:1,style:'solid'}}});
 text(s,'At High effort',930,190,302,42,31,C.ink,true);
 text(s,'4 / 39  No hint',930,253,302,40,29,C.blue);
 text(s,'18 / 39  General hint',930,312,302,40,29,C.gold);
 text(s,'30 / 39  Physics hint',930,371,302,40,29,C.teal,true);
 text(s,'The High no-hint and general-hint costs are incomplete.',930,451,302,89,24,C.muted);
 text(s,'Frontiers use 11 of 18 settings with complete cost records. All success rates appear next.',48,607,1184,40,23,C.muted);
 text(s,'Historical controls differ in concurrency and CPU limits. This is a descriptive comparison.',48,650,1184,26,19,C.gray);
 notes(s,60,'These are three separate cost-success frontiers for GPT-5.6 Luna, one for each prompt condition. Each point represents a reasoning-effort setting on all thirteen tasks with three trials each. Upper left is better. To avoid treating missing costs as zero, the curves use only the eleven settings with complete cost records. Seven settings remain outside this cost plot, and the next slide includes their success rates. The clearest High-effort comparison is four passes without hints, eighteen with a general hint, and thirty with a physics hint, all out of thirty-nine. The physics High point costs about 4.55 cents per attempt at the frozen API prices. The two other High costs are incomplete. Because the no-hint controls come from earlier batches with different concurrency and CPU quotas, these results are descriptive rather than an isolated causal estimate.','Historical baseline: https://github.com/liuyiqiandrew/PhysLiteBench/blob/c925ddf30d7729a629125f942bab67d9edc53f1f/results/luna-all-efforts-20261011-205240/report.md\nKnown-cost curve construction: retain nondominated points separately within each prompt condition, among cost-complete panels only. Missing costs do not participate in dominance comparisons.');
}

// Same model, all six reasoning settings, all three prompt conditions.
{
 const s=P.slides.items[7];await clear(s);head(s,8,'Luna: reasoning effort and hints','GPT-5.6 Luna, completed passes out of 39. All six effort settings are complete.');
 chart(s,'bar',{position:{left:48,top:166,width:1184,height:427},categories:E.map(e=>effort[e]),series:conds.map(c=>({name:labels[c],values:E.map(e=>D.luna.find(r=>r.condition===c&&r.effort===e).passes),fill:colors[c],line:{fill:'none',width:0}})),barOptions:{direction:'column',grouping:'clustered',gapWidth:65,overlap:0},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:23}},xAxis:{textStyle:{fontSize:23},line:{fill:C.grid,width:1}},yAxis:{title:'Completed passes out of 39',min:0,max:39,majorUnit:10,textStyle:{fontSize:21},majorGridlines:{fill:C.grid,width:1,style:'solid'}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:21,bold:true}}});
 text(s,'Physics hints have the highest observed success at every effort setting.',48,609,1184,39,29,C.ink,true);
 text(s,'No-hint controls are historical. Errors remain in denominators. Three trials per task limit precision.',48,650,1184,26,19,C.gray);
 notes(s,55,'Now hold the model fixed and compare all six reasoning settings. The three bars at each setting correspond to no hint, general hint, and physics hint. Physics hints have the highest observed success at every effort setting. At High the counts are four, eighteen, and thirty out of thirty-nine. At Max they are eight, twenty-seven, and thirty-five. More reasoning alone does not show a monotonic trend without hints, but the hinted results improve substantially from Low to Max. A physics hint also does not guarantee success, especially with little reasoning. Every panel is complete, including timeouts and the capacity rejection where applicable. The historical controls and small task-level samples limit causal interpretation and precision. The original multi-model no-hint effort comparison is preserved in backup slide seventeen.');
}

// Conclusion: replace only the pending conclusion, preserving the other content.
{
 const s=P.slides.items[10];
 for(const sh of s.shapes.items){
  sh.text?.replace('The hint comparison is the remaining experiment','Physics hints improve observed Luna success');
  sh.text?.replace('We will test whether targeted physical guidance improves success per dollar.','At High: 4/39 without hints, 18/39 with general hints, 30/39 with physics hints.');
 }
 text(s,'Completed collaborator study. The separate local High-effort comparison is still running.',64,642,1140,29,21,C.gray);
 notes(s,30,'A good calibration fit can hide the wrong physical model, as the bead example shows. Model choice, reasoning effort, and completing on time all affect benchmark success. The completed collaborator study adds a clear descriptive result: at Luna High, passes rise from four without hints to eighteen with general hints and thirty with physics hints. Our separate three-model High-effort comparison is still running. We keep those cohorts and the historical-control caveats separate.');
}

// Full collaborator results and error counts, including Terra and Sol.
{
 const s=P.slides.items[13];await clear(s);head(s,14,'Backup: completed collaborator hint study','Each cell shows passes / 39, with execution errors in parentheses. All models are GPT-5.6.');
 const other=JSON.parse(await fs.readFile(B+'/baseline-other.json','utf8'));
 const settings=[...E.map(e=>['gpt-5.6-luna',e]),['gpt-5.6-terra','low'],['gpt-5.6-terra','medium'],['gpt-5.6-sol','low'],['gpt-5.6-sol','medium']];
 const rows=settings.map(([model,e])=>{
  const base=model==='gpt-5.6-luna'?D.luna.find(x=>x.condition==='baseline'&&x.effort===e):other.find(x=>x.model===model&&x.reasoning_effort===e);
  const cells=conds.map(c=>{
   if(c==='baseline'){const err=base.errors??(Number(base.timeouts)+Number(base.other_errors));return `${base.passes}/39 (${err})`;}
   const x=D.all_hint_rows.find(x=>x.model===model&&x.reasoning_effort===e&&x.hint===c);return `${x.passes}/39 (${Number(x.timeouts)+Number(x.other_errors)})`;
  });return [model.replace('gpt-5.6-','').replace(/^./,c=>c.toUpperCase()),effort[e],...cells];
 });
 table(s,[['Model','Effort','No hint','General hint','Physics hint'],...rows],48,169,1184,435,[210,235,245,247,247],23);
 text(s,'780 hinted trials + 390 historical controls. Five pre-agent setup failures were retained separately.',48,621,1184,29,21,C.muted);
 text(s,'Published aggregates checked locally. Raw collaborator trajectories were not independently reviewed here.',48,657,1184,26,18,C.gray);
 notes(s,0,'The collaborator completed twenty hinted panels of thirty-nine model attempts, or 780 attempts. Its ten historical no-hint panels contribute another 390 controls. There are 440 clean hinted passes, 315 graded failures, 24 timeouts, and one API-capacity rejection. Five additional pre-agent setup failures are retained outside those matrix denominators, giving 785 orchestration records. The collaborator filled only those five setup slots, and reports that no model-phase failure was resampled. Our ongoing local study follows its own frozen retention rule and remains separate. Nineteen hinted attempts have unknown full cost. We checked aggregate arithmetic and published metadata locally, not raw remote sessions.');
}

// Preserve note text on unchanged slides; adjust only the timing allocation.
const timing=[10,45,85,60,55,60,60,55,55,85,30];
for(let i=0;i<timing.length;i++){
 const s=P.slides.items[i];s.speakerNotes.text=s.speakerNotes.text.replace(/^Target time: .*?\./,`Target time: ${timing[i]} seconds.`);
}
if(P.slides.items.length!==17)throw new Error('Unexpected slide count');
await fs.mkdir(B+'/draft-previews',{recursive:true});
for(const n of [5,7,8,11,14,17]){
 const s=P.slides.items[n-1];await fs.writeFile(`${B}/draft-previews/slide-${n}.png`,new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
}
const raw=B+'/draft.pptx',candidate=B+'/candidate.pptx';
await(await PresentationFile.exportPptx(P)).save(raw);
execFileSync(PY,[B+'/preserve_workbooks.py',INPUT,raw,candidate,B+'/workbook-preservation.json'],{stdio:'inherit'});
const final=W+'/output/physlitebench-presentation/PhysLiteBench-10-minute-collaborator-hints.pptx';
const tables=[2,4,5,6,13,14];
const result=await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:S+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:S+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:17,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[6,7,8,9,16,17],requiredEmbeddedWorkbookChartOwnerSlides:[6,9,16,17],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'user_request',families:[FONT]},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'});
await fs.writeFile(B+'/finalization-result.json',JSON.stringify(result,null,2));
const F=await PresentationFile.importPptx(await FileBlob.load(final));await fs.mkdir(B+'/final-previews',{recursive:true});
for(const [i,s] of F.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(`${B}/final-previews/${stem}.png`,new Uint8Array(await(await F.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(`${B}/final-previews/${stem}.json`,await(await s.export({format:'layout',detail:'full'})).text());
}
await fs.writeFile(B+'/final-previews/montage.webp',new Uint8Array(await(await F.export({format:'webp',montage:true,scale:.5})).arrayBuffer()));
console.log(JSON.stringify({final,slides:17,changedSlides:[5,7,8,11,14],backupAdded:17,timingSeconds:timing.reduce((a,b)=>a+b,0)}));
