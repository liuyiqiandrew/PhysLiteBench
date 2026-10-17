import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench',B=W+'/.presentation-local-final-20261015',R='/Users/yp9846/Desktop/PhysLiteBench';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const D=JSON.parse(await fs.readFile(B+'/data.json','utf8'));
if(D.status.recorded_attempts!==351||D.status.state!=='complete')throw Error('Full audited cohort required');
const current=R+'/output/presentation-20261014/PhysLiteBench-10-minute-combined-frontiers.pptx';
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
if(sha(await fs.readFile(current))!==sha(await fs.readFile(B+'/source.pptx')))throw Error('New user edits detected: reimport current file');
const P=await PresentationFile.importPptx(await FileBlob.load(B+'/source.pptx'));
const C={ink:'#111111',muted:'#444444',gray:'#666666',blue:'#1D4E89',gold:'#B45309',teal:'#0F766E',grid:'#D1D5DB',white:'#FFFFFF'};
function text(s,t,x,y,w,h,size=26,color=C.ink,bold=false){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'Times New Roman',fontSize:size,color,bold,wrap:'square',verticalAlignment:'top',insets:0};return a;}
async function layout(s){return JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());}
function table(s,values,x,y,w,h,widths,size=24){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({style:'solid',fill:'#D1D5DB',width:.7});
 for(let row=0;row<values.length;row++)for(let col=0;col<values[0].length;col++){
  const cell=t.getCell(row,col);cell.fill=row===0?'#374151':row%2===0?'#F3F4F6':'#FFFFFF';
  cell.text.style={typeface:'Times New Roman',fontSize:size,color:row===0?C.white:C.ink,bold:row===0,insets:9};
 }return t;
}
const s=P.slides.items[6];P.delete((await layout(s)).elements.map(x=>x.id));
text(s,'Hint comparison across models',48,22,1184,67,48,C.ink,true);
text(s,'Local study: GPT-5.6 Luna, GPT-5.6 Terra and DeepSeek V4.1 Flash, all at High effort.',48,99,1184,48,25,C.muted);
text(s,'Cost and completed passes',48,161,612,44,31,C.ink,true);
text(s,'All outcomes',700,161,532,44,31,C.ink,true);
const labels={baseline:'No hint',general:'General hint',physics:'Physics hint'},colors={baseline:C.blue,general:C.gold,physics:C.teal};
const series=['baseline','general','physics'].map(c=>{
 const points=[...D.frontiers[c]].sort((a,b)=>a.mean_estimated_api_usd-b.mean_estimated_api_usd);
 return {name:labels[c],xValues:points.map(x=>Number(x.mean_estimated_api_usd.toFixed(9))),values:points.map(x=>x.passes),line:{fill:colors[c],width:2.5,style:'solid'},marker:{symbol:c==='baseline'?'circle':c==='general'?'square':'triangle',size:9,fill:colors[c]},dataLabels:{showValue:false},dataLabelOverrides:points.map((x,i)=>({idx:i,text:x.model.includes('luna')?'Luna':'Terra',showValue:false,position:c==='physics'?'left':'top',textStyle:{fontSize:22,fill:C.ink,typeface:'Times New Roman',bold:false}}))};
});
const ch=s.charts.add('scatter',{position:{left:48,top:218,width:612,height:376},series,hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:22}},scatterOptions:{style:'lineWithMarkers'},xAxis:{title:'Estimated API cost / attempt (USD)',min:0,max:.32,majorUnit:.08,numberFormatCode:'$0.00',textStyle:{fontSize:20},line:{fill:C.grid,width:1}},yAxis:{title:'Completed passes out of 39',min:0,max:39,majorUnit:10,textStyle:{fontSize:20},majorGridlines:{fill:C.grid,width:1,style:'solid'}},chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0}});
applyPresentationChartFont(ch,{fontFamily:'Times New Roman'});
const names={'gpt-5.6-luna':'Luna','gpt-5.6-terra':'Terra','deepseek/deepseek-v4.1-flash':'DeepSeek\nV4.1 Flash'};
const values=[['Model','No hint','General','Physics']];
for(const m of Object.keys(names)){const r=[names[m]];for(const c of ['baseline','general','physics']){const v=D.panels.find(x=>x.model===m&&x.condition===c);r.push(`${v.passes}/39\n${v.errors} ${v.errors===1?'error':'errors'}`);}values.push(r);}
table(s,values,700,218,532,280,[172,120,120,120],24);
text(s,'Four settings have complete costs.\nFive settings are excluded from these frontiers.',48,608,612,56,22,C.gray);
text(s,'Passes / 39, with execution errors below.\nDeepSeek errors are mostly timeouts.',700,524,532,70,24,C.muted);
text(s,'Luna uses historical controls. Local concurrency changed from 3 trials to 6 sharing 8 CPUs.',48,674,1120,28,19,C.gray);
text(s,'7',1176,680,56,20,16,C.gray);
s.speakerNotes.text=`Target time: 50 seconds.\n\nAll 351 observations are recorded: 313 local attempts and 38 imported Luna baselines. Physics hints produced 32 completed passes for Luna and 36 for Terra out of 39 each. General hints produced seven and thirteen. DeepSeek had ${D.panels.filter(x=>x.model.startsWith('deepseek')).reduce((a,x)=>a+x.errors,0)} execution errors across its 117 attempts, mostly timeouts. These outcomes measure completion under our limits and do not establish a physics reasoning failure.\n\nThe left plot shows a separate observed frontier for each prompt condition among settings with complete costs. The blue and teal frontiers contain only Terra. General hints have two points, Luna and Terra. Five settings have missing or uncertain costs and cannot enter exact frontiers. The right table retains all results. Unknown cost is never zero. Upper provider reserves are not confirmed charges.\n\nLuna's 38 historical controls used Codex CLI 0.154.0 and concurrency four. Local trials used CLI 0.160.0, up to three concurrent trials initially, then up to six sharing Docker CPUs 0-7 after the user-authorized amendment. Resources, client and date differ, so the comparison does not isolate a causal hint effect. Hint length is not matched. All frozen task, prompt, grading and model/effort provenance checks passed. Published historical source metadata was checked locally, but raw remote transcripts were not independently verified. These local observations remain separate from the collaborator matrix and original GPT-6 sweep.\n\nSources:\n${R}/results/hint-comparison-20261014/results.json\n${R}/results/hint-comparison-20261014/final-audit-20261015.json\n${R}/results/hint-comparison-20261014/pricing.json\n${R}/results/hint-comparison-20261014/parallel-eight-cpus-amendment-20261015.json`;
const experiment=P.slides.items[4];const el=(await layout(experiment)).elements;
const tab=P.resolve(el.find(x=>x.kind==='table').id);tab.cells.set(3,2,'313 local attempts\n38 historical controls');
P.resolve(el.find(x=>x.text?.startsWith('Collaborator controls are historical')).id).text='Historical controls and changing runtime resources qualify the comparisons.';
experiment.speakerNotes.text=experiment.speakerNotes.text.replace('Slide 7 is reserved for our running study.','Slide 7 shows our completed local study.').replace('Our separate three-model High-effort study is still running.','Our separate three-model High-effort study has all 351 observations, comprising 313 local attempts and 38 imported historical Luna baselines.');
experiment.speakerNotes.text+='\n\nLocal-study runtime: imported Luna CLI 0.154.0/concurrency four versus local CLI 0.160.0, initially up to three trials then up to six sharing eight Docker CPUs. Tasks, prompts, grading and 600/60-second limits stayed frozen. Historical controls and the resource change prevent a clean causal interpretation. Sources: '+R+'/results/hint-comparison-20261014/plan.json and '+R+'/results/hint-comparison-20261014/final-audit-20261015.json';
const conclusion=P.slides.items[10];const ce=(await layout(conclusion)).elements;
P.resolve(ce.find(x=>x.text?.startsWith('Physics hints improve')).id).text='Physics hints show the highest observed success';
P.resolve(ce.find(x=>x.text?.startsWith('At High:')).id).text='Local High study: Luna 32/39 and Terra 36/39 with physics hints.';
P.resolve(ce.find(x=>x.text?.startsWith('Completed collaborator study.')).id).text='Historical controls and runtime differences limit causal claims. DeepSeek has many execution errors.';
conclusion.speakerNotes.text='Target time: 30 seconds.\n\nA good calibration fit can hide the wrong physical model, as the bead example shows. Cost and completing within the time limit matter alongside reasoning effort. Physics hints have the highest observed success in our local High study, with 32 of 39 completed passes for Luna and 36 for Terra. The separate collaborator study also shows higher observed success with physics hints. Historical controls and runtime differences limit causal claims. DeepSeek results mainly reflect execution problems under this harness.\n\nSources:\n'+R+'/results/hint-comparison-20261014/results.json\n'+R+'/results/hint-comparison-20261014/final-audit-20261015.json\nhttps://github.com/liuyiqiandrew/PhysLiteBench/blob/c925ddf30d7729a629125f942bab67d9edc53f1f/results/hint-comparison-20261015-005800/summary.md\nThe collaborator and local cohorts remain separate. Raw remote transcripts were not independently verified locally.';
const timing=[10,45,85,60,55,75,50,55,55,80,30];
for(let i=0;i<timing.length;i++)P.slides.items[i].speakerNotes.text=P.slides.items[i].speakerNotes.text.replace(/^Target time: .*?\./,`Target time: ${timing[i]} seconds.`);
if(timing.reduce((a,b)=>a+b,0)!==600||P.slides.items.length!==17)throw Error('Timing or slide count mismatch');
await fs.mkdir(B+'/draft-previews',{recursive:true});
for(const n of [5,7,11])await fs.writeFile(B+`/draft-previews/slide-${n}.png`,new Uint8Array(await(await P.export({slide:P.slides.items[n-1],format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(P)).save(B+'/draft.pptx');
execFileSync(PY,[B+'/preserve_workbooks.py',B+'/source.pptx',B+'/draft.pptx',B+'/candidate.pptx',B+'/workbook-preservation.json'],{stdio:'inherit'});
await fs.writeFile(B+'/edit-summary.json',JSON.stringify({changedVisibleSlides:[5,7,11],preservedUserExampleSlides:[3,4,10,16],timing,talkSeconds:600,qaSeconds:180,slideCount:17},null,2));
console.log('Created final-cohort draft from the current reviewed presentation.');
