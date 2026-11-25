import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const R='/Users/yp9846/Desktop/PhysLiteBench';
const B=path.join(W,'.presentation-plain-20261015');
const S='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const INPUT=R+'/output/presentation-20261014/PhysLiteBench-10-minute-CS-audience.pptx';
const FONT='Times New Roman';
const C={white:'#FFFFFF',ink:'#111111',muted:'#444444',gray:'#777777',light:'#F3F4F6',line:'#D1D5DB',blue:'#1D4E89',teal:'#0F766E',gold:'#B45309',red:'#A50026',purple:'#5B4B7C'};
const P=await PresentationFile.importPptx(await FileBlob.load(INPUT));
const old=[...P.slides.items];
const oldNotes=JSON.parse(await fs.readFile(W+'/.presentation-build-20261014/speaker-notes.json','utf8'));
function text(s,txt,x,y,w,h,size=26,color=C.ink,bold=false,align='left'){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=txt;sh.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',wrap:'square',insets:0};return sh;
}
function rect(s,x,y,w,h,fill=C.white,line=C.line,radius=0){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1.4,style:'solid'},borderRadius:radius});}
function circle(s,txt,x,y,d,fill,color=C.ink){const a=s.shapes.add({geometry:'ellipse',position:{left:x,top:y,width:d,height:d},fill,line:{fill:color,width:1.6,style:'solid'}});a.text=txt;a.text.style={typeface:FONT,fontSize:27,color,bold:true,alignment:'center',verticalAlignment:'middle',insets:0};return a;}
function arrow(s,x,y,w,color=C.ink){return s.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:16},fill:color,line:{fill:'none',width:0}});}
function connect(s,a,b,from='right',to='left',color=C.ink,kind='straight',two=false){return s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width:2,style:'solid'},tail:{type:'triangle',width:'sm',length:'sm'},...(two?{head:{type:'triangle',width:'sm',length:'sm'}}:{})});}
function base(title,sub){const s=P.slides.add();s.background.fill=C.white;text(s,title,48,22,1184,67,48,C.ink,true);if(sub)text(s,sub,48,96,1184,50,25,C.muted);return s;}
function footer(s,t){text(s,t,48,638,1184,40,18,C.gray);}
function note(s,seconds,body,refs){s.speakerNotes.text=`Target time: ${seconds?seconds+' seconds':'skip in current talk'}.\n\n${body}\n\nSources:\n${refs.map(r=>r.startsWith('/')?r:R+'/'+r).join('\n')}`;}
function chart(s,type,cfg){const ch=s.charts.add(type,{chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},...cfg});applyPresentationChartFont(ch,{fontFamily:FONT});for(let i=0;i<cfg.series.length;i++)ch.series.getItemAt(i).dataLabels={textStyle:{typeface:FONT,fontSize:18}};return ch;}

function table(s,values,x,y,w,h,widths,size=25){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.borders.assign({fill:C.line,width:.8,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#374151':r%2?C.white:C.light;
  cell.text.style={typeface:FONT,fontSize:size,bold:r===0,color:r===0?C.white:C.ink,verticalAlignment:'middle',wrap:'square'};
 }
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:12,right:12,top:8,bottom:8},anchor:'center'});
 return t;
}
async function replaceText(s,oldText,newText){for(const e of JSON.parse(await(await s.export({format:'layout',detail:'full'})).text()).elements)if(e.kind==='textbox'&&e.text===oldText)P.resolve(e.id).text=newText;}
async function clear(s){P.delete(JSON.parse(await(await s.export({format:'layout',detail:'full'})).text()).elements.map(e=>e.id));}
const physRefs=['tasks/single-file-memory/environment/README.md','tasks/single-file-memory/instruction.md','tasks/single-file-memory/environment/model.py','tasks/single-file-memory/solution/model.py','tasks/single-file-memory/AUTHOR.md'];

// Minimal title page, matching the existing dark Q&A treatment.
const cover=P.slides.add();cover.background.fill='#0B0B0F';
text(cover,'PhysLiteBench',64,238,1152,104,76,C.white,true);
text(cover,'Yue Pan and Andrew Liu',68,388,1120,58,36,'#D1D5DB');
note(cover,10,'We study how well coding agents can turn a physical setup and limited measurements into predictions.',[]);

const intro=old[0];
await replaceText(intro,'PhysLiteBench: physics beyond a good fit','Physics beyond a good fit');
await replaceText(intro,'Yue Pan, October 2026. Tasks were selected using GPT-5.6 Luna, so this is a deliberately challenging suite.','Tasks were selected using GPT-5.6 Luna, so this is a deliberately challenging suite.');

// Plain-language setup replaces the analytic chart in the main narrative.
const setup=base('Example: tracking one bead in a narrow tube','The beads move randomly and bump into each other, but cannot pass one another.');
const rows=[{y:166,caption:'Start every run with the same spacing',xs:[82,158,234,310,386,462]},{y:310,caption:'Observe the orange bead early in the run',xs:[74,148,274,331,408,484]},{y:454,caption:'Observe that same bead later in the run',xs:[65,126,306,364,422,490]}];
for(const {y,caption,xs} of rows){
 text(setup,caption,48,y,528,36,25,C.muted);
 rect(setup,48,y+47,528,2,C.gray,C.gray);rect(setup,48,y+105,528,2,C.gray,C.gray);
 xs.forEach((x,i)=>circle(setup,'',x,y+59,34,i===2?C.gold:'#E8ECEF',i===2?C.gold:C.gray));
}
text(setup,'One run shown. Each repeat resets the starting row.',48,587,546,35,21,C.gray);
text(setup,'What the agent gets',660,166,572,37,30,C.ink,true);
text(setup,'A setup description, starter code, and data\non how far the bead typically moves\nat several observation times.',660,216,572,104,27);
text(setup,'What the agent must predict',660,346,572,37,30,C.blue,true);
text(setup,'Fit the strength of random motion.\nThen predict how the bead’s early and later\npositions vary together across many runs.',660,396,572,104,27);
text(setup,'Physics it must account for',660,514,572,37,30,C.ink,true);
text(setup,'The initial spacing can affect the connection\nbetween early and later motion.',660,559,572,64,27);
footer(setup,'Illustrative setup. The benchmark asks for the statistical trend after long observation times, rather than one bead’s exact path.');

// Keep the requested three-condition comparison but remove specialized physics wording.
const hints=old[2];await clear(hints);
text(hints,'Three prompt conditions',48,22,1184,67,48,C.ink,true);
text(hints,'The task stays the same. Only the extra guidance changes.',48,96,1184,50,25,C.muted);
table(hints,[['Condition','Instruction or added guidance'],['No hint','Use the setup and supplied data to implement the model.'],['General hint','“Check your implementation carefully and test its correctness\nbefore finishing.”'],['Physics hint','The starting arrangement affects how early and later motion\nare connected. Account for that shared history.']],48,165,1184,300,[236,948],27);
text(hints,'For the bead example',48,505,1184,37,30,C.ink,true);
text(hints,'The general hint asks the agent to check its work.\nThe physics hint points to something its formula may be missing.',48,558,1184,72,29);
footer(hints,'No-hint instruction shortened. General hint quoted exactly. Physics hint summarized in plain language from the actual task hint.');

// A concrete account of the saved failure, without the task's specialist terminology.
const failure=old[8];await clear(failure);
text(failure,'Why the agent failed on the bead task',48,22,1184,67,48,C.ink,true);
text(failure,'GPT-5.6 Terra High, no hint. The saved run finished in about 80 seconds.',48,96,1184,50,25,C.muted);
const xs=[48,358,668,978];
const labels=[['Reads the setup','Same initial spacing'],['Trusts the starter','Keeps the formula'],['Fits the data','Adjusts motion strength'],['Runs the public test','The test passes']];
const nodes=labels.map(([a,b],i)=>{const box=rect(failure,xs[i],165,254,109,i===1?'#F9E7EB':C.light,i===1?C.red:C.line);text(failure,a,xs[i]+12,180,230,32,25,i===1?C.red:C.ink,true);text(failure,b,xs[i]+12,226,230,43,21);return box;});
for(let i=0;i<3;i++)connect(failure,nodes[i],nodes[i+1]);
text(failure,'The missed assumption',48,317,570,38,30,C.red,true);
text(failure,'Its formula treats motion over a given interval\nthe same, regardless of how long ago\nthe experiment started.',48,366,570,104,27);
text(failure,'The initial spacing still matters. A good fit\nat individual times does not check the link\nbetween two different times.',48,503,570,105,27);
text(failure,'The result',678,317,554,38,30,C.ink,true);
text(failure,'Fits the supplied measurements.\nFails the early-versus-later predictions.',678,366,554,72,27);
text(failure,'Check the diagnosis',678,469,554,37,29,C.ink,true);
text(failure,'Keep the fitted motion strength.\nChange only how the formula links two times.',678,518,554,64,25);
text(failure,'136%',678,590,195,44,35,C.red,true);arrow(failure,872,605,66,C.gray);text(failure,'0.003%',970,590,262,44,35,C.teal,true);
footer(failure,'Largest prediction error, before and after our formula correction. Allowed error: 4%. This check does not count as a new agent pass.');

const conclusion=old[9];
await replaceText(conclusion,'Calibration can hide the wrong physical model','A good fit can hide the wrong physical model');
await replaceText(conclusion,'The particle example fits equal-time motion but misses memory between times.','The bead example fits the data but gets the link between two times wrong.');

// Preserve the original editable illustration and all source workbooks as backup evidence.
const detail=old[1];
await replaceText(detail,'Example: particles that cannot pass','Backup: comparing the two prediction formulas');
note(detail,0,'Technical detail for questions. The channel represents an infinite line of identical Brownian point particles with reflecting collisions and fixed initial positions j/rho in each independent repeat. The task predicts the joint long-time covariance coefficient, not finite-time trajectories or the exact position in one realization. With A = sqrt(D/pi)/rho, the correct coefficient is A[sqrt(t+s)-sqrt(abs(t-s))]. The starter uses A/sqrt(2)[sqrt(t)+sqrt(s)-sqrt(abs(t-s))]. Both agree on every equal-time setting. The plotted curves use the same D and density, normalized to the equal-time value, with earlier time multiplier 0.5 and later multiplier from 0.5 to 4. They therefore isolate the time-dependence difference. This plot is analytic, not a measured trial result. The apparent memory comes from the fixed preparation and no-passing constraint, even though the random forcing on different particles is independent. The saved Terra fit is 0.949947415037877. Its parameter error is 0.005535%, and its prediction-group errors at later/earlier ratios 2, 4 and 8 are 36.60%, 77.88% and 136.06%. Replacing only the covariance expression at the fixed fitted D reduces all tested group errors to 0.002768%.', [...physRefs,'results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/metrics.json','results/trajectory-analysis-20261014/diagnostic-repairs.json']);

const order=[cover,intro,setup,hints,old[3],old[4],old[5],old[6],old[7],failure,conclusion,old[10],old[11],old[12],old[13],detail];
const ids=[];for(const s of order)ids.push(JSON.parse(await(await s.export({format:'layout'})).text()).slide.id);P.slides.reorder(ids);
const timing=[10,45,100,60,55,70,15,55,55,100,35,180,0,0,0,0];
const finalTiming=[10,45,85,60,55,60,60,55,55,85,30];
const rewrite={
2:`PhysLiteBench asks whether a coding agent can turn a physical description and limited measurements into reliable predictions. Think of the supplied measurements as training data. They cover only some settings of the experiment. The agent receives the setup, starter code and public tests. It can change the entire model. The evaluator then checks whether the program fits the measurements and makes correct predictions at other settings, within a fixed time budget. Our suite has thirteen tasks. They were selected using GPT-5.6 Luna, so this is a deliberately challenging sample. I will explain one example using beads moving inside a narrow tube.`,
3:`Imagine a long tube containing a row of beads. Random motion makes them bump around, but the tube is too narrow for one bead to pass another. We mark one bead orange and observe it. Every repeat starts with all the beads in exactly the same evenly spaced positions. We release them, then record the orange bead's position early and later in the same run. The next run starts from the original row but has different random motion. The pictures illustrate one possible run, not measured data. What does the agent know? It has this description, starter code, and measurements of how far the orange bead typically moves at several chosen times. These data let it estimate how strong the random motion is. What must it predict? Across many repeats, how strongly are the orange bead's early and later positions connected? For example, in runs where it is farther right early, does it also tend to be farther right later, and how strong is that relationship? The agent must write a numerical prediction for this relationship. The important physical detail is how we start the experiment. The beads begin in a regular row and cannot pass one another. That arrangement can still influence how motion at one time relates to motion later. Fitting the movement strength alone does not settle that question.`,
4:`There are three prompt conditions. The baseline gives the original task and public files. The general hint asks the agent to check its implementation and test correctness. It does not identify which assumption might need attention. The physics hint points to a specific feature of the experiment. For the beads, it emphasizes that the same starting arrangement can affect the relationship between positions measured at two times. The full hint also suggests how to calculate this relationship. The slide summarizes its physical idea in everyday language. The hint supplies a direction for thinking, not the final formula. We keep the task files and grading the same across these conditions.`,
10:`Now return to the orange bead. This saved Terra High run lets us trace what the agent actually did. It read the description, accepted the prediction formula in the starter, fitted the strength of the random motion, and passed the supplied test. The parameter fit is very accurate. The error occurs in the red box: it keeps a formula whose assumption does not match how the experiment begins. That formula treats the amount of motion over a given time interval the same whether the interval occurs earlier or later after release. In this experiment, the regular starting row still affects later motion. The agent even mentions the fixed starting positions in its comments, but does not change this part of its calculation. Why did the public test not catch it? The supplied measurements test the spread at individual times. Both formulas can fit those measurements. The hidden tests ask how positions at two different times are connected, and that is where the retained formula is wrong. The largest prediction-group error is about one hundred thirty-six percent, above the four-percent limit. To check our diagnosis, we kept the agent's fitted movement strength and replaced only the way the formula connects two times. The largest error then fell to about zero point zero zero three percent. This is our diagnostic correction to the saved program. The original agent attempt still counts as a failure.`,
11:`A good fit to the supplied measurements does not guarantee a correct new prediction. The bead example makes that concrete: the agent learns the strength of random motion but keeps the wrong rule for linking early and later positions. Across the suite, model cost, reasoning effort and timely completion all affect success. The remaining hint experiment will show whether pointing to a missing physical idea helps more than simply asking the agent to check its work. Its results remain pending.`
};
const refs={2:['README.md','docs/BENCHMARK.md','results/task-roster.json'],3:physRefs,4:['results/hint-comparison-20261014/plan.json','tasks/single-file-memory/hint.md'],10:[...physRefs,'results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/review.json','results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/model.diff','results/trajectory-analysis-20261014/diagnostic-repairs.json'],11:['results/reasoning-sweep-20261011/summary.csv','results/trajectory-analysis-20261014/analysis.json','results/hint-comparison-20261014/plan.json']};
for(const [i,s] of order.entries()){
 const n=i+1;
 if(rewrite[n])note(s,timing[i],rewrite[n],refs[n]);
 else s.speakerNotes.text=String(s.speakerNotes.text).replace(/^Target time: [^\n]+/,`Target time: ${timing[i]?timing[i]+' seconds':'skip in current talk'}.`);
 const l=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());
 const pages=l.elements.filter(e=>e.kind==='textbox'&&e.position.top>=675&&/^\d+$/.test(e.text));
 if(pages.length)P.delete(pages.map(e=>e.id));
 if(n!==1&&n!==12)text(s,String(n),1176,680,56,20,16,C.gray);
}
if(timing.slice(0,11).reduce((a,b)=>a+b,0)!==600||finalTiming.reduce((a,b)=>a+b,0)!==600)throw Error('Timing');
await fs.writeFile(B+'/slide-map.json',JSON.stringify({source:INPUT,order:order.map((s,i)=>({slide:i+1,title:String(s.shapes.items[0].text),seconds:timing[i]})),content_slides:10,title_slide:1,total_slides:16,example_slides:[3,4,10],hint_results_slide:7,experiment_slide:5,questions_slide:12,backup_slides:[13,14,15,16],timing_current_seconds:timing.slice(0,11),timing_final_seconds:finalTiming},null,2));
await fs.mkdir(B+'/previews',{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/previews/'+stem+'.png',new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/previews/'+stem+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
const raw=B+'/candidate-export.pptx',candidate=B+'/candidate.pptx';
await(await PresentationFile.exportPptx(P)).save(raw);
execFileSync(PY,[B+'/preserve_workbooks.py',INPUT,raw,candidate,B+'/workbook-preservation.json'],{stdio:'inherit'});
const name=process.env.DECK_FILENAME||'PhysLiteBench-10-minute-plain-language.pptx';
const final=W+'/output/physlitebench-presentation/'+name;
const tables=[2,4,5,6,13,14];
const result=await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:S+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:S+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[6,8,9,16],requiredEmbeddedWorkbookChartOwnerSlides:[6,8,9,16],materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'user_request',families:[FONT]},verifyArtifactToolImport:true,receiptPath:B+'/'+name+'.validation.json'});
await fs.writeFile(B+'/finalization-result.json',JSON.stringify(result,null,2));
const F=await PresentationFile.importPptx(await FileBlob.load(final));
await fs.mkdir(B+'/final-previews',{recursive:true});
for(const [i,s] of F.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/final-previews/'+stem+'.png',new Uint8Array(await(await F.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/final-previews/'+stem+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
console.log(JSON.stringify({final,slides:16,seconds:600,questions:180}));
