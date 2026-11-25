import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const R='/Users/yp9846/Desktop/PhysLiteBench';
const B=path.join(W,'.presentation-cs-20261015');
const S='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const INPUT=R+'/output/presentation-20261014/PhysLiteBench-10-minute-physics-example.pptx';
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

// One overview slide combines the title, background, and benchmark definition.
const intro=old[1];
await replaceText(intro,'Physics beyond the calibration regime','PhysLiteBench: physics beyond a good fit');
await replaceText(intro,'A model must fit the apparatus and predict behavior that calibration does not identify.','Can a coding agent infer the right physical model from a description and limited data?');
await replaceText(intro,'tasks spanning stochastic dynamics, transport,\ncorrelations, and quantum or statistical systems','tasks: random motion, heat and particle flow,\ncorrelations, and quantum systems');
const introTable=intro.tables.items[0];
introTable.getCell(1,0).value='Physical setup and starter code';introTable.getCell(1,1).value='Fit to observed data and model parameters';
introTable.getCell(2,0).value='Calibration data and public tests';introTable.getCell(2,1).value='Predictions beyond the calibration settings';
await replaceText(intro,'The roster was selected using GPT-5.6 Luna. Selection results and later evaluation results are separate.','Yue Pan, October 2026. Tasks were selected using GPT-5.6 Luna, so this is a deliberately challenging suite.');

// The user-requested apparatus schematic uses editable diagram objects.
const example=base('Example: particles that cannot pass','A narrow channel preserves particle order. Every experiment starts from the same positions.');
text(example,'Physical setup',48,161,515,34,29,C.ink,true);
text(example,'Reset to evenly spaced positions',48,207,515,32,25,C.muted);
rect(example,48,250,515,2,C.gray,C.gray);rect(example,48,310,515,2,C.gray,C.gray);
const start=[86,162,238,314,390,466];
start.forEach((x,i)=>circle(example,i===2?'●':'',x,263,36,i===2?C.gold:'#E8ECEF',i===2?C.gold:C.gray));
text(example,'Let the particles move randomly',48,350,515,32,25,C.muted);
rect(example,48,392,515,2,C.gray,C.gray);rect(example,48,452,515,2,C.gray,C.gray);
[74,145,279,334,408,484].forEach((x,i)=>circle(example,i===2?'●':'',x,405,36,i===2?C.gold:'#E8ECEF',i===2?C.gold:C.gray));
text(example,'Track the orange particle in each repeat.\nParticles collide but never overtake.',48,480,520,70,26);
text(example,'Why fitting alone is insufficient',650,161,582,34,29,C.ink,true);
const ratios=Array.from({length:57},(_,i)=>1+7*i/56);
const ref=ratios.map(r=>Number(((Math.sqrt(1+r)-Math.sqrt(r-1))/Math.sqrt(2)).toFixed(9)));
const shortcut=ratios.map(r=>Number(((1+Math.sqrt(r)-Math.sqrt(r-1))/2).toFixed(9)));
chart(example,'scatter',{position:{left:641,top:207,width:591,height:330},hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:21}},scatterOptions:{style:'line'},series:[{name:'Starter formula',xValues:ratios,values:shortcut,line:{fill:C.red,width:2.7},marker:{symbol:'none'}},{name:'Correct physics',xValues:ratios,values:ref,line:{fill:C.teal,width:2.7},marker:{symbol:'none'}}],xAxis:{min:1,max:8,majorUnit:1,title:'Later time / earlier time',textStyle:{typeface:FONT,fontSize:20},majorGridlines:null},yAxis:{min:0,max:1,majorUnit:.25,numberFormatCode:'0.00',title:'Shared motion (normalized)',textStyle:{typeface:FONT,fontSize:20},majorGridlines:{fill:C.line,width:.8,style:'solid'}}});
text(example,'Same time: both fit.\nDifferent times: memory changes the prediction.',650,551,582,68,26,C.ink,true);
text(example,'Agent task: fit the movement scale, then\npredict how two measurement times are related.',48,563,542,61,24,C.blue,true);
footer(example,'Schematic and analytic curves, not measured trajectories. “Shared motion” means covariance. The task asks for its long-time scaling coefficient.');

// Conditions are concrete, with the actual generic prompt and a plain-language physics paraphrase.
const hints=base('Three prompt conditions','The same example shows the difference between checking code and checking the physical assumption.');
table(hints,[['Condition','Instruction or added guidance'],['No hint','Implement the model using the setup and calibration data.'],['General hint','“Check your implementation carefully and test its correctness\nbefore finishing.”'],['Physics hint','Fixed starting positions leave a memory. Count crossings of a\nfixed location and retain their correlation between two times.']],48,165,1184,302,[236,948],26);
text(hints,'What the physics hint changes in this example',48,500,1184,36,29,C.ink,true);
const hn=[rect(hints,48,555,320,58,C.light,C.line),rect(hints,480,555,320,58,C.light,C.line),rect(hints,912,555,320,58,C.light,C.line)];
['Identical starting positions','Same path at two times','Preparation affects memory'].forEach((t,i)=>text(hints,t,[61,493,925][i],572,294,34,23,C.ink,false,'center'));
connect(hints,hn[0],hn[1]);connect(hints,hn[1],hn[2]);
footer(hints,'No-hint instruction shortened. General hint quoted exactly. Physics hint paraphrased from the frozen task prompt.');

// Experimental detail is shown after the task and hint design.
const experiment=old[4];await clear(experiment);
text(experiment,'Experiment and evaluation',48,22,1184,67,48,C.ink,true);
text(experiment,'13 frozen tasks, 3 repeats per setting. A pass requires all grading gates and timely completion.',48,96,1184,50,25,C.muted);
table(experiment,[['Study','Models and reasoning settings','Planned observations'],['No hint','GPT-6 Luna / Sol / Astra; GPT-5.6 Terra\nLow, High, Extra High\nDeepSeek V4.1 Flash: Low, High, Max','15 settings × 39 = 585'],['Hint comparison','GPT-5.6 Luna / Terra; DeepSeek V4.1 Flash\nHigh only, with 3 prompt conditions','9 settings × 39 = 351']],48,164,1184,292,[223,653,308],25);
text(experiment,'Same task files and grading across prompt conditions',48,488,1184,36,29,C.ink,true);
text(experiment,'Harbor 0.21.0 + Codex agent   /   600 s agent   /   60 s verifier',48,541,1184,36,26);
text(experiment,'Cost axis: estimated API cost per attempt. Execution errors count as non-passes.',48,590,1184,34,24,C.muted);
footer(experiment,'38 Luna hint-study baselines are historical: client 0.154.0 / concurrency 4 versus local 0.160.0 / 3. Hint differences are descriptive.');

const pareto=old[5];await replaceText(pareto,'Observed cost–success frontier','No-hint Pareto frontier');
await replaceText(pareto,'Each point contains 39 attempts. The frontier uses complete panels with reconciled costs.','Upper left is better: more passes at lower cost. Each point contains 39 attempts.');

// Leave results blank until the entire hint comparison is complete and audited.
const hintResults=base('Pareto frontiers with hints','No hint, general hint, and physics hint, with reasoning effort fixed at High.');
text(hintResults,'Pending completion',48,187,1184,65,38,C.muted,true);
text(hintResults,'Reserved for the completed 3-condition cost–success comparison',48,286,1184,48,29);
text(hintResults,'GPT-5.6 Luna, GPT-5.6 Terra, DeepSeek V4.1 Flash',48,361,1184,40,27,C.muted);
text(hintResults,'39 observations per model and prompt condition',48,424,1184,40,27,C.muted);
text(hintResults,'The final plot will compare these conditions within this cohort.',48,533,1184,48,27,C.blue);
footer(hintResults,'Results stay blank until all 351 observations and cost records are audited. Historical Luna controls limit causal interpretation.');

const effort=old[6];await replaceText(effort,'More reasoning does not guarantee more passes','Fixed model, different reasoning effort');
const failure=old[7];await replaceText(failure,'The failure modes differ sharply by model','Where the agents fail');
await replaceText(failure,'Recorded tool actions, saved programs, and verifier outcomes explain different bottlenecks.','585 no-hint attempts. Saved actions, submitted code, and verifier results separate the failure types.');

const trajectory=base('The example again: the wrong memory rule','GPT-5.6 Terra High, no hint, 80 seconds. Evidence from the saved actions and submitted code.');
const sx=[48,358,668,978],sw=[254,254,254,254];
const labels=[['Read the setup','Fixed initial positions'],['Keep starter formula','Memory assumption fails'],['Fit movement scale','Parameter error 0.0055%'],['Run public test','1 test passed']];
const sn=labels.map(([a,b],i)=>{const node=rect(trajectory,sx[i],165,sw[i],107,i===1?'#F9E7EB':C.light,i===1?C.red:C.line);text(trajectory,a,sx[i]+12,180,sw[i]-24,33,25,i===1?C.red:C.ink,true);text(trajectory,b,sx[i]+12,226,sw[i]-24,35,21);return node;});
for(let i=0;i<3;i++)connect(trajectory,sn[i],sn[i+1]);
text(trajectory,'The incorrect assumption',48,311,574,36,29,C.red,true);
text(trajectory,'Shifting the measurement window leaves\nthe statistics of position changes unchanged.',48,359,575,72,27);
text(trajectory,'But the repeated fixed starting positions\nleave a memory of the release time.',48,455,575,70,27);
text(trajectory,'The agent names the fixed preparation\nin its comments, but retains the wrong formula.',48,550,575,68,24,C.muted);
text(trajectory,'Hidden prediction error',678,311,554,36,29,C.ink,true);
table(trajectory,[['Later / earlier time','Group error'],['1 (equal times)','0.0028%'],['2 / 4 / 8','36.60% / 77.88% / 136.06%']],678,357,554,157,[237,317],23);
text(trajectory,'Replace only the memory formula',678,537,554,31,26,C.ink,true);
text(trajectory,'136.06%',678,580,195,46,35,C.red,true);arrow(trajectory,886,594,65,C.gray);text(trajectory,'0.0028%',976,580,256,46,35,C.teal,true);
footer(trajectory,'Analyst diagnostic with the fitted parameter fixed. Largest group error falls below the 4% tolerance. The original attempt remains a failure.');

const conclusion=old[11];await clear(conclusion);
text(conclusion,'Conclusions',48,22,1184,67,48,C.ink,true);
text(conclusion,'Calibration can hide the wrong physical model',64,169,1168,52,34,C.ink,true);
text(conclusion,'The particle example fits equal-time motion but misses memory between times.',64,233,1168,66,29,C.muted);
text(conclusion,'Cost, reasoning effort, and completion all matter',64,342,1168,52,34,C.ink,true);
text(conclusion,'Sol 6 Low and High reach the high-success end of the observed no-hint frontier.',64,406,1168,66,29,C.muted);
text(conclusion,'The hint comparison is the remaining experiment',64,515,1168,52,34,C.ink,true);
text(conclusion,'We will test whether targeted physical guidance improves success per dollar.',64,579,1168,48,29,C.muted);

const order=[intro,example,hints,experiment,pareto,hintResults,effort,failure,trajectory,conclusion,old[12],old[13],old[14],old[15]];
const keep=new Set(order);P.slides.keep(P.slides.items.map((s,i)=>i).filter(i=>keep.has(P.slides.items[i])));
const ids=[];for(const s of order)ids.push(JSON.parse(await(await s.export({format:'layout'})).text()).slide.id);P.slides.reorder(ids);
const timing=[50,100,65,55,75,15,55,55,95,35,180,0,0,0];
const finalTiming=[50,85,65,55,65,60,55,55,80,30];
const notes=[
`PhysLiteBench studies whether a coding agent can build a physical model, rather than only fit the data it is given. Think of calibration as training data: it measures the system under a limited set of conditions. The hidden tests ask for predictions at other settings, where an incorrect physical assumption can become visible. Each agent receives a description of the physical setup, starter code, measurements with known noise, and public tests. It can replace the predictor as well as fit the unknown parameters. The evaluator checks calibration, parameter accuracy, hidden predictions, and completion within a fixed time budget. Our frozen suite has thirteen tasks. We selected them using GPT-5.6 Luna, so the results describe a deliberately challenging suite rather than a random sample of physics. I will use one particle experiment throughout the talk.`,
`Imagine beads in a channel so narrow that they cannot pass. The actual mathematical model uses point particles on an infinite line, but the channel picture captures the constraint. Each independent experimental repeat begins from exactly the same evenly spaced positions. We release the particles and track the orange one as random thermal motion moves it around. The order of the particles never changes. The unknown parameter D sets the movement scale. The calibration measures the orange particle's spread at a single time. The hidden task measures how its positions at two different times vary together across repetitions, called covariance or shared motion here. This is about statistical memory, not whether an individual trajectory looks plausible. The plot compares two formulas using the same fitted movement scale. They agree exactly when the two measurement times coincide. That is all calibration probes. When the times differ, the starter predicts substantially more shared motion than the correct model. The difficulty is that fixed starting positions leave a memory of the release time. The starter instead assumes stationary increments: the statistics of a displacement over a time window depend only on that window's length. An agent can therefore fit every calibration setting and still keep the wrong memory model. The curves show analytic normalized long-time coefficients, not finite-time measurements or simulated trajectories. Here the earlier multiplier is 0.5 and the later one ranges from 0.5 to 4, within the documented domain.`,
`We compare three prompt conditions while holding model and configured effort fixed. With no hint, the agent gets the original implementation request and public files. The general hint is exactly the sentence shown: check the implementation and test correctness before finishing. It asks for more care, without pointing to a particular physical mechanism. The physics hint is task-specific. For the particles, it says that fixed preparation does not imply stationary increments and suggests deriving the two-time covariance from correlated crossings of independent Brownian paths. In plain terms, the same random path contributes to both measurements, so the preparation and shared history matter. The row paraphrases that cue for this audience. It does not provide the final covariance formula or hidden answers. The little flow diagram is the conceptual relationship the hint asks the agent to consider. We have not yet established that the hint improves outcomes.`,
`There are two distinct experiment cohorts. The completed no-hint sweep has fifteen model-effort settings, each covering thirteen tasks and three repeats, or five hundred eighty-five attempts. The hint study fixes effort at High for GPT-5.6 Luna, GPT-5.6 Terra, and DeepSeek V4.1 Flash. Nine model-condition settings make three hundred fifty-one logical observations. Those are different Luna generations, so their results should not be silently pooled. Both studies use the Harbor harness and a Codex agent, with ten minutes for the agent and one minute for verification. The files and grading stay frozen across the hint conditions. Thirty-eight Luna baselines were imported from historical matching-task runs with an older client and different concurrency. Thus the Luna hint comparison is descriptive and does not isolate a causal hint effect. Costs are API-equivalent estimates, not subscription usage.`,
`This is the observed no-hint cost-success frontier. Every point contains thirty-nine attempts. Higher is better success, and left is lower estimated cost. A Pareto point is one that no other eligible setting beats on both quantities. The dashed line connects those non-dominated settings. GPT-6 Luna Low passes four of thirty-nine at about four tenths of a cent per attempt. Luna Extra High passes twelve at about one point three cents. GPT-6 Sol Low passes thirty-seven at about seven point seven cents. Sol High passes all thirty-nine at about nineteen point eight cents. The exact coordinates in the table help distinguish the low-cost points clustered near the origin. More expensive settings do not exceed that maximum observed score. DeepSeek costs are lower bounds because interrupted requests can have incomplete accounting, so those points do not determine the exact-cost frontier. This comparison describes the selected tasks and fixed runtime budget. Three repeats per task are too few to infer a universal model ranking.`,
`This slot will compare the three prompt conditions in the High-effort cohort. The full hint study and cost audit are still pending, so I am leaving the results blank.`,
`Here we hold each model fixed and compare its effort settings on the same task suite. Terra increases from four to nine to nineteen completed passes. Luna rises substantially at Extra High, although its Low-to-High change goes the other way. Sol already succeeds on almost every attempt at Low. The Astra Extra High setting loses two completed passes because of timeouts. Both saved checkpoints pass the verifier afterward, but completion within the allowed budget is part of the score. These are descriptive comparisons: each point has only three repeats per task, and the High settings include earlier matching observations. Configuring more reasoning can improve a model's predictions, but it can also consume its execution budget, so completed success need not increase monotonically.`,
`The trajectory analysis combines recorded tool actions, final programs, and verifier outcomes. It does not expose private reasoning. Each bar contains one hundred seventeen attempts across three efforts for that model. Of one hundred ninety-one normally completed failures, one hundred eighty-six fail only hidden prediction checks. At least one hundred forty-one of those also show public-test passes in the recorded actions. This is the calibration-to-prediction gap illustrated by the particle task. DeepSeek has a different dominant bottleneck: one hundred five of its one hundred seventeen attempts end with execution errors, including eighty-three timeouts. A zero overall pass rate therefore cannot be interpreted as one hundred seventeen physics mistakes. The aggregate categories tell us where failures appear. A saved program and a targeted intervention are needed to diagnose a particular physical mistake.`,
`Return to the particles. This is one saved GPT-5.6 Terra High attempt without a hint, lasting about eighty seconds. The recorded actions show it reading the setup, accepting the starter covariance, fitting the movement scale, and running the public test. At step eight it writes, "The supplied covariance expression matches the deterministic (quenched) single-file limit." The source diff shows that it adds input validation and comments naming the fixed preparation but retains exactly the original time dependence. That is the red box in the flowchart. Its fitted movement parameter is accurate to about 0.0055 percent, and its public test passes. Equal-time hidden anchors are also accurate. Yet the errors grow to roughly thirty-seven, seventy-eight, and one hundred thirty-six percent when the later measurement is two, four, and eight times the earlier one. The four-percent prediction threshold is exceeded in each of these groups. This matches the task's intended trap: a familiar stationary-increment formula can reproduce all equal-time calibration while missing preparation-dependent memory. To test the diagnosis, the analyst replaces only the covariance formula in an in-memory copy and keeps the fitted parameter fixed. The largest group error falls to about 0.0028 percent. This intervention supports the identified source-level omission. It is not a hint-study outcome, and the original agent attempt still counts as a failure.`,
`The particle example shows why calibration accuracy alone cannot establish that a physical predictor is correct. The aggregate results reveal substantial differences in success per dollar, while the effort comparison shows that more reasoning also interacts with a fixed execution budget. The saved Terra program gives a concrete failure mechanism: the parameter fit is excellent, but the model retains the wrong temporal memory. The remaining experiment asks whether a targeted physics cue changes these outcomes more than a general request to check the code. That result stays open until the hint study is complete.`
];
const refs=[['README.md','docs/BENCHMARK.md','results/task-roster.json'],physRefs,['results/hint-comparison-20261014/plan.json','tasks/single-file-memory/hint.md'],['results/reasoning-sweep-20261011/README.md','results/hint-comparison-20261014/plan.json','results/hint-comparison-20261014/friend-reuse-amendment-20261015.json'],['results/reasoning-sweep-20261011/summary.csv','output/reasoning-sweep-20261011/pareto-frontier.json'],['results/hint-comparison-20261014/status.json','results/hint-comparison-20261014/plan.json'],['results/reasoning-sweep-20261011/summary.csv','results/trajectory-analysis-20261014/analysis.json'],['results/trajectory-analysis-20261014/analysis.json','output/trajectory-analysis-20261014/report.html'],['results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/review.json','results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/model.diff','results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/metrics.json','results/trajectory-analysis-20261014/diagnostic-repairs.json','jobs/pareto-gpt-5.6-terra-single-file-memory-20261011-135517/single-file-memory__PC6LumP/agent/trajectory.json',...physRefs],['results/reasoning-sweep-20261011/summary.csv','results/trajectory-analysis-20261014/analysis.json','results/hint-comparison-20261014/plan.json']];
for(let i=0;i<10;i++)note(order[i],timing[i],notes[i],refs[i]);
for(const [i,s] of order.entries()){
 const layout=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());
 const pages=layout.elements.filter(e=>e.kind==='textbox'&&e.position.top>=675&&/^\d+$/.test(e.text));
 pages.forEach(e=>P.resolve(e.id).text=String(i+1));
 if(i!==10&&!pages.length)text(s,String(i+1),1176,680,56,20,16,C.gray);
 if(i>=10)s.speakerNotes.text=String(s.speakerNotes.text).replace(/^Target time: [^\n]+/,`Target time: ${timing[i]?timing[i]+' seconds':'skip in current talk'}.`);
}
if(timing.slice(0,10).reduce((a,b)=>a+b,0)!==600||finalTiming.reduce((a,b)=>a+b,0)!==600)throw Error('Timing');
await fs.writeFile(B+'/slide-map.json',JSON.stringify({source:INPUT,order:order.map((s,i)=>({slide:i+1,title:String(s.shapes.items[0].text),seconds:timing[i]})),main_slides:10,total_slides:14,example_slides:[2,3,9],hint_results_slide:6,experiment_slide:4,questions_slide:11,backup_slides:[12,13,14],timing_current_seconds:timing.slice(0,10),timing_final_seconds:finalTiming},null,2));
await fs.writeFile(B+'/speaker-notes.json',JSON.stringify(notes.map((body,i)=>({slide:i+1,seconds:timing[i],body,sources:refs[i]})),null,2));
await fs.mkdir(B+'/previews',{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/previews/'+stem+'.png',new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/previews/'+stem+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
const raw=B+'/candidate-export.pptx',candidate=B+'/candidate.pptx';
await(await PresentationFile.exportPptx(P)).save(raw);
execFileSync(PY,[B+'/preserve_workbooks.py',INPUT,raw,candidate,B+'/workbook-preservation.json'],{stdio:'inherit'});
const name=process.env.DECK_FILENAME||'PhysLiteBench-10-minute-CS-audience.pptx';
const final=W+'/output/physlitebench-presentation/'+name;
const tables=[1,3,4,5,9,12,13];
const result=await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:S+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:S+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[2,5,7,8],requiredEmbeddedWorkbookChartOwnerSlides:[5,7,8],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'user_request',families:[FONT]},verifyArtifactToolImport:true,receiptPath:B+'/'+name+'.validation.json'});
await fs.writeFile(B+'/finalization-result.json',JSON.stringify(result,null,2));
const F=await PresentationFile.importPptx(await FileBlob.load(final));
await fs.mkdir(B+'/final-previews',{recursive:true});
for(const [i,s] of F.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/final-previews/'+stem+'.png',new Uint8Array(await(await F.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/final-previews/'+stem+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
console.log(JSON.stringify({final,slides:14,seconds:600,questions:180}));
