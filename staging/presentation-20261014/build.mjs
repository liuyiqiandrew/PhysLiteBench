import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, Presentation, PresentationFile } from '@oai/artifact-tool';
import { applyPresentationChartFont, finalizePresentation } from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';

const WORK='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const BUILD=path.join(WORK,'.presentation-build-20261014');
const OUT=path.join(WORK,'output/physlitebench-presentation');
const SKILL='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const D=JSON.parse(await fs.readFile(path.join(BUILD,'data.json'),'utf8'));
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#FCFBF8',ink:'#182B35',muted:'#596A73',teal:'#007E80',blue:'#406EA6',gold:'#B57C18',red:'#AE4E45',purple:'#79648E',grid:'#DFE4E4',white:'#FFFFFF'};
const FONT='Arial';
const modelNames={'gpt-6-luna':'GPT-6 Luna','gpt-6-sol':'GPT-6 Sol','gpt-5.6-terra':'GPT-5.6 Terra','gpt-6-astra':'GPT-6 Astra','deepseek/deepseek-v4.1-flash':'DeepSeek V4.1 Flash'};
const shortNames={'gpt-6-luna':'Luna 6','gpt-6-sol':'Sol 6','gpt-5.6-terra':'Terra 5.6','gpt-6-astra':'Astra 6','deepseek/deepseek-v4.1-flash':'DeepSeek'};
const colors={'gpt-6-luna':C.blue,'gpt-6-sol':C.teal,'gpt-5.6-terra':C.gold,'gpt-6-astra':C.purple,'deepseek/deepseek-v4.1-flash':C.red};
const efforts={low:'Low',medium:'Medium',high:'High',xhigh:'Extra High',max:'Max',none:'None'};
const notes=[];
const source=(f)=>path.join(D.repo,f);
function text(s,txt,x,y,w,h,size=28,color=C.ink,bold=false){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=txt; sh.text.style={typeface:FONT,fontSize:size,color,bold,wrap:'square',verticalAlignment:'top',insets:0};return sh;
}
function base(title,sub=''){
 const s=P.slides.add();s.background.fill=C.bg;
 text(s,title,64,44,1152,66,43,C.ink,true);
 if(sub) text(s,sub,66,119,1148,50,23,C.muted);
 text(s,String(P.slides.items.length).padStart(2,'0'),1174,677,42,22,16,C.muted);
 return s;
}
function note(s,seconds,body,refs=[]){
 const n={slide:P.slides.items.length,seconds,body,sources:refs};notes.push(n);
 s.speakerNotes.text=`Target time: ${seconds===0?'skip in current talk':seconds+' seconds'}.\n\n${body}\n\nSources:\n${refs.map(source).join('\n')}`;
}
function foot(s,t){text(s,t,66,634,1128,40,17,C.muted);}
function table(s,values,x,y,width,height,widths,size=22){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,values,columnWidths:widths});
 t.borders.assign({fill:C.bg,width:0,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?C.white:C.bg);
  cell.text.style={typeface:FONT,fontSize:size,color:r===0?C.white:C.ink,bold:r===0,insets:10};
 }
 return t;
}
function chart(s,type,config){
 const ch=s.charts.add(type,{chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},...config});
 applyPresentationChartFont(ch,{fontFamily:FONT});
 for(let i=0;i<(config.series?.length||0);i++){ const labels=ch.series.getItemAt(i).dataLabels; if(labels?.textStyle) labels.textStyle.typeface=FONT; }
 return ch;
}
const axisText={fontSize:19,fill:C.muted};
const grid={fill:C.grid,width:1,style:'solid'};

// 1. Title
{
 const s=P.slides.add();s.background.fill=C.ink;
 text(s,'PhysLiteBench',72,180,1100,110,78,C.white,true);
 text(s,'Cost, reasoning effort, and physics failure modes',76,322,1040,100,35,'#D6E4E6');
 text(s,'13 frozen tasks',76,509,650,45,25,'#8DD3CE');
 text(s,'Yue Pan   /   October 2026',76,593,900,35,23,'#D6E4E6');
 note(s,20,'This study asks how much reliable physics coding we get for the model compute we spend. I will show the cost–success frontier on thirteen frozen tasks, then use saved tool actions and final programs to locate failure modes. A separate experiment is testing whether general and physics hints change these outcomes.', ['results/reasoning-sweep-20261011/README.md']);
}
// 2. Benchmark setup
{
 const s=base('Physics beyond the calibration regime','A model must fit the apparatus and predict behavior that calibration does not identify.');
 table(s,[['Available to the agent','What the evaluator checks'],['Apparatus description and starter code','Calibration fit and fitted parameters'],['Public calibration data and tests','Predictions in held-out physical regimes'],['A fixed software interface','Completion within the time budget']],66,207,1146,285,[560,586],26);
 text(s,'13',70,526,120,75,58,C.teal,true);
 text(s,'tasks spanning stochastic dynamics, transport,\ncorrelations, and quantum or statistical systems',211,535,940,75,27);
 foot(s,'The roster was selected using GPT-5.6 Luna. Selection results and later evaluation results are separate.');
 note(s,55,'The benchmark separates parameter fitting from physical extrapolation. The agent receives an apparatus description, calibration observations, a starter program and public tests. The hidden evaluator checks the fit and parameter, but also asks for behavior beyond the calibration regime. A successful numerical fit can therefore coexist with an incorrect physical model. The thirteen tasks include transport, stochastic dynamics, constrained fluctuations and quantum or statistical systems. Private tests, reference solutions and physics hints stay outside the ordinary agent image. One important limitation is task selection: the roster was developed against GPT-5.6 Luna. We should interpret this as a deliberately challenging selected suite, not a random sample of all physics problems. A historical zero out of three selection result is also not a permanent guarantee of failure.', ['AGENTS.md','results/task-roster.json','TASKS.md']);
}
// 3. Protocol
{
 const s=base('The no-hint sweep','15 complete model–effort settings, each with 13 tasks and 3 repeats.');
 table(s,[['Model','Reasoning settings'],['GPT-6 Luna, Sol, Astra','Low, High, Extra High'],['GPT-5.6 Terra','Low, High, Extra High'],['DeepSeek V4.1 Flash','Low, High, Max']],66,190,1146,278,[620,526],25);
 text(s,'585 completed-panel attempts',70,510,850,48,36,C.teal,true);
 text(s,'600 s agent limit   /   60 s verifier limit   /   Codex 0.160.0',70,570,1110,34,25);
 foot(s,'Three DeepSeek None attempts form a separate partial panel. OpenAI High includes 30 reused observations per model.');
 note(s,40,'The complete comparison has fifteen settings and thirty-nine started attempts per setting, for five hundred eighty-five attempts. There are another three DeepSeek None observations, which are excluded from the complete-panel plots. All complete settings use the same Codex client and the same ten-minute agent and one-minute verifier budgets. Low and Extra High are fresh cohorts. The OpenAI High settings reuse thirty matching trials from the earlier ten-task comparison and add nine for the new tasks. The primary success metric requires normal completion and passing verification. A passing checkpoint saved before a timeout does not change that score. Costs are token-based API-equivalent estimates, not subscription charges.', ['results/reasoning-sweep-20261011/README.md','results/reasoning-sweep-20261011/summary.csv']);
}
// 4. Pareto chart. All 15 points are shown; lower-bound DeepSeek costs never enter frontier.
{
 const s=base('Observed cost–success frontier','Each point contains 39 attempts. The frontier uses complete panels with reconciled costs.');
 const rows=D.sweep.filter(r=>r.complete_panel);
 const series=rows.map(r=>({name:`${modelNames[r.model]} ${efforts[r.reasoning_effort]}`,xValues:[Number(r.mean_estimated_api_usd.toFixed(9))],values:[Number(r.pass_rate.toFixed(9))],line:{fill:'none',width:0},marker:{symbol:{low:'circle',high:'square',xhigh:'triangle',max:'diamond'}[r.reasoning_effort],size:12,fill:colors[r.model]}}));
 const f=D.frontier.points.map(p=>rows.find(r=>r.model===p.model&&r.reasoning_effort===p.effort));
 series.unshift({name:'Observed frontier',xValues:f.map(r=>Number(r.mean_estimated_api_usd.toFixed(9))),values:f.map(r=>Number(r.pass_rate.toFixed(9))),line:{fill:C.teal,width:2,style:'dashed'},marker:{symbol:'none'}});
 chart(s,'scatter',{position:{left:65,top:190,width:760,height:411},series,scatterOptions:{style:'lineWithMarkers'},hasLegend:false,xAxis:{title:'Estimated API cost per attempt (USD)',min:0,max:1.15,majorUnit:.2,numberFormatCode:'$0.00',textStyle:axisText,majorGridlines:null},yAxis:{title:'Completed pass rate',min:0,max:1.06,majorUnit:.25,numberFormatCode:'0%',textStyle:axisText,majorGridlines:grid}});
 text(s,'Frontier settings',858,190,345,35,26,C.teal,true);
 table(s,[['Setting','Pass','USD'],['Luna 6 Low','4/39','.0040'],['Luna 6 Extra High','12/39','.0126'],['Sol 6 Low','37/39','.0773'],['Sol 6 High','39/39','.1975']],850,235,362,237,[190,83,89],19);
 let yy=495;for(const m of ['gpt-6-luna','gpt-6-sol','gpt-5.6-terra','gpt-6-astra','deepseek/deepseek-v4.1-flash']){text(s,modelNames[m],860,yy,350,24,18,colors[m],true);yy+=24;}
 foot(s,'Circle = Low, square = High, triangle = Extra High, diamond = Max. DeepSeek costs are lower bounds.');
 note(s,90,'The vertical axis is the fraction of completed passes and the horizontal axis is estimated API cost per attempt. Upper left is desirable. The dashed curve connects settings that no other eligible setting beats on both cost and observed success. Four settings lie on this observed frontier. GPT-6 Luna Low costs about four tenths of a cent per attempt and passes four of thirty-nine. Luna Extra High increases that to twelve passes for about one point three cents. GPT-6 Sol Low passes thirty-seven at about seven point seven cents. Sol High reaches thirty-nine at about nineteen point eight cents. Sol Extra High and Astra cost more without improving the observed score over Sol High on this suite. The low-cost Luna points cluster at the left edge, so the adjacent table gives their exact coordinates. DeepSeek is shown for context, but its interrupted requests make these costs lower bounds, and those points do not enter the exact-cost frontier. This is an empirical frontier on a selected suite, not a universal model ranking. Three repeats per task give limited precision, and normal completion is part of the metric.', ['results/reasoning-sweep-20261011/summary.csv','output/reasoning-sweep-20261011/pareto-frontier.json','results/reasoning-sweep-20261011/README.md']);
}
// 5. Effort
{
 const s=base('More reasoning does not guarantee more passes','Completed passes out of 39, with model and task suite held fixed.');
 chart(s,'line',{position:{left:72,top:195,width:800,height:410},categories:['Low','High','Extra High'],lineOptions:{smooth:false},series:['gpt-6-luna','gpt-5.6-terra','gpt-6-sol','gpt-6-astra'].map(m=>({name:modelNames[m],smooth:false,values:['low','high','xhigh'].map(e=>D.sweep.find(r=>r.model===m&&r.reasoning_effort===e).passes),line:{fill:colors[m],width:3},marker:{symbol:'circle',size:9,fill:colors[m]},dataLabels:{showValue:['gpt-6-luna','gpt-5.6-terra'].includes(m),position:'top',textStyle:{fontSize:20,fill:C.ink}}})),hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},yAxis:{min:0,max:40,majorUnit:10,title:'Completed passes',textStyle:axisText,majorGridlines:grid},xAxis:{textStyle:axisText},dataLabels:{showValue:false}});
 text(s,'Terra 5.6',916,230,285,36,27,C.gold,true);text(s,'4, 9, 19 passes',916,276,285,62,27);
 text(s,'Astra Extra High',916,379,285,70,27,C.purple,true);text(s,'2 timeouts\nBoth checkpoints passed',916,460,285,96,25);
 foot(s,'Descriptive comparison on selected tasks. High reuses earlier observations. Timeout checkpoints do not count as completed passes.');
 note(s,55,'Here the cost axis is removed to show the relationship between configured effort and completed success. Terra improves from four to nine to nineteen passes, and Luna improves substantially at Extra High, although its Low-to-High change goes the other way. These are small samples, and the High cohort includes earlier observations, so I would not claim a precise causal response to the effort setting. Sol already performs strongly at Low. Astra Extra High drops from thirty-nine to thirty-seven completed passes because two attempts time out. Both saved checkpoints pass the verifier, so the apparent loss is about completing inside the prescribed budget. Keeping the original metric avoids changing the rules after seeing the result. That distinction motivates the trajectory analysis.', ['results/reasoning-sweep-20261011/summary.csv','results/trajectory-analysis-20261014/analysis.json']);
}
// 6. Failure modes
{
 const s=base('The failure modes differ sharply by model','Recorded tool actions, saved programs, and verifier outcomes explain different bottlenecks.');
 const models=['gpt-6-luna','gpt-5.6-terra','deepseek/deepseek-v4.1-flash','gpt-6-sol','gpt-6-astra'];
 const types=[['pass','Completed pass',C.teal],['hidden_prediction_only','Prediction only',C.gold],['other_verification_failure','Other grading',C.purple],['timeout','Timeout',C.red],['other_execution_error','Other execution','#AFB6BC']];
 const totals=(m,k)=>D.failures.filter(r=>r.model===m&&r.complete_panel).reduce((v,r)=>v+(r.outcomes[k]||0),0);
 chart(s,'bar',{position:{left:70,top:194,width:790,height:415},categories:models.map(m=>shortNames[m]),series:types.map(([k,n,c])=>({name:n,values:models.map(m=>totals(m,k)),fill:c})),barOptions:{direction:'bar',grouping:'stacked',gapWidth:55,overlap:100},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},xAxis:{textStyle:{fontSize:20,fill:C.ink},majorGridlines:null},yAxis:{min:0,max:117,majorUnit:39,title:'Attempts across three effort settings',textStyle:axisText,majorGridlines:grid}});
 text(s,'186 / 191',900,220,310,65,48,C.gold,true);
 text(s,'Normally completed failures\nfailed only hidden predictions',903,293,305,90,25);
 text(s,'105 / 117',900,420,310,65,48,C.red,true);
 text(s,'DeepSeek attempts ended\nwith execution errors',903,493,305,80,25);
 foot(s,'At least 141 of the 186 prediction-only failures show public-test passes. Errors alone do not establish a physics mistake.');
 note(s,75,'The stacked bars aggregate three complete effort settings per model, so every row contains one hundred seventeen attempts. Green means normal completion with a pass. Gold means the agent completed, fit calibration and the parameter correctly, and failed only hidden predictions. Almost every normally completed failure is in that category: one hundred eighty-six of one hundred ninety-one, or ninety-seven point four percent. At least one hundred forty-one of those also show a public-test pass in the tool record. This is strong evidence that fitting and public tests do not validate the extrapolated model. DeepSeek has a different dominant bottleneck: one hundred five execution errors in one hundred seventeen attempts, including eighty-three timeouts. Only twelve attempts complete normally, and those twelve fail predictions. A zero pass rate therefore combines physical errors with failures to deliver a completed solution. We must separate those causes before drawing conclusions about physics ability.', ['results/trajectory-analysis-20261014/analysis.json','output/trajectory-analysis-20261014/report.html']);
}
// 7. A specific trajectory
{
 const s=base('A successful fit can leave the wrong physics untouched','Entropy anomaly, GPT-6 Luna Low, one reviewed 21.6-second trajectory.');
 table(s,[['Observed action','Evidence'],['Read the apparatus and starter','Saved tool call'],['Fit the friction parameter','Only fit() changed'],['Run the public tests','2 tests passed'],['Submit the existing predictor','Conditional population correction stayed zero']],66,193,1146,290,[500,646],25);
 text(s,'77.38%',72,526,280,65,51,C.red,true);
 text(s,'largest hidden-group error\nin the saved submission',382,536,760,72,28);
 foot(s,'After an analyst restores the missing contact-population term with friction fixed, the largest tested error is 0.0443%.');
 note(s,80,'This case makes the aggregate pattern concrete. In one GPT-6 Luna Low run, the model reads the apparatus, constructs a friction scan, implements a weighted least-squares fit and runs the public tests. Both public tests pass. The saved source shows that the physical predictor is unchanged. In particular, the first correction to the conditional contact population remains identically zero. This matters because equal stationary probabilities for the valve do not guarantee equal position-conditioned contact residence in the finite-mass correction. The final program fits the observed parameter but misses that contribution to the extrapolated entropy response. The largest hidden-group error is seventy-seven point four percent. To check that this is the relevant omission, the diagnostic analysis holds the fitted friction fixed and restores only the population correction in an in-memory copy. The largest tested error then drops below five hundredths of a percent. That intervention supports the failure location in this saved program. It is an analyst diagnosis, not a new model success, and the original reward remains unchanged.', ['results/trajectory-analysis-20261014/cases/entropy-anomaly__6yBuZGL/review.json','results/trajectory-analysis-20261014/diagnostic-repairs.json']);
}
// 8. Diagnostic interventions
{
 const s=base('Four targeted corrections isolate physical omissions','Largest tested prediction-group error before and after an analyst intervention.');
 const v=[['Task / missing ingredient','Before','After']];
 const labels=['Entropy / conditional contact population','Magnetic tracer / mixed diffusion','Single-file memory / preparation covariance','Reaction–diffusion / constrained covariance'];
 D.diagnostics.forEach((d,i)=>v.push([labels[i],(100*Math.max(...Object.values(d.before))).toFixed(2)+'%',(100*Math.max(...Object.values(d.after))).toFixed(4)+'%']));
 table(s,v,66,194,1146,340,[746,190,210],25);
 text(s,'Fitted parameters stayed fixed. All tested groups fell below their original tolerances.',70,568,1118,58,27,C.teal,true);
 foot(s,'Diagnostic interventions do not add benchmark passes. Reaction–diffusion covers fluctuation groups only.');
 note(s,70,'The same diagnostic procedure was applied to four selected failures. First, it reproduced the saved errors. Then it changed one identified physical ingredient while holding the fitted parameter fixed. For entropy, that is the conditional contact population. For magnetic tracer, it is the mixed diffusion term that couples longitudinal motion and detector noise. For single-file memory, it is the two-time covariance appropriate to deterministic preparation. For reaction–diffusion, it is the charge-conditioned initial covariance instead of independent constituent counts. All tested groups fall below their original tolerances after the correction. This makes the physical diagnosis stronger than a qualitative reading of the source alone. The scope remains limited: these are four chosen cases, and the reaction–diffusion intervention checks only the fluctuation groups. It does not rerun the unchanged mean-transport solver. No benchmark result or submitted source was altered, and these repairs are not additional model passes.', ['results/trajectory-analysis-20261014/diagnostic-repairs.json','results/trajectory-analysis-20261014/audit.json']);
}
// 9. Hint experiment setup
{
 const s=base('The hint comparison holds effort at High','GPT-5.6 Luna, GPT-5.6 Terra, and DeepSeek V4.1 Flash across the same 13 tasks.');
 table(s,[['Prompt condition','Additional guidance'],['No hint','Original instruction'],['General hint','Check the implementation and test correctness'],['Physics hint','A task-specific conceptual physics cue']],66,193,1146,284,[368,778],26);
 text(s,'3 models × 13 tasks × 3 conditions × 3 repeats',70,521,1128,49,34,C.teal,true);
 text(s,'351 observations planned. Results remain pending.',70,581,1118,36,27);
 foot(s,'38 Luna baselines use a matching historical cohort with an older client. The Luna comparison therefore has a client-version confound.');
 note(s,55,'The next experiment asks whether guidance helps the agent repair the missing physics. It uses GPT-5.6 Luna, GPT-5.6 Terra and DeepSeek, always at High effort. Please note the generation difference: the earlier frontier used GPT-6 Luna. For each task and model there are three repeats under the original instruction, a general correctness reminder and a task-specific physics hint. The environment, task data and grading remain frozen. The total is three hundred fifty-one observations. To avoid duplicate spending, thirty-eight Luna baseline slots reuse the matching results supplied by our collaborator, and one already completed local baseline is preserved. Those imported trials use an older Codex client, so Luna is a comparison against historical controls, not a perfectly isolated causal hint experiment. The results slide is intentionally empty while the runs continue.', ['results/hint-comparison-20261014/plan.json','results/hint-comparison-20261014/friend-reuse-amendment-20261015.json']);
}
// 10. User-requested blank results section. Never populate with interim values.
{
 const s=base('Hint comparison results');
 text(s,'Pending completion',66,155,700,40,24,C.muted);
 note(s,0,'Reserved for the completed and audited hint comparison. No partial outcomes or estimates are shown. Skip this slide during the current talk. The final timing plan reserves sixty seconds here by shortening the Pareto, failure, trajectory and diagnostic discussions by fifteen seconds each. The result must report pass counts and execution errors separately, retain historical-control labels, and distinguish the GPT-5.6 Luna model from GPT-6 Luna in the original sweep.', ['results/hint-comparison-20261014/plan.json']);
}
// 11. Closing
{
 const s=base('What the completed experiments establish');
 text(s,'Cost and success vary substantially across settings',70,187,1125,60,34,C.teal,true);
 text(s,'Sol 6 Low and High occupy the high-success end of the observed frontier.',70,255,1100,70,28);
 text(s,'Most completed failures concern hidden predictions',70,355,1110,58,34,C.gold,true);
 text(s,'Source review and targeted corrections identify concrete physical omissions.',70,424,1100,70,28);
 text(s,'Execution failures need their own interpretation',70,522,1110,55,34,C.red,true);
 text(s,'The hint experiment will test whether guidance changes these outcomes.',70,586,1100,42,27);
 note(s,60,'The completed experiments support three conclusions. First, model choice and configured effort create a wide cost–success tradeoff on this frozen suite. The observed frontier reaches high success with GPT-6 Sol Low and High, while lower-cost Luna settings trade away substantial success. Second, a completed program can fit calibration and pass public tests while preserving an inappropriate physical approximation. The trajectory analysis identifies concrete examples, and controlled diagnostic corrections support four of those diagnoses. Third, execution reliability is a separate component of benchmark success, especially in the DeepSeek results and in the Astra Extra High timeouts. These distinctions are useful for interpreting the pending hint study. A gain from a physics hint would be most informative if it changed the final physical implementation, rather than only increasing tool use or time. The current data do not yet tell us whether that happens. The result remains open until the comparison is complete.', ['results/reasoning-sweep-20261011/summary.csv','results/trajectory-analysis-20261014/analysis.json','results/trajectory-analysis-20261014/diagnostic-repairs.json']);
}
// 12. Q&A
{
 const s=P.slides.add();s.background.fill=C.ink;
 text(s,'Questions',76,245,1100,100,74,C.white,true);
 text(s,'3 minutes',80,385,800,50,30,'#8DD3CE');
 note(s,180,'Suggested discussion prompts: How much does selecting tasks against Luna affect the ranking? How should timeout checkpoints enter a benchmark score? What evidence would isolate a causal hint effect? The backup slides give the full no-hint counts, the collaborator cohort and the task roster. Answer with the specific scope limits: selected tasks, three repeats per task, preserved failures, historical baselines and the time budget.', ['results/reasoning-sweep-20261011/README.md']);
}
// 13. Backup complete settings
{
 const s=base('Backup: all complete no-hint settings','Original sweep only. Costs are estimated USD per attempt.');
 const vals=[['Model','Effort','Passes','Execution errors','Mean cost']];
 D.sweep.filter(r=>r.complete_panel).forEach(r=>vals.push([modelNames[r.model],efforts[r.reasoning_effort],`${r.passes}/39`,String(r.errors),(r.model.startsWith('deepseek')?'≥ ':'')+'$'+r.mean_estimated_api_usd.toFixed(4)]));
 const t=table(s,vals,66,174,1146,433,[354,210,140,237,205],18);
  t.cells.block({row:0,column:0,rowCount:vals.length,columnCount:5}).assign({margins:{top:2,bottom:2,left:8,right:8}});
  for(let r=0;r<vals.length;r++) t.rows[r].height=27;
 foot(s,'DeepSeek None is partial at 3/39 and excluded. API-equivalent estimates differ from subscription usage.');
 note(s,0,'Use this slide for exact-count questions. Astra Extra High has thirty-seven completed passes but thirty-nine passing checkpoints. Terra Extra High has nineteen completed passes and twenty passing checkpoints. Luna Extra High has twelve completed passes and thirteen passing checkpoints. DeepSeek Low, High and Max costs remain lower bounds because of interrupted-response accounting. Four historical Terra High attempts were affected by host sleep, and three started DeepSeek High connection failures overlapped host sleep. Those outcomes are preserved in the primary data.', ['results/reasoning-sweep-20261011/summary.csv','results/reasoning-sweep-20261011/README.md']);
}
// 14. Collaborator results are not silently pooled with original sweep.
{
 const s=base('Backup: collaborator results remain a separate cohort','390 published no-hint trials on the matching 13-task snapshot.');
 const vals=[['Model','Effort','Passes / 39','Execution errors']];
 D.imported.forEach(r=>vals.push([r.model.replace('gpt-5.6-','GPT-5.6 ').replace(/\b(luna|sol|terra)\b/g,w=>w[0].toUpperCase()+w.slice(1)),efforts[r.reasoning_effort]||r.reasoning_effort,r.passes+' / 39',String(Number(r.timeouts)+Number(r.other_errors))]));
 table(s,vals,66,190,1146,380,[415,280,230,221],22);
 foot(s,'Codex 0.154.0 and concurrency 4, versus 0.160.0 and concurrency 3 locally. Model generations are distinct.');
 note(s,0,'The collaborator reports add a full GPT-5.6 Luna sweep with None, Low, Medium, High, Extra High and Max, plus GPT-5.6 Sol and Terra at Low and Medium. These are three hundred ninety unique published attempts. Their task and grading hashes match, but their client version and concurrency differ from the local experiment. They remain a separate cohort and are not silently pooled into the original GPT-6 frontier or trajectory totals. The current hint experiment reuses thirty-eight of the thirty-nine Luna High baseline observations. The one already completed local Luna slot remains local. Raw collaborator native logs were not available for independent trajectory review.', ['results/hint-comparison-20261014/imported-friend-results-20261015/import-manifest.json','results/hint-comparison-20261014/friend-reuse-amendment-20261015.json']);
}
// 15. Roster and scientific limits
{
 const s=base('Backup: the frozen 13-task roster');
 const left=D.tasks.slice(0,7),right=D.tasks.slice(7);
 left.forEach((n,i)=>text(s,n,76,165+i*56,548,43,27));
 right.forEach((n,i)=>text(s,n,664,165+i*56,548,43,27));
 foot(s,'Frozen snapshot: 34dedfbbd8ff. Task selection against GPT-5.6 Luna limits generalization to a broader physics population.');
 note(s,0,'This is the complete thirteen-task roster used in the current comparison. The inherited hydrodynamic-heating and thermoelastic-rod folders are outside the roster. The main experiment keeps task files, public instructions, grading and time limits fixed. The results describe this selected suite, and they do not establish population-wide success rates. A zero out of three observation also leaves substantial uncertainty about a task’s underlying pass probability.', ['results/task-roster.json','results/hint-comparison-20261014/plan.json','AGENTS.md']);
}

if(notes.slice(0,11).reduce((s,n)=>s+n.seconds,0)!==600) throw Error('Main talk timing must total 600 seconds');
await fs.writeFile(path.join(BUILD,'speaker-notes.json'),JSON.stringify(notes,null,2));
await fs.writeFile(path.join(BUILD,'speaker-notes.txt'),notes.map(n=>`SLIDE ${n.slide} / ${n.seconds}s\n${n.body}\nSources: ${n.sources.join(', ')}`).join('\n\n'));
await fs.mkdir(path.join(BUILD,'previews'),{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(path.join(BUILD,'previews',stem+'.png'),new Uint8Array(await (await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(BUILD,'previews',stem+'.layout.json'),await (await s.export({format:'layout'})).text());
}
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(P)).save(candidate);
const finalName=process.env.DECK_FILENAME||'PhysLiteBench-10-minute-draft.pptx';
const final=path.join(OUT,finalName);
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,3,4,7,8,9,13,14].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[2,3,4,7,8,9,13,14],requiredNativeChartOwnerSlides:[4,5,6],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,finalName+'.validation.json')});
await fs.writeFile(path.join(BUILD,'finalization-result.json'),JSON.stringify(result,null,2));
const importedDeck=await PresentationFile.importPptx(await FileBlob.load(final));
await fs.mkdir(path.join(BUILD,'final-previews'),{recursive:true});
for(const [i,s] of importedDeck.slides.items.entries()){
 await fs.writeFile(path.join(BUILD,'final-previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await (await importedDeck.export({slide:s,format:'png',scale:1})).arrayBuffer()));
}
console.log(JSON.stringify({final,slides:P.slides.items.length,mainTalkSeconds:600,questionSeconds:180,blankResultSlide:10}));
