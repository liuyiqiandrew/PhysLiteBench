import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {applyPresentationChartFont,finalizePresentation} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const R='/Users/yp9846/Desktop/PhysLiteBench';
const B=path.join(W,'.presentation-example-20261015');
const S='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const INPUT=R+'/output/presentation-20261014/PhysLiteBench-10-minute-committee-style.pptx';
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

// Example apparatus: a semantic, editable physics diagram with analytic profiles.
const apparatus=base('Entropy anomaly: the apparatus','A charged Brownian particle exchanges heat with one thermal reservoir at a time.');
text(apparatus,'Thermal contacts',48,165,300,35,29,C.ink,true);
text(apparatus,'Particle dynamics',452,165,355,35,29,C.ink,true);
text(apparatus,'Spatial calorimetry',900,165,332,35,29,C.ink,true);
const phase=Array.from({length:33},(_,i)=>i/32);
const vals=phase.map(z=>Math.cos(2*Math.PI*z));
chart(apparatus,'scatter',{position:{left:44,top:210,width:318,height:190},hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:18}},scatterOptions:{style:'line'},series:[[-1,C.blue,'s = −1'],[0,C.gray,'s = 0'],[1,C.red,'s = +1']].map(([v,c,name])=>({name,xValues:phase,values:vals.map(q=>Number(((1+.4*v)*(1+.4*q)).toFixed(9))),line:{fill:c,width:2.4},marker:{symbol:'none'}})),xAxis:{min:0,max:1,majorUnit:.5,title:'x / (2π)',textStyle:{typeface:FONT,fontSize:18},majorGridlines:null},yAxis:{min:0,max:2,majorUnit:1,title:'Tₛ(x)',textStyle:{typeface:FONT,fontSize:18},majorGridlines:{fill:C.line,width:.7,style:'solid'}}});
text(apparatus,'Three-contact valve',48,407,310,30,25,C.ink,true);
const st0=circle(apparatus,'−1',52,447,49,'#E7EFF7',C.blue);
const st1=circle(apparatus,'0',166,447,49,C.light,C.muted);
const st2=circle(apparatus,'+1',280,447,49,'#F9E7EB',C.red);
connect(apparatus,st0,st1,'right','left',C.muted,'straight',true);connect(apparatus,st1,st2,'right','left',C.muted,'straight',true);
connect(apparatus,st2,st0,'bottom','bottom',C.muted,'elbow',true);
text(apparatus,'Jump rates scale as 1/m',48,526,324,34,23,C.muted);
arrow(apparatus,373,330,63,C.teal);
text(apparatus,'one active\ncontact',366,371,87,56,20,C.teal,false,'center');
const domain=rect(apparatus,486,218,275,226,'#FAFAFA',C.gray);
domain.line={fill:C.gray,width:1.5,style:'dashed'};
text(apparatus,'periodic square',500,231,246,30,23,C.muted,false,'center');
// Charged particle and force/B symbols are the physical diagram, not sample trajectories.
const particle=circle(apparatus,'+',590,306,47,C.gold,C.white);
arrow(apparatus,646,321,82,C.ink);text(apparatus,'F',690,284,42,30,26,C.ink,true);
text(apparatus,'⊙ B',507,317,70,40,29,C.blue,true);
text(apparatus,'x, y  ∈  [0, 2π)',504,401,242,28,22,C.muted,false,'center');
text(apparatus,'Drag + thermal noise\nOnly unknown: friction γ',463,464,330,70,26,C.ink,false,'center');
arrow(apparatus,809,330,66,C.teal);
text(apparatus,'heat\nrecord',808,371,72,56,20,C.teal,false,'center');
chart(apparatus,'scatter',{position:{left:895,top:210,width:337,height:190},hasLegend:false,scatterOptions:{style:'line'},series:[{name:'Detector weight',xValues:phase,values:vals.map(q=>Number(((1+q)/2).toFixed(9))),line:{fill:C.teal,width:2.5},marker:{symbol:'none'}}],xAxis:{min:0,max:1,majorUnit:.5,title:'x / (2π)',textStyle:{typeface:FONT,fontSize:18},majorGridlines:null},yAxis:{min:0,max:1,majorUnit:.5,title:'w(x)',textStyle:{typeface:FONT,fontSize:18},majorGridlines:{fill:C.line,width:.7,style:'solid'}}});
text(apparatus,'Weight local bath entropy\nby the particle’s position',908,421,324,66,26);
text(apparatus,'Detector applies no force',908,505,324,34,23,C.muted);
rect(apparatus,48,569,1184,54,C.light,'#D1D5DB');
text(apparatus,'Stationary finite-mass readout: weighted bath entropy rate − Bₘ × measured residence. Then m → 0.',61,584,1157,35,23,C.ink);
footer(apparatus,'Illustrative profiles: T₀ = 1, bath and spatial contrasts = 0.4, k = 1, phase = 0. Bₘ is the matched uniform, force-free background rate.');
note(apparatus,65,'Here is the physical task we will follow throughout the talk. A charged particle moves in a two-dimensional periodic square. The thermal environment varies with position. In the prediction preparation shown here, a fast valve switches among three reservoirs, connecting exactly one at a time. The three temperature curves share the same spatial profile but have different amplitudes. The particle feels drag, thermal noise, a constant force and a perpendicular magnetic field. The magnetic field does no work. The common friction gamma is the only unknown parameter. On the right, the detector weights the bath entropy by position without exerting a force. To define the requested prediction, the apparatus first reaches stationarity at finite mass, measures weighted heat and actual residence, and subtracts a matched uniform, force-free background times that residence. Only then does it take mass to zero. The example curves illustrate valid settings; they are not measured benchmark outcomes. The two-contact calibration preparation is simpler than the three-contact prediction preparation.', ['tasks/entropy-anomaly/environment/README.md','tasks/entropy-anomaly/instruction.md']);

// Public prompt, the agent workflow, and prospective failure mechanisms.
const flow=base('Entropy anomaly: the agent’s task','Calibration identifies friction. The prediction must retain the physics of the full apparatus.');
rect(flow,48,160,1184,108,C.light,C.line);
text(flow,'Public request, shortened',64,172,1160,27,21,C.gray,true);
text(flow,'Implement model.py using the apparatus, calibration data, and prediction API.\nYou may replace any part of the implementation, including the predictor.',64,202,1150,63,25);
const xs=[48,362,676,990], widths=[258,258,258,242];
const nodes=xs.map((x,i)=>rect(flow,x,294,widths[i],152,C.white,i===3?C.teal:C.line));
text(flow,'Prompt + files',64,308,228,31,27,C.ink,true);
text(flow,'Apparatus equations\nCalibration records\nStarter implementation',64,351,230,83,23);
text(flow,'Agent edits code',378,308,228,31,27,C.ink,true);
text(flow,'fit(records): infer γ\npredict(settings):\ncompute excess entropy',378,351,230,83,23);
text(flow,'Public checks',692,308,228,31,27,C.ink,true);
text(flow,'Valid API and outputs\nFit calibration data\nTwo-contact regime',692,351,230,83,23);
text(flow,'Held-out tests',1006,308,212,31,27,C.teal,true);
text(flow,'Cyclic contacts\nAnisotropic drag\nMagnetic driving',1006,351,210,83,23);
for(let i=0;i<3;i++)connect(flow,nodes[i],nodes[i+1],'right','left',C.ink);
text(flow,'Possible failure points',48,475,1184,34,28,C.ink,true);
const failure=[['Observable','Change the order of subtraction\nand the small-mass limit.'],['Contact statistics','Assume equal contact populations\nat every position and order.'],['Validation','Treat a calibration pass as evidence\nfor the new prediction regime.']];
failure.forEach(([t,b],i)=>{const x=48+i*403;text(flow,t,x,522,376,31,26,C.gold,true);text(flow,b,x,563,376,61,24);});
footer(flow,'These are prospective failure mechanisms. No hints are supplied in the no-hint sweep. The later trajectory identifies the observed omission.');
note(flow,55,'The agent receives a deliberately compact request plus the complete apparatus description, calibration records and starter code. The prompt explicitly allows changes to any part of the implementation, including the predictor. Calibration uses the original two-contact preparation, equal reservoir temperatures and zero magnetic field. It determines the friction parameter but does not test the full prediction regime. The public tests check the interface and the calibration fit. The private evaluation then asks for cyclic-contact, anisotropic and magnetic cases. This workflow leaves several possible failure points. An agent could change the defined subtraction or limiting procedure. It could keep an equal-contact-population approximation beyond the order where it is valid. Or it could stop after passing public tests. These are possible mechanisms at this stage, not a claim that every failure has the same cause. After the aggregate results, we will return to one saved run to locate its actual failure.', ['tasks/entropy-anomaly/instruction.md','tasks/entropy-anomaly/environment/README.md','tasks/entropy-anomaly/environment/test_public.py','tasks/entropy-anomaly/tests/test_hidden.py']);

// Replace the old trajectory slide with a detailed record of this same example.
const trajectory=old[6];
const trajLayout=JSON.parse(await(await trajectory.export({format:'layout',detail:'full'})).text());
P.delete(trajLayout.elements.map(e=>e.id));
text(trajectory,'Entropy anomaly: the observed failure',48,22,1184,67,48,C.ink,true);
text(trajectory,'GPT-6 Luna Low, 21.6 seconds. Recorded tool actions and the submitted program.',48,96,1184,50,25,C.muted);
const steps=[['1  Read apparatus','README + starter'],['2  Scan friction','Best fit near γ = 1.1'],['3  Implement fit()','Predictor unchanged'],['4  Run public tests','2 tests passed']];
const txs=[48,362,676,990];
const stepNodes=steps.map(([a,b],i)=>{const box=rect(trajectory,txs[i],163,widths[i],100,C.light,C.line);text(trajectory,a,txs[i]+12,177,widths[i]-24,32,25,C.ink,true);text(trajectory,b,txs[i]+12,221,widths[i]-24,31,22);return box;});
for(let i=0;i<3;i++)connect(trajectory,stepNodes[i],stepNodes[i+1]);
text(trajectory,'Agent’s recorded explanation',48,302,557,34,27,C.ink,true);
text(trajectory,'“The existing model gives a good calibration fit\nnear friction 1.1, so the small-mass expansion\nis consistent with the records.”',48,346,570,100,25,C.muted);
text(trajectory,'What the submitted source retained',48,461,570,32,27,C.ink,true);
rect(trajectory,48,503,570,69,C.light,C.line);
text(trajectory,'density_correction = zeros(...)',62,518,544,34,26,C.red,true);
text(trajectory,'Abbreviated code: the conditional population correction is zero.',48,586,570,44,19,C.gray);
text(trajectory,'Physical diagnosis',672,302,560,34,27,C.ink,true);
text(trajectory,'Equal valve occupancies can still produce a\nposition-dependent population correction.',672,346,560,67,27);
text(trajectory,'Q n¹ = ∂ₓ j¹       with  ∑ₛ n¹ₛ = 0',672,430,560,41,29,C.blue,true);
text(trajectory,'Q: valve generator; j¹: contact current.\nn¹: first-order conditional population response.',672,479,560,56,22,C.muted);
text(trajectory,'77.38%',674,548,189,48,39,C.red,true);
arrow(trajectory,876,568,68,C.gray);
text(trajectory,'0.0443%',964,548,247,48,39,C.teal,true);
text(trajectory,'largest group error',674,599,205,28,21,C.muted);
text(trajectory,'after restoring n¹',964,599,244,28,21,C.muted);
footer(trajectory,'Analyst intervention with fitted friction fixed. All four tested groups then meet 4% tolerance. Original benchmark reward remains 0.');
note(trajectory,110,'Now return to the same calorimeter. In this GPT-6 Luna Low attempt, the saved tool record shows the agent reading the apparatus and the starter, scanning the friction parameter, implementing a bounded weighted least-squares fit and running the public tests. The first scan had a Python import error, which the agent corrected by setting the module path. That operational issue was resolved before submission. Both public tests passed. The quotation is the agent’s recorded explanation: the existing expansion agrees with the calibration records. That statement is correct about the calibration regime. The source diff shows that only fit() changed. The predictor kept its conditional contact-population correction at zero. Here is why that matters. The externally prescribed valve has uniform marginal state probabilities. But particle position and contact history are correlated, so the first-order position-conditioned population response need not vanish. The contact continuity equation relates that response to the divergence of the conditional spatial current. Setting it to zero removes a contribution to the finite excess calorimetry. The fitted friction is 1.09936, with only 0.0584 percent relative error, and the reduced calibration chi-square is 0.956. Yet the largest hidden prediction-group error is 77.38 percent, and two groups fail the four-percent tolerance. To test the diagnosis, the analyst restored only the population response in an in-memory copy while holding friction fixed. The largest group error becomes 0.0443 percent, with all four tested groups below tolerance. This is evidence about the saved implementation, not access to private reasoning and not an additional model pass.', ['results/trajectory-analysis-20261014/cases/entropy-anomaly__6yBuZGL/review.json','results/trajectory-analysis-20261014/cases/entropy-anomaly__6yBuZGL/model.diff','results/trajectory-analysis-20261014/cases/entropy-anomaly__6yBuZGL/model.py','results/trajectory-analysis-20261014/cases/entropy-anomaly__6yBuZGL/metrics.json','jobs/effort-gpt-6-luna-low-entropy-anomaly-20261011-191612/entropy-anomaly__6yBuZGL/agent/trajectory.json','results/trajectory-analysis-20261014/diagnostic-repairs.json','tasks/entropy-anomaly/AUTHOR.md']);

// Delete the requested old slide 8, then place the example before all aggregate statistics.
P.slides.keep(P.slides.items.map((s,i)=>i).filter(i=>i!==7));
const order=[old[0],old[1],apparatus,flow,old[2],old[3],old[4],old[5],trajectory,old[8],old[9],old[10],old[11],old[12],old[13],old[14]];
const canonicalIds=[];for(const s of order)canonicalIds.push(JSON.parse(await(await s.export({format:'layout'})).text()).slide.id);
P.slides.reorder(canonicalIds);
// Brief transition and closing now refer to the single running example.
for(const sh of old[10].shapes.items){
 if(String(sh.text)==='Source review and targeted corrections identify concrete physical omissions.')sh.text='The entropy example locates a missing contact-population response.';
}
const timing=[20,35,65,55,25,75,40,60,110,55,0,60,180,0,0,0];
const body={
1:'This study asks how much reliable physics coding we get for the model compute we spend. I will first show one concrete calorimetry task and the agent workflow, then compare model cost and success. Finally, we will return to the same task to diagnose a saved failure. A separate hint experiment is still running.',
2:'The benchmark separates parameter fitting from physical prediction. The agent receives an apparatus description, calibration data, starter code and public tests. The hidden evaluator also asks for predictions in regimes that the calibration does not establish. The suite has thirteen frozen tasks spanning transport, stochastic dynamics, correlations and quantum or statistical systems. It was selected using GPT-5.6 Luna, so it is deliberately challenging and not a random sample of physics. Let us make that design concrete with entropy anomaly.',
5:'Across the full suite, the no-hint sweep contains fifteen complete model-effort settings. Each has thirteen tasks and three repeats, for five hundred eighty-five attempts. The agent limit is ten minutes and the verifier limit one minute. OpenAI High settings reuse thirty matching earlier observations per model. Three DeepSeek None attempts form a separate partial panel and are excluded here.',
6:'Each point contains thirty-nine attempts. Higher means more completed passes, and farther left means lower estimated API cost. The dashed curve connects settings that no eligible setting beats on both observed success and cost. Four settings lie on this frontier: GPT-6 Luna Low passes four attempts at about four tenths of a cent per attempt. Luna Extra High passes twelve at about one point three cents. GPT-6 Sol Low passes thirty-seven at about seven point seven cents. Sol High passes all thirty-nine at about nineteen point eight cents. Sol Extra High and Astra cost more without exceeding that observed score. The table gives exact coordinates for the tightly clustered low-cost points. DeepSeek costs are lower bounds because some interrupted requests have incomplete accounting, so those points do not determine the exact-cost frontier. This is a descriptive result on a selected suite with three repeats per task, not a universal ranking.',
7:'Removing the cost axis shows how completed success varies with configured effort. Terra rises from four to nine to nineteen passes. Luna improves substantially at Extra High, while its Low-to-High change goes the other way. Sol already performs strongly at Low. Astra Extra High loses two completed passes because of timeouts, even though both saved checkpoints pass the verifier. Completion within the fixed budget is part of the score. These small samples and the reuse of earlier High observations make this a descriptive comparison, not a precise causal estimate of reasoning effort.',
8:'The bars separate outcomes across one hundred seventeen attempts per model. One hundred eighty-six of the one hundred ninety-one normally completed failures fail only hidden predictions. At least one hundred forty-one of those also show public-test passes in the saved tool record. That is the validation gap we anticipated in the example: a calibration fit can leave the extrapolated physical model wrong. DeepSeek has a different dominant limitation. One hundred five of its one hundred seventeen attempts end in execution errors, including eighty-three timeouts. Only twelve complete normally, and those twelve fail predictions. A zero total pass rate therefore mixes physical mistakes with failure to finish. We now return to entropy anomaly to see exactly where one completed program failed.',
12:'The completed experiments establish a broad cost-success tradeoff on this selected suite. GPT-6 Sol Low and High occupy the high-success end of the observed frontier, while lower-cost Luna settings trade away substantial success. The running example explains one important source of failure: fitting friction and passing public checks can preserve a physical approximation that fails in the prediction regime. The saved code and a targeted diagnostic correction locate the missing conditional contact-population response in that particular submission. The aggregate results show that prediction-only failures are common, but they do not establish that every task has the same omission. Execution reliability is another component of success, especially for DeepSeek and the Astra timeouts. The pending hint comparison will ask whether general or physics guidance changes these outcomes and, ideally, the final physical implementation. That result remains open until the study is complete.'
};
const sourceIndex=[0,1,null,null,2,3,4,5,6,8,9,10,11,12,13,14];
for(const [i,s] of order.entries()){
 const n=i+1;
 if(body[n]) note(s,timing[i],body[n],oldNotes[sourceIndex[i]].sources);
 else if(n!==3&&n!==4&&n!==9){
  const previous=String(s.speakerNotes.text);s.speakerNotes.text=previous.replace(/^Target time: [^\n]+/,`Target time: ${timing[i]?timing[i]+' seconds':'skip in current talk'}.`);
 }
 const layout=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());
 const pages=layout.elements.filter(e=>e.kind==='textbox'&&e.position.top>=675&&/^\d+$/.test(e.text));
 pages.forEach(el=>{P.resolve(el.id).text=String(n);});
 if(n!==1&&n!==13&&!pages.length)text(s,String(n),1176,680,56,20,16,C.gray);
}
// Keep the pending hint-results notes accurate after reordering and retiming.
note(old[9],0,'Reserved for the completed and audited hint comparison. No partial outcomes or estimates are shown. Skip this slide in the current talk. On completion, allocate sixty seconds here by shortening the frontier by fifteen seconds, the failure statistics by fifteen, the entropy trajectory by twenty, and the conclusion by ten. Report completed pass counts and execution errors separately. Preserve historical-control caveats and distinguish GPT-5.6 Luna from the GPT-6 Luna cohort.', ['results/hint-comparison-20261014/plan.json']);
if(timing.slice(0,12).reduce((a,b)=>a+b,0)!==600)throw Error('Timing');
await fs.writeFile(B+'/slide-map.json',JSON.stringify({source:INPUT,order:sourceIndex.map((v,i)=>({slide:i+1,original:v===null?null:v+1,title:String(order[i].shapes.items[0].text),seconds:timing[i]})),removed_original_slide:8,results_slide:11,hint_setup_slide:10,questions_slide:13,timing_current_seconds:timing.slice(0,12),timing_final_seconds:[20,35,65,55,25,60,40,45,90,55,60,50]},null,2));
await fs.mkdir(B+'/previews',{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/previews/'+stem+'.png',new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/previews/'+stem+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
const raw=B+'/candidate-export.pptx',candidate=B+'/candidate.pptx';
await(await PresentationFile.exportPptx(P)).save(raw);
execFileSync(PY,[B+'/preserve_workbooks.py',INPUT,raw,candidate,B+'/workbook-preservation.json'],{stdio:'inherit'});
const name=process.env.DECK_FILENAME||'PhysLiteBench-10-minute-physics-example.pptx';
const final=W+'/output/physlitebench-presentation/'+name;
const tables=[2,5,6,10,14,15];
const result=await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:S+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:S+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[3,6,7,8],requiredEmbeddedWorkbookChartOwnerSlides:[6,7,8],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'user_request',families:[FONT]},verifyArtifactToolImport:true,receiptPath:B+'/'+name+'.validation.json'});
await fs.writeFile(B+'/finalization-result.json',JSON.stringify(result,null,2));
const F=await PresentationFile.importPptx(await FileBlob.load(final));
await fs.mkdir(B+'/final-previews',{recursive:true});
for(const [i,s] of F.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/final-previews/'+stem+'.png',new Uint8Array(await(await F.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/final-previews/'+stem+'.json',await(await s.export({format:'layout',detail:'full'})).text());
}
console.log(JSON.stringify({final,slides:16,seconds:600,questions:180}));
