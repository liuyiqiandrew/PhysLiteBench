import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const R='/Users/yp9846/Desktop/PhysLiteBench';
const B=W+'/.presentation-trajectory-20261015';
const S='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const INPUT=R+'/output/presentation-20261014/PhysLiteBench-10-minute-plain-language.pptx';
const FONT='Times New Roman';
const C={white:'#FFFFFF',ink:'#111111',muted:'#444444',gray:'#777777',light:'#F3F4F6',line:'#D1D5DB',blue:'#1D4E89',teal:'#0F766E',gold:'#B45309',red:'#A50026'};
const P=await PresentationFile.importPptx(await FileBlob.load(INPUT));
const s=P.slides.items[9];
const before=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());
await fs.writeFile(B+'/before-slide-10.json',JSON.stringify(before,null,2));
await fs.writeFile(B+'/before-slide-10.png',new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
P.delete(before.elements.map(e=>e.id));
function text(txt,x,y,w,h,size=26,color=C.ink,bold=false,align='left'){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=txt;sh.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',wrap:'square',insets:0};return sh;
}
function rect(x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function circle(x,y,d,fill){return s.shapes.add({geometry:'ellipse',position:{left:x,top:y,width:d,height:d},fill,line:{fill:fill===C.gold?C.gold:C.gray,width:1.6,style:'solid'}});}
function tube(x,y,w,xs,d){rect(x,y,w,2,C.gray);rect(x,y+54,w,2,C.gray);xs.forEach((p,i)=>circle(x+p,y+15,d,i===2?C.gold:'#E8ECEF'));}
s.background.fill=C.white;
text('Why the agent failed on the bead task',48,22,1184,67,48,C.ink,true);
text('GPT-5.6 Terra High, no hint',48,96,1184,50,25,C.muted);

// Editable schematic of the same setup used on slide 3. Positions are illustrative.
text('Same evenly spaced start',48,171,574,36,28,C.ink,true,'center');
tube(214,220,242,[10,49,88,127,166,205],24);
s.shapes.add({geometry:'downArrow',position:{left:320,top:290,width:24,height:34},fill:C.gray,line:{fill:'none',width:0}});
text('Early position',48,349,242,33,27,C.ink,false,'center');
text('Later position',380,349,242,33,27,C.ink,false,'center');
tube(48,394,242,[10,46,92,123,169,204],24);
tube(380,394,242,[6,35,111,140,173,208],24);
s.shapes.add({geometry:'leftRightArrow',position:{left:305,top:411,width:60,height:20},fill:C.red,line:{fill:'none',width:0}});
text('Same bead in one run',48,470,574,32,24,C.muted,false,'center');
text('The agent overestimates the link',48,513,574,37,29,C.red,true,'center');

text('Fits the measured motion',694,171,538,39,31,C.teal,true);
text('Adjusts how far the bead wanders\nat each observation time.',694,225,538,78,29);
text('Misses the effect of the starting row',694,345,538,42,30,C.red,true);
text('Keeps the starter’s rule for linking\nits early and later positions.\nThat rule makes the link too strong.',694,401,538,111,29);

rect(48,575,1184,1.3,C.line);
text('Fits the supplied data. New predictions are wrong by up to 136%.',48,597,1184,40,31,C.ink,true);
text('Allowed error: 4%. Schematic positions. The link describes averages across repeated runs.',48,650,1184,30,19,C.gray);
text('10',1176,680,56,20,16,C.gray,false,'right');
s.speakerNotes.text=`Target time: 100 seconds.

Here is the same bead example, now using a saved Terra run. Every repeat starts with the neighbors evenly spaced. We record the orange bead early and again later. The red arrow means a statistical link across repeats, not a force between two beads.

The agent adjusted one number controlling the amount of random motion. Its code matched the supplied measurements of how far the bead wanders at each time, and the public test passed.

But it kept the starter's rule for connecting the two observations. In plain terms, that rule made an early displacement count too much when predicting the later motion. The evenly spaced starting row affects this connection, and its formula did not account for that effect correctly. Matching the spread at each time could not reveal this mistake.

The hidden predictions were wrong by as much as 136%, against a 4% allowance. We checked the diagnosis by keeping its fitted motion strength and replacing only the rule connecting two times. The largest error fell to about 0.003%. That is our diagnostic correction, not a new agent pass or a hint result. The conclusion comes from the saved code, tool actions and tests.

Sources:
${R}/tasks/single-file-memory/environment/README.md
${R}/tasks/single-file-memory/environment/model.py
${R}/tasks/single-file-memory/solution/model.py
${R}/results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/review.json
${R}/results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/model.diff
${R}/results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/metrics.json
${R}/results/trajectory-analysis-20261014/diagnostic-repairs.json

Technical clarification for questions: both formulas have correlated motion and agree at equal times. The saved formula overestimates the two-time covariance. It incorrectly treats increments as depending only on the time gap. The benchmark uses a joint long-time coefficient, not the finite-time positions drawn in this schematic. Equations and exact metrics remain in backup slide 16.`;

const raw=B+'/draft.pptx',candidate=B+'/candidate.pptx';
await(await PresentationFile.exportPptx(P)).save(raw);
execFileSync(PY,[B+'/preserve_workbooks.py',INPUT,raw,candidate,B+'/workbook-preservation.json'],{stdio:'inherit'});
const final=W+'/output/physlitebench-presentation/PhysLiteBench-10-minute-trajectory-visual-v3.pptx';
const tables=[2,4,5,6,13,14];
const result=await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:S+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:S+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[6,8,9,16],requiredEmbeddedWorkbookChartOwnerSlides:[6,8,9,16],materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'user_request',families:[FONT]},verifyArtifactToolImport:true,receiptPath:B+'/validation-v3.json'});
await fs.writeFile(B+'/finalization-result.json',JSON.stringify(result,null,2));
const F=await PresentationFile.importPptx(await FileBlob.load(final));
await fs.mkdir(B+'/final-previews',{recursive:true});
for(const [i,slide] of F.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(B+'/final-previews/'+stem+'.png',new Uint8Array(await(await F.export({slide,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(B+'/final-previews/'+stem+'.json',await(await slide.export({format:'layout',detail:'full'})).text());
}
console.log(JSON.stringify({final,slides:16,changedSlides:[10]}));
