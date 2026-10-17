import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
import { applyPresentationChartFont, finalizePresentation } from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';

const WORK='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const BUILD=path.join(WORK,'.presentation-restyle-20261014');
const OUT=path.join(WORK,'output/physlitebench-presentation');
const SKILL='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const INPUT=process.env.INPUT_DECK || '/Users/yp9846/Desktop/PhysLiteBench/output/presentation-20261014/PhysLiteBench-10-minute-draft.pptx';
const FONT='Times New Roman';
const C={white:'#FFFFFF',black:'#111111',muted:'#444444',footer:'#777777',dark:'#0B0B0F',band:'#F3F4F6',header:'#374151',blue:'#1D4E89',teal:'#0F766E',orange:'#B45309',red:'#A50026',purple:'#5B4B7C',grid:'#D1D5DB'};
const palette={'#FCFBF8':C.white,'#182B35':C.black,'#596A73':C.muted,'#007E80':C.teal,'#406EA6':C.blue,'#B57C18':C.orange,'#AE4E45':C.red,'#79648E':C.purple,'#DFE4E4':C.grid,'#D6E4E6':'#D1D5DB','#8DD3CE':'#D1D5DB'};
const P=await PresentationFile.importPptx(await FileBlob.load(INPUT));
P.theme.defaultFont=FONT;
P.theme.colorScheme={name:'Committee — Times New Roman',themeColors:{accent1:C.blue,accent2:C.teal,accent3:C.orange,accent4:C.red,accent5:C.purple,accent6:C.footer,bg1:C.white,bg2:C.band,tx1:C.black,tx2:C.muted,dk1:C.dark,dk2:C.header,lt1:C.white,lt2:C.band,hlink:C.blue,folHlink:C.purple}};
const snapshots=[];
const renamedTitles={7:'A good fit can preserve the wrong physics',8:'Targeted corrections identify physical omissions',14:'Backup: collaborator results remain separate'};
const columnCounts={2:2,3:2,4:3,7:2,8:3,9:2,13:5,14:4};
const fontSizes={2:27,3:27,4:21,7:27,8:27,9:27,13:20,14:24};
for(const [idx,s] of P.slides.items.entries()) {
 const n=idx+1;
 const layout=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());
 snapshots.push({slide:n,notes:String(s.speakerNotes.text),tables:layout.elements.filter(e=>e.kind==='table'),charts:layout.elements.filter(e=>e.kind==='chart')});
 const dark=n===1||n===12;
 s.background.fill=dark?C.dark:C.white;
 for(const el of layout.elements.filter(e=>e.kind==='textbox')) {
  const sh=P.resolve(el.id);
  const old=el.style||{};
  const pos={...el.position};
  let txt=el.text;
  let size=old.fontSize, color=palette[old.color]||old.color||C.black;
  const title=!dark&&pos.top===44;
  const subtitle=!dark&&pos.top===119;
  const footer=!dark&&pos.top===634;
  const page=!dark&&pos.top===677;
  if(title){txt=renamedTitles[n]||txt;Object.assign(pos,{left:48,top:22,width:1184,height:67});size=48;color=C.black;}
  else if(subtitle){Object.assign(pos,{left:48,top:96,width:1184,height:48});size=25;color=C.muted;}
  else if(footer){Object.assign(pos,{left:48,top:637,width:1184,height:40});size=18;color=C.footer;}
  else if(page){Object.assign(pos,{left:1176,top:680,width:56,height:20});size=16;color=C.footer;txt=String(n);}
  else if(!dark){pos.top-=16;if(pos.left<100)pos.left=48;}
  if(n===1){
   if(el.text==='PhysLiteBench'){Object.assign(pos,{left:64,top:158,width:1152,height:100});size=76;color=C.white;}
   if(el.text.startsWith('Cost,')){txt='Cost, reasoning effort,\nand physics failure modes';Object.assign(pos,{left:68,top:297,width:1090,height:116});size=40;color='#D1D5DB';}
   if(el.text==='13 frozen tasks'){Object.assign(pos,{left:68,top:489,width:900,height:48});size=29;color=C.white;}
   if(el.text.startsWith('Yue Pan')){Object.assign(pos,{left:68,top:600,width:950,height:40});size=27;color='#D1D5DB';}
  }
  if(n===12){pos.left=64;color=el.text==='Questions'?C.white:'#D1D5DB';size=el.text==='Questions'?76:32;}
  if(n===4&&el.text==='Frontier settings'){Object.assign(pos,{left:866,top:168,width:366,height:36});size=28;color=C.black;}
  if(n===4&&el.position.left===860){pos.left=870;pos.top=el.position.top-12;size=21;pos.height=28;}
  if(n===5&&!title&&!subtitle&&!footer&&!page){pos.left=920;pos.width=310;}
  if(n===6&&!title&&!subtitle&&!footer&&!page){pos.left=908;pos.width=324;if(size===48)size=50;else size=27;}
  if(n===8&&el.text.startsWith('Fitted parameters')){pos.width=1184;size=28;color=C.black;}
  if(n===11&&!title&&!page){pos.left=64;pos.width=1168;if(old.bold){color=C.black;size=35;}else{color=C.muted;size=29;}}
  if(n===15&&!title&&!footer&&!page){size=29;pos.width=550;if(pos.left===664)pos.left=668;}
  sh.text=txt;
  sh.position=pos;
  sh.text.style={typeface:FONT,fontSize:size,color,bold:old.bold||false,italic:old.italic||false,alignment:old.alignment||'left',verticalAlignment:'top',wrap:'square',insets:0};
 }
 for(const el of layout.elements.filter(e=>e.kind==='table')) {
  const t=P.resolve(el.id);
  const compact=n===4;
  const pos={...el.position,left:compact?860:48,top:el.position.top-16,width:compact?372:1184};
  if(n===13){pos.top=161;pos.height=448;}
  if(n===14){pos.top=174;pos.height=407;}
  t.position=pos;
  const cols=columnCounts[n];
  const rows=t.rows.length;
  const oldWidths=[];
  for(let c=0;c<cols;c++)oldWidths.push(t.columns.get(c).width);
  const scale=pos.width/oldWidths.reduce((a,b)=>a+b,0);
  for(let c=0;c<cols;c++)t.columns.get(c).width=oldWidths[c]*scale;
  const targetHeight=n===13?28:n===14?37:null;
  for(let r=0;r<rows;r++){
   if(targetHeight)t.rows[r].height=targetHeight;
   for(let c=0;c<cols;c++){
    const cell=t.getCell(r,c);
    cell.fill=r===0?C.header:(r%2?C.white:C.band);
    cell.text.style={typeface:FONT,fontSize:fontSizes[n],color:r===0?C.white:C.black,bold:r===0,verticalAlignment:'middle',insets:0};
   }
  }
  t.cells.block({row:0,column:0,rowCount:rows,columnCount:cols}).assign({margins:{top:(n===13||n===14)?2:7,bottom:(n===13||n===14)?2:7,left:12,right:12}});
  t.borders.assign({style:'solid',fill:'#C7CBD1',width:.6});
 }
 for(const ch of s.charts.items) {
  ch.chartFill=C.white;ch.plotAreaFill=C.white;
  ch.position={...ch.position,left:48,top:166,width:n===4?780:820,height:438};
  for(const axis of [ch.xAxis,ch.yAxis]) { const title=axis.title?.text; if(title) { axis.title=''; axis.title={text:title,textStyle:{typeface:FONT,fontSize:21,fill:C.muted}}; } }
  applyPresentationChartFont(ch,{fontFamily:FONT});
  ch.xAxis={textStyle:{fontSize:21,fill:C.muted,typeface:FONT}};
  ch.yAxis={textStyle:{fontSize:21,fill:C.muted,typeface:FONT},majorGridlines:{fill:C.grid,width:.7,style:'solid'}};
  ch.legend={textStyle:{fontSize:n===6?19:20,typeface:FONT,fill:C.black}};
  const count=n===4?16:n===5?4:5;
  for(let i=0;i<count;i++){
   const ser=ch.series.getItemAt(i);
   ser.dataLabels={textStyle:{typeface:FONT,fontSize:21,fill:C.black}};
   const name=ser.name;
   const color=n===6?[C.teal,C.orange,C.purple,C.red,'#AFB6BC'][i]:name.includes('Luna')?C.blue:name.includes('Sol')?C.teal:name.includes('Terra')?C.orange:name.includes('Astra')?C.purple:name.includes('DeepSeek')?C.red:C.teal;
   if(n===6)ser.fill=color;
   else if(n===4&&i===0){ser.line={fill:C.teal,width:2,style:'dashed'};}
   else {ser.marker.fill=color;if(n===5)ser.line={fill:color,width:3};}
  }
 }
}
await fs.mkdir(path.join(BUILD,'previews'),{recursive:true});
for(const [i,s] of P.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(path.join(BUILD,'previews',stem+'.png'),new Uint8Array(await(await P.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(BUILD,'previews',stem+'.json'),await(await s.export({format:'layout',detail:'full'})).text());
}
await fs.writeFile(path.join(BUILD,'original-native-content.json'),JSON.stringify(snapshots,null,2));
const candidate=path.join(BUILD,'candidate.pptx');
const rawExport=path.join(BUILD,'candidate-export.pptx');
await(await PresentationFile.exportPptx(P)).save(rawExport);
execFileSync(PY,[path.join(BUILD,'preserve_chart_workbooks.py'),INPUT,rawExport,candidate,path.join(BUILD,'workbook-preservation.json')],{stdio:'inherit'});
const final=path.join(OUT,'PhysLiteBench-10-minute-committee-style.pptx');
const owners=[2,3,4,7,8,9,13,14];
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:owners,requiredNativeChartOwnerSlides:[4,5,6],requiredEmbeddedWorkbookChartOwnerSlides:[4,5,6],fontPolicy:{basis:'user_request',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'PhysLiteBench-10-minute-committee-style.pptx.validation.json')});
await fs.writeFile(path.join(BUILD,'finalization-result.json'),JSON.stringify(result,null,2));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(final));
await fs.mkdir(path.join(BUILD,'final-previews'),{recursive:true});
for(const [i,s] of finalDeck.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(path.join(BUILD,'final-previews',stem+'.png'),new Uint8Array(await(await finalDeck.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(BUILD,'final-previews',stem+'.json'),await(await s.export({format:'layout',detail:'full'})).text());
}
console.log(JSON.stringify({final,slides:P.slides.items.length,font:FONT,mainTalkSeconds:600,questionSeconds:180,hintResultsSlide:10}));
