import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench';
const B=W+'/.presentation-no-pilot-20261016';
const U=W+'/output/PhysLiteBench-no-pilot-figures-20261016';
const current='/Users/yp9846/Desktop/PhysLiteBench/output/presentation-20261014/PhysLiteBench-10-minute-all-costs-reviewed.pptx';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
if(hash(await fs.readFile(current))!==hash(await fs.readFile(B+'/source.pptx')))throw Error('Source changed since snapshot');
const P=await PresentationFile.importPptx(await FileBlob.load(B+'/source.pptx'));
const slide=P.slides.items[5];
const layout=JSON.parse(await(await slide.export({format:'layout',detail:'full'})).text());
const pictures=layout.elements.filter(e=>e.kind==='image');
if(pictures.length!==1)throw Error('Unexpected slide 6 image count');
P.delete(pictures[0].id);
slide.images.add({blob:await fs.readFile(U+'/figures/09-no-hint-all-costs.png'),contentType:'image/png',position:{left:40,top:147,width:1200,height:527},fit:'contain',alt:'Dollar and token cost plots for all 24 primary no-hint settings. Pilot excluded. Open markers show recorded cost lower bounds. Dashed frontiers are provisional.'});
const subtitle=layout.elements.find(e=>e.text?.includes('plus a marked one-task pilot'));
if(!subtitle)throw Error('Expected pilot subtitle missing');
P.resolve(subtitle.id).text=subtitle.text.replace(', plus a marked one-task pilot','');
for(const n of [6,7]){
 const s=P.slides.items[n-1];
 s.speakerNotes.text=s.speakerNotes.text
  .replace(' The cross is a one-task pilot, shown separately from the thirteen-task benchmark.','')
  .replace('The three-attempt DeepSeek None pilot covers only one task and is marked separately on slide 6; it stays outside the benchmark totals and frontier.','The three-attempt DeepSeek None pilot covers only one task. It is excluded from the figures, benchmark totals and frontier; its source records are preserved.')
  .replaceAll('/PhysLiteBench-all-cost-figures-20261015/','/PhysLiteBench-no-pilot-figures-20261016/');
}
await(await PresentationFile.exportPptx(P)).save(B+'/candidate.pptx');
await fs.writeFile(B+'/changes.json',JSON.stringify({sourceSha256:hash(await fs.readFile(B+'/source.pptx')),changedVisibleSlides:[6],pilotDisplayed:false,primarySettings:24,attemptsUnchanged:2068,sourceUserEditsPreserved:true},null,2));
console.log('Slide 6 updated: pilot removed from dollar and token panels.');
