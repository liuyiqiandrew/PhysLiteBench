import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B='/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.presentation-local-final-20261015';
const p=await PresentationFile.importPptx(await FileBlob.load(B+'/source.pptx'));
await fs.mkdir(B+'/before',{recursive:true});
const summary=[];
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];const l=JSON.parse(await(await s.export({format:'layout',detail:'full'})).text());
 await fs.writeFile(B+`/before/slide-${i+1}.json`,JSON.stringify(l));
 summary.push({slide:i+1,elements:l.elements.map(x=>({id:x.id,kind:x.kind,text:x.text,position:x.position})),notes:s.speakerNotes.text});
 if([5,7,11].includes(i+1))await fs.writeFile(B+`/before/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
}
await fs.writeFile(B+'/source-summary.json',JSON.stringify(summary,null,2));
await fs.writeFile(B+'/before/montage.webp',new Uint8Array(await(await p.export({format:'webp',montage:true,scale:.5})).arrayBuffer()));
console.log(JSON.stringify(summary.filter(x=>[5,7,11].includes(x.slide))));
