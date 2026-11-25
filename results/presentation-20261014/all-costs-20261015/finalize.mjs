import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation} from '/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const W='/Users/yp9846/Documents/ChatGPT/phys-lite-bench',B=W+'/.presentation-costs-20261015';
const S='/Users/yp9846/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PY='/Users/yp9846/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const final=W+'/output/physlitebench-presentation/PhysLiteBench-10-minute-all-costs-reviewed.pptx';
const tables=[2,4,5,13,14,19];
// The user explicitly requested Python-rendered figures, not editable PowerPoint charts.
// Validate native tables and package geometry here; verify figure bytes/data separately.
const result=await finalizePresentation({workspaceDir:W,candidatePath:B+'/candidate.pptx',finalPath:final,pythonExecutable:PY,integrityValidatorPath:S+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:S+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:19,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'user_request',families:['Times New Roman']},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'});
await fs.writeFile(B+'/finalization-result.json',JSON.stringify(result,null,2));
const F=await PresentationFile.importPptx(await FileBlob.load(final));await fs.mkdir(B+'/final-previews',{recursive:true});
for(const [i,s] of F.slides.items.entries()){
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 await fs.writeFile(`${B}/final-previews/${stem}.png`,new Uint8Array(await(await F.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(`${B}/final-previews/${stem}.json`,await(await s.export({format:'layout',detail:'full'})).text());
}
await fs.writeFile(B+'/final-previews/montage.webp',new Uint8Array(await(await F.export({format:'webp',montage:true,scale:.5})).arrayBuffer()));
console.log(JSON.stringify({final,slides:19,changedVisibleSlides:[6,7],talkSeconds:600,qaSeconds:180}));
