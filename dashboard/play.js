const el=id=>document.getElementById(id);
let busy=false,output='';
async function loadRuns(){
 try{
  const response=await fetch('/api/runs');if(!response.ok)throw Error('Could not read training runs.');
  const data=await response.json(),runs=data.runs.filter(r=>r.checkpoint).sort((a,b)=>b.updated-a.updated);
  el('sampleRun').replaceChildren(...runs.map(r=>{const option=document.createElement('option');option.value=r.name;option.textContent=r.name;return option}));
  el('sampleGenerate').disabled=!runs.length;
  el('sampleStatus').textContent=runs.length?'Latest saved checkpoint':'No checkpoints yet. Save a training checkpoint first.';
 }catch(error){el('sampleStatus').textContent='Disconnected';el('sampleError').textContent=error.message+' Start or restart dashboard.py.';}
}
document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>{el('samplePrompt').value=button.dataset.prompt;el('samplePrompt').focus()}));
el('sampleForm').addEventListener('submit',async event=>{
 event.preventDefault();if(busy)return;
 const payload={run:el('sampleRun').value,prompt:el('samplePrompt').value,tokens:Number(el('sampleTokens').value),temperature:Number(el('sampleTemp').value)};
 if(!payload.prompt.trim()){el('sampleError').textContent='Enter a starting sentence.';return;}
 busy=true;el('sampleGenerate').disabled=true;el('sampleGenerate').textContent='Generating...';el('sampleError').textContent='';el('sampleStatus').textContent='Generating on CPU...';
 try{
  const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const result=await response.json();if(!response.ok)throw Error(result.error||'Generation failed. Restart dashboard.py if you just updated it.');
  output=result.text;el('sampleOutput').textContent=output;el('copySample').disabled=false;
  el('sampleMeta').textContent=`${payload.run} / saved step ${result.step.toLocaleString()} / ${result.tokens} new tokens / ${result.seconds}s / CPU`;
  el('sampleStatus').textContent='Sample complete';
 }catch(error){el('sampleError').textContent=error.message;el('sampleStatus').textContent='Could not generate. Restart dashboard.py if it is running an older version.';}
 finally{busy=false;el('sampleGenerate').disabled=false;el('sampleGenerate').textContent='Generate';}
});
el('copySample').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(output);el('sampleStatus').textContent='Copied'}catch{el('sampleError').textContent='Select the output text to copy it.'}});
loadRuns();
