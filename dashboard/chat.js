const $=id=>document.getElementById(id);let history=[],busy=false;
// history holds what the model sees ({role, content}); sources are kept beside it for display only.
function sourceList(sources){
  const box=document.createElement('details');box.className='chat-sources';
  const summary=document.createElement('summary');summary.textContent=`Library sources (${sources.length})`;box.append(summary);
  sources.forEach((s,i)=>{const item=document.createElement('div');item.className='chat-source';
    const title=document.createElement('strong');title.textContent=`[${i+1}] ${s.title}`;
    const text=document.createElement('p');text.textContent=s.text;item.append(title,text);box.append(item)});
  return box}
function draw(){$('messages').replaceChildren();for(const m of history){const card=document.createElement('section');card.className='saved-sample';const title=document.createElement('h2');title.textContent=m.role==='user'?'You':'daniellm';const text=document.createElement('div');text.className='sample-text';text.textContent=m.content;card.append(title,text);if(m.sources&&m.sources.length)card.append(sourceList(m.sources));$('messages').append(card)}}
function clear(){if(busy)return;history=[];draw()}
function lock(on){busy=on;for(const id of ['sendChat','chatRun','clearChat','useLibrary'])$(id).disabled=on}
$('clearChat').addEventListener('click',clear);$('chatRun').addEventListener('change',clear);
$('chatForm').addEventListener('submit',async e=>{e.preventDefault();if(busy)return;const content=$('message').value.trim();if(!content)return;
  const retrieve=$('useLibrary').checked;lock(true);history.push({role:'user',content});draw();
  $('chatStatus').textContent=retrieve?'Searching the library, then generating on CPU...':'Generating on CPU...';
  try{const messages=history.slice(-9).map(({role,content})=>({role,content}));
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:$('chatRun').value,messages,tokens:80,temperature:.7,retrieve})});
    const result=await r.json();if(!r.ok)throw Error(result.error);
    history.push({role:'assistant',content:result.continuation||'[No reply generated]',sources:result.sources||[]});$('message').value='';draw();
    const note=retrieve&&!(result.sources||[]).length?' / no library match, answered without sources':'';
    $('chatStatus').textContent=`Saved step ${result.step} / ${result.seconds}s${note}`}
  catch(e){history.pop();draw();$('chatStatus').textContent=e.message}finally{lock(false)}});
(async()=>{try{const r=await fetch('/api/runs');if(!r.ok)throw Error('Cannot load runs');const runs=(await r.json()).runs.filter(r=>r.checkpoint&&r.config.phase==='dialogue');$('chatRun').replaceChildren(...runs.map(r=>new Option(r.name,r.name)));$('sendChat').disabled=!runs.length;$('chatStatus').textContent=runs.length?'Ready':'No dialogue-trained checkpoint yet. Pretraining is still in progress; use Try for text completion.'}catch(e){$('chatStatus').textContent=e.message}})();
