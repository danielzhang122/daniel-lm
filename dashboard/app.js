const $=id=>document.getElementById(id);let state={runs:[]};
const fmt=n=>Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:0}):'—';
const quote=s=>"'"+String(s).replaceAll("'","''")+"'";
function selected(){return state.runs.find(r=>r.name===$('run').value)}
function chart(rows){
 const svg=$('chart');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
 const add=(tag,attrs,text)=>{const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;svg.appendChild(el)};
 const points=rows.filter(r=>Number.isFinite(r.train_loss)||Number.isFinite(r.val_loss));if(!points.length){add('text',{x:60,y:120,fill:'#929292'},'Waiting for training metrics…');return}
 const values=points.flatMap(r=>[r.train_loss,r.val_loss]).filter(Number.isFinite),lo=Math.floor(Math.min(...values)*2)/2,hi=Math.max(lo+1,Math.ceil(Math.max(...values)*2)/2),max=Math.max(1,...points.map(r=>r.step));
 const x=s=>50+680*s/max,y=v=>235-200*(v-lo)/(hi-lo);
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;add('line',{x1:50,x2:730,y1:y(v),y2:y(v),stroke:'#292929'});add('text',{x:8,y:y(v)+4,fill:'#929292','font-size':12},v.toFixed(1))}
 for(let i=0;i<=4;i++)add('text',{x:x(max*i/4),y:262,'text-anchor':'middle',fill:'#929292','font-size':11},fmt(max*i/4));
 for(const [key,color] of [['train_loss','#b8b8b8'],['val_loss','#e5484d']]){const data=points.filter(r=>Number.isFinite(r[key]));add('polyline',{points:data.map(r=>`${x(r.step)},${y(r[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5});for(const r of (key==='val_loss'?data:data.slice(-1))){const el=document.createElementNS(ns,'circle');el.setAttribute('cx',x(r.step));el.setAttribute('cy',y(r[key]));el.setAttribute('r',3);el.setAttribute('fill',color);const title=document.createElementNS(ns,'title');title.textContent=`Step ${r.step}: ${key} ${r[key].toFixed(4)}`;el.appendChild(title);svg.appendChild(el)}}
}
function commands(){const r=selected();if(!r)return;const last=r.rows.filter(x=>Number.isFinite(x.train_loss)).at(-1)||{};const target=Number($('goal').value),tps=r.config.tokens_per_step||8192;const valid=Number.isInteger(target)&&target>(last.step||0);$('copyTrain').disabled=!valid||!r.checkpoint;
 const data=r.config.data||`data/${r.name}`,out=`runs/${r.name}`,base='.\\.venv\\Scripts\\python.exe';
 const settings=r.config.training;let extra=settings?` --batch ${settings.batch} --accum ${settings.accum} --schedule-steps ${settings.horizon} --warmup ${settings.warmup} --lr ${settings.lr} --min-lr ${settings.min_lr}`:'';
 $('trainCommand').textContent=`${base} train.py --data ${quote(data)} --out ${quote(out)} --resume --steps ${valid?target:(last.step||0)+1000} --eval-every 100${extra}`;
 if(r.config.phase==='dialogue'){
  const cfg=r.config.sft_settings||{batch:2,accum:4,horizon:1000};
  $('trainCommand').textContent=`${base} train_dialogue.py --data ${quote(data)} --out ${quote(out)} --resume --steps ${valid?target:(last.step||0)+1000} --batch ${cfg.batch} --accum ${cfg.accum} --schedule-steps ${cfg.horizon}`;
 }
 $('generateCommand').textContent=`${base} generate.py --checkpoint ${quote(out+'/latest.pt')} --tokenizer ${quote(data+'/tokenizer.json')} --prompt ${quote($('prompt').value)} --tokens 150`;
 $('copyGenerate').disabled=!r.checkpoint;
 const seconds=(target-(last.step||0))*tps/last.tokens_per_second;
 $('estimate').textContent=!valid?'Choose a target above the latest step.':Number.isFinite(seconds)?`About ${Math.max(1,Math.ceil(seconds/60))} minutes at the latest logged speed. Estimate only.`:'An estimate appears after training reports its speed.';
}
function updateProgress(run,latest){
 const target=Number(run.config.target_steps),step=Number(latest.step),speed=Number(latest.tokens_per_second),tps=Number(run.config.tokens_per_step);
 const valid=Number.isFinite(target)&&target>0&&Number.isFinite(step)&&step>=0;
 $('progressBar').value=valid?Math.max(0,Math.min(100,100*step/target)):0;
 $('progressPercent').textContent=valid?`${(100*step/target).toFixed(1)}%`:'--';
 $('progressSteps').textContent=valid?`${fmt(step)} / ${fmt(target)} steps`:'No target for this run';
 const age=Math.max(0,Date.now()/1000-run.updated),seconds=(target-step)*tps/speed;
 if(!valid) $('progressEta').textContent='';
 else if(step>=target) $('progressEta').textContent='Target reached';
 else if(age>120) $('progressEta').textContent='No recent log updates; ETA unavailable';
 else if(Number.isFinite(seconds)&&seconds>=0){const minutes=Math.ceil(seconds/60),hours=Math.floor(minutes/60);$('progressEta').textContent=`About ${hours?hours+' hr ':''}${minutes%60} min remaining`;}
 else $('progressEta').textContent='Waiting for a speed estimate';
}
function render(){const r=selected();if(!r){$('activity').textContent='No training logs yet';chart([]);return}const rows=r.rows,last=rows.filter(x=>Number.isFinite(x.train_loss)).at(-1)||{},vals=rows.filter(x=>Number.isFinite(x.val_loss));$('step').textContent=fmt(last.step);$('target').textContent=r.config.target_steps?`Latest run target: ${fmt(r.config.target_steps)}`:'Pilot experiment';$('loss').textContent=vals.length?vals.at(-1).val_loss.toFixed(3):'—';$('speed').textContent=fmt(last.tokens_per_second);$('articles').textContent=fmt(r.dataset.articles?.train);$('dataNote').textContent=`${fmt(r.dataset.tokens?.train)} training tokens`;
 updateProgress(r,last);
 $('activity').textContent=`Last log: ${new Date(r.updated*1000).toLocaleTimeString()} · ${r.checkpoint?'Checkpoint available':'No checkpoint yet'}`;
 $('insight').textContent=vals.length>1?`Validation loss moved from ${vals[0].val_loss.toFixed(3)} to ${vals.at(-1).val_loss.toFixed(3)}. ${vals.at(-1).val_loss<vals[0].val_loss?'Prediction error is down; next, read a generated sample.':'Review the trend over several checks before extending the run.'}`:'Waiting for more validation checks. Early text can still be incoherent.';
 $('cdCommand').textContent='cd '+quote(state.project);chart(rows);commands();}
let refreshPending=false;
async function refresh(){if(refreshPending)return;refreshPending=true;try{
 const res=await fetch('/api/runs?compact=1');if(!res.ok)throw Error('Server returned '+res.status);
 state=await res.json();const previous=$('run').value,names=state.runs.map(r=>r.name);
 if([...$('run').options].map(o=>o.value).join('|')!==names.join('|')) $('run').replaceChildren(...names.map(name=>new Option(name,name)));
 $('run').value=names.includes(previous)?previous:[...state.runs].sort((a,b)=>b.updated-a.updated)[0]?.name||'';
 render();$('connection').textContent='Connected / refreshes every 3 seconds';
 }catch(error){$('connection').textContent='Could not update dashboard / check server or refresh';console.error(error);}finally{refreshPending=false;}}
$('run').addEventListener('change',render);$('goal').addEventListener('input',commands);$('prompt').addEventListener('input',commands);
for(const [button,source] of [['copyTrain','trainCommand'],['copyGenerate','generateCommand']])$(button).addEventListener('click',async()=>{try{await navigator.clipboard.writeText($(source).textContent);$('notice').textContent='Copied. Paste into your project’s PowerShell terminal.'}catch{$('notice').textContent='Clipboard unavailable. Select and copy the command above.'}});
refresh();setInterval(refresh,3000);
