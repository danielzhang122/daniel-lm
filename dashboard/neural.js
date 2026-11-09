/* The edges are real sampled MLP weights, not simulated activations. */
(()=>{
 const get=id=>document.getElementById(id),canvas=get('weightCanvas'),ctx=canvas.getContext('2d');
 let model=null,block=0,hover=null,pinned=null,points=[];
 const fmt=n=>Number(n).toLocaleString();
 function data(){return model.weight_samples[block]}
 function nodes(){const w=data(),inputs=w.expand[0].length,hidden=w.expand.length,outputs=w.project.length;return [inputs,hidden,outputs]}
 function dimensions(){const n=nodes(),w=model.config.width;get('weightDimensions').replaceChildren(...[w,w*4,w].map(x=>{const s=document.createElement('span');s.textContent=fmt(x)+' total';return s}));get('weightCaption').textContent=`Showing ${n[0]} input features, ${n[1]} hidden neurons, and ${n[2]} output features: ${fmt(n[0]*n[1]+n[1]*n[2])} real weights from this block. The full network is much larger. These are learned weights, not live activations. Attention, biases, normalization, and residual connections are not drawn here.`}
 function populate(){const sizes=nodes(),selection=get('weightFeature');selection.replaceChildren(new Option('All connections',''));['Input','Hidden','Output'].forEach((name,column)=>{const group=document.createElement('optgroup');group.label=name;for(let i=0;i<sizes[column];i++)group.append(new Option(`${name} ${i+1}`,`${column}:${i}`));selection.append(group)});selection.disabled=false;get('clearWeight').disabled=false;dimensions()}
 function coordinates(width,height){return nodes().map((count,column)=>Array.from({length:count},(_,i)=>({column,index:i,x:width*(.09+.41*column),y:22+(height-44)*(count===1?.5:i/(count-1))}))) }
 function draw(){if(!model)return;const rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;const ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,rect.width,rect.height);points=coordinates(rect.width,rect.height);const chosen=hover||pinned,w=data();
  function line(a,b,weight){const relevant=!chosen||(chosen.column===a.column&&chosen.index===a.index)||(chosen.column===b.column&&chosen.index===b.index);let strength=Math.min(Math.abs(weight)/.08,1);let alpha=relevant?(chosen ? 0.16+strength*0.72 : 0.035+strength*0.22):0.008;ctx.strokeStyle=weight>=0?`rgba(238,105,110,${alpha})`:`rgba(143,167,205,${alpha})`;ctx.lineWidth=(relevant&&chosen) ? 0.9 : 0.55;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
  for(let h=0;h<points[1].length;h++)for(let i=0;i<points[0].length;i++)line(points[0][i],points[1][h],w.expand[h][i]);
  for(let o=0;o<points[2].length;o++)for(let h=0;h<points[1].length;h++)line(points[1][h],points[2][o],w.project[o][h]);
  points.flat().forEach(p=>{const isChosen=chosen&&chosen.column===p.column&&chosen.index===p.index;ctx.beginPath();ctx.arc(p.x,p.y,isChosen?5:2.8,0,Math.PI*2);ctx.fillStyle=isChosen?'#fff':'#c5c5c5';ctx.fill();ctx.strokeStyle=isChosen?'#e5484d':'#383838';ctx.lineWidth=isChosen?1.8:1;ctx.stroke()});
  get('weightStatus').textContent=chosen?`${['Input feature','Hidden neuron','Output feature'][chosen.column]} ${chosen.index+1} selected · ${chosen.column===1?'Incoming and outgoing sampled weights highlighted.':'Sampled connections highlighted.'}`:`Block ${block+1} · ${fmt(nodes()[0]*nodes()[1]+nodes()[1]*nodes()[2])} sampled connections · choose a dot or use Highlight.`;
 }
 function pointAt(e){const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,bestDist=100;for(const p of points.flat()){const d=(p.x-x)**2+(p.y-y)**2;if(d<bestDist){best=p;bestDist=d}}return best}
 canvas.addEventListener('pointermove',e=>{const hit=pointAt(e);canvas.style.cursor=hit?'pointer':'default';if((hover?.column!==hit?.column)||(hover?.index!==hit?.index)){hover=hit;draw()}});
 canvas.addEventListener('pointerleave',()=>{hover=null;draw()});
 canvas.addEventListener('click',e=>{const hit=pointAt(e);pinned=hit?{column:hit.column,index:hit.index}:null;get('weightFeature').value=pinned?`${pinned.column}:${pinned.index}`:'';draw()});
 get('weightFeature').addEventListener('change',e=>{const [column,index]=e.target.value.split(':').map(Number);pinned=e.target.value?{column,index}:null;hover=null;draw()});
 get('clearWeight').addEventListener('click',()=>{pinned=null;hover=null;get('weightFeature').value='';draw()});
 get('weightBlock').addEventListener('change',e=>{block=Number(e.target.value);hover=null;pinned=null;populate();draw()});
 document.addEventListener('model-inspected',e=>{model=e.detail;block=0;hover=null;pinned=null;const select=get('weightBlock');select.replaceChildren(...model.weight_samples.map((_,i)=>new Option(`Block ${i+1}`,i)));select.disabled=false;populate();draw()});
 window.addEventListener('resize',draw);
})();
