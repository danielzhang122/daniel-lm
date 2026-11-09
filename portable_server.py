"""Dependency-free local web wrapper around exported CPU inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import socket
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
from library import search,article,status

ROOT=Path(__file__).parent
LOCK=threading.Lock()


class Server(ThreadingHTTPServer):
    allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()
PAGE='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>daniellm offline</title>
<style>body{max-width:780px;margin:40px auto;padding:20px;background:#111;color:#ddd;font:14px/1.7 monospace}textarea,input{box-sizing:border-box;width:100%;padding:12px;background:#191919;color:#ddd;border:1px solid #555}button{padding:10px;margin:10px 0;background:#222;color:#ddd;border:1px solid #555}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#ddd}</style>
<h1>daniellm</h1><p>Offline text completion. Experimental model: check facts against sources.</p><form id="f"><textarea id="p" rows="4" maxlength="2000">The Solar System consists of</textarea><button id="go">Generate</button></form><p id="s" role="status"></p><pre id="o"></pre>
<h2>Saved references</h2><form id="find"><input id="q" placeholder="Search the optional offline library"><button>Search</button></form><div id="results"></div>
<script>const $=id=>document.getElementById(id);$('f').onsubmit=async e=>{e.preventDefault();$('go').disabled=true;$('s').textContent='Generating on CPU...';try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:$('p').value})});const d=await r.json();if(!r.ok)throw Error(d.error);$('o').textContent=d.text;$('s').textContent='Step '+d.step+' / '+d.seconds+'s'}catch(e){$('s').textContent=e.message}finally{$('go').disabled=false}};
$('find').onsubmit=async e=>{e.preventDefault();try{const d=await(await fetch('/search?q='+encodeURIComponent($('q').value))).json();$('results').replaceChildren();if(!d.length)$('results').textContent='No match in the included library.';for(const a of d){const b=document.createElement('button');b.textContent=a.title;b.onclick=async()=>{const d=await(await fetch('/article?id='+a.id)).json();const t=document.createElement('pre');t.textContent=d.body+'\\nSource: '+d.url;$('results').replaceChildren(t)};const p=document.createElement('p');p.textContent=a.excerpt;$('results').append(b,p)}}catch(e){$('results').textContent=e.message}};</script>'''


class Handler(BaseHTTPRequestHandler):
    def reply(self,status,data,html=False):
        body=(data if html else json.dumps(data)).encode()
        self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json')
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        url=urlsplit(self.path);q=parse_qs(url.query)
        if url.path=='/': self.reply(200,PAGE,True)
        elif url.path=='/search': self.reply(200,search(q.get('q',[''])[0][:300]))
        elif url.path=='/article':
            try: a=article(int(q.get('id',['0'])[0]))
            except ValueError: a=None
            self.reply(200 if a else 404,a or {'error':'Not found'})
        else: self.reply(404,{'error':'Not found'})
    def do_POST(self):
        if self.path!='/generate': self.reply(404,{'error':'Not found'});return
        host=self.headers.get('Host','')
        if self.headers.get('Origin','http://'+host)!='http://'+host or self.headers.get_content_type()!='application/json':
            self.reply(403,{'error':'Use this server page'});return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<12000: raise ValueError('Invalid request size')
            prompt=json.loads(self.rfile.read(size)).get('prompt')
            if not isinstance(prompt,str) or not 1<=len(prompt)<=2000: raise ValueError('Prompt must be 1–2000 characters')
        except (ValueError,AttributeError): self.reply(400,{'error':'Invalid prompt'});return
        if not LOCK.acquire(False): self.reply(409,{'error':'Already generating'});return
        try:
            result=subprocess.run([sys.executable,str(ROOT/'generate.py'),'--checkpoint',str(ROOT/'model.pt'),'--tokenizer',str(ROOT/'tokenizer.json'),
                '--prompt',prompt,'--tokens','80','--device','cpu','--json'],capture_output=True,text=True,encoding='utf-8',timeout=120,
                env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            self.reply(200,json.loads(result.stdout)) if not result.returncode else self.reply(500,{'error':'Generation failed; check model and tokenizer files'})
        except (OSError,ValueError,subprocess.TimeoutExpired): self.reply(500,{'error':'Generation failed or timed out'})
        finally: LOCK.release()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    server=Server((a.host,a.port),Handler)
    print(f'daniellm: http://{a.host}:{server.server_port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
