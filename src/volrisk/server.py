"""Loopback-only research workspace. Credentials live in process memory only."""
import argparse
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .copilot import RiskEvidence,explain
from .portfolio import PortfolioLab


class Workspace(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,root):
        self.evidence=RiskEvidence(root)
        self.portfolio=PortfolioLab(root,self.evidence)
        self.token=secrets.token_urlsafe(32)
        self.api_key=os.environ.get('OPENAI_API_KEY','')
        self.model=os.environ.get('OPENAI_MODEL','gpt-4.1-mini')
        self.busy=threading.Semaphore(1)
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass  # Never log questions, headers or credentials.
    def send(self,status,data,kind='application/json; charset=utf-8'):
        raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if isinstance(data,(dict,list)) else data.encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(raw)
    def authorized(self,mutation=False):
        hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in hosts: return False
        if not mutation: return True
        origins={'http://'+h for h in hosts}
        return self.headers.get('Origin') in origins and secrets.compare_digest(self.headers.get('X-Workspace-Token',''),self.server.token)
    def do_GET(self):
        if not self.authorized(): return self.send(403,{'error':'Local access only'})
        path=urlparse(self.path)
        try:
            if path.path=='/':
                html=(Path(__file__).parent/'workspace.html').read_text().replace('__WORKSPACE_TOKEN__',self.server.token)
                return self.send(200,html,'text/html; charset=utf-8')
            if path.path in ['/workspace.js','/workspace.css']:
                file=Path(__file__).parent/path.path.lstrip('/')
                return self.send(200,file.read_text(),'text/javascript; charset=utf-8' if file.suffix=='.js' else 'text/css; charset=utf-8')
            if path.path=='/api/status': return self.send(200,{'configured':bool(self.server.api_key),'model':self.server.model,'provider':'OpenAI','verified':False})
            if path.path=='/api/catalogue': return self.send(200,self.server.evidence.catalogue())
            if path.path=='/api/portfolio/catalogue': return self.send(200,self.server.portfolio.catalogue())
            if path.path=='/api/day':
                q=parse_qs(path.query)
                return self.send(200,self.server.evidence.day_payload(q['protocol'][0],q['date'][0],q['model'][0]))
            return self.send(404,{'error':'Not found'})
        except (KeyError,ValueError): return self.send(400,{'error':'Invalid selection'})
    def do_POST(self):
        if not self.authorized(True): return self.send(403,{'error':'Local session token and origin required'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=20000: return self.send(413,{'error':'Request too large or empty'})
            body=json.loads(self.rfile.read(size))
            if self.path=='/api/configure':
                key=body.get('api_key','');model=body.get('model','gpt-4.1-mini')
                if not isinstance(key,str) or len(key)>300 or not isinstance(model,str) or len(model)>80: raise ValueError('Invalid local settings')
                self.server.api_key=key.strip();self.server.model=model.strip()
                return self.send(200,{'configured':bool(self.server.api_key),'model':self.server.model,'verified':False})
            if self.path=='/api/portfolio/evaluate':
                return self.send(200,self.server.portfolio.evaluate(body))
            if self.path not in ['/api/evidence','/api/chat','/api/portfolio/chat']: return self.send(404,{'error':'Not found'})
            if not isinstance(body.get('reveal',False),bool): raise ValueError('Reveal must be boolean')
            if self.path=='/api/portfolio/chat':
                context=self.server.portfolio.evaluate(body)['context']
            else:
                context=self.server.evidence.context(body['protocol'],body['date'],body['model'],body['ticker'],body.get('reveal',False))
            if self.path=='/api/evidence': return self.send(200,context)
            if not self.server.busy.acquire(blocking=False): return self.send(429,{'error':'A model request is already running'})
            try: result=explain(body.get('question',''),context,self.server.api_key,self.server.model)
            finally: self.server.busy.release()
            return self.send(200,result)
        except KeyError: return self.send(400,{'error':'Missing selection or request field'})
        except ValueError as exc: return self.send(400,{'error':str(exc)})
        except RuntimeError as exc: return self.send(503,{'error':str(exc)})
        except Exception: return self.send(500,{'error':'Workspace request failed. No credentials or question text were logged.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);args=p.parse_args()
    server=Workspace(('127.0.0.1',args.port),Path.cwd())
    print(f'Local risk workspace: http://127.0.0.1:{args.port}/',flush=True)
    print('OpenAI key configured' if server.api_key else 'OpenAI key not configured; use local AI settings.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__': main()
