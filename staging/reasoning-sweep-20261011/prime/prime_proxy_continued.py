"""Authenticated pass-through to Prime Responses, with a persistent usage ledger and optional USD cap.

Only the model namespace, output cap, and usage reporting are normalized.
No benchmark prompts or outputs are modified. Secrets are never logged.
"""
from pathlib import Path
from datetime import datetime,timezone
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from prime_catalog_probe import read_key
from openai import OpenAI
import argparse,hashlib,hmac,json,os,secrets,threading,uuid

ROOT=Path('/private/tmp/physlite-prime-session')
MODEL='deepseek/deepseek-v4.1-flash'
LOCK=threading.Lock()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def safe_summary(ledger):
    return dict(cap_usd=ledger['cap_usd'],accounted_usd=sum(x['accounted_usd'] for x in ledger['requests']),requests=len(ledger['requests']),in_flight=sum(x['status']=='pending' for x in ledger['requests']))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18764);a=parser.parse_args()
    ROOT.mkdir(mode=0o700,exist_ok=True);os.chmod(ROOT,0o700)
    auth=ROOT/'proxy-token';
    if not auth.exists():auth.write_text(secrets.token_urlsafe(40));os.chmod(auth,0o600)
    token=auth.read_text().strip()
    ledger_path=ROOT/'ledger.json'
    if not ledger_path.exists():write(ledger_path,dict(cap_usd=10,model=MODEL,requests=[dict(id='initial-responses-probe',status='completed',accounted_usd=.002,estimated_token_cost_usd=.00007392,purpose='Compatibility probe of V4 Flash0731; reserve conservatively retained',at=now())]))
    client=OpenAI(api_key=read_key(),base_url='https://api.pinference.ai/api/v1',max_retries=0,timeout=180)
    def update(reqid,**kwargs):
        with LOCK:
            ledger=json.loads(ledger_path.read_text());row=next(x for x in ledger['requests'] if x['id']==reqid);row.update(kwargs);write(ledger_path,ledger)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def error(self,status,message):
            body=json.dumps({'error':{'message':message,'type':'prime_budget_proxy'}}).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path!='/health':return self.error(404,'Unknown route')
            body=b'{"ok":true}';self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):return self.error(401,'Invalid local proxy credential')
            if self.path.rstrip('/') not in ['/v1/responses','/responses']:return self.error(404,'Only Responses inference is supported')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<8000000:return self.error(413,'Request too large')
            raw=self.rfile.read(size)
            try:payload=json.loads(raw)
            except Exception:return self.error(400,'Invalid JSON')
            if payload.get('model') not in [MODEL,'deepseek-v4.1-flash']:return self.error(400,'Model is outside the approved comparison')
            payload['model']=MODEL
            client_effort=(payload.get('reasoning') or {}).get('effort')
            provider_effort='max' if client_effort=='xhigh' else client_effort
            if provider_effort not in ['none','low','high','max']:return self.error(400,'Unsupported reasoning effort')
            payload['reasoning']={**(payload.get('reasoning') or {}),'effort':provider_effort}
            payload['max_output_tokens']=min(payload.get('max_output_tokens') or 128000,128000)
            payload['usage']={'include':True}
            # UTF-8 byte count plus generous special-token overhead bounds input
            # tokens. Reserve full output cap before allowing each paid request.
            bound=(len(json.dumps(payload,ensure_ascii=False).encode())+16384)*.30/1e6+payload['max_output_tokens']*1.20/1e6
            reqid=uuid.uuid4().hex
            with LOCK:
                ledger=json.loads(ledger_path.read_text())
                if ledger['cap_usd'] is not None and sum(x['accounted_usd'] for x in ledger['requests'])+bound>ledger['cap_usd']-.05:return self.error(402,'Authorized Prime spending cap reached')
                ledger['requests'].append(dict(id=reqid,at=now(),status='pending',reserved_usd=bound,accounted_usd=bound,request_sha256=hashlib.sha256(raw).hexdigest(),request_bytes=size,model=MODEL,client_reasoning_effort=client_effort,provider_reasoning_effort=provider_effort,forwarded_request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),stream=payload.get('stream',False)))
                write(ledger_path,ledger)
            usage=None;response_id=None;delivered=True
            try:
                headers={k:v for k,v in client.default_headers.items() if isinstance(v,(str,bytes))};headers.update(client.auth_headers);headers['Content-Type']='application/json'
                request=client._client.build_request('POST',str(client.base_url)+'responses',headers=headers,json=payload)
                upstream=client._client.send(request,stream=True)
                if upstream.status_code!=200:
                    body=upstream.read().decode(errors='replace')
                    update(reqid,status='http_error',http_status=upstream.status_code,error=body.replace(client.api_key,'[REDACTED]')[:1200])
                    upstream.close();return self.error(upstream.status_code,'Prime upstream rejected the request; see sanitized ledger')
                self.send_response(200);self.send_header('Content-Type',upstream.headers.get('Content-Type','application/json'));self.send_header('Connection','close');self.end_headers();self.close_connection=True
                if payload.get('stream'):
                    for line in upstream.iter_lines():
                        if line.startswith('data: '):
                            try:
                                event=json.loads(line[6:]);response=event.get('response',{})
                                if response.get('usage'):usage=response['usage'];response_id=response.get('id')
                            except (ValueError,TypeError):pass
                        if delivered:
                            try:self.wfile.write((line+'\n').encode());self.wfile.flush()
                            except (BrokenPipeError,ConnectionResetError):delivered=False
                else:
                    body=upstream.read();response=json.loads(body);usage=response.get('usage');response_id=response.get('id')
                    try:self.wfile.write(body)
                    except (BrokenPipeError,ConnectionResetError):delivered=False
                upstream.close()
                if usage:
                    ins=usage.get('input_tokens',usage.get('prompt_tokens',0));outs=usage.get('output_tokens',usage.get('completion_tokens',0))
                    estimated=(ins*.30+outs*1.20)/1e6;reported=usage.get('cost')
                    accounted=max(estimated,reported or 0)
                    update(reqid,status='completed',finished_at=now(),usage=usage,response_id=response_id,estimated_token_cost_usd=estimated,reported_cost_usd=reported,accounted_usd=accounted,delivered=delivered)
                else:update(reqid,status='missing_usage',finished_at=now(),delivered=delivered)
            except Exception as e:
                update(reqid,status='transport_error',error=str(e).replace(client.api_key,'[REDACTED]')[:600],finished_at=now())
                try:self.error(502,'Prime transport error; conservative spending reserve retained')
                except Exception:pass
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(json.dumps(dict(status='ready',port=a.port,model=MODEL,budget=safe_summary(json.loads(ledger_path.read_text())))),flush=True)
    try:server.serve_forever()
    finally:server.server_close();client.close()
if __name__=='__main__':main()
