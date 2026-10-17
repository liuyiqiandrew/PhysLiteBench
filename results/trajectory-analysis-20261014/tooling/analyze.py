from pathlib import Path
from datetime import datetime, timezone
import ast, collections, hashlib, json, re, statistics

R = Path('/Users/yp9846/Desktop/PhysLiteBench')
E = R / 'results/reasoning-sweep-20261011'
A = R / 'results/trajectory-analysis-20261014'
SOURCE = E / 'results.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        return node
    def strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node
    visit_Module = strip
    visit_ClassDef = strip
    visit_FunctionDef = strip

class WithoutFit(Normalize):
    def visit_FunctionDef(self, node):
        if node.name == 'fit':
            return None
        return super().visit_FunctionDef(node)
    def visit_Import(self, node):
        return None
    def visit_ImportFrom(self, node):
        return None

def functions(code):
    tree = Normalize().visit(ast.parse(code))
    out = {}
    def visit(nodes, prefix=''):
        for n in nodes:
            if isinstance(n, ast.FunctionDef):
                out[prefix+n.name] = ast.dump(n, include_attributes=False)
            elif isinstance(n, ast.ClassDef):
                visit(n.body, prefix+n.name+'.')
    visit(tree.body)
    return out

def sans_fit(code):
    return ast.dump(WithoutFit().visit(ast.parse(code)), include_attributes=False)

def public_summaries(s):
    # Tool observations only. Never inspect/export internal analysis messages.
    text = re.sub(r'\\+n', '\n', json.dumps(s.get('observation', {})))
    pattern = r'\b\d+ (?:passed|failed|errors?)(?:, \d+ (?:passed|failed|errors?|warnings?))* in [0-9.]+s'
    return list(dict.fromkeys(re.findall(pattern, text)))

def main():
    source = json.loads(SOURCE.read_text())
    rows = []
    integrity = []
    for r in source['trials']:
        p = R / r['path']
        final = p / 'artifacts/app/model.py'
        trajectory = p / 'agent/trajectory.json'
        starter = p.parent / 'frozen-task/environment/model.py'
        assert starter.exists(), starter
        for path, expected in [(final, r['source_sha256']), (trajectory, r['trajectory_sha256'])]:
            if expected:
                assert path.exists() and sha(path) == expected, path
        metrics_path = p / 'verifier/metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        test_path = p / 'verifier/test-stdout.txt'
        tests = test_path.read_text() if test_path.exists() else ''
        failed_tests = list(dict.fromkeys(re.findall(r'^(?:FAILED|ERROR)\s+([^\s]+)', tests, re.M)))
        pred_only = bool(failed_tests) and all(any(x in f for x in ['::test_prediction', '::test_fluctuation_prediction', '::test_coincidences']) for f in failed_tests)
        calibration = next((metrics[k] for k in ['calibration_chi2','calibration_reduced_chi2','chi2_per_record'] if k in metrics), None)
        parameter_error = next((metrics[k] for k in ['parameter_relative_error','parameter_relative_error_max','relative_diffusivity_error'] if k in metrics), None)
        calibration_ok = calibration is not None and calibration < 1.5
        parameter_ok = parameter_error is not None and parameter_error < (.05 if r['task']=='reaction-diffusion' else .03)
        exc = (r['exception'] or {}).get('exception_type')
        outcome = 'timeout' if exc == 'AgentTimeoutError' else 'other_execution_error' if exc else 'pass' if r['reward']==1 else 'hidden_prediction_only' if pred_only and calibration_ok and parameter_ok else 'other_verification_failure'
        changed, ast_error, preserved, exact = [], None, None, None
        if final.exists():
            a, b = starter.read_text(), final.read_text()
            exact = a == b
            try:
                fa, fb = functions(a), functions(b)
                changed = sorted(k for k in set(fa)|set(fb) if fa.get(k)!=fb.get(k))
                preserved = sans_fit(a)==sans_fit(b)
            except (SyntaxError, ValueError) as e:
                ast_error = type(e).__name__
        traj = json.loads(trajectory.read_text()) if trajectory.exists() else {}
        steps = []
        for s in traj.get('steps', []):
            calls = s.get('tool_calls', [])
            if not calls:
                continue
            args = '\n'.join(json.dumps(c.get('arguments', {})) for c in calls)
            tags = []
            if 'README.md' in args: tags.append('readme_reference')
            if re.search(r'pytest|python(?:3)?\s+[^\n]*test_public', args): tags.append('test_command_reference')
            if re.search(r'apply_patch|write_text|write_bytes|cat\s*>|sed\s+-i', args): tags.append('edit_command_reference')
            if re.search(r'python|numpy|scipy', args): tags.append('python_or_numerical_reference')
            if any(c.get('function_name') in ['write_stdin','wait'] for c in calls) or 'write_stdin' in args: tags.append('poll_or_wait')
            summaries = public_summaries(s)
            steps.append(dict(step_id=s.get('step_id'), timestamp=s.get('timestamp'), calls=len(calls), tags=tags, observed_test_summaries=summaries))
        observed_pass = any(re.search(r'\b\d+ passed in ', v) and not re.search(r'failed|error',v) for s in steps for v in s['observed_test_summaries'])
        row = {k:r[k] for k in ['model','reasoning_effort','task','trial','path','reward','reused']}
        row.update(outcome=outcome, exception_type=exc, checkpoint_pass=r['reward']==1, agent_seconds=r.get('agent_execution_seconds'), request_count=r.get('request_count'), token_usage=r.get('token_usage'), source_sha256=r['source_sha256'], trajectory_sha256=r['trajectory_sha256'], starter_sha256=sha(starter), final_code_exists=final.exists(), exact_starter_unchanged=exact, nonfit_ast_preserved=preserved, changed_functions=changed, ast_error=ast_error, calibration=calibration, calibration_ok=calibration_ok, parameter_error=parameter_error, parameter_ok=parameter_ok, failed_tests=failed_tests, hidden_metrics={k:metrics[k] for k in ['hidden','groups','hidden_nrmse','fluctuation_nrmse'] if k in metrics}, observed_public_test_pass=observed_pass, tool_call_count=sum(s['calls'] for s in steps), tool_steps=steps, verifier_summary=re.findall(r'\b\d+ (?:passed|failed|errors?)(?:, \d+ (?:passed|failed|errors?|warnings?))* in [0-9.]+s', tests)[-1:], evidence_files={str(f.relative_to(R)):sha(f) for f in [metrics_path,test_path] if f.exists()})
        rows.append(row)
    summaries=[]
    for s in source['summary']:
        rr=[r for r in rows if r['model']==s['model'] and r['reasoning_effort']==s['reasoning_effort']]
        c=collections.Counter(r['outcome'] for r in rr)
        hidden=[r for r in rr if r['outcome']=='hidden_prediction_only']
        summaries.append(dict(model=s['model'],effort=s['reasoning_effort'],attempts=len(rr),complete_panel=s['complete_panel'],outcomes=dict(c),unchanged_starter=sum(r['exact_starter_unchanged'] is True for r in rr),fit_only_hidden_failures=sum(r['nonfit_ast_preserved'] is True for r in hidden),public_pass_hidden_failure=sum(r['observed_public_test_pass'] for r in hidden),passing_timeout_checkpoints=sum(r['checkpoint_pass'] and r['exception_type'] is not None for r in rr),median_seconds=statistics.median(r['agent_seconds'] for r in rr if r['agent_seconds'] is not None),median_tool_calls=statistics.median(r['tool_call_count'] for r in rr)))
    A.mkdir(parents=True,exist_ok=True)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(SOURCE.relative_to(R)),source_sha256=sha(SOURCE),scope='All 588 primary recorded no-hint sweep attempts. 585 are in the 15 complete panels; 3 DeepSeek None attempts are preserved separately. No new model calls or regrading.',method='Extract tool actions, final Python AST and verifier evidence. Do not inspect or export internal analysis messages. Hidden-only means normal completion, passing calibration/parameter metrics, and only prediction/coincidence/fluctuation tests reported failed. AST preservation ignores fit methods, imports and docstrings; it is a structural indicator, not a full semantic proof.',trials=rows,summary=summaries)
    (A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
