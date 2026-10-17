from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random
import shutil
import subprocess

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
NAME = 'hint-comparison-20261014'
STAGING = ROOT / 'staging' / NAME
RESULTS = ROOT / 'results' / NAME
OLD = ROOT / 'staging/reasoning-sweep-20261011'
GENERIC = 'Check your implementation carefully and test its correctness before finishing.'
NEW_HINTS = {
    'ring-current-fluctuations': (
        'A current-conditioned ring need not be dominated by a homogeneous density profile. '
        'Check the stability of the homogeneous profile and allow a spatially modulated '
        'profile in the long-time variational problem. Translation invariance of the '
        'unaligned ensemble mean does not exclude such profiles.'
    ),
    'swirling-heat-tails': (
        'Heat differs from circulating-force work by the change in trap energy. '
        'Although that change is usually bounded on the time scale of the mean, '
        'rare initial or final energies can scale with observation time. Retain '
        'both endpoint factors when checking convergence of the exponential heat '
        'moment and its admissible counting-field domain.'
    ),
    'hard-rod-current': (
        'The exponential current average includes the freshly drawn rod-length '
        'order. A large current can select an atypical composition of the net '
        'crossed block. Retain the probability cost of that composition together '
        'with the free-particle crossing cost before optimizing; replacing the '
        'crossed lengths by their ordinary mean loses this contribution.'
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob('*'))
        if p.is_file() and not any(x in p.parts for x in ['__pycache__', '.pytest_cache', '.git'])
    }


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    assert not STAGING.exists() and not RESULTS.exists(), 'Experiment already exists; do not overwrite it'
    (STAGING / 'tooling').mkdir(parents=True)
    (RESULTS / 'tooling').mkdir(parents=True)
    prior = json.loads((ROOT / 'results/reasoning-sweep-20261011/plan.json').read_text())
    plan = {
        'experiment': NAME,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'models': ['gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-6-sol', 'gpt-6-astra', 'deepseek/deepseek-v4.1-flash'],
        'reasoning_effort': 'high',
        'conditions': ['baseline', 'general', 'physics'],
        'repeats': 3,
        'expected_attempts': 585,
        'openai_attempts': 468,
        'prime_attempts': 117,
        'agent_seconds': 600,
        'verifier_seconds': 60,
        'codex_cli': '0.160.0',
        'harbor': '0.21.0',
        'setup_timeout_multiplier': 3,
        'max_concurrent_trials': 3,
        'general_hint': GENERIC,
        'instruction_wrapper': '\n\nAdditional guidance:\n',
        'fresh_baselines': True,
        'reuse_historical_trials': False,
        'schedule_seed': 20261014,
        'tasks': {},
        'blocks': [],
        'protocol': [
            'Freeze all 13 task sources and all three instructions before the first trial.',
            'For each model/task/repeat block, launch one separate trial per condition, in seeded randomized order, up to three concurrently.',
            'Only instruction.md differs among conditions for a given provider and task. Match public environment, private grading, runtime and reasoning effort.',
            'Keep all started outcomes. Do not replace a timeout, a scored failure or a passing trial.',
            'Preserve pre-model setup failures separately; any replacement requires a recorded review before launch.',
            'Count a completed pass only when reward is 1 with no execution exception. Report passing checkpoints and execution errors separately.',
            'A failed grade is not classified as a physics error without reviewing the source and trajectory.',
            'All hints are fixed conceptual guidance, with no hidden cases, true fitted parameters, grading thresholds or executable solution.',
            'General and physics hints differ in length and specificity; this estimates the effect of these prompt packages, not a length-controlled semantic effect.',
            'Preserve GPT-5.6 Luna exactly; do not substitute GPT-6 Luna.',
            'The previous account ceiling is lifted by the user. Prime spending has no user-imposed cap. Record usage and stop on actual access/rate-limit failures.',
        ],
        'prior_context': 'The thirteen tasks were developed/selected against GPT-5.6 Luna. The old thermal-piston pilot and old reasoning-effort results are historical, not part of this experiment.',
    }
    audit = []
    for name, previous in prior['tasks'].items():
        source = ROOT / 'tasks' / name
        source_hashes = hashes(source)
        assert source_hashes == previous['qualified_source_sha256'], ('Task drift since earlier sweep', name)
        hint_path = source / 'hint.md'
        hint = hint_path.read_text().strip() if hint_path.exists() else NEW_HINTS[name]
        row = {
            'revision': previous['revision'],
            'source_sha256': source_hashes,
            'physics_hint': hint,
            'physics_hint_source': str(hint_path.relative_to(ROOT)) if hint_path.exists() else 'New conceptual hint derived from the current AUTHOR.md and public apparatus',
            'new_hint': not hint_path.exists(),
            'author_sha256': sha(source / 'AUTHOR.md'),
            'runtime': {},
        }
        baseline = (source / 'instruction.md').read_text()
        for provider in ['openai', 'prime']:
            configs = {}
            for condition in plan['conditions']:
                target = STAGING / 'tasks' / provider / condition / name
                shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.git'))
                if condition != 'baseline':
                    extra = GENERIC if condition == 'general' else hint
                    (target / 'instruction.md').write_text(baseline.rstrip() + plan['instruction_wrapper'] + extra + '\n')
                config = (target / 'task.toml').read_text()
                assert config.count('[agent]') == 1 and 'network_mode' not in config
                hosts = '["chatgpt.com", "*.openai.com"]' if provider == 'openai' else '["host.docker.internal"]'
                config = config.replace('[agent]', '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ' + hosts)
                (target / 'task.toml').write_text(config)
                hh = hashes(target)
                configs[condition] = hh
                row['runtime'][provider + '/' + condition] = {'path': str(target.relative_to(ROOT)), 'sha256': hh}
            without_instruction = [{k: v for k, v in value.items() if k != 'instruction.md'} for value in configs.values()]
            assert without_instruction[0] == without_instruction[1] == without_instruction[2]
            assert (STAGING / 'tasks' / provider / 'baseline' / name / 'instruction.md').read_bytes() == (source / 'instruction.md').read_bytes()
            audit.append({'task': name, 'provider': provider, 'only_instruction_differs': True, 'baseline_instruction_unchanged': True})
        plan['tasks'][name] = row

    for provider, original in [('openai', OLD / 'tooling/run_science.py'), ('prime', OLD / 'prime/run_science_v2.py')]:
        text = original.read_text()
        text = text.replace('parser.add_argument("--hint",', 'parser.add_argument("--prompt-condition", choices=("baseline", "general", "physics"), default="baseline")\n    parser.add_argument("--hint",')
        text = text.replace('"physics_hint": args.hint,', '"physics_hint": args.hint or args.prompt_condition == "physics",\n                    "prompt_condition": args.prompt_condition,')
        text = text.replace('/private/tmp/physlite-prime-session/auth.json', '/private/tmp/physlite-hints-prime-session/auth.json')
        text = text.replace(str(OLD / 'prime/codex-v2.toml'), str(STAGING / 'prime/codex.toml'))
        if provider == 'prime':
            text = text.replace('parser.add_argument("--job-name")', 'parser.add_argument("--job-name")\n    parser.add_argument("--prime-config", type=Path, required=True)')
            text = text.replace('"config=' + str(STAGING / 'prime/codex.toml') + '"', 'f"config={args.prime_config}"')
        (STAGING / 'tooling' / f'run_science_{provider}.py').write_text(text)
    (STAGING / 'prime').mkdir()
    for name in ['prime_catalog_probe.py', 'prime_proxy_v2.py']:
        text = (OLD / 'prime' / name).read_text().replace('/private/tmp/physlite-prime-session', '/private/tmp/physlite-hints-prime-session')
        if name == 'prime_proxy_v2.py':
            assert text.count("dict(id=reqid,at=now(),status='pending'") == 1
            text = text.replace("dict(id=reqid,at=now(),status='pending'", "dict(id=reqid,at=now(),trial_tag=self.headers.get('X-Physlite-Trial'),status='pending'")
        (STAGING / 'prime' / name).write_text(text)
    (STAGING / 'prime/codex.toml').write_text((OLD / 'prime/codex-v2.toml').read_text().replace('18764', '18765').replace('capped local relay', 'logged local relay'))
    for name in ['run_block.py', 'collect.py', 'plot.py', 'controller.py']:
        source = Path('/private/tmp/physlite-hint-tooling') / name
        if source.exists():
            shutil.copyfile(source, STAGING / 'tooling' / name)
    shutil.copyfile(OLD / 'tooling/prior_collector.py', STAGING / 'tooling/prior_collector.py')
    prices = json.loads((ROOT / 'results/reasoning-sweep-20261011/pricing.json').read_text())
    prime_prices = json.loads((ROOT / 'results/reasoning-sweep-20261011/prime/pricing-v2.json').read_text())
    prices['models']['deepseek/deepseek-v4.1-flash'].update({key: prime_prices[key] for key in ['input', 'cached_input', 'cache_write_input', 'output', 'apply_openai_long_context_rates']})
    prices['notes'].append('Carry forward the saved October 11 OpenAI and October 12 Prime rates solely as a fixed cost-comparison convention. These are not a current-price or subscription-quota quote.')
    write(RESULTS / 'pricing.json', prices)
    rng = random.Random(plan['schedule_seed'])
    names = list(plan['tasks'])
    index = 0
    for repeat in range(1, 4):
        for task_index, name in enumerate(names):
            models = plan['models'][:]
            if repeat > 1 or task_index > 0:
                rng.shuffle(models)
            for model in models:
                order = plan['conditions'][:]
                rng.shuffle(order)
                index += 1
                plan['blocks'].append({'index': index, 'task': name, 'model': model, 'repeat': repeat, 'condition_order': order})
    assert len(plan['blocks']) == 195
    plan['tooling_sha256'] = {str(p.relative_to(STAGING)): sha(p) for folder in ['tooling', 'prime'] for p in sorted((STAGING / folder).glob('*')) if p.is_file()}
    write(RESULTS / 'plan.json', plan)
    write(RESULTS / 'preflight-audit.json', {'checked_utc': plan['created_utc'], 'result': 'passed', 'canonical_task_count': 13, 'runtime_snapshots': 78, 'paired_checks': audit, 'new_hint_tasks': list(NEW_HINTS), 'canonical_tasks_unchanged': True, 'no_model_calls_made': True})
    write(RESULTS / 'budget-authorization.json', {'recorded_utc': plan['created_utc'], 'user_message': 'The previous account usage limit does not apply here; estimate account usage. Prime credit amount does not need a cap.', 'account_cap_used_percent': None, 'prime_cap_usd': None, 'initial_used_percent': 7, 'window_minutes': 10080, 'window_resets_at': 1792472328, 'estimate_additional_percentage_points': [45, 65], 'estimate_midpoint': 55, 'estimate_basis': 'Earlier High panels across four OpenAI models (156 attempts) overlapped roughly 17 account percentage points; three fresh prompt conditions require 468 OpenAI attempts. Broad range allows model substitution, hint effects, account rounding and orchestration; this is not a token-to-quota formula.'})
    write(RESULTS / 'status.json', {'state': 'prepared', 'expected_attempts': 585, 'recorded_attempts': 0, 'completed_blocks': 0})
    shutil.copytree(STAGING / 'tooling', RESULTS / 'tooling', dirs_exist_ok=True)
    shutil.copyfile(Path(__file__), RESULTS / 'tooling/prepare.py')
    write(RESULTS / 'model-access-preflight.json', {'source': 'Local Codex model catalog and official GPT-5.6 Luna model documentation', 'requested_models_present_in_local_catalog': plan['models'][:-1], 'high_effort_listed': True, 'live_access_checked': False, 'official_luna_documentation': 'https://developers.openai.com/api/docs/models/gpt-5.6-luna'})
    print(json.dumps({'prepared': NAME, 'tasks': 13, 'models': 5, 'conditions': 3, 'attempts': 585, 'new_hints': list(NEW_HINTS), 'paired_hash_checks': len(audit)}))


if __name__ == '__main__':
    main()
