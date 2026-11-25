from pathlib import Path
from datetime import datetime, timezone
import ast
import collections
import hashlib
import json
import shutil

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
STAGING = ROOT / 'staging/hint-comparison-20261014'
RESULTS = ROOT / 'results/hint-comparison-20261014'
MODELS = ['gpt-5.6-luna', 'gpt-5.6-terra', 'deepseek/deepseek-v4.1-flash']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    plan_path = RESULTS / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert len(plan['models']) == 5 and plan['expected_attempts'] == 585
    observed = []
    for path in sorted((RESULTS / 'blocks').glob('*/block.json')):
        block = json.loads(path.read_text())
        assert block['model'] in MODELS, 'An excluded model already has a block; preserve and report it separately'
        observed.append({'index': block['index'], 'model': block['model'], 'status': block['status']})
    history = RESULTS / 'history/before-three-model-amendment'
    history.mkdir(parents=True)
    for name in ['plan.json', 'budget-authorization.json', 'status.json', 'results.json', 'summary.csv', 'REPORT.txt']:
        path = RESULTS / name
        if path.exists():
            shutil.copyfile(path, history / name)
    shutil.copytree(STAGING / 'tooling', history / 'tooling')
    parent_sha = sha(plan_path)
    plan['models'] = MODELS
    plan['expected_attempts'] = 351
    plan['openai_attempts'] = 234
    plan['prime_attempts'] = 117
    removed = [b for b in plan['blocks'] if b['model'] not in MODELS]
    plan['blocks'] = [b for b in plan['blocks'] if b['model'] in MODELS]
    assert len(plan['blocks']) == 117
    slots = collections.Counter((b['model'], b['task']) for b in plan['blocks'])
    assert len(slots) == 39 and all(value == 3 for value in slots.values())
    stamp = datetime.now(timezone.utc).isoformat()
    amendment = {
        'recorded_utc': stamp,
        'user_request': 'Keep only GPT-5.6 Luna, GPT-5.6 Terra and DeepSeek; 13 tasks, 3 conditions, 3 repeats, always High effort.',
        'previous_plan_sha256': parent_sha,
        'previous_plan': str((history / 'plan.json').relative_to(ROOT)),
        'models': MODELS,
        'expected_attempts': 351,
        'removed_unstarted_blocks': [b['index'] for b in removed],
        'blocks_already_present': observed,
        'preserve_original_block_indices': True,
        'retained_trial_instructions_sources_grading_and_effort_unchanged': True,
        'removed_models_started_attempts': 0,
        'scope_change_is_user_requested_not_outcome_selected': True,
        'estimated_elapsed_hours': [14, 17],
        'timing_basis': 'Prior three-trial task maxima: Luna about 3.4 hours, Terra 4.1 hours, DeepSeek 7.1 hours for 39 blocks each; hints and provider interruptions can change duration.',
    }
    plan['scope_amendment'] = 'three-model-amendment.json'
    plan['previous_plan_sha256'] = parent_sha
    plan['protocol'].append('User narrowed the model roster after the first Luna block and during the first Terra block. Preserve these compatible observations and their original plan hashes; no prompt, source, grading or model-effort change.')
    path = STAGING / 'tooling/collect.py'
    text = path.read_text()
    text = text.replace("'expected_attempts': 585", "'expected_attempts': plan['expected_attempts']")
    text = text.replace("'Five fixed models, High effort, three fresh repeats per condition and task.'", "f\"{len(plan['models'])} fixed models, High effort, three fresh repeats per condition and task.\"")
    text = text.replace("{len(rows)}/585 attempts; {status['complete_model_condition_panels']}/15 complete panels.", "{len(rows)}/{plan['expected_attempts']} attempts; {status['complete_model_condition_panels']}/{len(plan['models'])*len(plan['conditions'])} complete panels.")
    path.write_text(text)
    path = STAGING / 'tooling/controller.py'
    text = path.read_text().replace("'attempts': 585", "'attempts': plan['expected_attempts']")
    path.write_text(text)
    path = STAGING / 'tooling/plot.py'
    text = path.read_text().replace("names = ['GPT-5.6 Luna', 'GPT-5.6 Terra', 'GPT-6 Sol', 'GPT-6 Astra', 'DeepSeek V4.1 Flash']", "display = {'gpt-5.6-luna': 'GPT-5.6 Luna', 'gpt-5.6-terra': 'GPT-5.6 Terra', 'deepseek/deepseek-v4.1-flash': 'DeepSeek V4.1 Flash'}\n    names = [display[model] for model in plan['models']]")
    text = text.replace('x = np.arange(5)', "x = np.arange(len(plan['models']))")
    text = text.replace("{data['status']['recorded_attempts']}/585 attempts recorded", "{data['status']['recorded_attempts']}/{plan['expected_attempts']} attempts recorded")
    path.write_text(text)
    for path in (STAGING / 'tooling').glob('*.py'):
        ast.parse(path.read_text())
        shutil.copyfile(path, RESULTS / 'tooling' / path.name)
    plan['tooling_sha256'] = {relative: sha(STAGING / relative) for relative in plan['tooling_sha256']}
    write(plan_path, plan)
    amendment['new_plan_sha256'] = sha(plan_path)
    write(RESULTS / 'three-model-amendment.json', amendment)
    budget = json.loads((RESULTS / 'budget-authorization.json').read_text())
    budget['original_five_model_estimate'] = {key: budget.pop(key) for key in ['estimate_additional_percentage_points', 'estimate_midpoint', 'estimate_basis']}
    budget['current_model_scope'] = MODELS
    budget['current_expected_attempts'] = 351
    budget['estimate_status'] = 'Previous 45–65-point estimate applied to five models and is superseded; do not reuse it as the three-model forecast.'
    write(RESULTS / 'budget-authorization.json', budget)
    shutil.copyfile(Path(__file__), RESULTS / 'tooling/amend_three_models.py')
    print(json.dumps({'models': MODELS, 'expected_attempts': 351, 'blocks': 117, 'removed_unstarted_blocks': len(removed), 'already_started_excluded_models': 0, 'prior_luna_results_preserved': True}))


if __name__ == '__main__':
    main()
