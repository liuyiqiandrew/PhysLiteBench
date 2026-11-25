"""Preserve the complete reviewed r7 candidate and its runnable dependencies."""
from pathlib import Path
import hashlib
import json
import shutil

STAGE = Path(__file__).resolve().parents[1]
REPO = STAGE.parents[1]
ARCHIVE = STAGE / 'archives/rotating-reservoir-r7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc'))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    assert not ARCHIVE.exists(), 'Never overwrite a preserved candidate'
    report = json.loads((STAGE / 'results/rotating-reservoir-r7-trial-review.json').read_text())
    assert report['physical_passes']==3 and report['frozen_sources_unchanged']
    for rel, digest in report['frozen_source_sha256'].items():
        assert sha(STAGE / rel)==digest
    copy(STAGE/'tasks/rotating-reservoir', ARCHIVE/'tasks/rotating-reservoir')
    for name in ['rotating_reservoir_baseline.py', 'validate_rotating_reservoir.py',
                 'review_rotating_r7.py', 'finalize_rotating_r7_review.py', 'archive_rotating_r7.py',
                 'collect_evaluation.py', 'freeze_evaluation.py']:
        copy(STAGE/'scripts'/name, ARCHIVE/'scripts'/name)
    for source in sorted((STAGE/'results').glob('rotating-reservoir-r7-*')):
        copy(source, ARCHIVE/'results'/source.name)
    for name in ['rotating-reservoir-r7-root', 'rotating-reservoir-r7-brownian']:
        copy(STAGE/'results/prototypes'/name, ARCHIVE/'results/prototypes'/name)
    # These complete r6 files preserve the inherited mean/noise/calibration
    # proof and the prior report's source hashes required by the r7 validator.
    copy(STAGE/'archives/rotating-reservoir-r6', ARCHIVE/'archives/rotating-reservoir-r6')
    jobs = sorted((REPO/'jobs').glob('rotating-reservoir-r7-*-20261006-042142'))
    assert len(jobs)==3
    for source in jobs:
        assert (source/'run-timing.json').exists()
        copy(source, ARCHIVE/'jobs'/source.name)
    # Original executed reviewers are untouched. Separate copies allow
    # rerunning the deterministic audit using the archive's local raw jobs.
    for name in ['review_rotating_r7.py', 'finalize_rotating_r7_review.py']:
        original = (ARCHIVE/'scripts'/name).read_text()
        if name=='review_rotating_r7.py':
            replacement=original.replace("JOB=REPO/'jobs/", "JOB=STAGE/'jobs/")
        else:
            replacement=original.replace("JOB = STAGE.parents[1] / 'jobs/", "JOB = STAGE / 'jobs/")
        assert replacement!=original
        (ARCHIVE/'scripts'/('reproduce_'+name)).write_text(replacement)
    (ARCHIVE/'README.md').write_text(
        '# Preserved rotating-reservoir revision 7\n\n'
        'This frozen candidate produced three physical passes, with one original '
        '600-second AgentTimeoutError retained alongside its installed passing code. '
        'It does not qualify for the requested zero-of-three target. No trials were rerun.\n\n'
        'The task, completed control, validator, exact executed review scripts, '
        'all three original jobs (plain/oracle/shortcut), native sessions, submitted '
        'files, exceptions, hashes, science evidence and previous full r6 dependencies '
        'are preserved. The r7 validator and prototypes resolve their dependencies '
        'inside this archive. Initial numerical-failure prototypes retain their '
        'original exploratory paths and are historical evidence, not production references.\n\n'
        'For deterministic review reproduction, use scripts/reproduce_review_rotating_r7.py '
        'then scripts/reproduce_finalize_rotating_r7_review.py; those copies change only '
        'the raw-job lookup to this archive. Original executed scripts are unchanged. '
        'The final report includes fixed-native-drag independent comparisons and a '
        'one-factor Gaussian-closure ablation; every original reward remains one.\n')
    files={str(path.relative_to(ARCHIVE)):sha(path) for path in sorted(ARCHIVE.rglob('*')) if path.is_file()}
    manifest=dict(task='rotating-reservoir', revision=7, files=files,
                  raw_passes=3, exceptions=1, clean_physical_failures=0,
                  trial_review_sha256=sha(STAGE/'results/rotating-reservoir-r7-trial-review.json'))
    target=ARCHIVE/'preservation-manifest.json'
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    assert all(sha(ARCHIVE/rel)==digest for rel,digest in files.items())
    assert all(sha(ARCHIVE/rel)==digest for rel,digest in report['frozen_source_sha256'].items())
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(files), manifest_sha256=sha(target), all_verified=True)), flush=True)


if __name__=='__main__':
    main()
