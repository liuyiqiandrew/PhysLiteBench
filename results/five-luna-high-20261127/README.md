# Five Luna High candidates

Each revision received three fresh, unhinted GPT-5.6 Luna High attempts.
An attempt passes when its saved model passes the public and private verifier.
The slab, spin and surfactant revisions each recorded **0/3 graded passes** in
their final batches. Eight of those nine attempts reached the 600-second
solve limit. All nine saved submissions were graded. No candidate produced
three timeout-free physics failures in its final batch.

The table counts passes per batch. Each batch launched three attempts.
Ungraded submissions are listed explicitly.

| Task | R1 | R2 | R3 |
|---|---:|---:|---:|
| [pore-echo-memory](../../tasks/pore-echo-memory/environment/README.md) | 3/3 | 3/3 | 2/3 |
| [slab-flux-pump](../../tasks/slab-flux-pump/environment/README.md) | 2/3 | 0/3 (2 solve timeouts) | — |
| [spin-ring-fringe](../../tasks/spin-ring-fringe/environment/README.md) | 3/3 | 0 passes, 2 ungraded (3 solve timeouts)<br>0/3 (confirmation, 3 solve timeouts) | — |
| [stellar-tracer-mixing](../../tasks/stellar-tracer-mixing/environment/README.md) | 3/3 (1 solve timeout) | 1 pass, 1 fail, 1 ungraded (3 solve timeouts) | 2 passes, 1 ungraded (2 solve timeouts) |
| [surfactant-film-response](../../tasks/surfactant-film-response/environment/README.md) | 3/3 | 2/3 | 0/3 (3 solve timeouts) |

## What the revisions test

- **Pore echo memory.** R1 tests frozen local fields shared between scans. R2 measures the connected survival of two isotopes among static defects. R3 measures transmission when the source-to-detector distance is near the length of the dominant surviving pore. In R3, two solvers handled the mode shapes at absorbing and reflecting walls. The third required the pore to contain the origin even when both sources had moved. Removing that extra constraint fixed its predictions at the same fitted intensity.
- **Slab flux pump.** R1 measures a prepared slab's voltage on its second field cycle. A failed submission integrated the first cycle. R2 adds a rotating field that produces simultaneous flux cutting and depinning. The two submissions that implemented the rotation predicted it accurately. The one that finished missed the axial voltage impulse, with a maximum error of 0.806 against a 0.008 limit. A timed-out submission missed both pickup fields and the impulse. The other timed-out submission left the fit unfinished.
- **Spin ring fringe.** R1 uses thermal interference in a periodic spin ring. R2 brackets a pulse with two local transverse spin rotations. This changes the parity sector during that pulse and the phase between arms. All three initial R2 attempts hit the solve limit. One received reward zero and two exceeded the verifier limit. All three saved submissions failed hidden predictions when regraded in the pinned image with one BLAS thread. Two also failed calibration. All three fresh confirmation attempts also hit the solve limit. Their artifacts graded 0/3 with one BLAS thread. Two fitted calibration but failed all three hidden families. One left the starter unfinished.
- **Stellar tracer mixing.** R1 distinguishes slow changes from abrupt changes followed by phase mixing. R2 adds repeated weak changes and an escape threshold. One R1 passing artifact was saved before its solve timeout. All three R2 attempts reached the solve timeout. The verifier passed one artifact and failed another that omitted the phase-volume factor from its tracer counts. The third exceeded the verifier limit. A separate 109-second diagnostic passed that artifact. R3 tests coherent pairs of changes and transport across orbital resonances. Two final artifacts passed. The third exceeded the verifier limit and a separate 180-second, single-thread diagnostic. It passed both public checks during the diagnostic, which ended before hidden metrics were available.
- **Surfactant film response.** R1 tests the hydrodynamic coupling of compression and normal motion. R2 adds spontaneous fluctuations in two nonisothermal fluid layers. One R2 submission used an incorrect cross traction and reversed the temperature weighting with depth. R3 measures correlations between modes at different wavevectors and frequencies under a travelling thermal grating. All three attempts hit the solve limit and graded zero. The closest submission incorrectly projected the scalar surface dilation by the cosine of the angle between the two wavevectors. Removing that factor at the fitted viscosity brought every group below 0.004% normalized RMS error.

After the stellar R3 batch, the private oracle was corrected for a numerical
loss of precision when evaluating a very small coherent kick near the escape
boundary. Fresh container controls returned 1 and 0. The public apparatus,
calibration and verifier still match the evaluated R3 revision.

## Controls and provenance

Every evaluated revision received two container controls. The physical solution
passed. The completed approximation fit the public calibration and recovered
the parameter, then failed a private prediction. Each task's AUTHOR.md
describes the derivation, reference calculation, error metric and numerical
checks. The [trial manifest](trials.json) records the Git source commit and
tree, frozen Harbor checksum, trial ID, reward, exceptions and verifier
metrics. Job paths in the manifest are relative to the worktree's ignored
`jobs/` directory. It contains final submissions, trajectories and grader
logs.

All batches used Harbor 0.21.0, Codex CLI 0.154.0, a 600-second solve limit
and a 60-second verifier limit. The agents had web search disabled and an
OpenAI-host network allowlist. Per-task images copied only the public environment
onto the cached runtime with NumPy 2.3.3, SciPy 1.16.3 and pytest 8.4.2.
Batches shared the local Docker host and some overlapped. Solver timeouts and
ungraded artifacts are identified separately. The spin confirmation changed
the BLAS threading environment. No graded result was discarded or replaced.

To run the branch's current candidate from the repository root, substitute
its task name below. The local validator reads checked-in data. The commands
stage a copy of the task to set the same host allowlist and explicitly disable
web search in Codex. For earlier revisions use the source commit in the
manifest. The delivered spin-ring Dockerfile sets the two thread variables used
in its confirmation.

```bash
task=slab-flux-pump
stem=${task//-/_}
OPENBLAS_NUM_THREADS=1 python3 "scripts/validate_${stem}.py"
python3 scripts/run_science.py "tasks/$task" --agent oracle --trials 1
python3 scripts/run_science.py "tasks/$task" --agent oracle --trials 1 \
  --solution-model "scripts/${stem}_baseline.py"
batch_dir=$(mktemp -d)
cp -R "tasks/$task" "$batch_dir/$task"
python3 - "$batch_dir/$task/task.toml" <<'PYTHON'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_text(p.read_text().replace(
    "[agent]", '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["chatgpt.com", "*.openai.com"]'
))
PYTHON
harbor run -p "$batch_dir/$task" -a codex --env docker -m gpt-5.6-luna \
  -k 3 -n 3 --ak version=0.154.0 --ak reasoning_effort=high \
  --ak web_search=disabled --ae "CODEX_AUTH_JSON_PATH=$HOME/.codex/auth.json" \
  --agent-setup-timeout-multiplier 3
```
