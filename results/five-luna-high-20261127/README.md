# Five physics tasks on GPT-5.6 Luna High

All five tasks scored **0/3** on unhinted GPT-5.6 Luna High. All 15 agents
returned a final answer, passed the public and private calibration checks and
produced finite predictions. Each failed at least one private physics check.
There were **zero solve timeouts, verifier timeouts or ungraded attempts** in
the five selected batches. The reference solution passed for each task.

The [grading and physics audit](AUDIT.md) reproduced all 25 original grades
and identified two public/private fit discrepancies. The current tasks include
the audit fixes. Replaying the archived submissions on the revised graders
gave 0/3 for each task. The fresh Luna runs below used the original revisions.

| Delivered task | Solve time across its three trials | Physics error observed |
|---|---:|---|
| Pore echo memory R4 | 56–82 s | Approximating the lengths of surviving valleys gave the wrong probability of detecting two tracers together. |
| Slab flux pump R4 | 459–712 s | The solvers mishandled magnetic screening or the direction and magnitude of the critical current in a superconducting film experiment. |
| Spin ring fringe R3 | 54–84 s | The fermion-mode overlap omitted parity strings between the selected physical spins. |
| Stellar tracer mixing R5 | 217–590 s | Two submissions used free streaming. The third gave the wrong amplitude for the later gravitational echo. |
| Surfactant film response R4 | 424–826 s | Treating molecular speeds as independent gave zero probability for neighbors travelling at the same speed. |

Times are rounded to seconds of host-awake elapsed time. The longest surfactant
trial spanned 1,946.8 wall-clock seconds. The host sleep log accounts for
1,121 seconds of that interval, leaving 825.8 seconds against a 1,800-second
solve limit. Its Codex stream ended with `turn.completed` and Harbor recorded
a grade and no exception.

## Revision history

Each batch used three fresh attempts. The bold results identify the delivered
revisions. The table includes all 63 Luna attempts in 21 batches, including
earlier timeouts and ungraded submissions.

| Task | R1 | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|
| [pore-echo-memory](../../tasks/pore-echo-memory/environment/README.md) | 3/3 | 3/3 | 2/3 | **0/3** | — |
| [slab-flux-pump](../../tasks/slab-flux-pump/environment/README.md) | 2/3 | 0/3 (2 solve timeouts) | 1/3 | **0/3** | — |
| [spin-ring-fringe](../../tasks/spin-ring-fringe/environment/README.md) | 3/3 | 0 passes, 2 ungraded (3 solve timeouts)<br>0/3 (confirmation, 3 solve timeouts) | **0/3** | — | — |
| [stellar-tracer-mixing](../../tasks/stellar-tracer-mixing/environment/README.md) | 3/3 (1 solve timeout) | 1 pass, 1 fail, 1 ungraded (3 solve timeouts) | 2 passes, 1 ungraded (2 solve timeouts) | 2/3 | **0/3** |
| [surfactant-film-response](../../tasks/surfactant-film-response/environment/README.md) | 3/3 | 2/3 | 0/3 (3 solve timeouts) | **0/3** | — |

## What the revisions test

- **Pore echo memory.** R1 tests frozen local fields shared between scans. R2 measures the connected survival of two isotopes among static defects. R3 measures transmission when the source-to-detector distance is near the length of the dominant surviving pore. In R3, two solvers handled the mode shapes at absorbing and reflecting walls. The third required the pore to contain the origin even when both sources had moved. Removing that extra constraint fixed its predictions at the same fitted intensity. R4 measures the probability that both tracers survive in the same valley and fall inside a specified detector window. All three submissions fit the sink intensity and retained the starter’s approximation for the lengths of surviving valleys. The largest probability error was 0.04884 against a 0.003 limit. Replacing the prediction with the reference calculation at each fitted intensity reduced the largest error below 2 × 10⁻⁹.
- **Slab flux pump.** R1 measures a prepared slab's voltage on its second field cycle. A failed submission integrated the first cycle. R2 adds a rotating field that produces simultaneous flux cutting and depinning. The two submissions that implemented the rotation predicted it accurately. The one that finished missed the axial voltage impulse, with a maximum error of 0.806 against a 0.008 limit. A timed-out submission missed both pickup fields and the impulse. The other timed-out submission left the fit unfinished. R3 adds a conducting sheet in a pre-threaded rotating slab. One submission passed. One dropped the sign of the normal induction and read the field on the wrong side of the sheet. Correcting both at the fitted pinning scale reduced its induction error from 0.03954 to less than 0.000000004. R4 adds a superconducting film that can screen the material below it and separates the bulk current limits for cutting and depinning. All three submissions reproduced the calibration and earlier conducting-sheet readouts. Each failed the film measurements. One used a conducting-sheet solution where the film screens the substrate. Another exceeded the depinning current limit by 28% in an allowed local state. The third held the depinning threshold fixed and dropped components of the current. Correcting the respective boundary and current relations at each fitted pinning scale brought every film error below 0.000000072.
- **Spin ring fringe.** R1 uses thermal interference in a periodic spin ring. R2 brackets a pulse with two local transverse spin rotations. This changes the parity sector during that pulse and the phase between arms. All three initial R2 attempts hit the solve limit. One received reward zero and two exceeded the verifier limit. All three saved submissions failed hidden predictions when regraded in the pinned image with one BLAS thread. Two also failed calibration. All three fresh confirmation attempts also hit the solve limit. Their artifacts graded 0/3 with one BLAS thread. Two fitted calibration but failed all three hidden families. One left the starter unfinished. R3 measures the overlap when two thermal rings exchange selected physical spins. All three submissions passed calibration and used the starter’s fermion-mode overlap for separated spins. Including the Jordan–Wigner strings through the intervening sites at each submission’s fitted visibility reduced the largest family RMS error from 0.2744 to 0.0000251. The limit is 0.008.
- **Stellar tracer mixing.** R1 distinguishes slow changes from abrupt changes followed by phase mixing. R2 adds repeated weak changes and an escape threshold. One R1 passing artifact was saved before its solve timeout. All three R2 attempts reached the solve timeout. The verifier passed one artifact and failed another that omitted the phase-volume factor from its tracer counts. The third exceeded the verifier limit. A separate 109-second diagnostic passed that artifact. R3 tests coherent pairs of changes and transport across orbital resonances. Two final artifacts passed. The third exceeded the verifier limit and a separate 180-second, single-thread diagnostic. It passed both public checks during the diagnostic, which ended before hidden metrics were available. R4 uses two drifting populations in a stellar stream. A velocity impulse leaves a pattern that reappears after a second impulse. The spectrometer measures that echo in a selected population and velocity band. Two submissions passed. The third recovered the dispersion and missed the band signal by 0.00611 against a 0.00045 tolerance. R5 also asks for the part of the signal that stays finite as the pulse separation grows, and for a later echo generated by the gravity of the first echo. All three R5 submissions recovered the dispersion. Two retained the starter’s free-streaming prediction and failed all five private groups. The third passed four groups and missed the later echo by 0.02455 against the same 0.00045 tolerance. Refining its velocity grid alone left an error of 0.01547. Also correcting the relay amplitude brought the error down to 0.0000243 at the original fitted dispersion.
- **Surfactant film response.** R1 tests the hydrodynamic coupling of compression and normal motion. R2 adds spontaneous fluctuations in two nonisothermal fluid layers. One R2 submission used an incorrect cross traction and reversed the temperature weighting with depth. R3 measures correlations between modes at different wavevectors and frequencies under a travelling thermal grating. All three attempts hit the solve limit and graded zero. The closest submission incorrectly projected the scalar surface dilation by the cosine of the angle between the two wavevectors. Removing that factor at the fitted viscosity brought every group below 0.004% normalized RMS error. R4 tracks neighbouring molecules on a driven monolayer. All three completed submissions assumed independent continuous speeds and assigned zero probability to adjacent molecules sharing a speed. For one preparation the exact probability is 0.2454256. Restoring the joint speed distribution at the original fitted frequency reduced all errors below 0.000000006.

After the stellar R3 batch, the private oracle was corrected for a numerical
loss of precision when evaluating a very small coherent kick near the escape
boundary. Fresh container controls returned 1 and 0. The public apparatus,
calibration and verifier matched the evaluated R3 revision after that correction.

## Controls and provenance

Every evaluated revision received two container controls. The physical solution
passed. The completed approximation fit the public calibration and recovered
the parameter, then failed a private prediction. Each task's AUTHOR.md
describes the derivation, reference calculation, error metric and numerical
checks. The [trial manifest](trials.json) records the Git source commit and
tree, frozen Harbor checksum, trial ID, reward, exceptions and verifier
metrics. Its physical-failure review records the submitted code hashes, solve and verifier durations, failed checks, diagnoses and fixed-parameter diagnostics for all 15 selected submissions. Job paths in the manifest are relative to the worktree's ignored
`jobs/` directory. It contains final submissions, trajectories and grader
logs.

All batches used Harbor 0.21.0 and Codex CLI 0.154.0. The initial study
allowed 600 seconds to solve and 60 seconds to verify. For the new slab, spin
and surfactant revisions, those limits were 1,800 and 180 seconds. Panel-specific
limits in the manifest override its original defaults. The agents had web search disabled and an
OpenAI-host network allowlist. Per-task images copied only the public environment
onto the cached runtime with NumPy 2.3.3, SciPy 1.16.3 and pytest 8.4.2.
Batches shared the local Docker host and some overlapped. Solver timeouts and
ungraded artifacts are identified separately. The spin R2 confirmation changed
the BLAS threading environment. No graded result was discarded or replaced.

To run one of the five tasks from the repository root, substitute
its task name below. The local validator reads checked-in data. The commands
stage a copy of the task to set the same host allowlist and explicitly disable
web search in Codex. For earlier revisions use the source commit in the
manifest. The spin-ring image uses a single BLAS thread.

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
