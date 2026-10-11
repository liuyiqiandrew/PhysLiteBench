# Excluded expansion candidates

Nineteen evaluated revisions are preserved here. Seventeen passed all three frozen
unhinted `gpt-5.6-luna` High trials. Hard-rod current revision 1 and counterion cloud revision 2 each passed one
of three, with two reviewed physical failures repaired by changing only
the response law at the original fitted parameter. All fifty-seven final
submissions and trajectories were reviewed. None meets the requested
0/3 criterion, and none was resampled unchanged.

| Candidate | Passes | Reviewed results |
|---|---:|---|
| Pulsed kinetic gas | 3/3 | [Results](results/expansion-pulsed-kinetic-gas-r1-results.json) |
| Driven jump ring | 3/3 | [Results](results/expansion-driven-jump-ring-r1-results.json) |
| Quenched spin activity | 3/3 | [Results](results/expansion-quenched-spin-activity-r1-results.json) |
| Compensated waveguide | 3/3 | [Results](results/expansion-compensated-waveguide-r1-results.json) |
| Accelerated detector pair | 3/3 | [Results](results/expansion-accelerated-detector-pair-r1-results.json) |
| Hard-rod current | 1/3 | [Results](results/expansion-hard-rod-current-r1-results.json) |
| Mobile-trap survival r1 | 3/3 | [Results](results/expansion-mobile-trap-survival-r1-results.json) |
| Driven mobile-trap survival r2 | 3/3 | [Results](mobile-trap-survival-r2/results/expansion-mobile-trap-survival-r2-results.json) |
| Annihilating gas target | 3/3 | [Results](results/expansion-annihilating-gas-target-r1-results.json) |
| Disordered wave chain | 3/3 | [Results](results/expansion-disordered-wave-chain-r1-results.json) |
| Coarse pointer | 3/3 | [Results](results/expansion-coarse-pointer-r1-results.json) |
| Trapped winding r1 | 3/3 | [Results](results/expansion-trapped-winding-r1-results.json) |
| Trapped winding r2 | 3/3 | [Results](trapped-winding-r2/results/expansion-trapped-winding-r2-results.json) |
| Finite-bath gravitating gas r1 | 3/3 | [Results](results/expansion-finite-bath-gravitating-gas-r1-results.json) |
| Binary gravitating gas r1 | 3/3 | [Results](results/expansion-binary-gravitating-gas-r1-results.json) |
| Counterion cloud r1 | 3/3 | [Results](results/expansion-counterion-cloud-r1-results.json) |
| Counterion cloud r2 | 1/3 | [Results](counterion-cloud-r2/results/expansion-counterion-cloud-r2-results.json) |
| Finite-ion cloud r1 | 3/3 | [Results](results/expansion-finite-ion-cloud-r1-results.json) |
| Finite-ion cloud r2 | 3/3 | [Results](finite-ion-cloud-r2/results/expansion-finite-ion-cloud-r2-results.json) |

Each package includes an oracle, a completed physically wrong control, an
independent reference, a 256-draw calibration-noise study, an independent
physics review, and fresh Docker controls. Scientific consistency did not
establish model difficulty. Reports retain the frozen hashes, original job
names and trial IDs. Evidence folders contain final submitted models and
verifier outputs; full trajectories and frozen jobs remain in the repository
root's ignored `jobs/` directory.

The original `gpt-6-luna` attempt was rejected as unsupported by the current
account. Those access failures are preserved separately and count as zero
scientific trials. The user then authorized `gpt-5.6-luna` High.

This archive move changes paths only. In the preserved reports, original
`tasks/`, `scripts/`, and `results/` paths resolve relative to this archive
unless they refer to root `jobs/`. References to pre-existing archive
families still use their original repository-root context; review text and
hashes are preserved. The validators also resolve their task
and report paths relative to this directory. Run them from the repository
root with the pinned numerical environment, for example:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/physlite-uv-cache \
uv run --offline --no-project --python 3.13 --with numpy==2.3.3 \
  --with scipy==1.16.3 --with pytest==8.4.2 python \
  archives/physics-expansion-20261011/scripts/validate_accelerated_detector_pair.py
```

The heat-Hall concept review is also retained here. It was dropped before
packaging because it reused the archived entropy-anomaly mechanism; no
model trials were run for it.

Driven mobile-trap revision 2 is self-contained in `mobile-trap-survival-r2/`; its original report and validator paths resolve relative to that nested archive. Its new force/temperature apparatus and exponential readout are kept separate from revision 1.

The quantum-oscillator and collisionless-echo concept reviews are retained under `results/`. Both were dropped before packaging because they reused archived physical mechanisms. No model trials were run. The gas report records shell-based literature retrieval despite the built-in web tool being disabled.

The driven-optical-binding concept was dropped before packaging because its proposed shortcut was not a consistent physical control. The review is retained in `results/`.

Trapped winding revision 2 is self-contained in `trapped-winding-r2/`; original source/report paths resolve from that directory. It changes the readout to joint winding and radial occupation, and supplies an exact-marginal independent-copy control. The isolated-gravitating-gas concept was dropped before packaging because its proposed stability flags were a weak test; its critique remains in `results/`.

Counterion cloud revision 2 is self-contained in `counterion-cloud-r2/`; original paths resolve from that directory. Its separately conserved ion species and complete mean-valence control distinguish it from revision 1.

The pulsed-kinetic-gas r1 result report uses `trajectory_sha256` for each
trial's native `agent/codex.txt`, rather than Harbor's parsed
`agent/trajectory.json`. All three hashes match those native logs. Later
expansion reports hash `trajectory.json`; the original reports are kept
unchanged.

Finite-ion cloud revision 2 is self-contained in `finite-ion-cloud-r2/`; original paths resolve from that directory. It measures finite-radius pressure and tests angular pair correlations rather than the dilution-limit free count of revision 1.
