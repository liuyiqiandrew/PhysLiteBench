# Ring-current-fluctuations: qualified task

This branch publishes only the ring-current-fluctuations task and its supporting scientific and evaluation evidence from local checkpoint `51e550ba5fe6bb226a313feba38d6465052cd148`. Other task work remains local. Existing project-wide status files are inherited from the base revision.

The frozen revision scored **0 passes in 3 unhinted GPT-5.6 Luna/high trials**. All three failures were reviewed as genuine physical-model failures, with no scored exceptions or retries. Each submission retained a homogeneous conditioned density profile and failed the two strong-field prediction groups while passing calibration and parameter recovery.

- [Public apparatus and API](../tasks/ring-current-fluctuations/environment/README.md)
- [Physical derivation](../tasks/ring-current-fluctuations/AUTHOR.md)
- [Qualification results](../results/zero-three-ring-current-r1-results.json)
- [Reviewed trial classifications](../results/zero-three-ring-current-r1-trial-reviews.json)
- [Submitted models and verifier evidence](../results/ring-current-r1-evidence/manifest.json)
- [Docker controls](../results/zero-three-ring-current-r1-controls-review.json)
- [Scientific validation](../results/ring-current-validation.json)

The task, both completed controls, calibration, private reference and grading are byte-identical to the qualified checkpoint. The self-contained scientific package under `staging/ring-current-fluctuations-r1/` retains its earlier numerical outcomes and reviews. Historical review paths to raw ignored `jobs/` directories remain provenance references; the raw jobs are not included in this branch.

Run the scientific validator from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_ring_current_fluctuations.py
```
