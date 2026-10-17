#!/bin/bash
set -eu
mkdir -p /logs/verifier
echo '{"error":"metrics unavailable"}' > /logs/verifier/metrics.json
cd /app
if PYTHONPATH=/app python -m pytest -q /app/test_public.py /tests/test_hidden.py; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
