# Rotating reservoir revision 6 scientific prototypes

The original prototype and spectral source/output pairs preserve exploratory evidence, including the first finite-volume prototype with its subsequently corrected tail gauge. Original main blocks retain their original output paths as provenance.

The `reproduce-*.py` copies change only path handling: they import the sibling finite-volume prototype and write distinct `reproduced-*.json` outputs beside themselves. They are runnable without temporary-directory dependencies. The spectral comparison explores a wider rotation range than the final bounded public apparatus; final revision-6 scientific validation uses the actual task and reports its public box separately.

Use `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B reproduce-rotating-hard-collision-spectral.py` for the independent continuous-generator comparison. Production task references are fully self-contained and import none of these prototypes.
