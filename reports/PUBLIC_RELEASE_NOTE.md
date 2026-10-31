# Public Release Note

This repository is a public portfolio and research release. It provides the final English reports, selected code, aggregate result tables, figures, checksum records, and automated checks needed to inspect the reported methods and results.

The following materials are intentionally excluded: raw clinical data, sample-level split assignments, individual-level out-of-fold predictions, complete bootstrap-resample draws, complete execution logs, intermediate feature caches, and the original per-experiment `learning_records`. These exclusions preserve the data-governance and privacy boundary described in the repository-root README.

ARF V1 is retained as an exploratory experimental record. Its archived scripts depend on some of the excluded materials and are included for methodological inspection rather than as a standalone public runtime. ARF V2 provides the current formal reliability evaluation. Publicly released aggregate outputs can be verified with `python scripts/verify_public_release.py`.
