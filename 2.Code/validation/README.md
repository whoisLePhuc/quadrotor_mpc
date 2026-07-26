# Validation Evidence

`stage8_30_seed/` is the original Phase 8 campaign committed with version
2.0.0. Its manifest reports source commit `9f4a1b3`, which predates the Phase 8
implementation commit. The numerical evidence is retained unchanged for
auditability, but it is not treated as release-provenance-complete evidence.

Version 2.0.1 fixes this class of error:

- a release campaign refuses a dirty Git tree by default;
- manifests record the commit, branch, clean state and a deterministic
  validation-source SHA-256;
- resume is rejected when the source snapshot changes;
- an explicitly allowed dirty-source smoke campaign is labeled
  `DIRTY_GIT_SNAPSHOT` and `FAIL_DIRTY_SOURCE`.

The 50-seed/cell release campaign must be generated after the v2.0.1 source is
committed. This sequencing prevents a report from naming a commit that does
not contain the code that generated it.
