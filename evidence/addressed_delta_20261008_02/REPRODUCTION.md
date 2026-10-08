# Addressed-delta development screen: reproduction and evidence map

Read `RESULTS.md`, `summary.json`, and `final_metrics.csv` first. This is a
four-block development screen: delta and additive each passed joint readiness
in 0/4 blocks; current passed in 2/4. The verdict is
`NO_DEVELOPMENTAL_SIGNAL`. Four paired blocks are descriptive evidence, not a
reliability or significance result. Old Full was not evaluated, and full
Boolean arrays were transient and are not persisted.

From the repository root, the CPU-only saved-data check is:

```powershell
python -X utf8 -B tools/export_addressed_delta.py --verify-only
```

It checks published hashes, the 71 qualified source bindings, frozen bank and
schedule hashes, all 12 final and 156 dense records, record-to-summary gates,
reporter aggregate and CSV equality, all 156 checkpoint completion markers,
and the 12 training curves' 300-update cadence. It performs no inference,
training, or optimizer updates and does not read local runs or checkpoints.

Fresh training requires the existing published bank dependency at
`evidence/port_relation_20261007_02/banks/` and
`PORT_RELATION_PUBLICATION_MANIFEST.json`, plus NumPy and the qualified
CUDA-enabled PyTorch environment. From the repository root:

```powershell
python -X utf8 -B new/addressed_delta/check_cells.py
python -X utf8 -B new/addressed_delta/reporting.py --self-test
python -X utf8 -u -B new/addressed_delta/run.py --check --out analyses/addressed_delta_fresh_qualification.json
python -X utf8 -u -B new/addressed_delta/run.py --out runs/addressed_delta_fresh --qualification analyses/addressed_delta_fresh_qualification.json
```

Use a new run name for each independent reproduction. The check command runs
the small actual-shape CUDA qualification; the final command starts a new
four-block training run. The publication includes the exact source banks but
does not include checkpoints or optimizer state. `interruption_and_recovery.json`
records that the published run inherited 22 committed stages after the parent
was cancelled and that the compatibility runner fixed JSON tuple/list config
comparison without changing numerical training code.
