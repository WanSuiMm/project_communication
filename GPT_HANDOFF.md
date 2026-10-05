# Incremental review: C8 read / R2 factorial

- Review base: `c05895d68ce5f3974ff9c16305308c2ea1b49554`.
- Evidence head: `c0677d89507000f895e16620b2b1948e234f4308`.
- This later commit changes only this handoff. Earlier frozen evidence and
  claims are unchanged. Export performed no new training or model inference.

## Read first

1. [Results](evidence/latent_factorial_20261005/RESULTS.md),
   [summary](evidence/latent_factorial_20261005/summary.json),
   [validation](evidence/latent_factorial_20261005/validation.json).
2. [Configuration](evidence/latent_factorial_20261005/config.json),
   [128 arm records](evidence/latent_factorial_20261005/perarm.json),
   [768 metric rows](evidence/latent_factorial_20261005/metrics.csv).
3. [Frozen protocol](new/latent_factorial/PROTOCOL.md),
   [code map](GPT_CONTEXT.md),
   [reproduction](evidence/latent_factorial_20261005/REPRODUCTION.md).

## New evidence and boundaries

COMPLETE128/128,32 paired training blocks,4361.047seconds (72m41s) for the
formal worker, excluding preflight and publication. Native C8, factorized C8,
native C8+R2 and factorized C8+R2 each have0/32 Full successes. D-A has0 wins,
0 losses, delta0 and exact two-sided p=1:
`NO_D_MINUS_A_RELIABILITY_QUALIFICATION`. Secondary Full comparisons likewise
have no discordances. All-zero outcomes do not prove model equivalence or
universal failure of sidecars, factorization or latents.

All arms share native effective core tensors and batch schedules within a
block; C/D also share extension tensors. R2 adds memory, computation and
parameters together, so the experiment cannot isolate memory necessity.
Evaluation cohorts are fixed and reused; training blocks are fresh.

All128 used CUDA Graph K8 training. The preflight checked five updates per arm
against eager K8, with bitwise-equal gradients, parameters and Adam states;
it is not300-update equivalence. Its passing eight-win statistics entry is
a synthetic threshold fixture, not the formal endpoint.

The package retains all numeric measurements losslessly,493.83MiB:128 complete
curves,256 paired Boolean trace banks,per-map summaries,frontier tables,
systems data,exact schedules and task banks. Raw data are secondary; start
with the small summaries above. Four CPU tests and saved-data validation pass.
Clone-safe verification passed with local run/qualification paths unavailable.
Checkpoint files and machine records stay local; checkpoint/tensor hashes
were checked locally during export and their public bindings are verified.

    python -X utf8 -B tools/export_latent_factorial.py --verify-only

Review questions: which Full components fail; do continuous metrics change
without meeting Full; what is identifiable when the concurrent native C8
baseline also has zero Full successes? Preserve the frozen primary verdict.
