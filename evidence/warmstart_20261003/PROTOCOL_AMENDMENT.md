# Protocol amendment and execution record

Protocol ID: `detached_warmstart_v1_nocap`.

The initial CPU01 and preflight01 records remain immutable. Preflight01 failed its former 2,400-second launch estimate gate: its estimate was 2,451.034 seconds, while its gradient-entry and memory gates passed. Before the formal run, the user removed that runtime cap. The amended CPU02 and preflight02 qualification passed; the time estimate remained informational and neither a runtime deadline nor a termination watchdog was enabled.

The runtime qualification rule changed before the formal run. The frozen implementation also contains a synchronization-only prefix-finiteness fix: device failure flags stay sticky across each eight-step window and host checks occur at window boundaries and at the final prefix step. CPU02 checks exact historical equivalence and a NaN/recovery fixture; the change preserves numerical states, gradients, and transient-error detection. The eight-arm recipe, training-state intervention, fixed data and batch schedules, four paired initializations, endpoint, full-phenotype thresholds, and stop conditions remained fixed. The completed formal run used 300 updates per arm. All four historical baseline replays passed. The result is `DEVELOPMENT_NOT_QUALIFIED` (baseline 1/4, warm-start 0/4); the failed old preflight is not a scientific treatment result.

See the exact frozen [protocol](../../new/warmstart/PROTOCOL.md), `config.json`, `source_bindings.json`, and hash-bound [provenance](provenance.json). This result is conditional on one bank and one batch schedule. It does not identify a phase transition, continuation closure, unique handoff mechanism, or general warm-start failure.
