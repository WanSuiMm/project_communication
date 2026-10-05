# Isolated native-width runtime benchmark

Question: can implementation overhead be reduced without changing the native
width model, K8 detach clock, FP32 computation, data or eager AdamW/clip recipe?
This is a systems prototype, not another efficacy experiment or a change to
the running frozen `native_latent_width_v1` suite. Do not edit its bound sources.

Compare original eager execution, the same math with batched finite checks,
and captured forward64/eight K8 backward windows via a CUDA Graph. Keep gradient
clipping and the original eager AdamW outside the capture. In capture, static
inputs are copied into fixed buffers before replay. Zero grads to None before
capture, so the first captured backward overwrites static grad allocations
and subsequent windows accumulate. Never set those grads to None during replay.
The W2 CPU phase is unrolled at capture: a fresh64-step trajectory always starts
at phase0 and ends at phase0; this is not valid for arbitrary external phase.

Finite checks still guard loss/final state, gradient norm and updated parameters;
only their host synchronization is batched. Nonfinite intermediate losses are
detected before the optimizer update rather than before each backward window.
This changes failure timing, not finite mathematical training. The full learned
step has no dynamical guarantee.

Fixed two representative arms W24 and W2-alternating; seed91981, nonzero F/Q
tails (.01 std) exercise recurrent gradients. Size32 batch8,24 generated maps
seed99432,24 fixed minibatches seed99433. Six paired optimizer updates check
all losses, final states, pre-clip gradients, parameters and Adam tensors.
Report bitwise equality separately from numerical tolerance; no close result
is labeled exact replay. Three warmup updates then six interleaved timing rounds
per method. All three methods use the same schedule within every round. Timing
counts are not independent training replicates. All benchmark models are
disposable and cannot enter the efficacy aggregate.

One profiler pass over eager execution reports kernel launches and scalar/sync
calls without saving a large trace. Original training may run concurrently in
another process: record this fact and interpret measured speedup as a pilot,
not isolated GPU throughput or a promised formal-run wall time. Record GPU,
backend, source hashes, setup/capture cost and peak CUDA memory.

Only if measured speed improves and exact trajectory replay is qualified can
one consider a separate uniformly accelerated experiment. Do not hot-patch or
splice the current frozen suite. Inexact replay is retained as an implementation
qualification failure, not a scientific architecture failure.

    python -X utf8 -B new/latent_runtime/bench.py --out analyses/NEW_RUNTIME_BENCH

Read RESULTS.md and summary.json; no trained checkpoint or full state array is
retained. This output should be small.
