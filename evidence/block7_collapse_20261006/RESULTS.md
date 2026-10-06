# Block7 collapse: result interpretation

The authorized zero-training audit completed32/32 matrix units and12/12
same-state single-step units. The worker completed normally in90.97seconds.
Eight native diagonals reproduced the prior original, flipped and paired
Boolean traces with ZERO differing bits. Saved-data validation checked192
view/cohort summaries against packed traces. No model fitting, parameter
change, optimizer update or readout alignment was performed.

Frozen artifacts: [RESULTS](frozen_RESULTS.md),
[matrix](matrix.json.gz),
[single-step](single_step.json.gz),
[validation](validation.json).

## Decision-relevant four cells

All values below use the SAME fixed u275 readout and source-paired correctness
on the all_changed component. These are terminal T256 coverage, not the
strict16<d<32 reach or preservation conditional on T64 correctness.

|Producer toT64|Consumer T64..256|size32 coverage|size64 coverage|
|---:|---:|---:|---:|
|275|275|0.950144|0.840536|
|275|300|0.970374|0.933684|
|300|275|0.489719|0.111351|
|300|300|0.425492|0.059384|

G300 preserves u275's native T64-correct cohort continuously with rates
0.999731(size32) and0.999708(size64). It also has greater sustained progress on
u275's initially wrong cells:0.828571/0.826714, versus G275's0.708075/0.602979.
The successful producer states from225 and250 similarly retain near-complete
preservation under G300, while their terminal coverage increases relative to
their own native continuation. Thus the finding is not confined to one275
cross cell.

On u300-produced states, G275 improves coverage and preservation somewhat,
but does not restore the successful regime. Older consumers225/250 also do
not restore it. Decoder-only changes are recorded explicitly; the main
pattern remains under producer, consumer and fixed275 observation views.

## Same-state one-step result

On u275's states at t64,128,192, G300 destruction under R275 is at most
0.000293(less than0.03%) at either size. G275 itself has small transient
destruction at size64. G300 is not an immediate broad destroyer of these
successful states; its net one-step coverage change is nonnegative at all
six state/size points.

## Bounded conclusion for the next architecture

The tested collapse cannot be adequately described as G300 losing its ability
to continue an already-successful execution state. The stronger distinction
is the state produced by the cold prefix0..64: u300's prefix does not supply
states that the tested consumers can continue successfully, whereas earlier
producer prefixes do. State production here includes BOTH the learned encoder
and64 recurrent steps; this audit does not separate them or identify which
state component is responsible.

For Hybrid v0, prioritize legal execution-state construction and write/update
interfaces during formation. A preservation mechanism applied only after
answers become correct is not directly supported as a sufficient repair.
This is guidance from a selected trajectory, not a proof of a specific explicit
contract, irreversible domain exit, or population-level success. The old fixed
u300 continuous-coverage qualification remains negative. Hybrid was not run.

Reading route: [summary](summary.json), [matrix CSV](matrix_summary.csv), [single-step CSV](single_step_summary.csv), [provenance](provenance.json), [reproduction](REPRODUCTION.md).
