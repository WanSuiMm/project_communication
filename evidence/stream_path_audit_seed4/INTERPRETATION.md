# Interpretation of the selected Streaming seed4 operator audit

The frozen seed4 solution is highly sensitive to BOTH tested pathways.
With all pathways present, historical evaluation replays with zero error:
size32 strict16<d<32 paired pooled correctness is 85.92%, 99.35%, 100%
at T64/T128/T256. Replacing T by identity or zeroing direct Laplacian inputs
in both F/Q gives 0/2918 primary hits at every one of these horizons.
Size64 d>32 is also zero for either knockout versus 12.48%, 38.60%, 59.43%
for the full model. The both-off no-communication control behaves as expected.

This rules out a narrow operational explanation: neither the unchanged
perception networks without streaming nor the unchanged streaming networks
without direct Laplacian inputs retain this successful behavior. The tested
frozen solution requires compatibility with the original hybrid update.
It does not establish a universally necessary architecture primitive.

The losses are not confined to remote pixels. At size32, no_perception has
0/1706 paired hits in d<8 already at T8; no_transport starts at 906/1706
at T8 and 1067/1706 at T16, then falls to zero by T32. At T64 all knockouts
have original BA 50.00% and flipped BA 48.44%. Thus it is not justified to
describe these interventions simply as a decrease in propagation speed.
The local computation/readout regime has also been disrupted.

Source-flip logit RMS on the changed component at size32/T64 is 7.289 for
full, 0.215 for no_transport and 0.027 for no_perception. Some counterfactual
response survives without correct decoding; nonzero response is not useful
task communication. State amplitude is not a sufficient explanation either:
full W RMS rises from 10.156 to 40.120 at T64 to T256 while the primary
improves, and knockout W amplitudes grow to a similar order of magnitude.
These descriptive RMS values do not establish stability or a mechanism.

Turning transport off also changes the first feature of F. Turning perception
off replaces its learned Laplacian feature slots, changes future states and
reduces the graph-hop upper bound from two to one per macro-step. Knockouts
are outside the model's original training regime. They cannot identify which
branch carries each semantic operation, prove that restoring Laplacians
would repair H/C/Z training, or show that a retrained knockout must fail.
The positive factorial interaction at the primary is numerical because only
the full model succeeds; it is not evidence of an additive causal decomposition.

Practical implication: if the next question is adding stationary workspace,
retain the original streaming width, F/Q networks, neighborhood inputs and
clock as the reference. Change one item with a neutral setting that reproduces
the reference. This audit provides no efficacy result for such an addition.
The prior multi-seed Streaming DEVELOPMENT_NO_GO and H/C/Z negative result
remain unchanged. This is one selected model, not training-seed replication.

Verification: the independent nonzero CPU reference passed; all six historical
size/horizon records replay exactly. Post-run CPU arithmetic validation checks
39 source bindings, 672 paired aggregates/denominators, 96 BA means, 672 CSV
rows and 600 contrasts. No model training or parameter updates occurred.
See [validation](validation.json). Its input_sha256 binds original run files;
private manifest/completion bytes are excluded, while the public copies
are bound separately in STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json.
