# Trajectory audit: finite-horizon interpretation

Inference-only replay completed for the existing seed-0 masked inertial model.
All15 size/horizon checkpoints reproduce published BA, BCE and H/V RMS with
maximum absolute error0. No training or parameter change was performed.
Temperature is fit separately on32 new calibration maps per size, never on the
16 frozen evaluation maps. Results and code are bound by manifest.json.

## What the audit establishes in this checkpoint

At size32, from T64 toT256:

- BA increases88.79% to91.55%; pooled open-pixel error fraction falls12.58%
  to9.99%. These use different aggregation weights and are not complements.
- Wrong signed-margin median changes -1.889 to -18.421. Wrong predictions
  contribute1.4514 of the total1.4519 BCE atT256. Increasingly confident
  remaining errors directly account for the observed loss deterioration.
- H RMS on open pixels rises14.48 to120.61; wall RMS rises22.71 to141.93.
  The growing norm is not explained solely by wall states.
- Raw BCE at T64/128/256 is0.2503/0.6565/1.4519. Independently fitted
  temperatures1.695/5.936/15.695 yield held-out BCE0.2569/0.2659/0.2492.
  This supports a substantial output-scale contribution to loss deterioration.
  It neither repairs classification errors nor establishes hidden-state stability.
- Open H energy in the componentwise-constant projection rises89.10% to97.04%.
  Projected V RMS rises0.4326 to0.6005. Component mean Laplacians cancel to
  about1e-8 and the mean velocity equation residual is about2e-8 atT256.
  This is consistent with a large drifting neutral-mode component.
- Mean per-map H cosine betweenT128 andT256 is0.9838, but component-mean
  velocity still changes22.0% in size-weighted vector norm. Alignment is high;
  convergence to a constant velocity or fixed direction is not established.

At size128/T256, componentwise-constant H energy reaches99.51% while BA is50%.
Independent calibration reduces BCE12.966 to0.6903 but cannot change BA.
High projection energy therefore does not mean correct component identity.
The large-domain failure includes wrong decisions, not merely mis-scaled logits.

## Boundaries and next comparison

This audit supports confidence amplification and substantial component-mean
drift over the tested horizon. It does not prove unbounded asymptotic growth,
projective convergence, a constant forcing limit, or a particular cause of
source-revision failure. The mean reaction depends on the full field; its
equation is not a closed autonomous dynamics in the mean alone. BCE is a proper
loss measurement, not by itself a calibration decomposition.

The finite-horizon audit does not change the separately frozen Masked Momentum
training recipe. That same-medium comparison remains necessary to assess
whether the explicit reaction/transport implementation adds value beyond a
generic recurrent update. No new architecture or temperature-scaled deployment
model is being proposed here.

Read RESULTS.md and summary.json first. components.json is secondary: exact
per-component H/V/reaction/Laplacian means and one-step residual ingredients.
