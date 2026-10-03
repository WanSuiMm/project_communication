# What the joint195/200 audit identifies

This is a zero-training, exploratory intervention audit of two checkpoints from one selected seed4 trajectory. All 492 conditions completed, including negative mixtures. The four primary-bank historical replay controls match exactly. Independent banks confirm across task maps; they do not replicate training trajectories. Rates in the main report are equal-map means, with eligible-map and cell denominators retained. Start at [RESULTS.md](RESULTS.md); use [profiles.csv](profiles.csv) to route to individual cases.

## 1. Conditional state production matters more than a readout change

On primary32 with readout195 fixed, state200 at64 under rule195 reaches strict T256 coverage1.000, whereas state195 under rule200 reaches.93333, close to the old diagonal. On the shared solved-at64 cohort, the corresponding all-steps survival is1.000 versus.93620. The state-origin intervention changes the whole pre64 computation; it does not isolate a single local cell operation. The200 continuation still improves acquisition timing. Readout is evaluated post hoc and never drives hidden dynamics.

See [S200/D195/R195](cases/primary32_crossS200_D195_R195.json), [S195/D200/R195](cases/primary32_crossS195_D200_R195.json), and the complete eight-case cube per bank in the index. This supports conditional state-production sensitivity, not a claim that195 dynamics are universally adequate or that200 states have a proven compositional type.

## 2. The parameter change is strongly coadapted

With R195 fixed, swapping only E or F into the195 background raises primary32 strict T256 to.99967 or1.000. Swapping Q alone lowers strict T128 from.90773 to.89035; with E/F200 in place, Q200 instead raises it from.95735 to.98780. At primary64, Q alone has strict T128.2577 and survival.2662, while the matching EFQ combination has strict T128.99975 and survival.99971. Confirmation64 retains damaging Q-alone versus strong matching-block behavior.

All16 coalitions and all24 orders are in [factorial.json](factorial.json). A Shapley average over these backgrounds is an attribution summary, not an intrinsic verdict on Q. The blocks cannot be ranked independently from these interventions. Fixed streaming has no parameter delta between checkpoints.

## 3. The apparent interpolation boundary is map specific

For primary32, lambda.4 has254/254 initial-correct cells on map10 exit by256; the other15 maps have none. At lambda.5, map10 has no exits, while map7 has one exit among154 initial-correct cells. Map10's paired-margin median at128 changes from-4.927 to+3.392, and at256 from-7.354 to+3.911. The median across map-level margin medians is positive on both sides. Independent32 maps already have perfect baseline195 long survival.

See [lambda.4](cases/primary32_lambda04.json) and [lambda.5](cases/primary32_lambda05.json). This localizes a sampled failure/remediation. The coarse eleven-point interpolation does not locate an exact threshold, establish a dynamical bifurcation, or support a population-wide computational phase transition.

## 4. Local swaps expose transient effects and robustness

The primary64 u195 frontier W swap gives150/162 target cells correct at128, compared with141/162 on the untreated baseline's exact same mask. Z and readout-null swaps each give141/162, including one gain and one loss; all reach162/162 by256. W's spillover is mixed: at128, rings1--4 have18 gains/8 losses, rings5--16 have1/37, and rings17--32 have2/29. Confirmation64 has the same W-over-Z direction at128, with both fully correct by256.

The null swap's immediate logit error is4.77e-7, but later paired margins change and two target Boolean outcomes differ at128. This is future sensitivity to a readout-relative null perturbation, with no net target outcome effect here. It is not a universal semantic null space. Primary32 u195 solved-cell W/Z/null/both swaps leave target Boolean traces identical to their exact-mask baseline, including the failures on map10.

See [frontier W](cases/primary64_u195_frontier_swapW.json), [frontier null](cases/primary64_u195_frontier_swapnull.json), and their `matched_baseline_cohorts`. These sparse, correctness-conditioned plans involve at most16 targets per map and can move states off manifold. Robustness or small effects on these plans do not prove future equivalence or rule out distributed mechanisms. Local cross-checkpoint transplants are retained even when they fail to rescue a whole-map phenotype.

## 5. Pulses affect acquisition timing; solved states keep computing

On the same363 selected primary64 u195 frontier cells, identity-stream sham has358/363 correct at128. A one-step Q knockout has320/363, F knockout328/363, and removing Q's LW input310/363. All recover to363/363 by256. Confirmation64 follows the same short-horizon direction. Size32 frontier pulses show no comparable endpoint separation, and200 frontier pulses largely hit ceilings.

See [drop Q](cases/primary64_u195_frontier_pulsedrop_Q.json), [without LW](cases/primary64_u195_frontier_pulseQ_without_LW.json), and the exact-mask baseline cohorts attached to each case. The same pulse mask is used for sham and knockout. These results identify conditional fast-acquisition sensitivity, not permanent Q necessity.

Actual-state update probes also show nonzero learned W and Z updates on solved cells. Some signed semantic increments are negative even when correctness survives. The fixed-linear-readout telescope is exact accounting in a specified order; nonlinear Q input contributions are not additive causal effects. Neither a zero-residual commitment gate nor a frozen solved-state identity mechanism has been established. The complete96 [update records](updates/) retain both readouts and all feature probes.

## Remaining uncertainty

The audit separates producer, continuation, readout and block-context effects. It does not identify the unique hidden-state organization explaining200, demonstrate infinite-horizon closure, establish a trainable default architecture, or increase the training-seed success rate. This is2D only. Earlier multi-seed architecture and warm-start negative verdicts remain unchanged. Any new mechanism experiment should target these unresolved alternatives rather than treat this selected checkpoint contrast as a new architecture qualification.
