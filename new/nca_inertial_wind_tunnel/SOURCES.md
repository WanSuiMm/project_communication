# Primary sources checked on 2026-09-30

These establish prior art and context; the algebra in THEORY.md is independently derived for the displayed discrete update. This list is not an exhaustive novelty audit.

1. Mordvintsev et al. **Growing Neural Cellular Automata** (2020).
   https://distill.pub/2020/growing-ca/
   Original persistent cellular dynamics and damage-training context.
2. Mordvintsev et al. **Differentiable Programming of Reaction-Diffusion Patterns** (2021).
   https://arxiv.org/abs/2107.06862
   https://selforglive.github.io/alife_rd_textures/
   Direct RD/NCA overlap. Merely separating reaction and diffusion is not a new contribution.
3. Sander et al. **Momentum Residual Neural Networks** (ICML 2021).
   https://proceedings.mlr.press/v139/sander21a.html
   https://arxiv.org/html/2102.07870v3
   Their two-state update matches the generic momentum form after scaling the residual; includes tied-weight examples and second-order interpretation.
4. Rusch et al. **Graph-Coupled Oscillator Networks** (ICML 2022).
   https://proceedings.mlr.press/v162/rusch22a.html
   Second-order controlled damped oscillators coupled over a graph; direct neighbor of inertial field computation. The provided momentum baseline is NOT an exact GraphCON reproduction.
5. Ruthotto & Haber. **Deep Neural Networks Motivated by Partial Differential Equations**.
   https://arxiv.org/abs/1804.04272
   Prior parabolic/hyperbolic architecture framework.
6. Turing. **The Chemical Basis of Morphogenesis** (1952).
   https://royalsocietypublishing.org/rstb/article/237/641/37/112910/The-chemical-basis-of-morphogenesis
   Historical diffusion-driven instability reference. The discrete Schur conditions in the notes are derived separately, not claimed as Turing's results.
7. PyTorch documentation: profiling and timing caveats.
   https://docs.pytorch.org/tutorials/beginner/profiler.html
   Warm-up and keep profiler overhead out of performance measurements.
