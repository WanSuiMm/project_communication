# Four short checks

1. **Stability.** L=B' diag(a) B is PSD for symmetric a>=0. Resolvents and the
   directional product are nonexpansive, with norm exactly one on constants.
   For F(H)=H+gamma*g(H), J_F=I+gamma*J_g; gamma*L_g<1 does NOT imply contraction
   (g(H)=H is a counterexample). A sufficient local scalar-step condition is
   sym(J_g)<=-mu*I, ||J_g||<=L and 0<gamma<2*mu/L^2. Dynamic edges add
   dM=T*dQ-tau*T*(dL)*M; phase caching does not delete that derivative.

2. **Propagation.** A radius-one local update reaches at most K cells after K
   updates. Positive unobstructed directional solves can give global dependence
   in one layer, but a fixed axis word cannot follow arbitrarily many turns
   behind hard barriers. A 1D constant-coefficient Green kernel is approximately
   exp(-|x|/sqrt(tau*a))/(2*sqrt(tau*a)): distance attenuation remains, and a
   doubled effective length typically needs four times tau*a. PCR also has
   logarithmic parallel stages; an implicit solve is not constant computation.

3. **Bandwidth.** With fixed medium and local inputs, one remote-to-target path
   through M_i in R^r has Jacobian rank at most min(C,r). Multiple rounds,
   neighboring reactions and state-dependent edges are additional information
   paths, so this is not a whole-network rank bound. Small r suffices only when
   the required remote statistic can be represented and delivered at that
   width. r near C weakens compression gains, not necessarily state retention.

4. **Operator class.** The palindromic directional product is symmetric,
   entrywise nonnegative and doubly stochastic. It is an invertible weighted
   averaging operator for finite scales, not arbitrary directed routing.
   O(GN) edge/scale variables determine a potentially dense spatial kernel.
   tau and a enter through their product: interpreting tau alone as distance is
   unjustified. Full nonlinear network responses need not be symmetric.

For one line solve A*M=Q, A'*V=upstream gives grad_Q=V and, for edge i--j,
grad_w=-sum_channels[(V_i-V_j)*(M_i-M_j)]. `transport.py` uses this adjoint,
including shared-weight accumulation across RHS and repeated solves.
