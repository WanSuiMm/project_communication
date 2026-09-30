# Inertial reaction–transport NCA：推导与边界

**日期：2026-09-30。状态：候选参数化；线性结论已数值复核；没有架构收益结论。**

本文从用户提出的两行更新推导，不把原有讨论中的直觉当成已证事实。下述计算是对候选系统的直接推导，不是宣称新的数学定理。相关架构近邻见 SOURCES.md。

## 1. 冻结研究对象

\[
V_{t+1}=B V_t+\eta R_\theta(H_t;X)-D\mathcal L H_t,\qquad
H_{t+1}=H_t+V_{t+1}.
\]

- \(H,V\in\mathbb R^{N\times C}\)，\(X\) 是可持续读取的局部外部证据。
- \(R_\theta\) 是逐位置、跨通道的共享非线性函数，不含全局池化、归一化或 attention。
- \(\mathcal L\succeq0\) 是正号图拉普拉斯；二维单位网格五点模板的谱上界为 8。实现采用复制边界的无通量模板。纯谱例子另用周期边界。
- \(B=\operatorname{diag}(\beta_c)\)，\(D=\operatorname{diag}(d_c)\)，空间上常数，\(0\le\beta_c<1\)，\(d_c\ge0\)。推导完整复谱公式时先取 \(B=\beta I\)。
- 同步更新；\(V_0=0\)；通常 \(H_0=E_\theta(X)\)，且 E 逐位置。随机 firing、alive-mask、时变系数不在此静态模态证明范围内。
- V 是 latent displacement/velocity，不是具有空间方向分量的物理 flux。它可以表达波动相位，但不自动提供寻址。

消去 V 得到精确离散二阶方程：

\[
H_{t+1}-2H_t+H_{t-1}+(I-B)(H_t-H_{t-1})
=\eta R_\theta(H_t;X)-D\mathcal L H_t.
\]

它具有阻尼波动的离散形式。\(B=0\) 时严格退化为一阶 reaction–diffusion 更新；这不是任意固定步长下都合法的连续奇异极限。

## 2. 纯 transport 的精确稳定区间

取一个通道、一条空间特征模态 \(\mathcal L\phi=\lambda\phi\)。令 \(a=d\lambda\)，则

\[
\binom{h_{t+1}}{v_{t+1}}=M(a,\beta)\binom{h_t}{v_t},\quad
M=\begin{pmatrix}1-a&\beta\\-a&\beta\end{pmatrix}.
\]

特征多项式：

\[
p(r)=r^2-(1+\beta-a)r+\beta.
\]

对 \(0\le\beta<1\)，Jury 条件给出严格 Schur 稳定的充要条件：

\[
\boxed{0<a<2(1+\beta).}
\]

证明：三个条件分别为 \(1-\beta>0\)、\(p(1)=a>0\)、\(p(-1)=2+2\beta-a>0\)。

所以 \(d_c\lambda_{\max}<2(1+\beta_c)\) 保证每个非零空间模态的纯 transport 渐近稳定。但注意：

1. \(a=0\) 的根是 \(1,\beta\)，是中性均值模态，不严格收缩。
2. \(a=2(1+\beta)\) 在负实轴出现单位根，不属于严格稳定区。
3. \(\beta=1\) 只可能是中性振荡，不是阻尼收敛；零模态可能出现线性漂移。
4. 稳定不是欧氏范数逐步不增。非正规性、重根都允许有限时间放大。
5. 空间均值满足 \(\bar V_{t+1}=B\bar V_t+\eta\overline{R(H_t;X)}\)。拉普拉斯无法抑制持续的均值注入。

纯 transport 为 \(\beta=.9\)、\(h_0=0,v_0=1\) 时，零模态 \(h_\infty=9\)，而不是恢复到 0。

实现可令

\[
\beta_c=.995\sigma(b_c),\quad
 d_c=.98\frac{2(1+\beta_c)}8\sigma(\delta_c).
\]

这只约束纯 transport；不能标注为“整个 NCA 稳定”。

## 3. 振荡频带与传播尺度

判别式为

\[
\Delta=(1+\beta-a)^2-4\beta.
\]

欠阻尼频带是

\[
\boxed{(1-\sqrt\beta)^2<a<(1+\sqrt\beta)^2.}
\]

该频带内

\[
r_\pm=\sqrt\beta e^{\pm i\omega},\quad
\cos\omega=\frac{1+\beta-a}{2\sqrt\beta}.
\]

振荡模态的渐近振幅因子是 \(\beta^{t/2}\)。
其 e-fold 记忆时间为 \(\tau_{amp}=-2/\log\beta\)，在 \(\beta\to1\) 时约为 \(2/(1-\beta)\)。

判别式大于零不一定都是连续时间意义的 overdamped：当根为负时，可能逐步翻转符号。严谨称“实根区”即可。

### 3.1 真正的局部因果界

半径一的空间模板、局部初始化/读出以及逐位置 reaction 意味着：

\[
\frac{\partial(H_t(i),V_t(i))}{\partial X(j)}=0
\quad\text{if }\operatorname{dist}_{grid}(i,j)>t
\]

（假定 X 从初始化便局部可读；首步的偏移约定不会改变线性尺度下界。）

归纳即可证明。惯性不能超越每步一格的依赖锥。普通 NCA 也可以实现 \(H_{t+1}(i)=H_t(i-1)\)，本来就能弹道传播。因此不能把“vanilla NCA 是 O(L²)，新 NCA 是 O(L)”当成一般性定理。

### 3.2 固定阻尼的大尺度极限

对固定 \(\beta<1\) 和小 \(a=d\lambda\)：

\[
r_{slow}=1-\frac{d\lambda}{1-\beta}+O(\lambda^2).
\]

规则大网格上 \(\lambda_{min,+}=\Theta(n^{-2})\)，故纯 transport 的慢模态衰减时间为

\[
\boxed{\Theta((1-\beta)n^2/d)}
\]

（固定误差因子，忽略与其有关的对数项）。所以固定阻尼并没有把无限尺度的扩散复杂度变成线性。

一维 \(\lambda(k)=4\sin^2(k/2)\sim k^2\)。对于 \(\beta\) 接近 1，欠阻尼低频截止约为

\[
|k|>\frac{1-\sqrt\beta}{\sqrt d}.
\]

可用波动传播尺度约为 \(\sqrt d/(1-\sqrt\beta)\)，忽略波长定义带来的 \(2\pi\) 常数。大于此尺度的低频模态仍是松弛型。

在 \(\beta=1\) 的无阻尼参考中：

\[
\sin(\omega/2)=\sqrt d\,|\sin(k/2)|,
\]

长波相速约 \(\sqrt d\)。对 \(\beta\approx1\) 的有限时间/有限频带也有近似波动行为；不能把这个近似推广到固定阻尼的任意大尺度。

### 3.3 何时能获得线性尺度的谱收敛率

在已知正定二次型谱区间 \([\mu,\Lambda]\) 上，经典 heavy-ball 的谱调参参考为

\[
d_*={4\over(\sqrt\Lambda+\sqrt\mu)^2},\quad
\beta_*={ (\sqrt\Lambda-\sqrt\mu)^2\over(\sqrt\Lambda+\sqrt\mu)^2},
\]

\[
\rho_*={\sqrt\Lambda-\sqrt\mu\over\sqrt\Lambda+\sqrt\mu}.
\]

当 \(\mu=\Theta(n^{-2})\) 时，\(1-\beta_*=\Theta(n^{-1})\)。渐近谱率改善为 \(1-O(n^{-1})\)，但依赖随尺度改变的参数，并排除了零模态。端点重根与条件数仍影响有限时间范数常数。

这只是经典二次迭代参考，不是网络已经具备的 size extrapolation。

## 4. 恢复 reaction：完整的局部稳定问题

设 Hbar 是均匀平衡，X 也均匀，\(J=\partial_H R(Hbar;X)\)。对空间模态 \(\lambda\)：

\[
K_\lambda=\eta J-\lambda D,\qquad
\mathbb M_\lambda=\begin{pmatrix}I+K_\lambda&B\\K_\lambda&B\end{pmatrix}.
\]

严格局部线性稳定要求所有实际网格模态满足 \(\rho(\mathbb M_\lambda)<1\)。

### 4.1 标量 momentum 的精确复数稳定区域

取 \(B=\beta I\)。K 的每一个（可为复数的）特征值 q 对应

\[
r^2-(1+\beta+q)r+\beta=0.
\]

所有根在单位圆内的充要条件是

\[
\boxed{
{(1+\beta+\Re q)^2\over(1+\beta)^2}
+{(\Im q)^2\over(1-\beta)^2}<1.
}
\]

证明：令 \(c=1+\beta+q\)。二阶 Schur–Cohn 条件为
\(|c-\beta\bar c|<1-\beta^2\)。展开即得到椭圆。

当 q 为实数时，退化成 \(-2(1+\beta)<q<0\)。但 \(\beta\to1\) 时，虚轴半宽 \(1-\beta\to0\)：高 momentum 对反应中的旋转型 channel coupling 更敏感。

具体反例：\(q=-.2+.1i\)。\(\beta=0\) 时 \(\rho=.8062258\)；\(\beta=.9\) 时 \(\rho=1.0646847\)。同一个稳定的一阶局部反应，加 momentum 后可以不稳定。

### 4.2 通道 momentum 与非均匀场

若 B 不是标量且与 J 不对易，必须检查上面的 \(2C\times2C\) 矩阵，不能逐个 channel 套椭圆。

若平衡图案非均匀，\(J_R(H^*(i);X(i))\) 随空间变化，Fourier 模态一般耦合。共享卷积权重并不足以让“围绕任意图案的线性化”被 Fourier 对角化。此时需要真实轨迹的 JVP/VJP、局部符号近似或完整空间 Jacobian。

## 5. Turing 条件：能推出起纹，不能直接推出修复

在一阶 RD 的连续生成元 \(J-\lambda D\) 中，取

\[
J=\begin{pmatrix}a&b\\c&e\end{pmatrix},\quad D=\operatorname{diag}(d_u,d_v),\quad d_u,d_v>0.
\]

均匀反应稳定条件：\(a+e<0\)、\(\det J=ae-bc>0\)。定义 \(s=d_v a+d_u e\)。

扩散诱导的静态不稳定需要

\[
s>0,\quad s^2>4d_ud_v\det J,
\]

且实际离散空间谱至少一个 \(\lambda\) 落入

\[
\lambda_\pm={s\pm\sqrt{s^2-4d_ud_v\det J}\over2d_ud_v}.
\]

这来自
\(\det(J-\lambda D)=\det J-s\lambda+d_ud_v\lambda^2\)。

对于实际离散惯性更新，还必须检验：

\[
\rho(\mathbb M_0)<1,\quad \exists\lambda>0:\rho(\mathbb M_\lambda)>1.
\]

而且要区分真实扩散驱动的正实增长率，和步长太大/惯性引起的离散不稳定。仅看到 \(\rho>1\) 不是 Turing 机制证据。

特别地，若 K 出现正实特征值 q，\(p(1)=-q<0\)，必有实根 r>1，惯性不会消灭这种静态 Turing 起纹。

**均匀态失稳产生图案，和成熟非均匀图案遭破坏后回归，是两个不同的稳定性问题。** 后者依赖成熟态 Jacobian、非线性饱和、输入条件以及吸引域。有限损伤后的再生不能由均匀态色散曲线直接推出。

线性检查用预先构造的 J=[[1,-2],[2,-3]]、D=diag(.1,1)，其不稳定区间为 (2,5)。取 eta=.01、beta=.8 后验证均匀模态稳定、部分空间模态增长。此例不包含学习、非线性饱和或再生。

## 6. 梯度穿越时间

令 \(S_t=(H_t,V_t)\)。对一般轨迹，

\[
K_t=\eta J_R(H_t;X)-D\mathcal L,\quad
M_t=\begin{pmatrix}I+K_t&B\\K_t&B\end{pmatrix},
\quad {\partial S_T\over\partial S_0}=M_{T-1}\cdots M_0.
\]

### 6.1 单步特征根稳定不保证乘积稳定

取 beta=.9，\(a_1=.1,a_2=3.7\)。各 \(M(a_i,.9)\) 的谱半径均为 \(\sqrt{.9}<1\)，但

\[
\rho(M(a_2,.9)M(a_1,.9))=4.8738054>1.
\]

所以每个训练状态都过 frozen-Jacobian 谱检验，也不足以证明时变轨迹的梯度稳定。这是直接可验证的二维反例。

### 6.2 阻尼明确收缩梯度体积

通过块行相减得到

\[
\det M_t=\det B.
\]

设 H 有 m 个标量，B=beta I，则

\[
\left|\det {\partial S_T\over\partial S_0}\right|=\beta^{mT}.
\]

2m 个奇异值的几何平均为 \(\beta^{T/2}\)。这不意味着每个梯度都消失，但排除了“所有方向都天然接近等距”的说法。

### 6.3 有条件的修复与梯度定理

若在一个平衡态 S* 的不变邻域内存在同一个 \(P\succ0\)、\(q<1\)，使

\[
DF(S)^\top P DF(S)\preceq q^2P,
\]

则

\[
\|F^k(S)-S^*\|_P\le q^k\|S-S^*\|_P,
\quad
\|\partial S_T/\partial S_t\|_P\le q^{T-t}.
\]

同一个条件既给局部修复，也给初态梯度衰减。我们没有证明训练后的 NCA 满足此条件，也没有由它界定可擦除的面积。

持续输入 X 改变了“收缩等于不能计算”的结论：

\[
{\partial S_T\over\partial X}
=\left(\prod_tM_t\right){\partial S_0\over\partial X}
+\sum_{s=0}^{T-1}\left(\prod_{t=s+1}^{T-1}M_t\right)E_s,
\]

其中 E_s 是当步输入注入 Jacobian。即使初态信息衰减，持续注入仍可支持有用的输入依赖解。不能简单把 Turing 不稳定当成“计算的必要条件”。

## 7. Elliptic equilibrium 到底保留了什么

平衡时

\[
V^*=0,\qquad D\mathcal L H^*=\eta R_\theta(H^*;X).
\]

固定 R,D,eta 后，平衡解集合完全不依赖 B。若 D 在相关通道上一致正定、边界条件合适，这有半线性椭圆平衡的形式；存在性、唯一性、吸引性并非自动成立。

有限 T 下仍只是局部演化。没有精确全局 solve，没有一步 global consistency，也没有突破局部依赖锥。多通道 d=0 时该通道还可能是退化而非严格椭圆。

因此 momentum 的潜在价值是路径、有限预算表现、吸引域和学习参数化；不是创造新的平衡解集。

## 8. 三条架构原则的当前结论

| 原则 | 已有推导 | 仍待验证 |
|---|---|---|
| 好训练 | 纯 transport 可约束；完整线性稳定域明确 | 非线性训练、梯度乘积、优化景观；高 beta 可能更差 |
| machine-native | stencil + pointwise MLP + 两状态更新，无求解器 | 小算子带宽/launch 开销、训练显存、matched-quality GPU 时间 |
| 正确的信息流 | 有可控欠阻尼频带、严格局部因果界 | 标签/语义是否到达正确位置；振幅传播不等于信息传递 |

逐步 MLP 成本为 O(B N C W)，W 与 C 同阶时是 O(B N C²)，不是对通道数线性。
推理状态为 2BNC 个数；普通 BPTT 激活开销随 T 增加。代码没有实现 reversible backprop。

## 9. 可继续的工作假设

> 在参数、状态容量与实际计算预算受控时，显式 reaction–transport 分解配合惯性，是否比 generic momentum NCA 更容易学到可延续、可纠正、携带输入身份的信息传播？

这比“两个状态比一个状态强”“wave 比 heat 传播快”更接近真正的架构命题。

Momentum ResNet、GraphCON、RD-NCA 已覆盖本候选的大量构件。当前两行公式不是已确立的新贡献；应以它作为对照明确的候选实验，而不是先命名新架构。
