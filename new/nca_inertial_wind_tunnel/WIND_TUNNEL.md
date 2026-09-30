# 一个任务、四个对照的最小 wind-tunnel

## 要回答什么

**显式空间 transport + 惯性，能否比 generic momentum NCA 更容易学会可持续、可修正的空间计算？**

不检验“PDE 比神经网络高级”，也不把重新学出热方程、生成更漂亮的条纹当作架构成功。

## 任务：Seeded region identity

给定二维障碍/区域图，每个连通区域有一个 + 或 - 的身份种子。所有区域的身份随机指定，与绝对位置、形状无关。模型必须输出每个可通行像素所属区域的身份。

输入只有三个通道：区域 mask、正种子、负种子。每步可在当前位置读取 X。目标用连通分量 BFS 生成，不用任何 PDE 模拟或 PDE teacher。

地图包含带门的房间、弯折连接、断开的近邻区域。缩放画布会增大房间和路径尺度；不是只复制更多固定尺寸小问题。真实图距离仅供评估，不输入模型。

X 作为外部证据始终保留；因此这里测试的是 **conditional state repair**，不是“所有外部信息丢失后的自主形态再生”。所有模型接受相同输入保留方式。

空间算子为普通五点 Laplacian，不按目标连通分量重新连图，不提供 component id，不把答案预编码进传输系数。任务边界要由输入和 learned reaction 使用。

## 四个 arm

| Arm | 更新 | 目的 |
|---|---|---|
| nca_state_matched | U'=U+eta F(U,L U,X)，U 有 2C 通道 | 排除仅仅增加 latent state 容量 |
| momentum_nca | V'=B V+eta F(H,L H,X)，H'=H+V' | 最重要近邻；排除只有 momentum 的收益 |
| rd_nca | H'=H+eta R(H,X)-D L H | beta=0 的自然对照 |
| inertial_rd | V'=B V+eta R(H,X)-D L H，H'=H+V' | 候选 |

F/R 都是两层 pointwise MLP，中间 Tanh；最后一层零初始化。F 读取 H 和 LH，R 不读取邻居。这里的 NCA 是同步、各向同性 perception NCA，不声称复现原始 Growing NCA 的全部训练配方。

默认 C=16、候选 MLP 隐层宽 128。hidden widths 对齐到 8 的倍数，在此条件下近似匹配参数。所有 arm 的参数量、state 标量数、宽度、计时都写入 JSON。**不能把参数近似匹配写成 FLOP、激活内存、表达能力完全相同。**

D 和 beta 是空间共享的逐通道参数；beta 初值统一 .9，不预指定哪个通道负责哪种语义。eta=.1，d 初值 .1。初值不是扫参得到的最优值。

## 冻结第一轮

训练 32x32，512 个固定生成样本，batch 8，800 个 AdamW 更新，lr 1e-3，weight decay 1e-4，clip norm 1。

每步 rollout 从 {32,48,64} 均匀抽取，在 T-4 和 T 两个晚期位置监督。50% batch 在 rollout 中点擦除约 6.25% 正方形区域的 H 和 V，X 不变。四臂使用相同 minibatch、rollout、damage 随机序列。

先只跑 seed 0。若出现实质差异，只给最相关的两臂补 seed 1、2。没有 hyperparameter sweep，没有先做一串诊断 gate 的要求。

**800 更新是初筛配置，不是“不成功便否定整个 PDE/NCA 方向”的结论预算。** 若所有 arm 都学不会，先判断任务/训练配方是否具备可用性，而不是随即给候选加 attention 或 hierarchy。

## 测量的不是一个 accuracy 数字

### A. Correct information flow

32、64、128 网格；T={16,32,64,128,256}。记录 open pixels 上的 balanced accuracy、BCE、按源到目标图距离分桶的准确率。

另构造两张几何完全相同、只翻转一个区域种子身份的反事实输入，要求对应区域两种答案都正确。固定模板或纯 smooth pattern 无法通过这个检验。

图距离衡量任务结构跨度；有限因果锥的理论距离是底层网格距离，二者不要混用。

用固定检查点估计到达并持续保持 95% balanced accuracy 的步数，未达到记为未达到，而不是用最大步数代替成功。更高 accuracy 也要结合实际延迟。

### B. Continue / repair / revise

从 T=64 的状态擦除 H 和 V，测试 6.25%（训练内）与 25%（训练外）面积；追加 {8,16,32,64} 步。

同批比较 damaged continuation、undamaged continuation、相同额外步数的 cold restart。

主恢复统计只在损伤前每样本 balanced accuracy >=95% 的样本上计算。没有合格样本则输出 null/0 eligible，不能声称“修复失败率”或“修复成功”。同时保留全批结果和资格覆盖率。

然后只翻转一个种子身份、保留已有 state，测正确改变该区域并保持其他区域不变的能力。必须区分“恢复旧答案”和“根据新证据修正答案”。

### C. Trainability / actual compute

记录训练曲线、未剪裁梯度范数、剪裁频率、数值失败。

给两个 rollout 深度记录 loss 对初始状态的梯度范数。该值不是完整 Jacobian 最大奇异值，也不是 loss landscape 证明。

另用相同 batch/网格/horizon，warm-up 后测完整 encoder + rollout + readout 延迟。GPU 测量前后同步；不把 CPU 计时称为 GPU 收益。

最终比较 matched-quality time-to-solution 与任务质量—延迟曲线；单步 kernel 快慢、等步数优劣都不单独决定架构胜负。

## 如何解释第一轮结果

- 只比 rd_nca 好，却不优于 momentum_nca：当前收益可以由一般 momentum 解释，尚无 PDE transport 的增量价值证据。
- 比两个 momentum/state-matched 对照更早学会，且远程反事实正确率、同预算持续计算或修复至少一个明显改善：值得继续研究这个参数化。不同种子复核，而不是立即宣布通用 backbone。
- 同步数更准但实际用时更长：可能是计算分配 trade-off；保留 quality–time 曲线，不写 speedup。
- 学会单一尺寸但大尺寸退化：与固定阻尼的有限波动尺度相容；不能推翻所有 PDE-typed state，也不能宣称已有无限尺寸外推。
- beta 通道没有分化：不直接判死。是否存在不同有效动力学，还取决于 D、reaction Jacobian、任务和频带；标量参数分布不是机制本身。

这里不把“出现 Turing 不稳定”列为成功条件。好的下游推理未必需要从均匀态起纹。

## 与原始 2x2 audit 的关系

这四臂不是传播图 × 权重共享的总诊断。它们只检验当前两行候选中两项具体改变：显式 transport 与惯性，并控制状态容量。没有扩展到 attention、解耦权重、multiscale 或 PDE fitting。
