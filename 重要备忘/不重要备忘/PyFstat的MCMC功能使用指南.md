### PyFstat的MCMC功能使用指南

本文档旨在为您详细介绍PyFstat中蒙特卡洛（MCMC）搜索功能的核心理论、参数配置及具体使用方法，作为您日后进行相关研究的备忘。

---

### 1. 核心思想：贝叶斯参数推断

与寻找唯一最佳点的格点搜索不同，MCMC的目的是在贝叶斯统计的框架下，完整地描绘出信号参数的**后验概率分布（Posterior Probability Distribution）**。

其理论基础是贝叶斯定理：
`P(θ|D) ∝ P(D|θ) × P(θ)`
即：**后验概率 ∝ 似然函数 × 先验概率**

*   **后验概率**: 我们最终的目标，即在观测到数据后，参数为某值的可信度。
*   **似然函数**: 连接数据和理论模型的桥梁，体现了数据对参数的支持程度。
*   **先验概率**: 我们在分析前对参数已有的认知或假设。

---

### 2. 算法实现：`emcee` 与 `ptemcee`

PyFstat的MCMC功能，其核心是基于一个非常流行和强大的Python库——**`emcee`**（及其支持并行回火的变体`ptemcee`）来实现的。

它采用了一种高效的“系综采样器”（Ensemble Sampler），通过一个“行者（Walker）”团体在参数空间中智能地探索，最终这些“行者”的足迹就描绘出了后验概率分布。

---

### 3. Likelihood 与 Prior 的计算细节

#### 3.1 Likelihood (似然函数)

这是MCMC与F统计量结合的**最关键一步**。在PyFstat中，对数似然函数 `log(L)` 与F统计量 `F` 存在一个精确的线性关系。

*   **理论关系**: F统计量本身是对数似然比，`2F = log(L_signal) - log(L_noise)`。在MCMC采样中，`log(L_noise)`项为常数，因此 `log(L_signal) ∝ F`。
*   **PyFstat实现**: PyFstat为了进行严谨的贝叶斯证据计算，使用了被正确归一化的对数似然函数。其具体形式为：
    `logL = 2F * 0.5 + log(70 / ρ_max⁴) = F + log(70 / ρ_max⁴)`
    *   **比例系数 `0.5`**: 将 `2F` 转换回理论上与 `logL` 成正比的 `F`。
    *   **加法系数 `log(70 / ρ_max⁴)`**: 一个基于振幅先验和用户设定的最大信噪比 `ρ_max` 计算出的归一化常数。对于纯粹的参数推断，此项不影响结果；但对于严谨的贝叶斯模型选择，此项至关重要。

#### 3.2 Prior (先验概率)

先验通过一个名为 `theta_prior` 的字典来定义，它告诉MCMC每个参数允许的范围和分布。

*   **实现**: MCMC的“行者”每尝试走出先验设定的边界，其对数先验概率 `log(P)` 就会变为负无穷大，从而阻止该移动，将搜索约束在有效区域内。

---

### 4. 先验设置通用原则

1.  **均匀先验 (Uniform Prior)** - **最常用**
    *   **含义**: `{'type': 'unif', 'lower': ..., 'upper': ...}`
    *   **用途**: 当你只知道参数的大致范围，但无任何倾向性时使用。这是最保守、最诚实的先验，让数据自己说话。

2.  **固定值 (Delta函数先验)** - **用于已知参数**
    *   **含义**: `'Alpha': 1.23`
    *   **用途**: 当参数已通过其他方式精确得知时（如定向搜索），将其固定可以降低搜索维度，节省大量计算资源。

3.  **信息先验 (Informative Prior)** - **用于有额外物理知识时**
    *   **含义**: `{'type': 'norm', 'loc': ..., 'scale': ...}` (正态分布)
    *   **用途**: 如果有理论或其它观测证据表明参数可能服从某种分布（如高斯分布），使用信息先验可以加速收敛并提高精度。

---

### 5. MCMC 配置与使用指导

以下是一个典型的、从数据生成到MCMC搜索和可视化的完整流程。

```python
import pyfstat
import numpy as np
import os

# --- 步骤一：准备数据 ---
outdir = "pyfstat_mcmc_memo_data"
logger = pyfstat.set_up_logger(label="mcmc_memo", outdir=outdir)

data_params = {
    'tstart': 1000000000,
    'duration': 100 * 86400,
    'detectors': 'H1,L1',
    'sqrtSX': '4e-23',
}
tref = data_params['tstart']

injection_params = {
    'F0': 150.0,
    'F1': -1e-10,
    'F2': 0,
    'Alpha': 1.5,
    'Delta': 0.5,
    'h0': float(data_params['sqrtSX']) / 70, # depth=70, 较弱信号
    'cosi': 0.5,
    'tref': tref,
}

data_writer = pyfstat.Writer(
    label="mcmc_memo", outdir=outdir, **data_params, **injection_params
)
data_writer.make_data()
sft_path_pattern = data_writer.sftfilepath

# --- 步骤二：定义先验 (MCMC的核心配置) ---
# 模拟一个定向搜索场景：已知天空位置和F2，搜索F0和F1
theta_prior = {
    'F0': {'type': 'unif', 'lower': 149.999, 'upper': 150.001},
    'F1': {'type': 'unif', 'lower': -2e-10, 'upper': 0},
    'F2': injection_params['F2'],
    'Alpha': injection_params['Alpha'],
    'Delta': injection_params['Delta'],
}

# --- 步骤三：实例化并运行MCMCSearch ---
mcmc_search = pyfstat.MCMCSearch(
    label="mcmc_memo_search",
    outdir=outdir,
    sftfilepattern=sft_path_pattern,
    theta_prior=theta_prior,
    tref=tref,
    nsteps=[200, 500],  # [燃烧步数, 采样步数]
    nwalkers=100,       # “行者”数量
    ntemps=2,           # 温度阶梯数（用于并行回火）
)

mcmc_search.run()

# --- 步骤四：可视化结果 ---
# 绘制角图（Corner Plot），展示参数的后验分布和协方差
mcmc_search.plot_corner(truths=injection_params)

print(f"MCMC分析完成，结果已保存在 {outdir} 目录。")
```
