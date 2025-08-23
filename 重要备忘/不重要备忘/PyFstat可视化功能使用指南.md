### PyFstat 可视化功能使用指南

本文档旨在为您详细总结PyFstat中内置的各类可视化功能，并提供具体的使用案例和参数配置说明，作为您日后的速查手册。

---

### 准备工作：生成一份多维搜索数据

为了更好地演示`plot_1D`和`plot_2D`在高维数据下的行为，我们首先生成一份3D格点搜索的结果。后续的示例将基于这份名为`search_3D`的搜索结果。

```python
import pyfstat
import numpy as np
import os

# --- 1. 准备数据 ---
outdir = "pyfstat_viz_memo_data"
logger = pyfstat.set_up_logger(label="viz_memo", outdir=outdir)

data_params = {
    'tstart': 1000000000,
    'duration': 10 * 86400, # 10天，用于快速生成
    'detectors': 'H1',
    'sqrtSX': '1e-23',
}
tref = data_params['tstart']

injection_params = {
    'F0': 100.0,
    'F1': -1e-11,
    'F2': 0,
    'Alpha': 1.0,
    'Delta': 0.5,
    'h0': float(data_params['sqrtSX']) / 20, # depth=20, 强信号
    'cosi': 0.8,
    'tref': tref,
}

data_writer = pyfstat.Writer(
    label="viz_memo", outdir=outdir, **data_params, **injection_params
)
data_writer.make_data()
sft_path_pattern = data_writer.sftfilepath

# --- 2. 执行一个3D格点搜索 ---
search_3D = pyfstat.GridSearch(
    label="grid_search_3D",
    outdir=outdir,
    sftfilepattern=sft_path_pattern,
    F0s=[99.9999, 100.0001, 1e-5],
    F1s=[-2e-11, 0, 2e-12],
    Alphas=[0.9, 1.1, 0.02],
    Deltas=[injection_params['Delta']], # 将Delta维度固定
    tref=tref
)
search_3D.run()

print("3D格点搜索数据已生成完毕，可以开始进行可视化。")
```

---

### 1. 格点搜索 (`GridSearch`) 可视化

#### `plot_1D(xkey, ...)`

*   **功能与解读**: 此函数用于绘制F统计量随**单个参数**的变化。然而，在处理**多维搜索结果**时，它的行为需要被精确理解：
    *   **它不是投影或边缘化**。它只是将数据表中`xkey`对应的整个列和`2F`对应的整个列直接绘制出来。
    *   **如何解读**: 对于高维数据，您看到的不是一条平滑曲线，而是一个“**散点带**”。这个带子显示了，在所有其他维度（本例中是`F1`和`Alpha`）取遍所有值的情况下，`xkey`参数（如`F0`）在每个取值点上对应的**所有`2F`值的集合**。这个带子的“上边缘”轮廓，才大致反映了最大`2F`值随该参数的变化趋势。

*   **参数配置**: 核心参数是`xkey`，一个字符串，用于指定X轴是哪个物理参数，如`'F0'`。

*   **使用案例**:
    ```python
    # 假设 search_3D 对象已在上一步中生成并运行
    search_3D.plot_1D(
        xkey='F0',
        xlabel="Frequency (Hz)",
        ylabel="2F Statistic"
    )
    # 这将生成一张名为 grid_search_3D_1D_F0_twoF.png 的图片
    ```

#### `plot_2D(xkey, ykey, ...)`

*   **功能与解读**: 此函数将高维的F统计量数据投影到一个二维平面上，并以**热力图**的形式展示。它采用的降维方式是**最大值投影 (Maximization Projection)**。
    *   **工作原理**: 当您指定绘制`xkey`和`ykey`两个维度时，对于图中每一个像素点`(x, y)`，它的颜色值所代表的`2F`，是在所有**被隐藏**的维度（本例中是`Alpha`）的所有取值中，所能找到的那个**最大**的`2F`值。
    *   **如何解读**: 这张图回答了这样一个问题：“固定X和Y轴的参数，在所有其他参数的搜索范围内，我能得到的**最乐观（最大）**的`2F`值是多少？” 这与MCMC中通过积分进行边缘化以降维的意义完全不同。

*   **参数配置**: 核心参数是`xkey`和`ykey`，两个字符串，分别定义X轴和Y轴的物理参数。

*   **使用案例**:
    ```python
    # 假设 search_3D 对象已在上一步中生成并运行
    search_3D.plot_2D(
        xkey='F0',
        ykey='F1'
    )
    # 这将生成一张名为 grid_search_3D_2D_F0_F1_twoF.png 的图片
    ```

---

### 2. MCMC 搜索 (`MCMCSearch`) 可视化

*注意：运行以下案例前，需要先运行一份MCMC搜索，例如`PyFstat的MCMC功能使用指南.md`中的代码。这里只展示绘图方法本身。*

#### `plot_walkers()`

*   **功能**: 绘制所有“行者”在MCMC采样过程中的**轨迹**。
*   **用途**: **诊断MCMC收敛性**的最重要工具。通过观察行者轨迹是否从分散的初始状态收敛到平稳的分布，可以判断“燃烧期”是否足够，采样是否有效。
*   **使用案例**:
    ```python
    # 假设 mcmc_search 对象已运行完毕
    mcmc_search.plot_walkers()
    ```

#### `plot_corner()`

*   **功能**: 绘制所有被搜索参数的**角图（Corner Plot）**。
*   **用途**: **展示MCMC最终结果**的核心和标准方式。它在一个图中同时展示了每个参数的**一维后验概率分布**（看最佳值和不确定度）和每两个参数之间的**二维联合后验概率分布**（看相关性）。
*   **使用案例**:
    ```python
    # 假设 mcmc_search 对象已运行完毕
    # truths 参数可以在图上用线条标出注入的真实值，非常便于对比
    mcmc_search.plot_corner(truths=injection_params)
    ```

#### `plot_prior_posterior()`

*   **功能**: 将每个参数的**先验（Prior）**和**后验（Posterior）**分布绘制在同一张图上进行对比。
*   **用途**: 直观地判断数据对我们知识的更新程度。如果后验分布远比先验窄，说明数据提供了很强的约束。
*   **使用案例**:
    ```python
    # 假设 mcmc_search 对象已运行完毕
    mcmc_search.plot_prior_posterior(injection_parameters=injection_params)
    ```

---

### 3. 通用及数据检查可视化

这些功能通常在搜索前用于检查数据，或用于对单个候选点进行诊断。

#### `plot_spectrogram()`

*   **功能**: 生成并绘制SFT数据的**语图（Spectrogram）**。
*   **用途**: 检查数据质量，查看是否存在明显的噪声线，或一个极强的信号是否肉眼可见。
*   **使用案例**:
    ```python
    # 需要一个Search对象来加载数据
    # 这里的GridSearch只是为了加载数据，不运行
    data_viewer = pyfstat.GridSearch(sftfilepattern=sft_path_pattern, outdir=outdir)
    data_viewer.plot_spectrogram(fmin=99.9, fmax=100.1)
    ```

#### `plot_twoF_cumulative()`

*   **功能**: 绘制在**一个固定参数点**上，`2F`值随观测时间累积增长的过程。
*   **用途**: 诊断候选信号。真实信号的`2F`值会随时间大致线性增长，而噪声尖峰则不会。
*   **使用案例**:
    ```python
    # 需要一个ComputeFstat对象
    calculator = pyfstat.ComputeFstat(sftfilepattern=sft_path_pattern, **injection_params)
    
    # CFS_input 字典包含了要计算的那个固定点的参数
    calculator.plot_twoF_cumulative(CFS_input=injection_params)
    ```
