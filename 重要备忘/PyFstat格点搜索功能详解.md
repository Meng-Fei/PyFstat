### PyFstat F-统计量格点搜索功能详解

本文档旨在详细阐述在使用 PyFstat 库进行F统计量格点扫描（Grid Search）时，所涉及的信号参数、噪声/探测器参数以及扫描机制，并提供一份完整的代码示例。

---

#### 1. 核心概念

F统计量是一种在引力波数据处理中广泛应用的最佳匹配滤波（matched-filtering）统计量，用于搜寻来自旋转中子星的、微弱的、持续的引力波信号（Continuous Waves, CWs）。

**格点扫描（Grid Search）** 的基本思想是：
1.  在一个多维的信号参数空间中（例如：频率 `F0`、一阶频率导数 `F1`等）定义一个离散的网格。
2.  在每一个网格点上，都假设存在一个具有该点参数的引力波信号。
3.  计算数据与该假设信号模板的匹配程度，即F统计量的值。
4.  通过遍历所有网格点，找到F统计量最大的点，该点对应的参数即为最可能的信号源参数。

---

#### 2. 信号注入参数

在进行模拟和测试时，我们首先需要向噪声中“注入”一个已知参数的模拟信号。这些参数定义了引力波源的物理属性。在 `GridSearch` 中，它们不仅用于生成模拟信号，更重要的是，它们**定义了格点扫描的中心或固定值**。

| 参数名 | 物理含义 | 单位 | 说明 |
| :--- | :--- | :--- | :--- |
| `F0` | 参考时刻的引力波频率 | Hz | 这是最核心的参数之一。 |
| `F1` | 频率的一阶时间导数（Spindown） | Hz/s | 描述了由于能量辐射导致的频率下降速率。 |
| `F2` | 频率的二阶时间导数 | Hz/s² | 更高阶的Spindown项。 |
| `Alpha` | 天球坐标系下的赤经 | rad | 信号源的天空位置。 |
| `Delta` | 天球坐标系下的赤纬 | rad | 信号源的天空位置。 |
| `h0` | 引力波应变幅度 | (无) | 信号的内在强度。在F统计量计算中，此参数被解析地最大化，故无需扫描。 |
| `cosi` | 中子星自转轴与观测者视线的夹角的余弦 | (无) | `iota` 是倾角。此参数影响信号的偏振。同样被解析地最大化。 |
| `psi` | 引力波的偏振角 | rad | 描述了引力波椭圆在天空平面上的方向。同样被解析地最大化。 |
| `phi0` | 参考时刻的初始相位 | rad | 信号的初始相位。同样被解析地最大化。 |
| `tref` | 参考时刻 | GPS秒 | 所有与时间相关的参数的参考时间点。**说明**: 此参数是一个规范选择（Gauge Choice），定义了其他参数的计算基准。它在搜索中被固定，而不是一个被扫描或最大化的物理自由度。

---

#### 3. 噪声与探测器参数

这些参数定义了SFT数据的属性和来源。

| 参数名 | 含义 | 默认值/设置方式 |
| :--- | :--- | :--- |
| `sftfilepattern` | SFT文件的路径和匹配模式 | **必须指定**。例如 `data/*.sft`。 |
| `minStartTime` | 开始分析的最早GPS时间 | `None`。若为`None`，则从SFT文件中自动读取。 |
| `maxStartTime` | 结束分析的最晚GPS时间 | `None`。若为`None`，则从SFT文件中自动读取。 |
| `detectors` | 使用的探测器 | `None`。若为`None`，则从SFT文件中自动推断。可指定为逗号分隔的字符串，如 `'H1,L1'`。 |
| `assumeSqrtSX` | 噪声强度的假设值 | `None`。用于模拟，代表单边噪声功率谱密度 `sqrt(S_h)`。

---

#### 4. 格点扫描参数 (`GridSearch` 类)

这些参数直接定义了多维参数空间的网格。

| 参数名 | 含义 | 设置方式与说明 |
| :--- | :--- | :--- |
| `label` | 搜索任务的标签 | 字符串，用于构成输出文件的名字。 |
| `outdir` | 输出目录 | 字符串，存放搜索结果和图表的目录。 |
| `F0s`, `F1s`, ... | **核心扫描维度** | 定义格点扫描范围的关键。格式为元组 `(min, max, step)`。例如 `F0s=(100.0, 100.1, 1e-4)`。若只想固定参数，则传入单元素列表，如 `Alphas=[1.23]`。 |
| `nsegs` | 数据分段数 | `1`。`nsegs=1` 表示全相干搜索。大于1表示半相干搜索。

---

#### 5. 维度配置与未指定维度的行为

*   **在格点中的维度**: PyFstat 会在您定义的 `(min, max, step)` 范围内进行遍历。
*   **不在格点中的物理维度**: 在计算时，这些参数的值会被**固定**为您在初始化时提供的值。
*   **解析最大化的维度** (`h0`, `cosi`, `psi`, `phi0`): 这四个振幅参数**永远不需要被扫描**，F统计量的数学构造本身就包含了对它们的解析最大化。

### 6. 完整工作流程示例

以下代码展示了从头开始的一个完整流程：创建包含注入信号的数据 -> 设置并运行一个二维格点搜索 -> 将结果可视化。

```python
import pyfstat
import numpy as np
import os

# --- 1. 定义基础设置 ---
# 所有输出（SFT数据、计算结果、图片）都将保存在这个目录
outdir = "PyFstat_GridSearch_Example"
# label用于区分不同的任务，会出现在文件名中
label = "MyFirstGridSearch"
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# --- 2. 创建模拟数据 (Writer) ---
# 数据参数
data_params = {
    'tstart': 1000000000,
    'duration': 10 * 86400,  # 10天，用于快速演示
    'detectors': 'H1',
    'sqrtSX': '1e-23',
}
tref = data_params['tstart']

# 注入信号的参数
injection_params = {
    'F0': 100.0,
    'F1': -1e-10,
    'F2': 0,
    'Alpha': 1.0,
    'Delta': 0.5,
    'h0': float(data_params['sqrtSX']) / 25, # depth=25, 一个较强的信号
    'cosi': 0.8,
    'tref': tref,
}

# 实例化Writer并生成SFT数据
data_writer = pyfstat.Writer(
    label=label, outdir=outdir, **data_params, **injection_params
)
data_writer.make_data()

# 获取SFT文件的路径模式，这是连接数据和搜索的关键
sft_path_pattern = data_writer.sftfilepath

# --- 3. 设置并运行格点搜索 (GridSearch) ---
# 定义一个二维的搜索空间：F0 vs F1
# 其他参数（如Alpha, Delta）将使用注入时的固定值
search = pyfstat.GridSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=sft_path_pattern,
    F0s=[99.9999, 100.0001, 2e-6],
    F1s=[-1.1e-10, -0.9e-10, 1e-12],
    Alphas=[injection_params['Alpha']], # 固定天空位置
    Deltas=[injection_params['Delta']],
    tref=tref
)
# 执行搜索
search.run()

# --- 4. 可视化结果 ---
# 将二维搜索结果绘制成热力图
search.plot_2D(xkey='F0', ykey='F1')

print(f"\n流程执行完毕！所有结果已保存在 '{outdir}' 目录下。")
```

#### 结果保存在哪里？

当您运行完上面的脚本后，所有的输出都会被保存在您定义的 `outdir` 目录（在此例中是 `PyFstat_GridSearch_Example`）中：

1.  **SFT数据**: 由`Writer`生成的SFT文件。它们的文件名会遵循 `outdir/label-GPS-时长.sft` 的模式，例如：
    *   `PyFstat_GridSearch_Example/MyFirstGridSearch-1000000000-1800.sft`

2.  **计算结果**: `search.run()` 的主要输出是一个文本文件，里面包含了每个格点的参数值和对应的`2F`统计量。其命名模式为 `outdir/label_detectors_ClassName.txt`，例如：
    *   `PyFstat_GridSearch_Example/MyFirstGridSearch_H1_GridSearch.txt`

3.  **可视化结果**: `search.plot_2D()` 生成的图片。其命名模式为 `outdir/label_plot-type_params_statistic.png`，例如：
    *   `PyFstat_GridSearch_Example/MyFirstGridSearch_2D_F0_F1_twoF.png`
