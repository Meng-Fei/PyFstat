### PyFstat 核心计算内核使用备忘

本文档旨在为您日后进行自定义算法探索（如量子加速、梯度下降等）提供一个关于如何使用PyFstat核心计算功能的、可直接上手的代码备忘。

---

### 目标：构建 `f(params) -> 2F` 核心函数

我们的核心目标是创建一个Python函数，该函数接收一组信号参数，并返回在该参数点上的F统计量值（`2F`）。这个函数是所有高级算法的基础。

`f(params) -> 2F_value`

我们将通过以下三个步骤实现和使用它。

---

### 步骤一：准备数据（信号注入与SFT生成）

在进行任何计算之前，我们需要一个包含信号和噪声的SFT数据集。以下代码将生成一个用于测试的数据集。

*   **场景**: 观测时长10天，注入一个相对较强 (`depth=30`) 的信号，使用H1和L1两个探测器。

```python
import pyfstat
import numpy as np
import os

# --- 1. 定义输出目录和日志 ---
outdir = "pyfstat_kernel_memo_data"
logger = pyfstat.set_up_logger(label="kernel_memo", outdir=outdir)

# --- 2. 定义数据和信号参数 ---
data_params = {
    'tstart': 1000000000,
    'duration': 10 * 86400,  # 10天，用于快速测试
    'detectors': 'H1,L1',
    'sqrtSX': '4e-23', # 模拟O3噪声水平
}
tref = data_params['tstart']

injection_params = {
    'F0': 150.0,
    'F1': -1e-10,
    'F2': 0,
    'Alpha': 1.5,
    'Delta': 0.5,
    'h0': float(data_params['sqrtSX']) / 30, # depth=30, 较强信号
    'cosi': 0.5,
    'tref': tref,
}

# --- 3. 生成SFT数据 ---
data_writer = pyfstat.Writer(
    label="kernel_memo",
    outdir=outdir,
    **data_params,
    **injection_params
)
data_writer.make_data()

# --- 4. 获取SFT文件路径模式（关键！） ---
sft_path_pattern = data_writer.sftfilepath
print(f"数据已生成，SFT路径模式为: {sft_path_pattern}")
```

---

### 步骤二：封装核心计算内核

现在，我们来创建那个核心的 `f(params) -> 2F` 函数。我们将使用 `pyfstat.ComputeFstat` 类，因为我们只需要单点的计算功能。

```python
# (接上文代码)

# 将固定参数打包，方便传入
# 注意：ComputeFstat需要一个明确的tref，即使注入参数里有
setup_args = {
    "tref": tref,
    "minStartTime": data_params['tstart'],
    "maxStartTime": data_params['tstart'] + data_params['duration'],
    "sftfilepattern": sft_path_pattern,
}

def get_2F_value(search_params):
    """
    核心计算函数 (Kernel)
    接收一个包含搜索参数的字典，返回2F值。
    """
    # 1. 将搜索参数与固定参数合并
    all_params = setup_args.copy()
    all_params.update(search_params)

    # 2. 实例化F统计量计算器
    # 我们只计算单点，所以search_ranges可以设为None
    calculator = pyfstat.ComputeFstat(search_ranges=None, **all_params)

    # 3. 执行计算并返回结果
    twoF = calculator.get_det_stat()
    return twoF

# --- 测试一下核心函数 ---
# 使用注入的真实参数进行测试，应该会得到一个很高的2F值
true_signal_params = {
    'F0': injection_params['F0'],
    'F1': injection_params['F1'],
    'F2': injection_params['F2'],
    'Alpha': injection_params['Alpha'],
    'Delta': injection_params['Delta'],
}

max_2F = get_2F_value(true_signal_params)
print(f"在真实信号点的2F值为: {max_2F:.2f}")
```

---

### 步骤三：使用与并行化技巧

当您需要为自定义算法（如计算梯度）进行多次核心函数调用时，并行化是必不可少的。Python的 `multiprocessing` 库可以与我们的核心函数完美结合。

```python
# (接上文代码)
import multiprocessing

# --- 示例：通过有限差分计算F0方向的梯度 ---
# 1. 定义需要并行计算的参数点列表
dF0 = 1e-9 # 一个很小的频率偏移量

point_center = true_signal_params.copy()
point_plus_dF0 = true_signal_params.copy()
point_plus_dF0['F0'] += dF0

points_to_evaluate = [point_center, point_plus_dF0]
print(f"\n即将并行计算 {len(points_to_evaluate)} 个点...")

# 2. 使用multiprocessing.Pool进行并行计算
# 注意：在脚本中运行时，通常需要将并行代码放在 if __name__ == "__main__": 块下
if __name__ == "__main__":
    # 使用的CPU核心数，可根据您的机器配置调整
    num_processes = os.cpu_count()
    print(f"使用 {num_processes} 个CPU核心...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # map函数会将列表中的每个元素作为参数，传递给get_2F_value函数
        # 并行地执行它们，然后收集所有返回值
        results = pool.map(get_2F_value, points_to_evaluate)

    # 3. 利用返回的结果计算梯度
    gradient_F0 = (results[1] - results[0]) / dF0
    print(f"在中心点的2F值为: {results[0]:.2f}")
    print(f"在F0+dF0点的2F值为: {results[1]:.2f}")
    print(f"估算的F0方向梯度为: {gradient_F0:.2e}")
```

### 总结

通过将 `pyfstat.ComputeFstat` 封装成一个简单的Python函数，您可以将其作为一个强大的“黑盒子”计算核心，并利用Python丰富的并行计算生态来高效地驱动您的自定义算法探索。

```