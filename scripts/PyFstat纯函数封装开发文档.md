# PyFstat纯函数封装

## 项目概述

将PyFstat的F统计量计算功能封装为纯函数：`f(F0, F1, Alpha, Delta) -> 2F`

### 核心特性
- ✅ 纯函数设计，无副作用
- ✅ 支持任意参数点的单点计算
- ✅ 可通过partial固定变量实现降维
- ✅ 兼容scipy、matplotlib等标准工具

## 技术架构

### 1. SignalConfig配置类

```python
@dataclass
class SignalConfig:
    # 数据参数
    tstart: int = 1000000000    # GPS开始时间
    duration: int = 86400        # 观测时长（秒）
    detectors: str = 'H1'        # 探测器
    sqrtSX: float = 1e-23        # 噪声水平
    Tsft: int = 1800             # SFT时长
    
    # 信号强度
    depth: float = 30.0          # 信号深度
    
    # 注入信号参数
    injection_F0: float = 30.0      # 注入频率
    injection_F1: float = -1e-10    # 频率导数
    injection_Alpha: float = 1.0    # 赤经
    injection_Delta: float = 0.5    # 赤纬
    
    # 固定参数
    F2: float = 0.0              # 二阶频率导数
    cosi: float = -1            # cos(倾角)
    psi: float = 0.0             # 偏振角
    phi: float = 0.0             # 初始相位
    
    @property
    def h0(self) -> float:
        return self.sqrtSX / self.depth
    
    @property
    def tref(self) -> int:
        return self.tstart + self.duration // 2
```

### 2. FStatFunction核心类

```python
class FStatFunction:
    def __init__(self, config: SignalConfig):
        # 生成SFT数据
        writer = pyfstat.Writer(
            label="fstatfunc",
            outdir=self.tempdir,
            Band=5.0,  # 5Hz带宽
            F0=config.injection_F0,
            F1=config.injection_F1,
            Alpha=config.injection_Alpha,
            Delta=config.injection_Delta,
            # ... 其他参数
        )
        writer.make_data()
        
        # 初始化ComputeFstat（纯计算模式）
        self.compute_fstat = pyfstat.ComputeFstat(
            sftfilepattern=writer.sftfilepath,
            tref=config.tref,
            minStartTime=config.tstart,
            maxStartTime=config.tstart + config.duration,
            minCoverFreq=-0.5,  # 自动推断
            maxCoverFreq=-0.5,  # 自动推断
        )
    
    def __call__(self, F0: float, F1: float, Alpha: float, Delta: float) -> float:
        """纯函数接口"""
        return self.compute_fstat.get_fullycoherent_twoF(
            F0=F0, F1=F1, F2=self.config.F2,
            Alpha=Alpha, Delta=Delta
        )
```

### 3. 数值梯度计算

```python
def compute_gradient(f, F0, F1, Alpha, Delta, free_vars, deltas=None):
    """
    中心差分法计算梯度
    ∂f/∂x ≈ (f(x+δ) - f(x-δ)) / 2δ
    """
    default_deltas = {
        'F0': 1e-9,
        'F1': 1e-16,
        'Alpha': 1e-6,
        'Delta': 1e-6
    }
    
    if deltas:
        default_deltas.update(deltas)
    
    gradients = {}
    base_params = {'F0': F0, 'F1': F1, 'Alpha': Alpha, 'Delta': Delta}
    
    for var in free_vars:
        delta = default_deltas[var]
        
        # 前向点
        params_plus = base_params.copy()
        params_plus[var] += delta
        f_plus = f(**params_plus)
        
        # 后向点
        params_minus = base_params.copy()
        params_minus[var] -= delta
        f_minus = f(**params_minus)
        
        # 中心差分
        gradients[var] = (f_plus - f_minus) / (2 * delta)
    
    return gradients
```

## 使用指南

### 基本使用

```python
from signal_config import SignalConfig
from fstat_function import FStatFunction

# 初始化
config = SignalConfig(duration=86400, depth=20)
f = FStatFunction(config)

# 单点计算
twoF = f(30.0, -1e-10, 1.0, 0.5)
print(f"2F = {twoF:.2f}")
```

### 降维使用

```python
from functools import partial
import numpy as np

# 1D函数：固定3个变量
f_1d = partial(f, F1=-1e-10, Alpha=1.0, Delta=0.5)
x = np.linspace(28.0, 31.9, 100)
y = [f_1d(F0=xi) for xi in x]

# 2D函数：固定2个变量
f_2d = lambda F0, Alpha: f(F0, -1e-10, Alpha, 0.5)
Z = [[f_2d(x, y) for x in f0_range] for y in alpha_range]
```

### 梯度计算

```python
from numerical_gradient import compute_gradient

grad = compute_gradient(
    f, 30.0, -1e-10, 1.0, 0.5,
    free_vars=['F0', 'Alpha'],
    deltas={'F0': 1e-9, 'Alpha': 1e-6}
)

print(f"∂2F/∂F0 = {grad['F0']:.2e}")
print(f"∂2F/∂Alpha = {grad['Alpha']:.2e}")
```

### 与标准工具集成

```python
# scipy优化
from scipy.optimize import minimize

result = minimize(
    lambda x: -f(x[0], -1e-10, x[1], 0.5),
    x0=[30.0, 1.0],
    bounds=[(28.0, 31.9), (0, 2*np.pi)],
    method='L-BFGS-B'
)

# matplotlib绘图
import matplotlib.pyplot as plt

X, Y = np.meshgrid(f0_range, alpha_range)
Z = np.array([[f(x, -1e-10, y, 0.5) for x in f0_range] for y in alpha_range])
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='2F')
plt.xlabel('F0 (Hz)')
plt.ylabel('Alpha (rad)')
```

## 性能指标

### 频率范围
- **支持范围**: 28.0 - 31.9 Hz（约4Hz带宽）
- **频率分辨率**: ~0.000556 Hz (1/Tsft)
- **边界限制原因**: PyFstat考虑多普勒效应的保守估计

### 计算性能
- **初始化时间**: 2-3秒（生成SFT文件）
- **单点计算**: <10ms
- **内存占用**: ~50MB（主要是SFT数据）

### 参数范围
| 参数 | 有效范围 | 单位 |
|-----|---------|-----|
| F0 | 28.0 - 31.9 | Hz |
| F1 | -1e-8 - 1e-8 | Hz/s |
| Alpha | 0 - 2π | rad |
| Delta | -π/2 - π/2 | rad |

## 文件结构

```
scripts/
├── signal_config.py       # 配置类定义
├── fstat_function.py      # 核心函数封装
├── numerical_gradient.py  # 梯度计算工具
├── test_fstat_function.py # 单元测试
└── example_usage.py       # 使用示例
```

## 关键技术点

1. **纯函数设计**: 无状态、无副作用，输入确定则输出确定
2. **频率离散性**: SFT频率是离散bin，分辨率=1/Tsft
3. **多普勒补偿**: PyFstat自动处理地球运动引起的频移
4. **避免网格搜索**: 不使用search_ranges参数，实现单点计算