# PyFstat纯函数封装开发文档

## 项目目标

将PyFstat的F统计量计算功能封装为一个纯函数，使其像 `f(x,y) = x² + y²` 一样简单易用，支持：
- 任意固定变量实现降维
- 数值梯度计算
- 标准Python工具集成（scipy、matplotlib等）

## 核心设计理念（基于Linus哲学）

1. **纯函数设计**：`f(F0, F1, Alpha, Delta) -> 2F`，没有隐式行为
2. **显式控制**：必须提供全部4个参数，不允许默认值
3. **零防御编程**：出错立即失败，不隐藏问题
4. **标准接口**：兼容Python生态系统

## 遇到的问题与解决方案

### 问题1：参数命名错误

**现象**：
```
Writer.__init__() got an unexpected keyword argument 'phi0'. Did you mean 'phi'?
```

**原因**：
PyFstat使用`phi`而不是`phi0`作为初始相位参数名。

**解决方案**：
将`signal_config.py`中的`phi0`改为`phi`。

### 问题2：SFT文件标签规范

**现象**：
```
Label 'fstat_func' is not alphanumeric, which is incompatible with the SFTv3 naming specification
```

**原因**：
SFTv3命名规范要求标签只能包含字母数字，不允许下划线、连字符等。

**解决方案**：
将标签从`fstat_func`改为`fstatfunc`。

### 问题3：频率范围匹配问题（最核心的问题）

**现象**：
```
[minCoverFreq,maxCoverFreq]=[30.000000,34.000000] Hz incompatible with SFT files content [30.000000,33.999444] Hz
```

**原因分析**：
1. SFT文件的频率是离散的bin，不是连续的
2. 频率bin宽度 = 1/Tsft（例如Tsft=1800秒时，bin宽度=0.000556Hz）
3. ComputeFstat对频率范围做严格检查，微小差异都会报错
4. 使用`search_ranges`参数时，PyFstat只为指定的参数点优化内部数据结构，无法计算其他参数值

**错误的解决尝试**：
1. ❌ 手动指定`minCoverFreq/maxCoverFreq`：总是有微小误差
2. ❌ 使用`search_ranges`单点优化：导致只能计算该点，其他频率报错
3. ❌ 尝试各种频率范围调整：边界总是对不上

**最终解决方案**：
```python
# 方案1：使用Writer自动计算的最小覆盖带宽（不指定Band）
writer = pyfstat.Writer(
    # ... 其他参数
    # 不指定Band，让PyFstat自动计算
)

# 方案2：指定适中的Band，然后精确匹配
writer = pyfstat.Writer(
    Band=5.0,  # 明确的带宽
    # ... 其他参数
)

# 使用实际的SFT频率范围，稍微收缩以避免边界问题
actual_fmax = writer.fmin + writer.Band
df = 1.0 / writer.Tsft  # bin宽度
self.compute_fstat = pyfstat.ComputeFstat(
    minCoverFreq=writer.fmin + df,  # 跳过第一个bin
    maxCoverFreq=actual_fmax - 2*df  # 跳过最后两个bin
)
```

### 问题4：频率范围超出导致的I/O错误

**现象**：
```
ERROR: data gap or overlap at first bin of SFT#0
XLAL Error - XLALLoadSFTs: I/O error
```

**原因**：
当请求的频率范围超出SFT文件实际包含的范围时，LALSuite底层会报I/O错误。

**解决方案**：
确保ComputeFstat的频率范围严格在SFT文件范围内，并留有余量。

### 问题5：并行计算中的pickle问题

**现象**：
```
cannot pickle 'lalpulsar.SFTCatalog' object
```

**原因**：
PyFstat内部使用C扩展对象（如SFTCatalog），这些对象无法被Python的pickle序列化，因此无法在多进程间传递。

**解决方案**：
在梯度计算时使用`parallel=False`参数，改用串行计算：
```python
grad = compute_gradient(f, 30.0, -1e-10, 1.0, 0.5,
                       free_vars=['F0', 'Alpha'],
                       parallel=False)  # 避免pickle问题
```

## 最终架构设计

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
    
    # 信号强度控制
    depth: float = 30.0          # 信号深度
    
    # 固定参数
    F2: float = 0.0              # 频率二阶导数
    cosi: float = 0.0            # cos(倾角)
    psi: float = 0.0             # 偏振角
    phi: float = 0.0             # 初始相位
    
    @property
    def h0(self) -> float:
        """引力波应变幅度"""
        return self.sqrtSX / self.depth
    
    @property
    def tref(self) -> int:
        """参考时间（观测中点）"""
        return self.tstart + self.duration // 2
```

### 2. FStatFunction纯函数类

```python
class FStatFunction:
    def __init__(self, config: SignalConfig):
        # 1. 创建临时目录
        self.tempdir = tempfile.mkdtemp()
        
        # 2. 生成SFT数据
        writer = pyfstat.Writer(
            label="fstatfunc",
            outdir=self.tempdir,
            Band=5.0,  # 适中的带宽
            # ... 配置参数
        )
        writer.make_data()
        
        # 3. 初始化ComputeFstat
        self.compute_fstat = pyfstat.ComputeFstat(
            sftfilepattern=writer.sftfilepath,
            minCoverFreq=...,  # 精确匹配SFT范围
            maxCoverFreq=...
        )
    
    def __call__(self, F0: float, F1: float, Alpha: float, Delta: float) -> float:
        """纯函数接口：4参数进，1个2F值出"""
        params = {'F0': F0, 'F1': F1, 'F2': self.config.F2, 
                  'Alpha': Alpha, 'Delta': Delta}
        return self.compute_fstat.get_fullycoherent_twoF(params=params)
```

### 3. 数值梯度计算

```python
def compute_gradient(f, F0, F1, Alpha, Delta, free_vars, deltas=None, parallel=False):
    """
    对任何4参数函数计算数值梯度
    使用中心差分法：∂f/∂x ≈ (f(x+δ) - f(x-δ)) / 2δ
    """
    # 构建评估点
    # 并行或串行计算
    # 返回梯度字典
```

## 使用方法

### 基本使用

```python
# 1. 初始化
from signal_config import SignalConfig
from fstat_function import FStatFunction

config = SignalConfig(duration=86400, depth=20)
f = FStatFunction(config)

# 2. 单点计算（必须提供全部4个参数）
twoF = f(30.0, -1e-10, 1.0, 0.5)
```

### 降维使用

```python
from functools import partial

# 1D函数：固定3个变量
f_1d = partial(f, F1=-1e-10, Alpha=1.0, Delta=0.5)
y = [f_1d(x) for x in np.linspace(29.9, 30.1, 100)]

# 2D函数：固定2个变量
f_2d = lambda x, y: f(x, -1e-10, y, 0.5)
Z = [[f_2d(x, y) for x in x_range] for y in y_range]

# 3D函数：固定1个变量
f_3d = lambda x, y, z: f(x, -1e-10, y, z)
```

### 梯度计算

```python
from numerical_gradient import compute_gradient

# 计算梯度（注意使用parallel=False）
grad = compute_gradient(
    f, 30.0, -1e-10, 1.0, 0.5,
    free_vars=['F0', 'Alpha'],  # 指定自由变量
    deltas={'F0': 1e-9, 'Alpha': 1e-6},  # 步长
    parallel=False  # 避免pickle问题
)

print(f"∂2F/∂F0 = {grad['F0']:.2e}")
print(f"∂2F/∂Alpha = {grad['Alpha']:.2e}")
```

### 与标准工具集成

```python
# 1. scipy优化
from scipy.optimize import minimize

result = minimize(
    lambda x: -f(x[0], -1e-10, x[1], 0.5),  # 最大化2F
    x0=[30.0, 1.0],
    method='Nelder-Mead'
)

# 2. matplotlib绘图
import matplotlib.pyplot as plt

X, Y = np.meshgrid(f0_range, alpha_range)
Z = [[f(x, -1e-10, y, 0.5) for x in f0_range] for y in alpha_range]
plt.contourf(X, Y, Z)

# 3. 数值积分
from scipy.integrate import quad

integral = quad(lambda x: f(x, -1e-10, 1.0, 0.5), 29.9, 30.1)
```

## 性能与限制

### 性能特点
- 初始化开销：生成SFT文件需要几秒
- 单点计算：毫秒级
- 内存占用：主要是SFT文件缓存

### 使用限制
1. **频率范围**：默认配置下约27.5-32.5Hz，超出范围会报错
2. **并行限制**：梯度计算建议串行，避免pickle问题
3. **参数范围**：Delta必须在[-π/2, π/2]内

## 文件清单

```
scripts/
├── signal_config.py       # 配置类
├── fstat_function.py      # 纯函数封装
├── numerical_gradient.py  # 梯度计算
├── test_fstat_function.py # 测试用例
├── example_usage.py       # 使用示例
└── common.py             # 共享常量（保留）
```

## 关键经验总结

1. **不要与底层对抗**：PyFstat和LALSuite有严格的参数检查，微小误差都会失败。解决方案是理解其工作原理，而不是试图绕过。

2. **频率离散性是关键**：SFT的频率是离散bin，不是连续值。理解这一点才能正确设置频率范围。

3. **简单优于复杂**：最初试图用`search_ranges`很"聪明"地解决问题，实际上引入了更多限制。简单的`minCoverFreq/maxCoverFreq`反而更灵活。

4. **显式优于隐式**：要求用户提供全部4个参数看似麻烦，实际上消除了歧义，使行为完全可预测。

5. **接受限制**：并行计算的pickle问题无法完美解决，接受串行计算的限制比强行并行更实际。

## 后续改进建议

1. **扩展频率范围**：可以在初始化时生成更宽的SFT频带
2. **缓存优化**：可以缓存SFT文件，避免重复生成
3. **批量计算**：可以添加向量化接口，一次计算多个点
4. **GPU加速**：PyFstat支持GPU，可以探索GPU加速选项

---

*按照Linus Torvalds的哲学编写：简单、直接、无防御编程*