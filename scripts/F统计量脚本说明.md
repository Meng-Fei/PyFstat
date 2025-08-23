# F统计量计算脚本实现文档

## 概述
在PyFstat项目的`scripts/`目录下创建了`计算F统计量.py`脚本，用于快速计算给定参数的F统计量值。

## 调研过程

### 1. PyFstat项目结构调查
- 项目包含examples目录，有多个示例脚本展示不同搜索方法
- 核心计算类位于`pyfstat/core.py`中的`ComputeFstat`类
- F统计量是引力波数据分析的核心统计量，用于检测连续引力波信号

### 2. ComputeFstat类分析
关键方法调查：
- `__init__`: 初始化计算对象，需要配置时间参数、数据源、频率覆盖范围
- `get_fullycoherent_twoF`: 计算完全相干的2F统计量（F统计量的2倍）
- `get_fullycoherent_detstat`: 更通用的探测统计量计算接口

核心参数：
- `tref`: 参考时间（GPS秒）
- `minStartTime/maxStartTime`: 数据时间范围
- `minCoverFreq/maxCoverFreq`: 频率覆盖范围（必需参数）
- `sftfilepattern`: SFT数据文件模式（可选，不提供则生成模拟数据）

### 3. 示例代码分析
参考`examples/grid_examples/PyFstat_example_grid_search_F0.py`：
- 使用Writer类生成模拟数据
- 使用GridSearch进行网格搜索
- 核心计算通过ComputeFstat实现

## 实现方案

### 设计原则（遵循Linus哲学）
1. **极简主义**：只做一件事 - 计算F统计量
2. **无防御性编程**：出错立即报错，不做智能fallback
3. **最小代码**：直接调用核心API，无冗余封装
4. **纯净输出**：只输出结果值，无调试信息

### 脚本功能
接受命令行参数，计算并返回F统计量值。

#### 必需参数
- `--F0`: 频率 (Hz)
- `--Alpha`: 赤经 (radians)  
- `--Delta`: 赤纬 (radians)

#### 可选参数
- `--F1`: 频率一阶导数（默认0）
- `--F2`: 频率二阶导数（默认0）
- `--tref`: 参考时间（默认1000000000 GPS秒）
- `--tstart`: 开始时间
- `--duration`: 持续时间（秒）
- `--sftfile`: SFT文件路径模式
- `--sqrtSX`: 噪声水平
- `--detectors`: 探测器（默认H1）

### 技术要点

#### 1. 环境配置
使用shebang直接指定conda环境Python解释器：
```python
#!/home/inkecy/miniconda/envs/pyfstat-dev/bin/python
```
这样可以直接运行脚本，无需手动激活conda环境。

#### 2. 日志抑制
PyFstat默认输出大量日志，通过自定义NullHandler完全禁用：
```python
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logging.getLogger().addHandler(NullHandler())
logging.getLogger().setLevel(logging.CRITICAL + 1)
```

#### 3. 频率覆盖范围
ComputeFstat要求明确指定minCoverFreq和maxCoverFreq，设置为F0±1Hz：
```python
minCoverFreq=args.F0 - 1,
maxCoverFreq=args.F0 + 1
```

## 使用示例

直接运行：
```bash
./scripts/计算F统计量.py --F0 30.0 --Alpha 1.0 --Delta 1.5
```

输出：
```
3.8643760681152344
```

使用SFT文件：
```bash
./scripts/计算F统计量.py --F0 30.0 --Alpha 1.0 --Delta 1.5 --sftfile "path/to/*.sft"
```

## 代码位置
- 脚本路径：`/home/inkecy/PyFstat/scripts/计算F统计量.py`
- 文档路径：`/home/inkecy/PyFstat/scripts/F统计量脚本说明.md`

## 核心实现
脚本核心逻辑极简：
1. 解析命令行参数
2. 创建ComputeFstat对象（配置数据源和频率范围）
3. 调用get_fullycoherent_twoF计算F统计量
4. 打印结果值

总代码量约60行，实际核心逻辑不到20行。