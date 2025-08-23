### PyFstat格点搜索结果数据详解

本文档旨在为您提供一份关于`PyFstat.GridSearch`结果数据的、内容详尽的参考指南，作为您日后进行数据操作和二次开发的编程备忘。

---

### 1. 概述：两种形式的结果数据

在您成功运行一次格点搜索后（`search.run()`），其结果会以两种形式存在：

1.  **内存中的对象**: `search.data` 属性。它在Python会话中立即可用，适合进行快速的结果检查和连续操作。
2.  **磁盘上的文件**: 一个`.txt` 格式的纯文本文件。它被自动保存在您指定的`outdir`中，用于永久存储、分享以及日后的离线分析。

这两种形式代表的是同一份数据，我们接下来将分别详细介绍它们的结构和使用方法。

---

### 2. 内存中的 `search.data`：NumPy结构化数组

#### 2.1 核心概念：一维的“超级表格”

`search.data` 的类型是 **NumPy结构化数组 (Structured Array)**。理解它的关键在于，它**不是**一个传统意义上的二维矩阵。

*   **根本区别**: 
    *   一个**标准的二维NumPy数组**（矩阵）的 `shape` 是 `(行数, 列数)`，且所有元素的**数据类型必须相同**（如`float64`）。
    *   而`search.data`是一个**一维数组**，其 `shape` 是 `(总点数,)`。它的特殊之处在于，这个一维数组的**每一个元素**，都是一个包含了多个**命名字段**的“结构体（struct）”。每个字段（列）都可以有自己独立的数据类型。

正是因为“命名字段”的存在，我们才可以用类似字典的方式 `['列名']` 来操作它，这是普通二维数组做不到的。

#### 2.2 结构解析：它包含什么？

`search.data` 的具体结构由您的`GridSearch`配置动态决定。

*   **行数 (结构体总数)**: 等于您定义的**格点空间中所有点的总和**。如果`F0s`有100个点，`F1s`有50个点，那么总行数就是 `100 * 50 = 5000`。

*   **列名 (字段名称)**: 由以下几部分构成：
    1.  **您搜索的参数**: 您在`GridSearch`中设定了范围的每一个参数（如`F0s`, `F1s`等），都会成为一个字段（列）。
    2.  **核心计算结果**: **永远会有一个**名为 `'twoF'` 的字段，代表该点的F统计量值。
    3.  **可选计算结果**: 
        *   若 `singleFstats=True`，则会额外包含每个探测器的字段，如 `'twoFH1'`, `'twoFL1'`。
        *   若 `BSGL=True`，则会额外包含 `'log10BSGL'` 字段。

*   **如何精确查看结构**: 
    ```python
    # 假设 search 对象已运行完毕
    # 1. 查看总共有多少行
    print(f"Shape of search.data: {search.data.shape}")
    # 2. 查看所有列的名称和数据类型
    print(f"Dtype of search.data: {search.data.dtype}")
    ```

#### 2.3 操作指南：如何使用 `search.data`

以下是四种最核心的操作方法：

1.  **访问整列数据**: 使用 `['列名']`，返回一个标准的一维NumPy数组。
    ```python
    all_twoFs = search.data['twoF']
    ```

2.  **访问单行数据**: 使用 `[行号]`，返回一个代表该行的结构体对象。
    ```python
    first_row = search.data[0]
    # 可以继续通过列名访问行内元素
    f0_in_first_row = first_row['F0']
    ```

3.  **访问单个元素**: 组合使用 `['列名'][行号]`。
    ```python
    twoF_of_10th_point = search.data['twoF'][9]
    ```

4.  **筛选与查找 (布尔掩码)**: 这是最强大的功能。
    ```python
    # 示例A：找到2F值最大的一行
    loudest_row = search.data[np.argmax(search.data['twoF'])]
    print(f"Loudest point: {loudest_row}")

    # 示例B：找到所有2F值大于50的点
    significant_points = search.data[search.data['twoF'] > 50]
    print(f"Found {len(significant_points)} significant points.")
    ```

#### 2.4 最佳实践：转换为Pandas DataFrame

对于复杂的探索性数据分析，强烈推荐将`search.data`转换为功能更丰富的Pandas DataFrame。

```python
import pandas as pd
df = pd.DataFrame(search.data)
# 现在，您可以使用df.sort_values(), df.query(), df.groupby()等所有Pandas的强大功能
```

---

### 3. 磁盘上的 `.txt` 文件：可移植的纯文本格式

#### 3.1 文件结构 (File Structure)

`search.run()` 自动保存的`.txt`文件是人类可读的，其内部结构清晰，包含三部分：

1.  **注释头 (Metadata Header)**:
    *   文件的前几十行是以注释符（`%`）开头的元数据。
    *   这里面包含了这次搜索的所有关键信息：运行日期、用户名、PyFstat和LALSuite的版本、以及您在创建`GridSearch`对象时传入的完整参数字典。
    *   这保证了结果的**可复现性**。

2.  **列名行 (Column-Name Header)**:
    *   紧跟在元数据注释头后面的，是另一行以`%`开头的注释，其中用空格分隔了所有数据列的名称。

3.  **数据体 (Data Body)**:
    *   文件的剩余部分就是纯粹的数值数据，不带注释。
    *   每一行代表格点搜索中的一个点。
    *   每一列对应一个参数或计算结果，并用空格隔开。

##### 文件内容示例 (`..._GridSearch.txt`)
```text
% date: 2023-10-27 10:30:00.12345
% user: your_username
% hostname: your_computer
% PyFstat: 1.8.0
% lalapps_ComputeFstatistic_v2: ...
% parameters: 
% { 'F0s': [99.9999, 100.0001, 2e-06],
%   'F1s': [-1.1e-10, -0.9e-10, 1e-12],
%   'label': 'MyFirstGridSearch',
%   ...
% }
% F0 F1 twoF
9.999990e+01 -1.100000e-10 3.45123
9.999992e+01 -1.100000e-10 4.12345
9.999994e+01 -1.100000e-10 5.78901
...
```

#### 3.2 文件命名规则

文件会自动保存在您指定的`outdir`中，其命名模式为：
`{outdir}/{label}_{detectors}_{ClassName}.txt`

*   **示例**: `PyFstat_GridSearch_Example/MyFirstGridSearch_H1_GridSearch.txt`

#### 3.3 读取与操作

您可以使用以下方法从文件中加载数据，进行离线分析。

*   **使用Pandas (强烈推荐)**:
    ```python
    import pandas as pd
    filepath = 'PyFstat_GridSearch_Example/MyFirstGridSearch_H1_GridSearch.txt'
    df = pd.read_csv(filepath, delim_whitespace=True, comment='%')
    print(df.head())
    ```

*   **使用NumPy**:
    ```python
    import numpy as np
    filepath = 'PyFstat_GridSearch_Example/MyFirstGridSearch_H1_GridSearch.txt'
    data = np.genfromtxt(filepath, names=True, comments='%')
    # 这会返回一个与search.data类型完全相同的NumPy结构化数组
    ```
