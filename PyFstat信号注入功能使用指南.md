### PyFstat 信号注入功能使用指南

本文档旨在为您详细介绍PyFstat中注入单个及多个连续波信号的原理和具体操作方法，作为您日后进行相关研究的编程备忘。

---

### 1. 概述：为何要注入信号？

在引力波数据分析中，信号注入是一项至关重要的技术，其主要目的有两个：

1.  **算法验证 (Algorithm Verification)**: 在一个已知信号存在的数据中运行您的搜索流程，以验证整套算法能够正确地找到信号。这是确保代码和逻辑正确性的基础。
2.  **灵敏度评估 (Sensitivity Assessment)**: 通过注入一系列不同强度 `h0` 的信号，可以系统性地评估您的搜索流程在特定噪声背景下，能够探测到的最微弱信号有多弱。这常用于计算探测深度的“上限（Upper Limits）”。

PyFstat提供了两种注入信号的方式，分别对应单信号和多信号场景。

---

### 2. 注入单个信号：Python字典法

这是最常用、最直接的方法，非常适合进行快速的算法测试和单一目标的注入研究。

*   **原理**: `pyfstat.Writer` 的初始化方法可以直接接收以关键字参数（`**kwargs`）形式传入的信号参数。PyFstat在内部会将这些参数收集起来，填充到一个LALSuite的 `PulsarParams` 结构体中，代表一个独立的信号源。

*   **操作方法**: 
    1.  创建一个标准的Python字典，其中包含您想注入信号的所有物理参数。
    2.  在实例化 `pyfstat.Writer` 时，使用 `**` 语法将该字典解包传入。

*   **代码示例**:
    ```python
    import pyfstat

    # 1. 定义单个信号的参数字典
    single_injection_params = {
        'F0': 100.0,
        'F1': -1e-10,
        'F2': 0,
        'Alpha': 1.5,
        'Delta': 0.5,
        'h0': 5e-24,
        'cosi': 0.5,
        'tref': 1000000000,
    }

    # 2. 在创建Writer时，使用 ** 解包字典
    writer = pyfstat.Writer(
        label="single_injection_test",
        outdir="data/",
        tstart=1000000000,
        duration=100*86400,
        sqrtSX='1e-23',
        detectors='H1,L1',
        **single_injection_params
    )
    
    # 3. 生成包含这个单信号的SFT数据
    writer.make_data()
    ```

---

### 3. 注入多个信号：参数文件法

当您需要模拟一个更复杂的场景，比如数据中包含多个信号源时，需要采用基于文件的方法。

*   **原理**: 当传递给 `pyfstat.Writer` 的 `injectSources` 参数是一个**字符串（文件路径）**而非字典时，PyFstat会启用一种不同的内部机制。它会调用底层的 `lalpulsar.PulsarParamsFromFile` 函数，这个函数专门用于解析特定格式的文本文件，并从中构建一个包含**多个** `PulsarParams` 结构体的向量（`PulsarParamsVector`）。最终，这个向量中的所有信号都会被同时注入到数据中。

*   **操作方法**: 这是一个两步过程。

    **步骤 3.1: 创建参数文件 (`.cff` 或 `.par`)**

    您需要首先创建一个纯文本文件，按照“键 = 值”的格式定义每个信号的参数。不同的信号源之间用一个唯一的段落标记（如 `[Pulsar_2]`）来分隔。

    **文件内容示例: `my_multiple_signals.cff`**
    ```text
    # --- 第一个信号：一个较强的信号 ---
    # 频率和自旋下降
    Freq = 100.0
    f1dot = -1.0e-10
    
    # 天空位置 (rad)
    Alpha = 1.5
    Delta = 0.5
    
    # 振幅参数
    h0 = 8.0e-24
    cosi = 0.5
    
    # 所有信号共用的参考时间
    refTime = 1000000000
    
    # 使用一个唯一的段落标记来分隔不同的信号源
    # 这个标记可以是任意的，只要它不和参数名冲突即可
    [Pulsar_2]
    
    # --- 第二个信号：一个位于不同位置、较弱的信号 ---
    Freq = 150.25
    f1dot = -2.5e-11
    Alpha = 3.1
    Delta = -0.2
    h0 = 2.0e-24
    cosi = 0.1
    refTime = 1000000000
    ```

    **步骤 3.2: 在PyFstat中调用该文件**

    在Python脚本中，将上面创建的**文件名**作为字符串，赋值给`injectSources`参数。

    ```python
    # 1. 定义参数文件的路径
    injection_filepath = 'my_multiple_signals.cff'

    # 2. 将文件名字符串传递给 injectSources 参数
    multi_writer = pyfstat.Writer(
        label="multi_injection_test",
        outdir="data/",
        tstart=1000000000,
        duration=100*86400,
        sqrtSX='1e-23',
        detectors='H1,L1',
        injectSources=injection_filepath
    )

    # 3. 生成包含多个信号的SFT数据
    multi_writer.make_data()
    print(f"包含多个信号的数据已在'{multi_writer.outdir}'中生成完毕。")
    ```

*   **如何验证注入是否成功 (高级技巧)**

    生成数据后，您可以分别在每个注入信号的精确参数点上，调用一次核心计算函数，检查`2F`值是否都显著高于噪声水平（通常远大于10）。

    ```python
    # 使用我们之前备忘录中的 get_2F_value 函数
    # ... (此处省略 get_2F_value 函数的定义) ...

    # 验证第一个信号
    params_sig1 = {'F0': 100.0, 'F1': -1.0e-10, 'Alpha': 1.5, 'Delta': 0.5, 'F2': 0}
    # twoF1 = get_2F_value(params_sig1, sftfilepattern=multi_writer.sftfilepath, ...)
    # print(f"在信号1位置的2F值为: {twoF1}")

    # 验证第二个信号
    params_sig2 = {'F0': 150.25, 'F1': -2.5e-11, 'Alpha': 3.1, 'Delta': -0.2, 'F2': 0}
    # twoF2 = get_2F_value(params_sig2, sftfilepattern=multi_writer.sftfilepath, ...)
    # print(f"在信号2位置的2F值为: {twoF2}")
    ```
