# F统计量热图功能实现总结

## 调研发现

### 1. GridSearch的多维处理机制
- **关键发现**：当搜索超过2个参数时，`plot_2D`方法使用`flatten_method`参数来处理非绘图维度
- **默认行为**：`flatten_method=np.max`，即对其他维度取**最大值**，而非边缘化
- **代码位置**：`pyfstat/grid_based_searches.py`的`plot_2D`方法（第595-761行）

### 2. 参数范围限制
- **Delta参数**：必须在`[-π/2, π/2]`范围内（赤纬的物理限制）
- **Alpha参数**：通常在`[0, 2π]`范围内（赤经）
- **频率参数**：使用metric mismatch计算最优步长

### 3. PyFstat的设计模式
- 使用列表格式定义搜索范围：`[start, end, step]`
- 固定参数使用单元素列表：`[value]`
- GridSearch会自动生成所有参数组合的笛卡尔积

## 实现细节

### 脚本功能
创建了`scripts/绘制F统计量热图.py`，支持：

1. **命令行参数**
   - `--xkey`：指定x轴参数（默认F0）
   - `--ykey`：指定y轴参数（默认F1）  
   - `--search_params`：指定搜索哪些参数（逗号分隔）

2. **动态搜索范围**
   - 绘图维度：50×50网格点
   - 非绘图维度：10×10网格点（用于展示flatten效果）
   - 未搜索参数：固定在注入值

3. **使用示例**
```bash
# 默认F0-F1热图
python scripts/绘制F统计量热图.py

# Alpha-Delta天空位置热图
python scripts/绘制F统计量热图.py --xkey Alpha --ykey Delta --search_params Alpha,Delta

# 4维搜索，绘制F0-F1，其他维度取最大值
python scripts/绘制F统计量热图.py --xkey F0 --ykey F1 --search_params F0,F1,Alpha,Delta
```

## 关键代码片段

### 搜索范围设置逻辑
```python
# 构建搜索范围字典
if 'F0' in search_params:
    if args.xkey == 'F0' or args.ykey == 'F0':
        N = 50  # 绘图维度用更多点
        search_ranges['F0s'] = [F0_min, F0_max, dF0]
    else:
        N = 10  # 非绘图维度用较少点
        search_ranges['F0s'] = [F0_min, F0_max, dF0]
else:
    search_ranges['F0s'] = [inj_params["F0"]]  # 固定值
```

### Delta范围限制
```python
# 确保不超出[-pi/2, pi/2]范围
min_delta = max(inj_params["Delta"] - DeltaDelta/2, -np.pi/2)
max_delta = min(inj_params["Delta"] + DeltaDelta/2, np.pi/2)
```

## 测试结果

1. **F0-F1热图**
   - 输出目录：`F_stat_heatmap_F0_F1/`
   - 最大2F值：156.06
   - 搜索点数：2500（50×50）

2. **Alpha-Delta热图**
   - 输出目录：`F_stat_heatmap_Alpha_Delta/`
   - 最大2F值：104.57
   - 搜索点数：1683（因Delta范围受限）

## 设计原则遵循

1. **无防御性编程**：代码直接执行，无try-except
2. **最小化实现**：复用GridSearch现有功能，不重新实现
3. **硬编码默认值**：减少配置复杂度
4. **无智能fallback**：参数错误直接报错

## 注意事项

- 多维搜索计算量大，建议控制网格点数
- Delta参数必须在有效范围内
- 非绘图维度的处理是取最大值，不是真正的边缘化