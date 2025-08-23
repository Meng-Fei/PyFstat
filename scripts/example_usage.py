#!/usr/bin/env python
"""
FStatFunction使用示例
展示如何使用纯函数接口进行各种计算
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from signal_config import SignalConfig
from fstat_function import FStatFunction
from numerical_gradient import compute_gradient

# 关闭日志输出
import logging
logging.disable(logging.CRITICAL)

print("FStatFunction 使用示例")
print("=" * 50)

# 1. 初始化
print("\n1. 初始化配置")
config = SignalConfig(
    duration=10*86400,  # 10天
    depth=25,           # 信号强度
    sqrtSX=1e-23       # 噪声水平
)
print(config)

print("\n2. 创建函数对象")
f = FStatFunction(config)
print(f"SFT频率范围: {f.sft_fmin:.2f} - {f.sft_fmin + f.sft_band:.2f} Hz")

# 2. 单点计算
print("\n3. 单点计算")
twoF = f(30.0, -1e-10, 1.0, 0.5)
print(f"2F(30.0, -1e-10, 1.0, 0.5) = {twoF:.2f}")

# 3. 1D扫描（使用partial固定3个变量）
print("\n4. 1D扫描示例")
f_1d = partial(f, F1=-1e-10, Alpha=1.0, Delta=0.5)
f0_range = np.linspace(29.98, 30.02, 5)
print("F0        2F")
print("-" * 20)
for f0 in f0_range:
    try:
        val = f_1d(f0)
        print(f"{f0:.4f}  {val:8.2f}")
    except:
        print(f"{f0:.4f}  无法计算")

# 4. 2D网格（使用lambda）
print("\n5. 2D网格示例（3x3）")
f_2d = lambda x, y: f(x, -1e-10, y, 0.5)
print("      Alpha=0.98  1.00  1.02")
print("F0    " + "-" * 25)
for f0 in [29.99, 30.00, 30.01]:
    values = []
    for alpha in [0.98, 1.00, 1.02]:
        try:
            val = f_2d(f0, alpha)
            values.append(f"{val:6.1f}")
        except:
            values.append("   N/A")
    print(f"{f0:.2f}  " + "  ".join(values))

# 5. 梯度计算（串行，避免pickle问题）
print("\n6. 数值梯度计算")
grad = compute_gradient(
    f, 30.0, -1e-10, 1.0, 0.5,
    free_vars=['F0', 'Alpha'],
    parallel=False  # 避免pickle问题
)
for var, value in grad.items():
    print(f"∂2F/∂{var} = {value:.2e}")

# 6. 优化示例
print("\n7. 使用scipy优化")
try:
    from scipy.optimize import minimize
    
    # 定义目标函数（最大化2F = 最小化-2F）
    def objective(x):
        try:
            return -f(x[0], -1e-10, x[1], 0.5)
        except:
            return 1e10  # 超出范围时返回大值
    
    # 初始猜测
    x0 = [30.0, 1.0]
    
    # 运行优化
    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': 50})
    
    if result.success:
        print(f"优化成功:")
        print(f"  最优参数: F0={result.x[0]:.6f}, Alpha={result.x[1]:.6f}")
        print(f"  最大2F = {-result.fun:.2f}")
    else:
        print("优化未收敛")
except ImportError:
    print("scipy未安装，跳过优化示例")

# 7. 参数空间可视化（如果matplotlib可用）
print("\n8. 可视化（需要matplotlib）")
try:
    # 小范围2D扫描用于可视化
    f0_grid = np.linspace(29.99, 30.01, 10)
    alpha_grid = np.linspace(0.99, 1.01, 10)
    F0, Alpha = np.meshgrid(f0_grid, alpha_grid)
    
    Z = np.zeros_like(F0)
    for i in range(len(f0_grid)):
        for j in range(len(alpha_grid)):
            try:
                Z[j, i] = f(F0[j, i], -1e-10, Alpha[j, i], 0.5)
            except:
                Z[j, i] = np.nan
    
    # 绘制等高线图
    plt.figure(figsize=(8, 6))
    contour = plt.contour(F0, Alpha, Z, levels=10)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('F0 [Hz]')
    plt.ylabel('Alpha [rad]')
    plt.title('2F统计量等高线图')
    plt.colorbar(contour, label='2F')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig('fstat_contour.png', dpi=100)
    print("等高线图已保存为 fstat_contour.png")
    plt.close()
    
except Exception as e:
    print(f"可视化失败: {e}")

print("\n" + "=" * 50)
print("示例完成！")
print("\n关键要点:")
print("1. FStatFunction是纯函数：f(F0, F1, Alpha, Delta) -> 2F")
print("2. 必须提供全部4个参数")
print("3. 使用partial/lambda固定变量实现降维")
print("4. 兼容标准Python工具（scipy, matplotlib等）")
print("5. 梯度计算使用parallel=False避免序列化问题")