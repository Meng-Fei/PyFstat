#!/usr/bin/env python
"""
FStatFunction测试用例
验证纯函数封装的正确性
"""
import sys
import numpy as np
from signal_config import SignalConfig
from fstat_function import FStatFunction
from numerical_gradient import compute_gradient, compute_hessian


def test_basic_functionality():
    """测试1：基本功能"""
    print("测试1：基本功能")
    print("-" * 40)
    
    config = SignalConfig(duration=86400, depth=20)
    f = FStatFunction(config)
    
    # 测试函数可调用
    twoF = f(30.0, -1e-10, 1.0, 0.5)
    assert twoF > 0, f"2F值应该为正，实际得到: {twoF}"
    print(f"✓ 函数调用成功: 2F = {twoF:.2f}")
    
    # 测试不同参数返回不同值
    twoF1 = f(30.0, -1e-10, 1.0, 0.5)
    twoF2 = f(30.1, -1e-10, 1.0, 0.5)
    diff = abs(twoF1 - twoF2)
    assert diff > 0.01, f"不同F0应该给出明显不同的2F，差异: {diff}"
    print(f"✓ 参数敏感性正常: |2F(30.0) - 2F(30.1)| = {diff:.2f}")
    
    # 测试极端参数
    twoF3 = f(100.0, -1e-8, 2.0, 1.0)
    assert twoF3 > 0, "极端参数也应该返回正值"
    print(f"✓ 极端参数测试通过: 2F(100Hz) = {twoF3:.2f}")
    
    print()
    return True


def test_consistency():
    """测试2：一致性"""
    print("测试2：一致性")
    print("-" * 40)
    
    config = SignalConfig(duration=86400, depth=25)
    f = FStatFunction(config)
    
    # 相同参数应该给出相同结果
    params = (30.0, -1e-10, 1.0, 0.5)
    results = [f(*params) for _ in range(5)]
    
    mean_val = np.mean(results)
    std_val = np.std(results)
    
    assert std_val < 1e-10, f"相同参数应该给出相同结果，std = {std_val}"
    print(f"✓ 一致性测试通过:")
    print(f"  5次调用平均值: {mean_val:.6f}")
    print(f"  标准差: {std_val:.2e}")
    
    print()
    return True


def test_gradient():
    """测试3：梯度计算"""
    print("测试3：梯度计算")
    print("-" * 40)
    
    config = SignalConfig(duration=86400, depth=20)
    f = FStatFunction(config)
    
    # 在信号附近计算梯度
    center = (30.0, -1e-10, 1.0, 0.5)
    
    # 测试单变量梯度
    grad_f0 = compute_gradient(f, *center, free_vars=['F0'])
    assert 'F0' in grad_f0
    print(f"✓ 单变量梯度: ∂2F/∂F0 = {grad_f0['F0']:.2e}")
    
    # 测试多变量梯度（并行）
    grad_multi = compute_gradient(f, *center, 
                                 free_vars=['F0', 'Alpha', 'Delta'])
    assert len(grad_multi) == 3
    print(f"✓ 多变量梯度（并行）:")
    for var, val in grad_multi.items():
        print(f"  ∂2F/∂{var} = {val:.2e}")
    
    # 测试串行计算
    grad_serial = compute_gradient(f, *center,
                                  free_vars=['F0', 'Alpha'],
                                  parallel=False)
    print(f"✓ 串行梯度计算成功")
    
    # 验证并行和串行结果一致
    diff_f0 = abs(grad_multi['F0'] - grad_serial['F0'])
    assert diff_f0 < 1e-6, f"并行和串行结果应该一致，差异: {diff_f0}"
    print(f"✓ 并行/串行一致性: 差异 < {1e-6}")
    
    print()
    return True


def test_2d_scan():
    """测试4：2D参数扫描"""
    print("测试4：2D参数扫描")
    print("-" * 40)
    
    config = SignalConfig(duration=86400, depth=15)
    f = FStatFunction(config)
    
    # 小范围2D扫描
    f0_range = np.linspace(29.98, 30.02, 8)
    alpha_range = np.linspace(0.98, 1.02, 8)
    
    print(f"扫描范围: F0∈[{f0_range[0]:.2f}, {f0_range[-1]:.2f}]")
    print(f"         Alpha∈[{alpha_range[0]:.2f}, {alpha_range[-1]:.2f}]")
    
    # 构建2D网格
    Z = np.zeros((len(f0_range), len(alpha_range)))
    for i, f0 in enumerate(f0_range):
        for j, alpha in enumerate(alpha_range):
            Z[i, j] = f(f0, -1e-10, alpha, 0.5)
    
    # 找最大值
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    max_f0 = f0_range[max_idx[0]]
    max_alpha = alpha_range[max_idx[1]]
    max_2F = Z.max()
    min_2F = Z.min()
    
    print(f"✓ 2D扫描完成:")
    print(f"  网格大小: {Z.shape}")
    print(f"  2F范围: [{min_2F:.2f}, {max_2F:.2f}]")
    print(f"  最大值位置: F0={max_f0:.4f}, Alpha={max_alpha:.4f}")
    
    # 验证有变化
    assert max_2F > min_2F + 1, "2D扫描应该显示明显的参数依赖性"
    print(f"✓ 参数依赖性明显: Δ2F = {max_2F - min_2F:.2f}")
    
    print()
    return True


def test_hessian():
    """测试5：Hessian矩阵（二阶导数）"""
    print("测试5：Hessian矩阵")
    print("-" * 40)
    
    config = SignalConfig(duration=86400, depth=20)
    f = FStatFunction(config)
    
    # 计算Hessian
    center = (30.0, -1e-10, 1.0, 0.5)
    hess = compute_hessian(f, *center, free_vars=['F0', 'Alpha'])
    
    # 验证对称性
    sym_error = abs(hess['F0']['Alpha'] - hess['Alpha']['F0'])
    assert sym_error < 1e-6, f"Hessian应该对称，误差: {sym_error}"
    
    print(f"✓ Hessian计算成功:")
    print(f"  ∂²2F/∂F0² = {hess['F0']['F0']:.2e}")
    print(f"  ∂²2F/∂F0∂Alpha = {hess['F0']['Alpha']:.2e}")
    print(f"  ∂²2F/∂Alpha² = {hess['Alpha']['Alpha']:.2e}")
    print(f"  对称性误差: {sym_error:.2e}")
    
    # 对于峰值附近，Hessian对角元应该为负（凸函数）
    if hess['F0']['F0'] < 0:
        print(f"✓ F0方向呈凸形（峰值特征）")
    
    print()
    return True


def test_partial_application():
    """测试6：偏函数应用"""
    print("测试6：偏函数应用")
    print("-" * 40)
    
    from functools import partial
    
    config = SignalConfig(duration=86400)
    f = FStatFunction(config)
    
    # 创建1D函数
    f_1d = partial(f, F1=-1e-10, Alpha=1.0, Delta=0.5)
    result_1d = f_1d(30.0)
    print(f"✓ 1D函数(只有F0): 2F = {result_1d:.2f}")
    
    # 创建2D函数
    f_2d = partial(f, F1=-1e-10, Delta=0.5)
    result_2d = f_2d(30.0, 1.0)  # F0和Alpha
    print(f"✓ 2D函数(F0,Alpha): 2F = {result_2d:.2f}")
    
    # 验证等价性
    full_result = f(30.0, -1e-10, 1.0, 0.5)
    assert abs(result_1d - full_result) < 1e-10
    assert abs(result_2d - full_result) < 1e-10
    print(f"✓ 偏函数与完整调用等价")
    
    # Lambda表达式
    f_lambda = lambda x, y: f(x, -1e-10, y, 0.5)
    result_lambda = f_lambda(30.0, 1.0)
    assert abs(result_lambda - full_result) < 1e-10
    print(f"✓ Lambda表达式正常工作")
    
    print()
    return True


def main():
    """运行所有测试"""
    print("=" * 50)
    print("FStatFunction 测试套件")
    print("=" * 50)
    print()
    
    tests = [
        test_basic_functionality,
        test_consistency,
        test_gradient,
        test_2d_scan,
        test_hessian,
        test_partial_application
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            print()
            failed += 1
    
    print("=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed > 0:
        print("存在失败的测试！")
        sys.exit(1)
    else:
        print("所有测试通过！✓")
        print("=" * 50)


if __name__ == "__main__":
    main()