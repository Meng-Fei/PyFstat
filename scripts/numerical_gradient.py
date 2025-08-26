"""
数值梯度计算 - 中心差分法
对任何4参数函数计算梯度的纯串行实现
"""
import numpy as np
from typing import List, Dict, Callable, Optional


def compute_gradient(
    f: Callable[[float, float, float, float], float],
    F0: float,
    F1: float,
    Alpha: float,
    Delta: float,
    free_vars: List[str],
    deltas: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    计算数值梯度（中心差分法）
    
    Parameters
    ----------
    f : callable
        必须接受(F0, F1, Alpha, Delta)四个参数的函数
    F0, F1, Alpha, Delta : float
        计算梯度的中心点
    free_vars : list of str
        需要计算梯度的变量名列表
    deltas : dict, optional
        各变量的步长，默认使用合理值
        
    Returns
    -------
    dict
        各自由变量的梯度值
        
    Example
    -------
    >>> grad = compute_gradient(f, 30.0, -1e-10, 1.0, 0.5,
    ...                        free_vars=['F0', 'Alpha'])
    >>> print(f"∂2F/∂F0 = {grad['F0']}")
    """
    
    # 默认步长（根据参数的典型尺度设置）
    if deltas is None:
        deltas = {
            'F0': 1e-9,       # Hz
            'F1': 1e-15,      # Hz/s
            'Alpha': 1e-6,    # rad
            'Delta': 1e-6     # rad
        }
    
    # 参数映射
    param_names = ['F0', 'F1', 'Alpha', 'Delta']
    base_values = [F0, F1, Alpha, Delta]
    
    # 构建所有需要评估的点
    eval_tasks = []
    for var in free_vars:
        if var not in param_names:
            raise ValueError(f"未知变量: {var}")
        
        idx = param_names.index(var)
        delta = deltas.get(var, 1e-6)
        
        # 正向扰动
        values_plus = base_values.copy()
        values_plus[idx] += delta
        
        # 负向扰动
        values_minus = base_values.copy()
        values_minus[idx] -= delta
        
        eval_tasks.append({
            'var': var,
            'plus': values_plus,
            'minus': values_minus,
            'delta': delta
        })
    
    # 串行计算所有函数值
    results = []
    for task in eval_tasks:
        results.append(f(*task['plus']))
        results.append(f(*task['minus']))
    
    # 计算梯度
    gradients = {}
    for i, task in enumerate(eval_tasks):
        f_plus = results[2*i]
        f_minus = results[2*i + 1]
        gradients[task['var']] = (f_plus - f_minus) / (2 * task['delta'])
    
    return gradients