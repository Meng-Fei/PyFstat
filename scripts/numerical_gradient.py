"""
数值梯度计算 - 中心差分法
可以对任何4参数函数计算梯度，支持并行加速
"""
import numpy as np
from multiprocessing import Pool
from typing import List, Dict, Callable, Optional


def compute_gradient(
    f: Callable[[float, float, float, float], float],
    F0: float,
    F1: float,
    Alpha: float,
    Delta: float,
    free_vars: List[str],
    deltas: Optional[Dict[str, float]] = None,
    parallel: bool = True
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
    parallel : bool
        是否使用并行计算（默认True）
        
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
    
    # 计算函数值
    if parallel and len(eval_tasks) > 1:
        # 并行计算
        all_points = []
        for task in eval_tasks:
            all_points.append(task['plus'])
            all_points.append(task['minus'])
        
        with Pool() as pool:
            results = pool.starmap(f, all_points)
    else:
        # 串行计算
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


def compute_hessian(
    f: Callable[[float, float, float, float], float],
    F0: float,
    F1: float,
    Alpha: float,
    Delta: float,
    free_vars: List[str],
    deltas: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, float]]:
    """
    计算数值Hessian矩阵（二阶导数）
    
    使用五点公式提高精度
    
    Parameters
    ----------
    参数同compute_gradient
    
    Returns
    -------
    dict of dict
        Hessian矩阵，hessian[var1][var2] = ∂²f/∂var1∂var2
    """
    
    if deltas is None:
        deltas = {
            'F0': 1e-8,
            'F1': 1e-14,
            'Alpha': 1e-5,
            'Delta': 1e-5
        }
    
    param_names = ['F0', 'F1', 'Alpha', 'Delta']
    base_values = [F0, F1, Alpha, Delta]
    
    hessian = {}
    
    for i, var_i in enumerate(free_vars):
        hessian[var_i] = {}
        idx_i = param_names.index(var_i)
        delta_i = deltas.get(var_i, 1e-6)
        
        for j, var_j in enumerate(free_vars):
            if j < i:
                # 利用对称性
                hessian[var_i][var_j] = hessian[var_j][var_i]
                continue
                
            idx_j = param_names.index(var_j)
            delta_j = deltas.get(var_j, 1e-6)
            
            if i == j:
                # 对角元素：∂²f/∂x²
                values = base_values.copy()
                f_center = f(*values)
                
                values[idx_i] = base_values[idx_i] + delta_i
                f_plus = f(*values)
                
                values[idx_i] = base_values[idx_i] - delta_i
                f_minus = f(*values)
                
                hessian[var_i][var_j] = (f_plus - 2*f_center + f_minus) / delta_i**2
            else:
                # 非对角元素：∂²f/∂x∂y
                values_pp = base_values.copy()
                values_pp[idx_i] += delta_i
                values_pp[idx_j] += delta_j
                
                values_pm = base_values.copy()
                values_pm[idx_i] += delta_i
                values_pm[idx_j] -= delta_j
                
                values_mp = base_values.copy()
                values_mp[idx_i] -= delta_i
                values_mp[idx_j] += delta_j
                
                values_mm = base_values.copy()
                values_mm[idx_i] -= delta_i
                values_mm[idx_j] -= delta_j
                
                f_pp = f(*values_pp)
                f_pm = f(*values_pm)
                f_mp = f(*values_mp)
                f_mm = f(*values_mm)
                
                hessian[var_i][var_j] = (f_pp - f_pm - f_mp + f_mm) / (4 * delta_i * delta_j)
    
    return hessian