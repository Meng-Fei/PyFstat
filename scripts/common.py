"""PyFstat脚本共享配置和工具函数"""

import logging
import numpy as np

# 默认参数
DEFAULT_SQRTSX = 1e-23
DEFAULT_TSTART = 1000000000
DEFAULT_DURATION = 86400
DEFAULT_DETECTORS = 'H1'
DEFAULT_TSFT = 1800

# 注入信号默认参数
DEFAULT_INJECTION = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 1.0, 
    "Delta": 1.5,
    "cosi": 0.0
}

# 参数范围限制
DELTA_MIN = -np.pi/2
DELTA_MAX = np.pi/2

def disable_logging():
    """完全禁用日志输出"""
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    
    logging.getLogger().addHandler(NullHandler())
    logging.getLogger().setLevel(logging.CRITICAL + 1)

def calculate_tref(tstart, duration):
    """计算参考时间"""
    return tstart + duration / 2

def calculate_metric_mismatch_step(duration, m=0.01):
    """基于metric mismatch计算最优步长"""
    return {
        'F0': np.sqrt(12 * m) / (np.pi * duration),
        'F1': np.sqrt(180 * m) / (np.pi * duration**2),
        'Alpha': 0.01,
        'Delta': 0.01
    }

def clip_to_valid_range(value, min_val, max_val):
    """限制参数在有效范围内"""
    return max(min_val, min(max_val, value))