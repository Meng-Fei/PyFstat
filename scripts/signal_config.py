"""
信号配置类 - 统一管理PyFstat参数
基于Linus哲学：简单、直接、无防御编程
"""
from dataclasses import dataclass


@dataclass
class SignalConfig:
    """
    F统计量计算的配置参数
    所有参数都有合理默认值，但可以根据需要覆盖
    """
    # 数据参数
    tstart: int = 1000000000  # GPS开始时间
    duration: int = 86400      # 观测时长（秒），默认1天
    detectors: str = 'H1'      # 探测器
    sqrtSX: float = 1e-23      # 噪声功率谱密度平方根
    Tsft: int = 1800           # SFT时长（秒），标准值30分钟
    
    # 信号强度控制
    depth: float = 30.0        # 信号深度（SNR相关）
    
    # 其他固定参数（F统计量中被解析最大化的参数）
    F2: float = 0.0            # 频率二阶导数，通常为0
    cosi: float = 0.0          # cos(倾角)
    psi: float = 0.0           # 偏振角
    phi: float = 0.0           # 初始相位
    
    @property
    def h0(self) -> float:
        """引力波应变幅度"""
        return self.sqrtSX / self.depth
    
    @property
    def tref(self) -> int:
        """参考时间（观测中点）"""
        return self.tstart + self.duration // 2
    
    def __str__(self) -> str:
        """打印配置摘要"""
        return (
            f"SignalConfig(\n"
            f"  观测: {self.duration/86400:.1f}天 @ {self.detectors}\n"
            f"  噪声: sqrtSX={self.sqrtSX:.1e}\n"
            f"  信号: depth={self.depth:.1f} (h0={self.h0:.2e})\n"
            f"  SFT: {self.Tsft}秒\n"
            f")"
        )