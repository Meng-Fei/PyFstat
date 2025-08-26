"""
FStatFunction - 纯函数封装
将PyFstat的F统计量计算封装为纯函数：f(F0, F1, Alpha, Delta) -> 2F
"""
import os
import shutil
import tempfile
import logging
import numpy as np
import pyfstat
from signal_config import SignalConfig


# 完全禁用日志
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logging.getLogger().addHandler(NullHandler())
logging.getLogger().setLevel(logging.CRITICAL + 1)


class FStatFunction:
    """
    F统计量纯函数封装
    
    使用方式：
    config = SignalConfig(duration=86400, depth=30)
    f = FStatFunction(config)
    twoF = f(30.0, -1e-10, 1.0, 0.5)
    """
    
    def __init__(self, config: SignalConfig):
        """
        初始化函数
        
        Parameters
        ----------
        config : SignalConfig
            配置参数对象
        """
        self.config = config
        
        # 创建临时目录存放SFT文件
        self.tempdir = tempfile.mkdtemp(prefix="fstat_func_")
        
        # 生成SFT数据（只做一次）
        # Band=5.0Hz，实际频率搜索范围：27.5-32.5Hz
        writer = pyfstat.Writer(
            label="fstatfunc",
            outdir=self.tempdir,
            tstart=config.tstart,
            duration=config.duration,
            detectors=config.detectors,
            sqrtSX=config.sqrtSX,
            Tsft=config.Tsft,
            Band=5.0,  # 5Hz带宽，实际频率范围：27.5-32.5Hz
            # 注入信号参数（从config读取）
            F0=config.injection_F0,
            F1=config.injection_F1,
            F2=config.F2,
            Alpha=config.injection_Alpha,
            Delta=config.injection_Delta,
            h0=config.h0,
            cosi=config.cosi,
            psi=config.psi,
            phi=config.phi,
            tref=config.tref
        )
        writer.make_data()
        
        # 保存SFT信息供后续使用
        self.sft_fmin = writer.fmin
        self.sft_band = writer.Band
        
        # 初始化ComputeFstat对象
        # 不使用search_ranges，让ComputeFstat作为纯计算器
        # minCoverFreq=-0.5让PyFstat自动从SFT文件推断频率覆盖
        self.compute_fstat = pyfstat.ComputeFstat(
            sftfilepattern=writer.sftfilepath,
            tref=config.tref,
            minStartTime=config.tstart,
            maxStartTime=config.tstart + config.duration,
            minCoverFreq=-0.5,  # 自动从SFT文件推断频率覆盖范围
            maxCoverFreq=-0.5,  # -0.5是特殊值，表示自动推断
            # 不设置search_ranges！这是关键
        )
        
    def __call__(self, F0: float, F1: float, Alpha: float, Delta: float) -> float:
        """
        计算F统计量
        
        必须提供全部4个参数，没有默认值
        
        Parameters
        ----------
        F0 : float
            频率 (Hz)
        F1 : float
            频率一阶导数 (Hz/s)
        Alpha : float
            赤经 (radians)
        Delta : float
            赤纬 (radians)
        
        Returns
        -------
        float
            2F统计量值
        """
        # 直接传递参数，不使用params字典
        return self.compute_fstat.get_fullycoherent_twoF(
            F0=F0,
            F1=F1, 
            F2=self.config.F2,
            Alpha=Alpha,
            Delta=Delta
        )
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'tempdir') and os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
    
    def __repr__(self):
        """对象表示"""
        return f"FStatFunction({self.config})"