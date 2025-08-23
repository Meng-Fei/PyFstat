"""
FStatFunction - 纯函数封装
将PyFstat的F统计量计算封装为纯函数：f(F0, F1, Alpha, Delta) -> 2F
"""
import os
import shutil
import tempfile
import logging
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
        # 使用适中的Band以支持合理的频率范围
        writer = pyfstat.Writer(
            label="fstatfunc",
            outdir=self.tempdir,
            tstart=config.tstart,
            duration=config.duration,
            detectors=config.detectors,
            sqrtSX=config.sqrtSX,
            Tsft=config.Tsft,
            Band=5.0,  # 5Hz带宽，覆盖27.5-32.5Hz
            # 注入参数（弱信号，主要是为了生成有效的SFT）
            F0=30.0,
            F1=-1e-10,
            F2=config.F2,
            Alpha=1.0,
            Delta=0.5,
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
        # 使用实际的SFT频率范围，稍微收缩以避免边界问题
        actual_fmax = writer.fmin + writer.Band
        # SFT频率bin的宽度
        df = 1.0 / writer.Tsft
        # 稍微收缩范围以避免边界错误
        self.compute_fstat = pyfstat.ComputeFstat(
            sftfilepattern=writer.sftfilepath,
            tref=config.tref,
            minStartTime=config.tstart,
            maxStartTime=config.tstart + config.duration,
            minCoverFreq=writer.fmin + df,  # 跳过第一个bin
            maxCoverFreq=actual_fmax - 2*df  # 跳过最后两个bin
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
        params = {
            'F0': F0,
            'F1': F1,
            'F2': self.config.F2,  # 使用配置中的F2
            'Alpha': Alpha,
            'Delta': Delta
        }
        
        return self.compute_fstat.get_fullycoherent_twoF(params=params)
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'tempdir') and os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
    
    def __repr__(self):
        """对象表示"""
        return f"FStatFunction({self.config})"