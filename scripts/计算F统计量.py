#!/home/inkecy/miniconda/envs/pyfstat-dev/bin/python
"""计算给定参数的F统计量"""

import sys
import os
import argparse
import logging

# 完全禁用所有日志
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logging.getLogger().addHandler(NullHandler())
logging.getLogger().setLevel(logging.CRITICAL + 1)

import pyfstat

def main():
    parser = argparse.ArgumentParser(description='计算F统计量')
    
    # 必需参数
    parser.add_argument('--F0', type=float, required=True, help='频率 (Hz)')
    parser.add_argument('--Alpha', type=float, required=True, help='赤经 (radians)')
    parser.add_argument('--Delta', type=float, required=True, help='赤纬 (radians)')
    
    # 可选参数
    parser.add_argument('--F1', type=float, default=0, help='频率一阶导数')
    parser.add_argument('--F2', type=float, default=0, help='频率二阶导数')
    parser.add_argument('--tref', type=int, default=1000000000, help='参考时间 (GPS seconds)')
    parser.add_argument('--tstart', type=int, default=1000000000, help='开始时间')
    parser.add_argument('--duration', type=int, default=86400, help='持续时间 (seconds)')
    parser.add_argument('--sftfile', type=str, default=None, help='SFT文件路径模式')
    parser.add_argument('--sqrtSX', type=str, default='1e-23', help='噪声水平')
    parser.add_argument('--detectors', type=str, default='H1', help='探测器')
    parser.add_argument('--gradient', action='store_true', help='计算梯度')
    
    args = parser.parse_args()
    
    # 创建ComputeFstat对象
    if args.sftfile:
        # 使用已有SFT文件
        cf = pyfstat.ComputeFstat(
            tref=args.tref,
            sftfilepattern=args.sftfile,
            minStartTime=args.tstart,
            maxStartTime=args.tstart + args.duration,
            minCoverFreq=args.F0 - 1,
            maxCoverFreq=args.F0 + 1
        )
    else:
        # 生成模拟数据
        cf = pyfstat.ComputeFstat(
            tref=args.tref,
            minStartTime=args.tstart,
            maxStartTime=args.tstart + args.duration,
            Tsft=1800,
            injectSqrtSX=args.sqrtSX,
            detectors=args.detectors,
            minCoverFreq=args.F0 - 1,
            maxCoverFreq=args.F0 + 1
        )
    
    # 计算F统计量
    params = {
        'F0': args.F0,
        'F1': args.F1,
        'F2': args.F2,
        'Alpha': args.Alpha,
        'Delta': args.Delta
    }
    
    twoF = cf.get_fullycoherent_twoF(params=params)
    
    if args.gradient:
        # 计算梯度 - 最简单的有限差分
        import numpy as np
        
        # 步长
        dF0 = np.sqrt(12 * 0.001) / (np.pi * args.duration)
        dAlpha = 1e-5
        dDelta = 1e-5
        
        # F0方向梯度
        params_plus = params.copy()
        params_plus['F0'] = args.F0 + dF0
        twoF_F0_plus = cf.get_fullycoherent_twoF(params=params_plus)
        
        params_minus = params.copy()
        params_minus['F0'] = args.F0 - dF0
        twoF_F0_minus = cf.get_fullycoherent_twoF(params=params_minus)
        
        grad_F0 = (twoF_F0_plus - twoF_F0_minus) / (2 * dF0)
        
        # Alpha方向梯度
        params_plus = params.copy()
        params_plus['Alpha'] = args.Alpha + dAlpha
        twoF_Alpha_plus = cf.get_fullycoherent_twoF(params=params_plus)
        
        params_minus = params.copy()
        params_minus['Alpha'] = args.Alpha - dAlpha
        twoF_Alpha_minus = cf.get_fullycoherent_twoF(params=params_minus)
        
        grad_Alpha = (twoF_Alpha_plus - twoF_Alpha_minus) / (2 * dAlpha)
        
        # Delta方向梯度
        params_plus = params.copy()
        params_plus['Delta'] = args.Delta + dDelta
        twoF_Delta_plus = cf.get_fullycoherent_twoF(params=params_plus)
        
        params_minus = params.copy()
        params_minus['Delta'] = args.Delta - dDelta
        twoF_Delta_minus = cf.get_fullycoherent_twoF(params=params_minus)
        
        grad_Delta = (twoF_Delta_plus - twoF_Delta_minus) / (2 * dDelta)
        
        # 输出F值和梯度
        print(f"{twoF}")
        print(f"梯度_F0: {grad_F0}")
        print(f"梯度_Alpha: {grad_Alpha}")
        print(f"梯度_Delta: {grad_Delta}")
    else:
        # 输出结果
        print(twoF)

if __name__ == '__main__':
    main()