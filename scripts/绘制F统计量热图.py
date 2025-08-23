#!/usr/bin/env python
"""绘制F统计量参数空间的热图"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyfstat

# 固定参数
sqrtSX = 1e-23
tstart = 1000000000
duration = 10 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFOs = "H1"

# 注入信号参数
inj_params = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 1.0, 
    "Delta": 1.5,
    "h0": sqrtSX / 20,
    "cosi": 0.0
}

def main():
    parser = argparse.ArgumentParser(description='绘制F统计量热图')
    parser.add_argument('--xkey', type=str, default='F0', help='x轴参数 (F0/F1/Alpha/Delta)')
    parser.add_argument('--ykey', type=str, default='F1', help='y轴参数 (F0/F1/Alpha/Delta)')
    parser.add_argument('--search_params', type=str, default='F0,F1', 
                       help='搜索参数列表，逗号分隔 (如: F0,F1 或 F0,F1,Alpha,Delta)')
    
    args = parser.parse_args()
    search_params = args.search_params.split(',')
    
    # 创建输出目录
    outdir = f"F_stat_heatmap_{args.xkey}_{args.ykey}"
    os.makedirs(outdir, exist_ok=True)
    
    # 生成注入数据
    data = pyfstat.Writer(
        label="heatmap",
        outdir=outdir,
        tstart=tstart,
        duration=duration,
        sqrtSX=sqrtSX,
        detectors=IFOs,
        tref=tref,
        F0=inj_params["F0"],
        F1=inj_params["F1"],
        F2=inj_params["F2"],
        Alpha=inj_params["Alpha"],
        Delta=inj_params["Delta"],
        h0=inj_params["h0"],
        cosi=inj_params["cosi"],
    )
    data.make_data()
    
    # 设置搜索参数范围
    m = 0.01
    dF0 = np.sqrt(12 * m) / (np.pi * duration)
    dF1 = np.sqrt(180 * m) / (np.pi * duration**2)
    dAlpha = 0.01  # 天空分辨率
    dDelta = 0.01
    
    # 构建搜索范围字典
    search_ranges = {}
    
    # F0范围
    if 'F0' in search_params:
        if args.xkey == 'F0' or args.ykey == 'F0':
            N = 50  # 绘图维度用更多点
            DeltaF0 = N * dF0
            search_ranges['F0s'] = [inj_params["F0"] - DeltaF0/2, inj_params["F0"] + DeltaF0/2, dF0]
        else:
            # 非绘图维度用较少点
            N = 10
            DeltaF0 = N * dF0  
            search_ranges['F0s'] = [inj_params["F0"] - DeltaF0/2, inj_params["F0"] + DeltaF0/2, dF0]
    else:
        search_ranges['F0s'] = [inj_params["F0"]]
    
    # F1范围
    if 'F1' in search_params:
        if args.xkey == 'F1' or args.ykey == 'F1':
            N = 50
            DeltaF1 = N * dF1
            search_ranges['F1s'] = [inj_params["F1"] - DeltaF1/2, inj_params["F1"] + DeltaF1/2, dF1]
        else:
            N = 10
            DeltaF1 = N * dF1
            search_ranges['F1s'] = [inj_params["F1"] - DeltaF1/2, inj_params["F1"] + DeltaF1/2, dF1]
    else:
        search_ranges['F1s'] = [inj_params["F1"]]
    
    # Alpha范围
    if 'Alpha' in search_params:
        if args.xkey == 'Alpha' or args.ykey == 'Alpha':
            N = 50
            DeltaAlpha = N * dAlpha
            search_ranges['Alphas'] = [inj_params["Alpha"] - DeltaAlpha/2, inj_params["Alpha"] + DeltaAlpha/2, dAlpha]
        else:
            N = 10
            DeltaAlpha = N * dAlpha
            search_ranges['Alphas'] = [inj_params["Alpha"] - DeltaAlpha/2, inj_params["Alpha"] + DeltaAlpha/2, dAlpha]
    else:
        search_ranges['Alphas'] = [inj_params["Alpha"]]
    
    # Delta范围（限制在[-pi/2, pi/2]内）
    if 'Delta' in search_params:
        if args.xkey == 'Delta' or args.ykey == 'Delta':
            N = 50
            DeltaDelta = N * dDelta
            # 确保不超出[-pi/2, pi/2]范围
            min_delta = max(inj_params["Delta"] - DeltaDelta/2, -np.pi/2)
            max_delta = min(inj_params["Delta"] + DeltaDelta/2, np.pi/2)
            search_ranges['Deltas'] = [min_delta, max_delta, dDelta]
        else:
            N = 10
            DeltaDelta = N * dDelta
            min_delta = max(inj_params["Delta"] - DeltaDelta/2, -np.pi/2)
            max_delta = min(inj_params["Delta"] + DeltaDelta/2, np.pi/2)
            search_ranges['Deltas'] = [min_delta, max_delta, dDelta]
    else:
        search_ranges['Deltas'] = [inj_params["Delta"]]
    
    # F2始终固定
    search_ranges['F2s'] = [inj_params["F2"]]
    
    # 执行网格搜索
    search = pyfstat.GridSearch(
        label=f"heatmap_{args.xkey}_{args.ykey}",
        outdir=outdir,
        sftfilepattern=data.sftfilepath,
        F0s=search_ranges['F0s'],
        F1s=search_ranges['F1s'],
        F2s=search_ranges['F2s'],
        Alphas=search_ranges['Alphas'],
        Deltas=search_ranges['Deltas'],
        tref=tref,
        minStartTime=tstart,
        maxStartTime=tend,
    )
    search.run()
    
    # 设置轴标签
    labels = {
        'F0': '$f - f_0$ [Hz]',
        'F1': '$\\dot{f} - \\dot{f}_0$ [Hz/s]',
        'Alpha': '$\\alpha - \\alpha_0$ [rad]',
        'Delta': '$\\delta - \\delta_0$ [rad]'
    }
    
    # 绘制热图
    search.plot_2D(
        xkey=args.xkey, 
        ykey=args.ykey, 
        colorbar=True,
        x0=inj_params[args.xkey], 
        y0=inj_params[args.ykey],
        xlabel=labels.get(args.xkey, args.xkey),
        ylabel=labels.get(args.ykey, args.ykey)
    )
    
    print(f"热图已保存到 {outdir}/")
    
    # 输出搜索信息
    max_dict = search.get_max_twoF()
    print(f"最大2F值: {max_dict['twoF']:.2f}")
    print(f"搜索参数: {search_params}")
    print(f"绘图维度: {args.xkey} vs {args.ykey}")
    
    if len(search_params) > 2:
        other_params = [p for p in search_params if p != args.xkey and p != args.ykey]
        print(f"其他维度 {other_params} 通过取最大值展平")

if __name__ == '__main__':
    main()