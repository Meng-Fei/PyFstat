#!/usr/bin/env python
"""
3D F-statistic Surface Plotting Script

Plot F-statistic 3D surface in either:
- F0-F1 parameter space (with fixed sky coordinates)
- Alpha-Delta parameter space (with fixed frequency parameters)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time

# 添加scripts目录到路径
sys.path.insert(0, 'scripts')

from signal_config import SignalConfig
from fstat_function import FStatFunction

# ============= 核心开关 =============
PLOT_SKY_COORDINATES = False  # True: Alpha vs Delta, False: F0 vs F1
# ====================================

def plot_fstat_3d_surface():
    """Plot F-statistic 3D surface"""
    
    print("Initializing F-statistic function...")
    
    # 使用默认配置初始化
    config = SignalConfig()
    f = FStatFunction(config)
    
    print(f"Configuration: {config}")
    print(f"Injected signal parameters: F0={config.injection_F0}, F1={config.injection_F1}")
    print(f"Sky coordinates: Alpha={config.injection_Alpha}, Delta={config.injection_Delta}")
    
    # 参数空间选择逻辑
    if PLOT_SKY_COORDINATES:
        print("\n📍 Mode: Sky Coordinates (Alpha-Delta) Parameter Space")
        
        # 天球坐标模式: 变化Alpha, Delta，固定F0, F1
        alpha_center = config.injection_Alpha
        delta_center = config.injection_Delta
        
        x_min = alpha_center - 1
        x_max = alpha_center + 1
        y_min = delta_center - 1
        y_max = 3.14/2
        x_points = 500
        y_points = 500
        
        x_array = np.linspace(x_min, x_max, x_points)
        y_array = np.linspace(y_min, y_max, y_points)
        
        fixed_f0 = config.injection_F0
        fixed_f1 = config.injection_F1
        
        # 标签
        xlabel = 'Right Ascension Alpha (rad)'
        ylabel = 'Declination Delta (rad)'
        zlabel = 'F-statistic 2F'
        
        print(f"\nParameter ranges:")
        print(f"Alpha: [{x_min:.3f}, {x_max:.3f}] rad ({x_points} points)")
        print(f"Delta: [{y_min:.3f}, {y_max:.3f}] rad ({y_points} points)")
        print(f"Fixed F0: {fixed_f0:.1f} Hz")
        print(f"Fixed F1: {fixed_f1:.2e} Hz/s")
        
    else:
        print("\n📍 Mode: Frequency (F0-F1) Parameter Space")
        
        # 频率模式: 变化F0, F1，固定Alpha, Delta
        x_min, x_max = 29, 31  # F0范围
        x_points = 1000
        
        # F1范围（全负数）
        f1_center = config.injection_F1
        f1_range = 9e-8
        y_min = f1_center - f1_range   # 更负
        y_max = f1_center + f1_range    # 较小负值
        y_points = 1000
        
        x_array = np.linspace(x_min, x_max, x_points)
        y_array = np.linspace(y_min, y_max, y_points)
        
        fixed_alpha = config.injection_Alpha
        fixed_delta = config.injection_Delta
        
        # 标签
        xlabel = 'Frequency F0 (Hz)'
        ylabel = 'Frequency derivative F1 (Hz/s)'
        zlabel = 'F-statistic 2F'
        
        print(f"\nParameter ranges:")
        print(f"F0: [{x_min:.3f}, {x_max:.3f}] Hz ({x_points} points)")
        print(f"F1: [{y_min:.2e}, {y_max:.2e}] Hz/s ({y_points} points)")
        print(f"Fixed Alpha: {fixed_alpha:.3f} rad")
        print(f"Fixed Delta: {fixed_delta:.3f} rad")
    
    # 生成网格
    X_grid, Y_grid = np.meshgrid(x_array, y_array)
    
    # Calculate F-statistic values
    print(f"\nStarting calculation of {x_points}x{y_points}={x_points*y_points} points...")
    
    TwoF_grid = np.zeros_like(X_grid)
    total_points = x_points * y_points
    
    start_time = time.time()
    
    for i in range(y_points):  # Y_grid的行
        for j in range(x_points):  # X_grid的列
            current_point = i * x_points + j + 1
            
            x_val = X_grid[i, j]
            y_val = Y_grid[i, j]
            
            # 计算2F值 - 根据模式调用不同参数顺序
            if PLOT_SKY_COORDINATES:
                # 天球坐标模式
                twoF = f(fixed_f0, fixed_f1, x_val, y_val)
            else:
                # 频率模式
                twoF = f(x_val, y_val, fixed_alpha, fixed_delta)
            
            TwoF_grid[i, j] = twoF
            
            # Show progress
            if current_point % 100 == 0 or current_point == total_points:
                elapsed = time.time() - start_time
                progress = current_point / total_points * 100
                eta = elapsed * (total_points - current_point) / current_point if current_point > 0 else 0
                print(f"Progress: {current_point:4d}/{total_points} ({progress:5.1f}%) - "
                      f"Current 2F={twoF:6.2f} - ETA: {eta:.1f}s")
    
    elapsed_total = time.time() - start_time
    print(f"\nCalculation completed! Total time: {elapsed_total:.1f}s")
    print(f"2F statistics: min={np.min(TwoF_grid):.2f}, max={np.max(TwoF_grid):.2f}, "
          f"mean={np.mean(TwoF_grid):.2f}")
    
    # Create 3D plot
    print("\nGenerating 3D surface plot...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(X_grid, Y_grid, TwoF_grid, 
                          cmap='viridis',
                          vmin=np.min(TwoF_grid),
                          vmax=np.max(TwoF_grid),
                          alpha=0.8,
                          linewidth=0,
                          antialiased=True)
    
    # 标注线自动适配
    if PLOT_SKY_COORDINATES:
        # 天球坐标的注入位置
        inj_x, inj_y = config.injection_Alpha, config.injection_Delta
        label_x = f'Injection α={inj_x:.3f} rad'
        label_y = f'Injection δ={inj_y:.3f} rad'
    else:
        # 频率的注入位置
        inj_x, inj_y = config.injection_F0, config.injection_F1
        label_x = f'Injection F0={inj_x:.1f} Hz'
        label_y = f'Injection F1={inj_y:.1e} Hz/s'
    
    # Add red lines on Z=0 plane to mark injection parameters
    if x_min <= inj_x <= x_max and y_min <= inj_y <= y_max:
        # Create lines at Z=0 plane
        y_line = np.full_like(x_array, inj_y)
        x_line = np.full_like(y_array, inj_x)
        
        # Plot red lines at Z=0 plane
        ax.plot(x_array, y_line, np.zeros_like(x_array), color='red', linewidth=4, 
               label=label_x)
        ax.plot(x_line, y_array, np.zeros_like(y_array), color='red', linewidth=4, 
               linestyle='--', label=label_y)
        ax.legend()
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    
    # 根据模式设置标题
    if PLOT_SKY_COORDINATES:
        title = (f'F-statistic 3D Surface in Sky Coordinates\n'
                f'Fixed parameters: F0={fixed_f0:.1f} Hz, F1={fixed_f1:.1e} Hz/s\n'
                f'Observation: {config.duration/86400:.1f} day, Signal depth: {config.depth:.1f}')
    else:
        title = (f'F-statistic 3D Surface in Frequency Space\n'
                f'Fixed parameters: α={fixed_alpha:.3f} rad, δ={fixed_delta:.3f} rad\n'
                f'Observation: {config.duration/86400:.1f} day, Signal depth: {config.depth:.1f}')
    
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('2F value', fontsize=12)
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Save plot - 根据模式选择文件名
    if PLOT_SKY_COORDINATES:
        output_path = 'temp_results/fstat_3d_surface_sky.png'
    else:
        output_path = 'temp_results/fstat_3d_surface_freq.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n3D surface plot saved to: {output_path}")
    
    # Display plot information
    print(f"Image resolution: 300 DPI")
    print(f"Grid density: {x_points}x{y_points}")
    
    return output_path, X_grid, Y_grid, TwoF_grid, config, xlabel, ylabel


def plot_2d_heatmap(X_grid, Y_grid, TwoF_grid, config, xlabel, ylabel, PLOT_SKY_COORDINATES):
    """Plot 2D heatmap of F-statistic"""
    
    print("\nGenerating 2D heatmap...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    im = ax.contourf(X_grid, Y_grid, TwoF_grid, levels=50, cmap='viridis')
    
    # 添加等高线
    contours = ax.contour(X_grid, Y_grid, TwoF_grid, levels=10, colors='white', 
                          linewidths=0.5, alpha=0.3)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
    
    # 标注注入参数位置
    if PLOT_SKY_COORDINATES:
        inj_x, inj_y = config.injection_Alpha, config.injection_Delta
        title = (f'F-statistic 2D Heatmap in Sky Coordinates\n'
                f'Fixed: F0={config.injection_F0:.1f} Hz, F1={config.injection_F1:.1e} Hz/s')
    else:
        inj_x, inj_y = config.injection_F0, config.injection_F1
        title = (f'F-statistic 2D Heatmap in Frequency Space\n'
                f'Fixed: α={config.injection_Alpha:.3f} rad, δ={config.injection_Delta:.3f} rad')
    
    # 标记注入点
    ax.axvline(x=inj_x, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Injection {xlabel.split()[0]}')
    ax.axhline(y=inj_y, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Injection {ylabel.split()[0]}')
    
    # 在注入点添加十字标记
    ax.plot(inj_x, inj_y, 'r*', markersize=20, markeredgecolor='white', 
            markeredgewidth=2, label='Injection point')
    
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(TwoF_grid), TwoF_grid.shape)
    max_x = X_grid[max_idx]
    max_y = Y_grid[max_idx]
    max_2F = TwoF_grid[max_idx]
    
    # 标记最大值点
    ax.plot(max_x, max_y, 'w^', markersize=15, markeredgecolor='yellow', 
            markeredgewidth=2, label=f'Max 2F={max_2F:.1f}')
    
    # 设置标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('2F value', fontsize=12)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', color='white')
    
    # 保存图片
    if PLOT_SKY_COORDINATES:
        output_path = 'temp_results/fstat_2d_heatmap_sky.png'
    else:
        output_path = 'temp_results/fstat_2d_heatmap_freq.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D heatmap saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("F-statistic 3D Surface and 2D Heatmap Plotting")
    print("=" * 60)
    
    # 显示当前模式
    mode_str = "Sky Coordinates (Alpha-Delta)" if PLOT_SKY_COORDINATES else "Frequency (F0-F1)"
    print(f"Current mode: {mode_str}")
    print("To switch mode, change PLOT_SKY_COORDINATES variable")
    print("=" * 60)
    
    try:
        # 运行主函数，获取3D图路径和计算好的数据
        output_3d, X_grid, Y_grid, TwoF_grid, config, xlabel, ylabel = plot_fstat_3d_surface()
        
        # 使用已计算的数据绘制2D热图
        output_2d = plot_2d_heatmap(X_grid, Y_grid, TwoF_grid, config, 
                                    xlabel, ylabel, PLOT_SKY_COORDINATES)
        
        print("\n✅ All tasks completed!")
        print(f"3D surface: {output_3d}")
        print(f"2D heatmap: {output_2d}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise