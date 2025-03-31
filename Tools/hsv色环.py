import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle

def plot_hsv_wheel_with_marker(hsv_input=(0, 1, 1), marker_radius=0.95):
    """带颜色标记的HSV色环生成器
    hsv_input: 元组格式 (H, S, V)
    H: 0-360
    S: 0-100 或 0-1
    V: 0-100 或 0-1
    marker_radius: 标记点半径位置（0-1之间）"""
    
    # 输入参数解析
    h, s, v = hsv_input
    
    # 自动检测输入范围并归一化
    h_norm = h % 360  # 处理超过360度的角度
    s_norm = s/100 if s > 1 else s
    v_norm = v/100 if v > 1 else v
    
    # 生成色环
    size = 500
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    theta_positive = np.where(theta < 0, theta + 2*np.pi, theta)
    
    # 生成基础色环（S=1, V=1）
    hue = theta_positive/(2*np.pi)
    hsv = np.dstack((hue, np.ones_like(hue), np.ones_like(hue)))
    rgb = hsv_to_rgb(hsv)
    rgb[r > 1] = 1
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin='lower', extent=(-1, 1, -1, 1))
    
    # 添加角度标注系统
    text_radius = 1.12
    for deg in np.arange(0, 360, 30):
        rad = np.deg2rad(deg)
        x = text_radius * np.cos(rad)
        y = text_radius * np.sin(rad)
        
        # 智能对齐
        if 45 < deg < 135:    ha, va = 'center', 'bottom'
        elif 225 < deg < 315: ha, va = 'center', 'top'
        elif deg <=45 or deg>=315: ha, va = 'left', 'center'
        else: ha, va = 'right', 'center'
        
        ax.text(x, y, f"{deg}°", ha=ha, va=va, fontsize=10, color='black')

    # 绘制输入颜色标记
    marker_color = hsv_to_rgb([[h_norm/360, s_norm, v_norm]]).reshape(1, -1)
    marker_angle = np.deg2rad(h_norm)
    
    # 计算标记位置
    mx = marker_radius * np.cos(marker_angle)
    my = marker_radius * np.sin(marker_angle)
    
    # 创建标记图形
    marker = Circle((mx, my), radius=0.03, 
                   facecolor=marker_color, 
                   edgecolor='black',
                   linewidth=1.5,
                   zorder=3)
    ax.add_patch(marker)
    
    # 添加说明文字
    info_text = f"HSV: ({h},{s},{v})\n→ ({h_norm:.1f}°, {s_norm:.2f}, {v_norm:.2f})"
    ax.text(0, -1.2, info_text, 
           ha='center', va='top',
           fontsize=12, 
           bbox=dict(facecolor='white', alpha=0.8))

    ax.axis('off')
    plt.tight_layout()
    plt.show()

# 使用示例（修改此处参数测试不同颜色）
hsv_color = (64,1,1)  # 标准绿色


plot_hsv_wheel_with_marker(hsv_color)