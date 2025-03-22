import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

class HSVColorPicker:
    def __init__(self):
        # 初始化图形界面
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._init_color_wheel()
        self._init_pointer()
        self._init_display()
        
        # 交互状态跟踪
        self.dragging = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def _init_color_wheel(self):
        """创建HSV色环"""
        size = 500
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # 计算极坐标
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        theta_positive = np.where(theta < 0, theta + 2*np.pi, theta)
        hue = theta_positive / (2*np.pi)
        
        # 生成RGB图像
        hsv = np.dstack((hue, np.ones_like(hue, dtype=float), np.ones_like(hue, dtype=float)))
        rgb = hsv_to_rgb(hsv)
        rgb[r > 1] = 1  # 白色背景
        
        # 显示色环
        self.ax.imshow(rgb, origin='lower', extent=(-1, 1, -1, 1))
        self.ax.axis('off')

    def _init_pointer(self):
        """初始化颜色指针"""
        self.pointer = self.ax.plot(
            [0], [0], 
            marker='o', 
            markersize=12,
            markeredgecolor='black',
            markerfacecolor='white',
            linestyle='',
            visible=False
        )[0]
        
        # 添加方向箭头
        self.arrow = self.ax.arrow(
            0, 0, 0, 0, 
            head_width=0.05, 
            head_length=0.1, 
            fc='black', 
            ec='black',
            visible=False
        )

    def _init_display(self):
        """初始化信息显示"""
        self.info_text = self.ax.text(
            0.05, 0.95,
            "HSV: (0°, 1.00, 1.00)",
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.9)
        )
        
    def update_display(self, x, y, update_color_code=False):
        """更新指针位置和显示信息"""
        # 计算极坐标
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        h_deg = np.degrees(theta) % 360
        
        # 更新指针位置
        self.pointer.set_data([x], [y])
        self.pointer.set_visible(True)
        
        # 更新箭头方向
        self.arrow.set_visible(True)
        arrow_length = 0.7 * r if r > 0.2 else 0.5
        self.arrow.set_data(
            x=0, y=0, 
            dx=x*arrow_length, 
            dy=y*arrow_length
        )
        
        # 更新颜色显示
        current_color = hsv_to_rgb([[h_deg/360, 1.0, 1.0]])[0]  # 确保形状为 (3,)
        self.pointer.set_markerfacecolor(current_color)
        
        # 只有在按下鼠标时才更新颜色代码
        if update_color_code:
            # 确保 current_color * 255 是一个形状为 (3,) 的数组
            rgb_values = np.round(current_color * 255).astype(int)
            self.info_text.set_text(
                f"H: {h_deg:.1f}°\nS: 1.00\nV: 1.00\n"
                f"RGB: {tuple(rgb_values)}"
            )
            self.fig.canvas.draw_idle()  # 强制刷新画布

    def on_press(self, event):
        """鼠标按下事件处理"""
        if event.inaxes != self.ax:
            return
        self.dragging = True
        self.update_pointer_position(event.xdata, event.ydata, update_color_code=True)

    def on_motion(self, event):
        """鼠标移动事件处理"""
        if self.dragging and event.inaxes == self.ax:
            self.update_pointer_position(event.xdata, event.ydata, update_color_code=True)

    def on_release(self, event):
        """鼠标释放事件处理"""
        self.dragging = False

    def update_pointer_position(self, x, y, update_color_code=False):
        """更新指针到指定坐标"""
        # 限制在色环范围内
        r = np.sqrt(x**2 + y**2)
        if r > 1:
            x = x/r
            y = y/r
        
        self.update_display(x, y, update_color_code)

# 运行颜色选择器
picker = HSVColorPicker()
plt.show()