import numpy as np  # 导入模块，供后续使用
import matplotlib  # 导入模块，供后续使用
matplotlib.use('QtAgg')  # 调用函数/方法执行某个动作或计算
from matplotlib.figure import Figure  # 从模块导入符号，供后续调用
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # 从模块导入符号，供后续调用
from matplotlib.patches import Polygon, Rectangle, Circle, Arc  # 从模块导入符号，供后续调用
import matplotlib.pyplot as plt  # 导入模块，供后续使用
import seaborn as sns  # 导入模块，供后续使用
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy  # 从模块导入符号，供后续调用
from PySide6.QtCore import Qt, QPoint  # 从模块导入符号，供后续调用
from PySide6.QtGui import QCursor, QPixmap  # 从模块导入符号，供后续调用
import io  # 导入模块，供后续使用

# Set dark theme for matplotlib
plt.style.use('dark_background')  # 调用函数/方法执行某个动作或计算
# Configure Chinese font support
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']  # 执行当前语句（保持与上文逻辑一致）
plt.rcParams['axes.unicode_minus'] = False  # 执行当前语句（保持与上文逻辑一致）

class MplCanvas(FigureCanvasQTAgg):  # 定义类（封装数据与行为）
    def __init__(self, parent=None, width=5, height=4, dpi=100):  # 定义函数（封装可复用逻辑）
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)  # 给对象属性 self.fig 赋值/初始化（来自当前语句右侧表达式）
        self.fig.patch.set_facecolor('#1e1e1e') # Dark background
        self.axes = self.fig.add_subplot(111)  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
        self.axes.set_facecolor('#1e1e1e')  # 调用函数/方法执行某个动作或计算
        super().__init__(self.fig)  # 调用函数/方法执行某个动作或计算
        self.setParent(parent)  # 调用函数/方法执行某个动作或计算
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 调用函数/方法执行某个动作或计算
        self.updateGeometry()  # 调用函数/方法执行某个动作或计算
        self._hover_preview = None  # 给对象属性 self._hover_preview 赋值/初始化（来自当前语句右侧表达式）
        self.setMouseTracking(True)  # 调用函数/方法执行某个动作或计算

    def _ensure_preview(self):  # 定义函数（封装可复用逻辑）
        if self._hover_preview is None:  # 条件分支判断并选择执行路径
            w = QWidget(None)  # 将 w 设为一次调用/构造的返回值
            w.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 调用函数/方法执行某个动作或计算
            w.setAttribute(Qt.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算
            lay = QVBoxLayout(w)  # 将 lay 设为一次调用/构造的返回值
            lay.setContentsMargins(4, 4, 4, 4)  # 调用函数/方法执行某个动作或计算
            lbl = QLabel(w)  # 将 lbl 设为一次调用/构造的返回值
            lbl.setStyleSheet("background-color: #202020; border: 1px solid #444;")  # 调用函数/方法执行某个动作或计算
            lay.addWidget(lbl)  # 调用函数/方法执行某个动作或计算
            self._hover_preview = w  # 给对象属性 self._hover_preview 赋值/初始化（来自当前语句右侧表达式）
            self._hover_label = lbl  # 给对象属性 self._hover_label 赋值/初始化（来自当前语句右侧表达式）

    def _make_preview_pixmap(self):  # 定义函数（封装可复用逻辑）
        buf = io.BytesIO()  # 将 buf 设为一次调用/构造的返回值
        try:  # 开始异常捕获保护块
            self.fig.canvas.draw()  # 调用函数/方法执行某个动作或计算
            self.fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')  # 调用函数/方法执行某个动作或计算
            pix = QPixmap()  # 将 pix 设为一次调用/构造的返回值
            pix.loadFromData(buf.getvalue())  # 调用函数/方法执行某个动作或计算
            target_w = int(self.width() * 1.6)  # 将 target_w 设为一次调用/构造的返回值
            target_h = int(self.height() * 1.6)  # 将 target_h 设为一次调用/构造的返回值
            return pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 从函数返回结果
        except Exception:  # 捕获异常并进行处理
            return None  # 从函数返回结果

    def enterEvent(self, event):  # 定义函数（封装可复用逻辑）
        self._ensure_preview()  # 调用函数/方法执行某个动作或计算
        pm = self._make_preview_pixmap()  # 将 pm 设为一次调用/构造的返回值
        if pm:  # 条件分支判断并选择执行路径
            self._hover_label.setPixmap(pm)  # 调用函数/方法执行某个动作或计算
            gp = QCursor.pos()  # 将 gp 设为一次调用/构造的返回值
            self._hover_preview.move(gp + QPoint(20, 20))  # 调用函数/方法执行某个动作或计算
            self._hover_preview.show()  # 调用函数/方法执行某个动作或计算
        super().enterEvent(event)  # 调用函数/方法执行某个动作或计算

    def mouseMoveEvent(self, event):  # 定义函数（封装可复用逻辑）
        if self._hover_preview and self._hover_preview.isVisible():  # 条件分支判断并选择执行路径
            gp = QCursor.pos()  # 将 gp 设为一次调用/构造的返回值
            self._hover_preview.move(gp + QPoint(20, 20))  # 调用函数/方法执行某个动作或计算
        super().mouseMoveEvent(event)  # 调用函数/方法执行某个动作或计算

    def leaveEvent(self, event):  # 定义函数（封装可复用逻辑）
        if self._hover_preview:  # 条件分支判断并选择执行路径
            self._hover_preview.hide()  # 调用函数/方法执行某个动作或计算
        super().leaveEvent(event)  # 调用函数/方法执行某个动作或计算

    def cleanup(self):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.fig.clf()  # 调用函数/方法执行某个动作或计算

class RadarChart(MplCanvas):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent, width=5, height=4, dpi=100)  # 调用函数/方法执行某个动作或计算
        self.axes.remove()  # 调用函数/方法执行某个动作或计算
        self.axes = self.fig.add_subplot(111, polar=True)  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
        self.axes.set_facecolor('#1e1e1e')  # 调用函数/方法执行某个动作或计算
        
    def plot(self, p1_stats, p2_stats, categories):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        N = len(categories)  # 将 N 设为一次调用/构造的返回值
        angles = [n / float(N) * 2 * np.pi for n in range(N)]  # 初始化变量 angles 为一个容器/表达式结果
        angles += angles[:1] # Close the loop
        
        # Draw Player 1
        values1 = list(p1_stats.values())  # 将 values1 设为一次调用/构造的返回值
        values1 += values1[:1]  # 执行当前语句（保持与上文逻辑一致）
        self.axes.plot(angles, values1, linewidth=2, linestyle='solid', label='球员1', color='#ff4d4d')  # 调用函数/方法执行某个动作或计算
        self.axes.fill(angles, values1, '#ff4d4d', alpha=0.25)  # 调用函数/方法执行某个动作或计算
        
        # Draw Player 2
        values2 = list(p2_stats.values())  # 将 values2 设为一次调用/构造的返回值
        values2 += values2[:1]  # 执行当前语句（保持与上文逻辑一致）
        self.axes.plot(angles, values2, linewidth=2, linestyle='solid', label='球员2', color='#4d79ff')  # 调用函数/方法执行某个动作或计算
        self.axes.fill(angles, values2, '#4d79ff', alpha=0.25)  # 调用函数/方法执行某个动作或计算
        
        # Labels
        self.axes.set_xticks(angles[:-1])  # 调用函数/方法执行某个动作或计算
        self.axes.set_xticklabels(categories, color='white', size=10)  # 调用函数/方法执行某个动作或计算
        
        # Y labels
        self.axes.set_rlabel_position(0)  # 调用函数/方法执行某个动作或计算
        self.axes.set_yticks([20, 40, 60, 80, 100])  # 调用函数/方法执行某个动作或计算
        self.axes.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=7)  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 100)  # 调用函数/方法执行某个动作或计算
        
        # Grid color
        self.axes.grid(color='grey', alpha=0.3)  # 调用函数/方法执行某个动作或计算
        self.axes.spines['polar'].set_visible(False)  # 调用函数/方法执行某个动作或计算
        
        # Legend
        self.axes.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='#333', edgecolor='none', labelcolor='white')  # 调用函数/方法执行某个动作或计算
        
        self.draw()  # 调用函数/方法执行某个动作或计算

class CourtMapBase(MplCanvas):  # 定义类（封装数据与行为）
    def draw_court(self, ax=None):  # 定义函数（封装可复用逻辑）
        if ax is None: ax = self.axes  # 条件分支判断并选择执行路径
        
        # Court dimensions (standard) - mapped to image coords roughly if needed
        # Or just use standard dimensions 13.4 x 6.1 and normalize data points to it
        # Here we assume data points are in image pixels (e.g. 1280x720)
        # We need a way to map them. For now, let's assume we plot in pixel coordinates directly
        # and overlay a "schematic" court is hard without homography.
        # BETTER APPROACH: Just draw the data points (heatmap) on black, 
        # assuming the user knows the court shape from the points.
        # OR: Use a generic rectangle if we don't have homography.
        
        # Since we want "Professional", we ideally project points to top-down view.
        # But we don't have the homography matrix here easily. 
        # However, TrackNet usually outputs ball_x, ball_y in screen coordinates.
        # For a top-down view, we need a homography transform. 
        # IF we don't have it, we display in "Screen View" (Perspective).
        # Let's stick to Screen View for now, but flip Y so 0 is bottom? No, image coords 0 is top.
        
        ax.invert_yaxis() # Match image coords
        ax.set_aspect('equal')  # 调用函数/方法执行某个动作或计算
        ax.axis('off')  # 调用函数/方法执行某个动作或计算

class HitPoint3D(MplCanvas):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent, width=5, height=4)  # 调用函数/方法执行某个动作或计算
        self.axes.remove()  # 调用函数/方法执行某个动作或计算
        self.axes = self.fig.add_subplot(111, projection='3d')  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
        self.axes.set_facecolor('#1e1e1e')  # 调用函数/方法执行某个动作或计算
        self.axes.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))  # 调用函数/方法执行某个动作或计算
        self.axes.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))  # 调用函数/方法执行某个动作或计算
        self.axes.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))  # 调用函数/方法执行某个动作或计算

    def plot(self, df):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        hits = df[df['is_hit'] == 1]  # 将表达式计算结果赋给变量 hits
        if hits.empty: return  # 条件分支判断并选择执行路径
        
        # x, y, z(approx height)
        xs = hits['ball_x'].values  # 将表达式计算结果赋给变量 xs
        ys = hits['ball_y'].values  # 将表达式计算结果赋给变量 ys
        ys = 720 - ys  # 将表达式计算结果赋给变量 ys
        
        # Z heuristic: assume linear relationship with Y for depth? 
        # Actually without 3D calibration, we use Y as depth, and 720-Y as height is wrong.
        # But let's assume standard camera angle:
        # X is width. Y is depth (slanted). Z is height.
        # Here we only have 2D (x, y). 
        # We can map Y to Depth, and try to infer Height? No, impossible without calibration.
        # So we just plot X, Y and "Color" as speed or type.
        # OR: We make a 3D scatter where Z is "Speed" or "Height Proxy"
        
        # Let's visualize: X=Width, Y=Depth, Z=Height (Proxy: 720 - ball_y, but that's wrong for depth)
        # Better: Z = Hit Impact Height (Low vs High)
        # Let's assume High Y (small pixel val) is Far Depth. Low Y (large pixel val) is Near Depth.
        # Height is unknown.
        # Let's plot 3D: X=Width, Y=Depth (Y-coord), Z=Speed
        
        zs = hits['ball_speed'].values  # 将表达式计算结果赋给变量 zs
        
        scatter = self.axes.scatter(xs, ys, zs, c=hits['hit_player'], cmap='coolwarm', s=50, depthshade=True)  # 将 scatter 设为一次调用/构造的返回值
        
        self.axes.set_xlabel("宽度 (X)")  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("深度 (Y)")  # 调用函数/方法执行某个动作或计算
        self.axes.set_zlabel("球速")  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("3D击球分析（球速）", color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class LoadChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, df):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        # Calculate cumulative distance (Load)
        # We need time series of distance for P1 and P2
        
        # Create a time index
        df_sorted = df.sort_values('time_seconds')  # 将 df_sorted 设为一次调用/构造的返回值
        t = df_sorted['time_seconds']  # 将表达式计算结果赋给变量 t
        
        # Cumulative sum of speed * dt = distance
        # Fill NA speeds
        v1 = df_sorted['p1_speed'].fillna(0)  # 将 v1 设为一次调用/构造的返回值
        v2 = df_sorted['p2_speed'].fillna(0)  # 将 v2 设为一次调用/构造的返回值
        dt = df_sorted['time_seconds'].diff().fillna(0.04)  # 将 dt 设为一次调用/构造的返回值
        
        dist1 = (v1 * dt).cumsum()  # 初始化变量 dist1 为一个容器/表达式结果
        dist2 = (v2 * dt).cumsum()  # 初始化变量 dist2 为一个容器/表达式结果
        
        self.axes.plot(t, dist1, color='#ff4d4d', label='球员1负荷', linewidth=2)  # 调用函数/方法执行某个动作或计算
        self.axes.plot(t, dist2, color='#4d79ff', label='球员2负荷', linewidth=2)  # 调用函数/方法执行某个动作或计算
        
        self.axes.fill_between(t, dist1, color='#ff4d4d', alpha=0.1)  # 调用函数/方法执行某个动作或计算
        self.axes.fill_between(t, dist2, color='#4d79ff', alpha=0.1)  # 调用函数/方法执行某个动作或计算
        
        self.axes.set_xlabel("时间（秒）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("累计距离（像素）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("体能负荷（距离）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.legend(facecolor='#333', edgecolor='none', labelcolor='white')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        
        self.draw()  # 调用函数/方法执行某个动作或计算

class SankeyChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, data):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.axes.axis('off')  # 调用函数/方法执行某个动作或计算
        
        if not data or not data['sources']:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无桑基数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
            
        # Simplified Sankey using parallel coordinates or just connection lines
        # Since matplotlib doesn't have built-in Sankey for complex flows easily,
        # We simulate a 2-stage flow: Service (Left) -> Outcome (Right)
        
        sources = data['sources'] # List of source labels
        targets = data['targets'] # List of target labels
        values = data['values']   # List of counts
        
        # Get unique nodes and map to y-positions
        unique_src = sorted(list(set(sources)))  # 将 unique_src 设为一次调用/构造的返回值
        unique_tgt = sorted(list(set(targets)))  # 将 unique_tgt 设为一次调用/构造的返回值
        
        src_y = np.linspace(0.8, 0.2, len(unique_src))  # 将 src_y 设为一次调用/构造的返回值
        tgt_y = np.linspace(0.8, 0.2, len(unique_tgt))  # 将 tgt_y 设为一次调用/构造的返回值
        
        src_map = {k: v for k, v in zip(unique_src, src_y)}  # 初始化变量 src_map 为一个容器/表达式结果
        tgt_map = {k: v for k, v in zip(unique_tgt, tgt_y)}  # 初始化变量 tgt_map 为一个容器/表达式结果
        
        # Draw connections
        max_val = max(values) if values else 1  # 将 max_val 设为一次调用/构造的返回值
        
        for s, t, v in zip(sources, targets, values):  # 循环遍历序列/迭代器
            y1 = src_map[s]  # 将表达式计算结果赋给变量 y1
            y2 = tgt_map[t]  # 将表达式计算结果赋给变量 y2
            width = (v / max_val) * 10  # 初始化变量 width 为一个容器/表达式结果
            
            # Draw bezier curve
            self.draw_bezier(0.2, y1, 0.8, y2, width, color='#00ffcc', alpha=0.5)  # 调用函数/方法执行某个动作或计算
            
        # Draw Nodes
        for k, y in src_map.items():  # 循环遍历序列/迭代器
            self.axes.text(0.1, y, k, ha='right', va='center', color='white', fontsize=10)  # 调用函数/方法执行某个动作或计算
            self.axes.scatter(0.2, y, color='#ff4d4d', s=100, zorder=3)  # 调用函数/方法执行某个动作或计算
            
        for k, y in tgt_map.items():  # 循环遍历序列/迭代器
            self.axes.text(0.9, y, k, ha='left', va='center', color='white', fontsize=10)  # 调用函数/方法执行某个动作或计算
            self.axes.scatter(0.8, y, color='#4d79ff', s=100, zorder=3)  # 调用函数/方法执行某个动作或计算
            
        self.axes.set_xlim(0, 1)  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 1)  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("战术流向：发球 → 结局", color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算
        
    def draw_bezier(self, x1, y1, x2, y2, width, color, alpha):  # 定义函数（封装可复用逻辑）
        t = np.linspace(0, 1, 100)  # 将 t 设为一次调用/构造的返回值
        # Cubic Bezier with control points
        xc1 = x1 + 0.3  # 将表达式计算结果赋给变量 xc1
        xc2 = x2 - 0.3  # 将表达式计算结果赋给变量 xc2
        
        x = (1-t)**3*x1 + 3*(1-t)**2*t*xc1 + 3*(1-t)*t**2*xc2 + t**3*x2  # 初始化变量 x 为一个容器/表达式结果
        y = (1-t)**3*y1 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y2  # 初始化变量 y 为一个容器/表达式结果
        
        self.axes.plot(x, y, linewidth=width, color=color, alpha=alpha)  # 调用函数/方法执行某个动作或计算

class TransitionHeatmap(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, df):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        if df.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无转移数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
            
        sns.heatmap(df, ax=self.axes, cmap='viridis', annot=True, fmt='.2f',   # 执行当前语句（保持与上文逻辑一致）
                    cbar=False, annot_kws={"size": 8})  # 将表达式计算结果赋给变量 cbar
        
        self.axes.set_title("战术转移概率", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_xlabel("响应击球", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("来球类型", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white', rotation=45)  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class TransitionChordChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, chord_data: dict, title="战术转移弦图"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.axes.axis('off')  # 调用函数/方法执行某个动作或计算
        nodes = chord_data.get('nodes', [])  # 将 nodes 设为一次调用/构造的返回值
        links = chord_data.get('links', [])  # 将 links 设为一次调用/构造的返回值
        if not nodes or not links:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无转移数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        labels = [n['name'] for n in nodes]  # 初始化变量 labels 为一个容器/表达式结果
        n = len(labels)  # 将 n 设为一次调用/构造的返回值
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)  # 将 angles 设为一次调用/构造的返回值
        radius = 1.0  # 将表达式计算结果赋给变量 radius
        node_xy = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]  # 初始化变量 node_xy 为一个容器/表达式结果
        cmap = plt.get_cmap('Set2')  # 将 cmap 设为一次调用/构造的返回值
        label_color = {labels[i]: cmap(i % cmap.N) for i in range(n)}  # 初始化变量 label_color 为一个容器/表达式结果
        # nodes
        for i, (x, y) in enumerate(node_xy):  # 循环遍历序列/迭代器
            self.axes.scatter([x], [y], s=160, color=label_color[labels[i]], edgecolors='white', linewidths=0.8, alpha=0.9)  # 调用函数/方法执行某个动作或计算
            self.axes.text(x*1.12, y*1.12, labels[i], color='white', ha='center', va='center', fontsize=10)  # 调用函数/方法执行某个动作或计算
        # links
        max_v = max(l.get('value', 1) for l in links) if links else 1  # 将 max_v 设为一次调用/构造的返回值
        idx_map = {labels[i]: i for i in range(n)}  # 初始化变量 idx_map 为一个容器/表达式结果
        for l in links:  # 循环遍历序列/迭代器
            s = l['source']; t = l['target']; v = l.get('value', 1)  # 将 s 设为一次调用/构造的返回值
            if s not in idx_map or t not in idx_map: continue  # 条件分支判断并选择执行路径
            i = idx_map[s]; j = idx_map[t]  # 将表达式计算结果赋给变量 i
            x1, y1 = node_xy[i]; x2, y2 = node_xy[j]  # 执行当前语句（保持与上文逻辑一致）
            ctrl = ((x1+x2)/2.0, (y1+y2)/2.0)  # 初始化变量 ctrl 为一个容器/表达式结果
            tlin = np.linspace(0, 1, 120)  # 将 tlin 设为一次调用/构造的返回值
            bx = (1-tlin)**2*x1 + 2*(1-tlin)*tlin*ctrl[0] + tlin**2*x2  # 初始化变量 bx 为一个容器/表达式结果
            by = (1-tlin)**2*y1 + 2*(1-tlin)*tlin*ctrl[1] + tlin**2*y2  # 初始化变量 by 为一个容器/表达式结果
            lw = 0.6 + 6.0*(v/max_v)  # 将 lw 设为一次调用/构造的返回值
            color = label_color[s]  # 将表达式计算结果赋给变量 color
            self.axes.plot(bx, by, color=color, linewidth=lw, alpha=0.75)  # 调用函数/方法执行某个动作或计算
        circle = plt.Circle((0,0), radius, color='#333', fill=False, linewidth=1.0)  # 将 circle 设为一次调用/构造的返回值
        self.axes.add_artist(circle)  # 调用函数/方法执行某个动作或计算
        self.axes.axis('equal')  # 调用函数/方法执行某个动作或计算
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算
class ThemeRiverChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, river_data: dict, title="战术类型流图"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        times = river_data.get('times', [])  # 将 times 设为一次调用/构造的返回值
        series = river_data.get('series', {})  # 将 series 设为一次调用/构造的返回值
        if not times or not series:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无战术数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        t = np.array(times, dtype=float)  # 将 t 设为一次调用/构造的返回值
        types = list(series.keys())  # 将 types 设为一次调用/构造的返回值
        colors = plt.cm.Set2(np.linspace(0, 1, len(types)))  # 将 colors 设为一次调用/构造的返回值
        vals = np.vstack([np.array(series[tp], dtype=float) for tp in types])  # 将 vals 设为一次调用/构造的返回值
        # Normalize rows to ensure stack sums ~1
        vals = np.clip(vals, 0, 1)  # 将 vals 设为一次调用/构造的返回值
        cum = np.cumsum(vals, axis=0)  # 将 cum 设为一次调用/构造的返回值
        base = np.zeros_like(t)  # 将 base 设为一次调用/构造的返回值
        for i, tp in enumerate(types):  # 循环遍历序列/迭代器
            upper = base + vals[i]  # 将表达式计算结果赋给变量 upper
            self.axes.fill_between(t, base, upper, color=colors[i], alpha=0.8, label=tp)  # 调用函数/方法执行某个动作或计算
            base = upper  # 将表达式计算结果赋给变量 base
        self.axes.set_title(f"{title}（滑窗{river_data.get('window_sec', 2.0)}秒）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_xlabel("时间（秒）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("占比", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 1)  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333', alpha=0.3)  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.legend(facecolor='#333', edgecolor='none', labelcolor='white', ncol=min(3, len(types)))  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算
class MomentumChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, rallies):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        # Calculate momentum: (Hit Count * Duration) as proxy for intensity
        x = range(1, len(rallies) + 1)  # 将 x 设为一次调用/构造的返回值
        y = [r.hit_count * r.duration_sec for r in rallies] # Intensity
        
        # Color by hit count
        colors = [r.hit_count for r in rallies]  # 初始化变量 colors 为一个容器/表达式结果
        
        if not rallies:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无回合数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            scatter = self.axes.scatter(x, y, c=colors, cmap='viridis', s=50, zorder=2)  # 将 scatter 设为一次调用/构造的返回值
            if len(rallies) > 1:  # 条件分支判断并选择执行路径
                self.axes.plot(x, y, color='grey', alpha=0.5, zorder=1)  # 调用函数/方法执行某个动作或计算
        
        self.axes.set_xlabel("回合序号", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("强度（拍数×时长）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        
        # Force integer ticks for X axis if few rallies
        if len(rallies) < 10:  # 条件分支判断并选择执行路径
            from matplotlib.ticker import MaxNLocator  # 从模块导入符号，供后续调用
            self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))  # 调用函数/方法执行某个动作或计算
            
        self.axes.grid(True, color='#333')  # 调用函数/方法执行某个动作或计算
        
        # Add colorbar
        # cbar = self.fig.colorbar(scatter, ax=self.axes)
        # cbar.ax.yaxis.set_tick_params(color='white')
        
        self.draw()  # 调用函数/方法执行某个动作或计算

class RallyComplexityChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, rallies):  # 定义函数（封装可复用逻辑）
        import pandas as pd  # 导入模块，供后续使用
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        if not rallies:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无回合数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        rows = []  # 初始化变量 rows 为一个容器/表达式结果
        for r in rallies:  # 循环遍历序列/迭代器
            df = r.trajectory  # 将表达式计算结果赋给变量 df
            avg_speed = float(pd.to_numeric(df.get('ball_speed', pd.Series()), errors='coerce').dropna().mean()) if df is not None else 0.0  # 将 avg_speed 设为一次调用/构造的返回值
            rows.append({'hits': r.hit_count, 'duration': r.duration_sec, 'avg_speed': avg_speed})  # 调用函数/方法执行某个动作或计算
        d = pd.DataFrame(rows)  # 将 d 设为一次调用/构造的返回值
        if d.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "No Rally Stats", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        try:  # 开始异常捕获保护块
            sns.kdeplot(x=d['hits'], y=d['duration'], ax=self.axes, fill=True, cmap='plasma', levels=30, thresh=0.05, alpha=0.7)  # 调用函数/方法执行某个动作或计算
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        sc = self.axes.scatter(d['hits'], d['duration'], c=d['avg_speed'], cmap='viridis', s=40, edgecolors='none')  # 将 sc 设为一次调用/构造的返回值
        self.axes.set_xlabel("拍数", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("时长（秒）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("回合复杂度密度", color='white')  # 调用函数/方法执行某个动作或计算
        cb = self.fig.colorbar(sc, ax=self.axes, fraction=0.046, pad=0.04)  # 将 cb 设为一次调用/构造的返回值
        cb.ax.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class RadarBarsChart(MplCanvas):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent, width=5, height=4, dpi=100)  # 调用函数/方法执行某个动作或计算
        self.axes.remove()  # 调用函数/方法执行某个动作或计算
        self.axes = self.fig.add_subplot(111, polar=True)  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
        self.axes.set_facecolor('#1e1e1e')  # 调用函数/方法执行某个动作或计算
    def plot(self, p1_stats, p2_stats, categories):  # 定义函数（封装可复用逻辑）
        import numpy as np  # 导入模块，供后续使用
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        N = len(categories)  # 将 N 设为一次调用/构造的返回值
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)  # 将 angles 设为一次调用/构造的返回值
        width = (2*np.pi / N) * 0.35  # 初始化变量 width 为一个容器/表达式结果
        vals1 = np.array(list(p1_stats.values()))  # 将 vals1 设为一次调用/构造的返回值
        vals2 = np.array(list(p2_stats.values()))  # 将 vals2 设为一次调用/构造的返回值
        self.axes.bar(angles - width*0.6, vals1, width=width, color='#ff4d4d', alpha=0.6, label='球员1')  # 调用函数/方法执行某个动作或计算
        self.axes.bar(angles + width*0.6, vals2, width=width, color='#4d79ff', alpha=0.6, label='球员2')  # 调用函数/方法执行某个动作或计算
        self.axes.set_xticks(angles)  # 调用函数/方法执行某个动作或计算
        self.axes.set_xticklabels(categories, color='white', fontsize=10)  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 100)  # 调用函数/方法执行某个动作或计算
        self.axes.grid(color='grey', alpha=0.3)  # 调用函数/方法执行某个动作或计算
        self.axes.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='#333', edgecolor='none', labelcolor='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算
class ShotQualityViolinChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, df):  # 定义函数（封装可复用逻辑）
        import pandas as pd  # 导入模块，供后续使用
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        hits = df[df.get('is_hit', 0) == 1].copy()  # 将 hits 设为一次调用/构造的返回值
        if hits.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无击球数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        hits['Type'] = hits['stroke_type_name'].astype(str)  # 调用函数/方法执行某个动作或计算
        type_labels = {  # 初始化变量 type_labels 为一个容器/表达式结果
            '杀': '杀球', 'smash': '杀球',  # 执行当前语句（保持与上文逻辑一致）
            '抽': '抽球', 'drive': '抽球',  # 执行当前语句（保持与上文逻辑一致）
            '吊': '吊球', 'drop': '吊球',  # 执行当前语句（保持与上文逻辑一致）
            '网': '网前', 'net': '网前',  # 执行当前语句（保持与上文逻辑一致）
            '挑': '挑球', 'lift': '挑球',  # 执行当前语句（保持与上文逻辑一致）
            '高': '高远', 'clear': '高远'  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        def norm(x):  # 定义函数（封装可复用逻辑）
            xl = x.lower()  # 将 xl 设为一次调用/构造的返回值
            for k, v in type_labels.items():  # 循环遍历序列/迭代器
                if k in xl:  # 条件分支判断并选择执行路径
                    return v  # 从函数返回结果
            return '其他'  # 从函数返回结果
        hits['Type'] = hits['Type'].apply(norm)  # 调用函数/方法执行某个动作或计算
        hits = hits[hits['Type'] != '其他']  # 将表达式计算结果赋给变量 hits
        if hits.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "No Typed Hit Data", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        sns.violinplot(data=hits, x='Type', y='ball_speed', ax=self.axes, inner='quartile', cut=0, palette='Set3')  # 调用函数/方法执行某个动作或计算
        self.axes.set_xlabel("击球类型", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("球速（px/s）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("击球质量（速度分布）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white', axis='x', rotation=20)  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white', axis='y')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333', alpha=0.3)  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class ParallelCoordinatesChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, engine):  # 定义函数（封装可复用逻辑）
        import pandas as pd  # 导入模块，供后续使用
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        df = engine.df.copy()  # 将 df 设为一次调用/构造的返回值
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        hits = df[df.get('is_hit', 0) == 1].copy()  # 将 hits 设为一次调用/构造的返回值
        if hits.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无击球数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        hits['Height'] = 720 - hits['ball_y'].astype(float)  # 调用函数/方法执行某个动作或计算
        hits['Speed'] = hits['ball_speed'].astype(float)  # 调用函数/方法执行某个动作或计算
        p1_speed = hits['p1_speed'].astype(float)  # 将 p1_speed 设为一次调用/构造的返回值
        p2_speed = hits['p2_speed'].astype(float)  # 将 p2_speed 设为一次调用/构造的返回值
        hits['PlayerSpeed'] = p1_speed.where(hits['hit_player'] == 1, p2_speed)  # 调用函数/方法执行某个动作或计算
        dims = ['Speed', 'Height', 'PlayerSpeed']  # 初始化变量 dims 为一个容器/表达式结果
        scaled = hits[dims].copy()  # 将 scaled 设为一次调用/构造的返回值
        for d in dims:  # 循环遍历序列/迭代器
            mn = float(scaled[d].min())  # 将 mn 设为一次调用/构造的返回值
            mx = float(scaled[d].max())  # 将 mx 设为一次调用/构造的返回值
            if mx - mn <= 1e-6:  # 条件分支判断并选择执行路径
                scaled[d] = 0.5  # 执行当前语句（保持与上文逻辑一致）
            else:  # 条件分支的否则路径
                scaled[d] = (scaled[d] - mn) / (mx - mn)  # 调用函数/方法执行某个动作或计算
        x = np.arange(len(dims))  # 将 x 设为一次调用/构造的返回值
        self.axes.set_xticks(x)  # 调用函数/方法执行某个动作或计算
        self.axes.set_xticklabels(['球速','高度','选手速度'], color='white')  # 调用函数/方法执行某个动作或计算
        for _, row in scaled.iterrows():  # 循环遍历序列/迭代器
            self.axes.plot(x, row.values, color='#00ffcc', alpha=0.3, linewidth=1.0)  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 1)  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333', alpha=0.4)  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("并行坐标：球速/高度/选手速度", color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算
class FeatureCourt3D(MplCanvas):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent, width=6, height=5, dpi=100)  # 调用函数/方法执行某个动作或计算
        self.axes.remove()  # 调用函数/方法执行某个动作或计算
        self.axes = self.fig.add_subplot(111, projection='3d')  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
        # Transparent background
        self.fig.patch.set_alpha(0.0)  # 调用函数/方法执行某个动作或计算
        self.axes.set_facecolor((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
        try:  # 开始异常捕获保护块
            self.axes.xaxis.set_pane_color((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
            self.axes.yaxis.set_pane_color((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
            self.axes.zaxis.set_pane_color((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        try:  # 开始异常捕获保护块
            self.axes.set_box_aspect((1, 1, 0.8))  # 调用函数/方法执行某个动作或计算
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        try:  # 开始异常捕获保护块
            self.axes.set_anchor('C')  # 调用函数/方法执行某个动作或计算
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        self._color_by_speed = True  # 给对象属性 self._color_by_speed 赋值/初始化（来自当前语句右侧表达式）
        self._show_hulls = True  # 给对象属性 self._show_hulls 赋值/初始化（来自当前语句右侧表达式）
        self._show_lines = True  # 给对象属性 self._show_lines 赋值/初始化（来自当前语句右侧表达式）
        self._dist = 9.0  # 给对象属性 self._dist 赋值/初始化（来自当前语句右侧表达式）
        self._engine = None  # 给对象属性 self._engine 赋值/初始化（来自当前语句右侧表达式）
        self._last_data = {}  # 给对象属性 self._last_data 赋值/初始化（来自当前语句右侧表达式）
        self._cbar = None  # 给对象属性 self._cbar 赋值/初始化（来自当前语句右侧表达式）
    def wheelEvent(self, event):  # 定义函数（封装可复用逻辑）
        try:  # 开始异常捕获保护块
            delta = event.angleDelta().y()  # 将 delta 设为一次调用/构造的返回值
        except Exception:  # 捕获异常并进行处理
            delta = 0  # 将表达式计算结果赋给变量 delta
        if delta > 0:  # 条件分支判断并选择执行路径
            self.zoom(1.12)  # 调用函数/方法执行某个动作或计算
        elif delta < 0:  # 条件分支判断并选择执行路径
            self.zoom(1/1.12)  # 调用函数/方法执行某个动作或计算
        event.accept()  # 调用函数/方法执行某个动作或计算
    def _normalize(self, arr):  # 定义函数（封装可复用逻辑）
        arr = np.asarray(arr, dtype=float)  # 将 arr 设为一次调用/构造的返回值
        if arr.size == 0:  # 条件分支判断并选择执行路径
            return arr  # 从函数返回结果
        mn = np.nanmin(arr)  # 将 mn 设为一次调用/构造的返回值
        mx = np.nanmax(arr)  # 将 mx 设为一次调用/构造的返回值
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-6:  # 条件分支判断并选择执行路径
            return np.full_like(arr, 0.5)  # 从函数返回结果
        return (arr - mn) / (mx - mn)  # 从函数返回结果
    def _type_weight(self, name):  # 定义函数（封装可复用逻辑）
        if not name:  # 条件分支判断并选择执行路径
            return 1.0  # 从函数返回结果
        n = str(name).lower()  # 将 n 设为一次调用/构造的返回值
        if 'smash' in n or '杀' in n:  # 条件分支判断并选择执行路径
            return 1.30  # 从函数返回结果
        if 'drive' in n or '抽' in n:  # 条件分支判断并选择执行路径
            return 1.10  # 从函数返回结果
        if 'drop' in n or '吊' in n:  # 条件分支判断并选择执行路径
            return 0.90  # 从函数返回结果
        if 'net' in n or '网' in n:  # 条件分支判断并选择执行路径
            return 0.80  # 从函数返回结果
        if 'lift' in n or 'clear' in n or '挑' in n or '高' in n:  # 条件分支判断并选择执行路径
            return 1.00  # 从函数返回结果
        return 1.00  # 从函数返回结果
    def set_color_mode(self, mode: str):  # 定义函数（封装可复用逻辑）
        self._color_by_speed = (mode == 'speed')  # 给对象属性 self._color_by_speed 赋值/初始化（来自当前语句右侧表达式）
        self._replot()  # 调用函数/方法执行某个动作或计算
    def set_show_hulls(self, flag: bool):  # 定义函数（封装可复用逻辑）
        self._show_hulls = flag  # 给对象属性 self._show_hulls 赋值/初始化（来自当前语句右侧表达式）
        self._replot()  # 调用函数/方法执行某个动作或计算
    def set_show_lines(self, flag: bool):  # 定义函数（封装可复用逻辑）
        self._show_lines = flag  # 给对象属性 self._show_lines 赋值/初始化（来自当前语句右侧表达式）
        self._replot()  # 调用函数/方法执行某个动作或计算
    def zoom(self, factor: float):  # 定义函数（封装可复用逻辑）
        if factor <= 0:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._dist = max(3.0, min(25.0, self._dist / factor))  # 给对象属性 self._dist 赋值/初始化（来自当前语句右侧表达式）
        try:  # 开始异常捕获保护块
            self.axes.dist = self._dist  # 给对象属性 self.axes.dist 赋值/初始化（来自当前语句右侧表达式）
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        self.draw()  # 调用函数/方法执行某个动作或计算
    def reset_zoom(self):  # 定义函数（封装可复用逻辑）
        self._dist = 9.0  # 给对象属性 self._dist 赋值/初始化（来自当前语句右侧表达式）
        try:  # 开始异常捕获保护块
            self.axes.dist = self._dist  # 给对象属性 self.axes.dist 赋值/初始化（来自当前语句右侧表达式）
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        self.draw()  # 调用函数/方法执行某个动作或计算
    def _replot(self):  # 定义函数（封装可复用逻辑）
        if self._engine is not None:  # 条件分支判断并选择执行路径
            self.plot(self._engine)  # 调用函数/方法执行某个动作或计算
    def plot(self, engine):  # 定义函数（封装可复用逻辑）
        import pandas as pd  # 导入模块，供后续使用
        # Recreate axes using GridSpec: main 3D axes + right-side colorbar axes
        try:  # 开始异常捕获保护块
            self.fig.clear()  # 调用函数/方法执行某个动作或计算
            gs = self.fig.add_gridspec(1, 2, width_ratios=[25, 1])  # 将 gs 设为一次调用/构造的返回值
            self.axes = self.fig.add_subplot(gs[0], projection='3d')  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
            self._cax = self.fig.add_subplot(gs[1])  # 给对象属性 self._cax 赋值/初始化（来自当前语句右侧表达式）
            # Transparent backgrounds
            self.fig.patch.set_alpha(0.0)  # 调用函数/方法执行某个动作或计算
            self.axes.set_facecolor((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
            self._cax.set_facecolor((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
            try:  # 开始异常捕获保护块
                self.axes.xaxis.set_pane_color((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
                self.axes.yaxis.set_pane_color((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
                self.axes.zaxis.set_pane_color((0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
            except Exception:  # 捕获异常并进行处理
                pass  # 控制流语句：改变当前代码块的执行方式
            try:  # 开始异常捕获保护块
                self.axes.set_box_aspect((1, 1, 0.8))  # 调用函数/方法执行某个动作或计算
                self.axes.set_anchor('C')  # 调用函数/方法执行某个动作或计算
            except Exception:  # 捕获异常并进行处理
                pass  # 控制流语句：改变当前代码块的执行方式
            # Reset previous colorbar
            self._cbar = None  # 给对象属性 self._cbar 赋值/初始化（来自当前语句右侧表达式）
        except Exception:  # 捕获异常并进行处理
            # Fallback to clearing existing axes
            self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self._engine = engine  # 给对象属性 self._engine 赋值/初始化（来自当前语句右侧表达式）
        df = engine.df  # 将表达式计算结果赋给变量 df
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "No Data", color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        hits = df[df.get('is_hit', 0) == 1].copy()  # 将 hits 设为一次调用/构造的返回值
        if hits.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "No Hit Data", color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        W = float(df['ball_x'].max() if 'ball_x' in df else 1280)  # 将 W 设为一次调用/构造的返回值
        H = float(df['ball_y'].max() if 'ball_y' in df else 720)  # 将 H 设为一次调用/构造的返回值
        # 3D coordinates
        X = hits['ball_x'].astype(float) / max(W, 1.0)  # 将 X 设为一次调用/构造的返回值
        Y = hits['ball_y'].astype(float) / max(H, 1.0)  # 将 Y 设为一次调用/构造的返回值
        Z = (H - hits['ball_y'].astype(float)) / max(H, 1.0)  # 初始化变量 Z 为一个容器/表达式结果
        speed = hits['ball_speed'].astype(float).fillna(0.0)  # 将 speed 设为一次调用/构造的返回值
        speed_n = self._normalize(speed)  # 将 speed_n 设为一次调用/构造的返回值
        # size by threat
        type_names = hits.get('stroke_type_name', pd.Series([''] * len(hits)))  # 将 type_names 设为一次调用/构造的返回值
        weights = np.array([self._type_weight(t) for t in type_names], dtype=float)  # 将 weights 设为一次调用/构造的返回值
        sizes = 26.0 + 22.0 * np.power(speed_n, 1.3) * weights  # 将 sizes 设为一次调用/构造的返回值
        sizes = np.maximum(sizes, 18.0)  # 将 sizes 设为一次调用/构造的返回值
        # color by mode
        cm = plt.get_cmap('turbo')  # 将 cm 设为一次调用/构造的返回值
        colors = None  # 将表达式计算结果赋给变量 colors
        if self._color_by_speed:  # 条件分支判断并选择执行路径
            colors = speed_n  # 将表达式计算结果赋给变量 colors
        else:  # 条件分支的否则路径
            # Discrete palette by type or player
            palette = plt.cm.Set2(np.linspace(0, 1, 8))  # 将 palette 设为一次调用/构造的返回值
            categories = type_names.fillna('').astype(str)  # 将 categories 设为一次调用/构造的返回值
            # fallback to player color if type empty
            if (categories == '').all():  # 条件分支判断并选择执行路径
                cats = hits['hit_player'].astype(int).astype(str)  # 将 cats 设为一次调用/构造的返回值
            else:  # 条件分支的否则路径
                cats = categories  # 将表达式计算结果赋给变量 cats
            uniq = sorted(list(set(cats)))  # 将 uniq 设为一次调用/构造的返回值
            cmap_map = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}  # 初始化变量 cmap_map 为一个容器/表达式结果
            colors = np.array([cmap_map[u] for u in cats])  # 将 colors 设为一次调用/构造的返回值
        # draw court grid (floor)
        floor_x = np.linspace(0, 1, 20)  # 将 floor_x 设为一次调用/构造的返回值
        floor_y = np.linspace(0, 1, 20)  # 将 floor_y 设为一次调用/构造的返回值
        FX, FY = np.meshgrid(floor_x, floor_y)  # 调用函数/方法执行某个动作或计算
        FZ = np.zeros_like(FX)  # 将 FZ 设为一次调用/构造的返回值
        self.axes.plot_wireframe(FX, FY, FZ, color='#555555', rstride=2, cstride=2, alpha=0.25)  # 调用函数/方法执行某个动作或计算
        # draw net plane at mid-depth
        net_y = 0.5  # 将表达式计算结果赋给变量 net_y
        net_x = np.array([0, 1, 1, 0, 0])  # 将 net_x 设为一次调用/构造的返回值
        net_y_poly = np.array([net_y, net_y, net_y, net_y, net_y])  # 将 net_y_poly 设为一次调用/构造的返回值
        net_z = np.array([0, 0.15, 0.15, 0, 0])  # 将 net_z 设为一次调用/构造的返回值
        self.axes.plot(net_x, net_y_poly, net_z, color='#8888ff', alpha=0.45)  # 调用函数/方法执行某个动作或计算
        # scatter points
        glow = self.axes.scatter(X, Y, Z, c=colors, cmap=cm if self._color_by_speed else None, s=sizes*2.0, depthshade=True, alpha=0.14, edgecolors='none')  # 将 glow 设为一次调用/构造的返回值
        sc = self.axes.scatter(X, Y, Z, c=colors, cmap=cm if self._color_by_speed else None, s=sizes, depthshade=True, alpha=0.95, edgecolors='white', linewidths=0.6)  # 将 sc 设为一次调用/构造的返回值
        # colorbar
        if self._color_by_speed:  # 条件分支判断并选择执行路径
            try:  # 开始异常捕获保护块
                self._cbar = self.fig.colorbar(sc, cax=self._cax)  # 给对象属性 self._cbar 赋值/初始化（来自当前语句右侧表达式）
                self._cbar.ax.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
                self._cbar.set_label("球速（归一化）", color='white')  # 调用函数/方法执行某个动作或计算
            except Exception:  # 捕获异常并进行处理
                self._cbar = self.fig.colorbar(sc, ax=self.axes, fraction=0.02, pad=0.02)  # 给对象属性 self._cbar 赋值/初始化（来自当前语句右侧表达式）
                self._cbar.ax.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
                self._cbar.set_label("球速（归一化）", color='white')  # 调用函数/方法执行某个动作或计算
        # rally lines
        if self._show_lines and engine.rallies:  # 条件分支判断并选择执行路径
            for r in engine.rallies:  # 循环遍历序列/迭代器
                r_hits = r.trajectory  # 将表达式计算结果赋给变量 r_hits
                r_hits = r_hits[r_hits.get('is_hit', 0) == 1]  # 将 r_hits 设为一次调用/构造的返回值
                if len(r_hits) < 2:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                x = r_hits['ball_x'].astype(float) / max(W, 1.0)  # 将 x 设为一次调用/构造的返回值
                y = r_hits['ball_y'].astype(float) / max(H, 1.0)  # 将 y 设为一次调用/构造的返回值
                z = (H - r_hits['ball_y'].astype(float)) / max(H, 1.0)  # 初始化变量 z 为一个容器/表达式结果
                sp = r_hits['ball_speed'].astype(float).fillna(0.0)  # 将 sp 设为一次调用/构造的返回值
                c = self._normalize(sp).mean()  # 将 c 设为一次调用/构造的返回值
                self.axes.plot(x, y, z, color=(cm(c) if self._color_by_speed else '#00ffcc'), alpha=0.65, linewidth=2.0)  # 调用函数/方法执行某个动作或计算
                if len(x) >= 2:  # 条件分支判断并选择执行路径
                    dx = np.diff(x)  # 将 dx 设为一次调用/构造的返回值
                    dy = np.diff(y)  # 将 dy 设为一次调用/构造的返回值
                    dz = np.diff(z)  # 将 dz 设为一次调用/构造的返回值
                    self.axes.quiver(x[:-1], y[:-1], z[:-1], dx, dy, dz, length=1.0, normalize=True, color=(cm(c) if self._color_by_speed else '#00ffcc'), alpha=0.35, linewidth=0.8)  # 调用函数/方法执行某个动作或计算
                # Start/End markers
                try:  # 开始异常捕获保护块
                    self.axes.scatter([x[0]], [y[0]], [z[0]], s=70, marker='^', color='#ffff66', edgecolors='white', linewidths=0.8, alpha=0.95)  # 调用函数/方法执行某个动作或计算
                    self.axes.scatter([x[-1]], [y[-1]], [z[-1]], s=70, marker='s', color='#66ffcc', edgecolors='white', linewidths=0.8, alpha=0.95)  # 调用函数/方法执行某个动作或计算
                except Exception:  # 捕获异常并进行处理
                    pass  # 控制流语句：改变当前代码块的执行方式
        # convex hulls per type
        if self._show_hulls and 'stroke_type_name' in hits.columns:  # 条件分支判断并选择执行路径
            from scipy.spatial import ConvexHull  # 从模块导入符号，供后续调用
            for tname, grp in hits.groupby('stroke_type_name'):  # 循环遍历序列/迭代器
                if len(grp) < 20:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                pts = np.vstack([  # 将表达式计算结果赋给变量 pts
                    grp['ball_x'].astype(float) / max(W, 1.0),  # 执行当前语句（保持与上文逻辑一致）
                    grp['ball_y'].astype(float) / max(H, 1.0),  # 执行当前语句（保持与上文逻辑一致）
                    (H - grp['ball_y'].astype(float)) / max(H, 1.0)  # 调用函数/方法执行某个动作或计算
                ]).T  # 执行当前语句（保持与上文逻辑一致）
                try:  # 开始异常捕获保护块
                    hull = ConvexHull(pts)  # 将 hull 设为一次调用/构造的返回值
                    for simplex in hull.simplices:  # 循环遍历序列/迭代器
                        tri = pts[simplex]  # 将表达式计算结果赋给变量 tri
                        self.axes.plot_trisurf(tri[:,0], tri[:,1], tri[:,2], color='#00ffcc', alpha=0.12, linewidth=0)  # 调用函数/方法执行某个动作或计算
                    # centroid
                    c = pts.mean(axis=0)  # 将 c 设为一次调用/构造的返回值
                    self.axes.scatter([c[0]],[c[1]],[c[2]], s=40, color='#00ffcc', alpha=0.6)  # 调用函数/方法执行某个动作或计算
                    self.axes.text(c[0], c[1], c[2]+0.02, str(tname), color='white', fontsize=8)  # 调用函数/方法执行某个动作或计算
                except Exception:  # 捕获异常并进行处理
                    continue  # 控制流语句：改变当前代码块的执行方式
        # labels and view
        self.axes.set_xlabel("宽度（左→右）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("深度（前→后）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_zlabel("高度（低→高）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.view_init(elev=28, azim=-62)  # 调用函数/方法执行某个动作或计算
        try:  # 开始异常捕获保护块
            self.axes.dist = self._dist  # 给对象属性 self.axes.dist 赋值/初始化（来自当前语句右侧表达式）
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        self.axes.set_xlim(0, 1)  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 1)  # 调用函数/方法执行某个动作或计算
        self.axes.set_zlim(0, 1)  # 调用函数/方法执行某个动作或计算
        # XY density projection
        try:  # 开始异常捕获保护块
            gx = np.linspace(0,1,40)  # 将 gx 设为一次调用/构造的返回值
            gy = np.linspace(0,1,40)  # 将 gy 设为一次调用/构造的返回值
            Hxy, xedges, yedges = np.histogram2d(X, Y, bins=[gx, gy])  # 调用函数/方法执行某个动作或计算
            Xg, Yg = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)  # 调用函数/方法执行某个动作或计算
            self.axes.contourf(Xg, Yg, Hxy.T, zdir='z', offset=0, cmap='inferno', alpha=0.25)  # 调用函数/方法执行某个动作或计算
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        # highlight top-speed hits
        try:  # 开始异常捕获保护块
            thresh = np.quantile(speed_n, 0.95)  # 将 thresh 设为一次调用/构造的返回值
            idx = np.where(speed_n >= thresh)[0]  # 将 idx 设为一次调用/构造的返回值
            self.axes.scatter(np.asarray(X)[idx], np.asarray(Y)[idx], np.asarray(Z)[idx], s=np.asarray(sizes)[idx]*2.2, color='#ffcc00', alpha=0.95, marker='^', edgecolors='white', linewidths=0.8)  # 调用函数/方法执行某个动作或计算
        except Exception:  # 捕获异常并进行处理
            pass  # 控制流语句：改变当前代码块的执行方式
        self.axes.set_title("击球落点-高度 3D 战术图", color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算
class VoronoiMap(CourtMapBase):  # 定义类（封装数据与行为）
    def plot(self, p1_pos, p2_pos):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.draw_court()  # 调用函数/方法执行某个动作或计算
        
        # p1_pos, p2_pos are (x, y) tuples
        points = np.array([p1_pos, p2_pos])  # 将 points 设为一次调用/构造的返回值
        
        # Define bounds (screen size)
        bounds = [0, 1280, 0, 720] # xmin, xmax, ymin, ymax
        
        # Simple Voronoi for 2 points is a perpendicular bisector line
        # We can just fill two polygons
        
        # Perpendicular bisector
        mid = (points[0] + points[1]) / 2  # 初始化变量 mid 为一个容器/表达式结果
        vec = points[1] - points[0]  # 将表达式计算结果赋给变量 vec
        # Normal vector (-dy, dx)
        normal = np.array([-vec[1], vec[0]])  # 将 normal 设为一次调用/构造的返回值
        
        # This is a bit complex to clip to rectangle manually with matplotlib polygons quickly
        # Alternative: nearest neighbor classification for a grid of points (contourf)
        
        grid_x, grid_y = np.mgrid[0:1280:20j, 0:720:20j]  # 执行当前语句（保持与上文逻辑一致）
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]  # 将 grid_points 设为一次调用/构造的返回值
        
        # Distances to P1 and P2
        d1 = np.linalg.norm(grid_points - points[0], axis=1)  # 将 d1 设为一次调用/构造的返回值
        d2 = np.linalg.norm(grid_points - points[1], axis=1)  # 将 d2 设为一次调用/构造的返回值
        
        # Mask: 0 for P1, 1 for P2
        mask = (d2 < d1).astype(int)  # 初始化变量 mask 为一个容器/表达式结果
        mask = mask.reshape(grid_x.shape)  # 将 mask 设为一次调用/构造的返回值
        
        self.axes.contourf(grid_x, grid_y, mask, levels=[-0.1, 0.5, 1.1],   # 执行当前语句（保持与上文逻辑一致）
                           colors=['#ff4d4d', '#4d79ff'], alpha=0.3)  # 初始化变量 colors 为一个容器/表达式结果
        
        # Draw players
        self.axes.scatter(*points[0], c='#ff4d4d', s=200, label='球员1', edgecolors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.scatter(*points[1], c='#4d79ff', s=200, label='球员2', edgecolors='white')  # 调用函数/方法执行某个动作或计算
        
        self.axes.set_title("空间控制（Voronoi）", color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class SpeedHeightScatter(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, df):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        # x: speed, y: y-coordinate (height proxy in 2D image)
        # In image coords, smaller y is higher. So we invert y.
        
        # Filter hits
        hits = df[df['is_hit'] == 1]  # 将表达式计算结果赋给变量 hits
        if hits.empty: return  # 条件分支判断并选择执行路径

        x = hits['ball_speed']  # 将表达式计算结果赋给变量 x
        y = hits['ball_y'] # Pixel height (0 is top)
        
        # Invert Y to show "Height" (0 at bottom) - approx
        # Assuming 720p
        y_height = 720 - y  # 将表达式计算结果赋给变量 y_height
        
        scatter = self.axes.scatter(x, y_height, c=hits['hit_player'], cmap='coolwarm', alpha=0.7)  # 将 scatter 设为一次调用/构造的返回值
        
        self.axes.set_xlabel("球速（px/s）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("击球高度（px，自底向上）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("球速-击球高度分布", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333')  # 调用函数/方法执行某个动作或计算
        
        self.draw()  # 调用函数/方法执行某个动作或计算

class HeatmapChart(CourtMapBase):  # 定义类（封装数据与行为）
    def plot(self, x, y, title="热力图"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.draw_court()  # 调用函数/方法执行某个动作或计算
        
        if len(x) > 10:  # 条件分支判断并选择执行路径
            # KDE plot
            sns.kdeplot(x=x, y=y, ax=self.axes, fill=True, cmap='inferno', alpha=0.8, levels=20, thresh=0.05)  # 调用函数/方法执行某个动作或计算
            # Scatter on top for density
            self.axes.scatter(x, y, color='white', s=1, alpha=0.3)  # 调用函数/方法执行某个动作或计算
        
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class ShotTypePie(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, type_counts, title="击球类型"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        labels = list(type_counts.keys())  # 将 labels 设为一次调用/构造的返回值
        sizes = list(type_counts.values())  # 将 sizes 设为一次调用/构造的返回值
        
        # Filter small
        total = sum(sizes)  # 将 total 设为一次调用/构造的返回值
        labels = [l for l, s in zip(labels, sizes) if s/total > 0.02]  # 初始化变量 labels 为一个容器/表达式结果
        sizes = [s for s in sizes if s/total > 0.02]  # 初始化变量 sizes 为一个容器/表达式结果
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))  # 将 colors 设为一次调用/构造的返回值
        
        wedges, texts, autotexts = self.axes.pie(  # 执行当前语句（保持与上文逻辑一致）
            sizes, labels=labels, autopct='%1.1f%%',  # 执行当前语句（保持与上文逻辑一致）
            startangle=90, colors=colors,  # 将表达式计算结果赋给变量 startangle
            textprops=dict(color="w")  # 将 textprops 设为一次调用/构造的返回值
        )  # 执行当前语句（保持与上文逻辑一致）
        
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class PhysicalKPI(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, kpis: dict, zones: dict, title="运动表现指标"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.axes.axis('off')  # 调用函数/方法执行某个动作或计算
        y0 = 0.9  # 将表达式计算结果赋给变量 y0
        dy = 0.1  # 将表达式计算结果赋给变量 dy
        items = [  # 初始化变量 items 为一个容器/表达式结果
            ("平均速度", f"{kpis.get('avg_speed',0):.1f}"),  # 执行当前语句（保持与上文逻辑一致）
            ("95%分位速度", f"{kpis.get('p95_speed',0):.1f}"),  # 执行当前语句（保持与上文逻辑一致）
            ("最大速度", f"{kpis.get('max_speed',0):.1f}"),  # 执行当前语句（保持与上文逻辑一致）
            ("加速度峰值", f"{kpis.get('accel_peak',0):.2f}"),  # 执行当前语句（保持与上文逻辑一致）
            ("加速度均值", f"{kpis.get('accel_mean',0):.2f}"),  # 执行当前语句（保持与上文逻辑一致）
            ("前场占比", f"{zones.get('front',0)*100:.1f}%"),  # 执行当前语句（保持与上文逻辑一致）
            ("后场占比", f"{zones.get('back',0)*100:.1f}%"),  # 执行当前语句（保持与上文逻辑一致）
            ("左/右占比", f"{zones.get('left',0)*100:.1f}% / {zones.get('right',0)*100:.1f}%"),  # 执行当前语句（保持与上文逻辑一致）
            ("近网攻势指数", f"{zones.get('net_aggr',0):.2f}")  # 调用函数/方法执行某个动作或计算
        ]  # 执行当前语句（保持与上文逻辑一致）
        self.axes.text(0.02, 0.98, title, color='white', fontsize=12, va='top')  # 调用函数/方法执行某个动作或计算
        for i, (k, v) in enumerate(items):  # 循环遍历序列/迭代器
            self.axes.text(0.05, y0 - i*dy, f"{k}: {v}", color='#00ffcc', fontsize=10)  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class SpeedHistogram(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, series_df, title="速度分布"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        s = series_df['speed'].dropna()  # 将 s 设为一次调用/构造的返回值
        if s.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        sns.histplot(s, bins=30, kde=True, ax=self.axes, color='#4d79ff', alpha=0.7)  # 调用函数/方法执行某个动作或计算
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_xlabel("速度", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("频数", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class AccelTimeline(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, accel_df, title="加速度时间序列"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        if accel_df.empty:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "暂无数据", ha='center', color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        t = accel_df['time_seconds']  # 将表达式计算结果赋给变量 t
        a = accel_df['accel']  # 将表达式计算结果赋给变量 a
        self.axes.plot(t, a, color='#ffcc00', linewidth=1.2)  # 调用函数/方法执行某个动作或计算
        thr = np.nanpercentile(np.abs(a), 90) if np.isfinite(a).all() else 0.0  # 将 thr 设为一次调用/构造的返回值
        mask = np.abs(a) >= thr  # 将 mask 设为一次调用/构造的返回值
        self.axes.scatter(t[mask], a[mask], s=15, color='#ff4d4d', alpha=0.8)  # 调用函数/方法执行某个动作或计算
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_xlabel("时间（秒）", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("加速度", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333', alpha=0.3)  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class CoverageQuantile(CourtMapBase):  # 定义类（封装数据与行为）
    def plot(self, x, y, title="覆盖分位等值线"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.draw_court()  # 调用函数/方法执行某个动作或计算
        if len(x) < 10:  # 条件分支判断并选择执行路径
            self.axes.text(0.5, 0.5, "数据不足", color='white')  # 调用函数/方法执行某个动作或计算
            self.draw()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        sns.kdeplot(x=x, y=y, ax=self.axes, fill=False, cmap='viridis', levels=[0.2, 0.5, 0.8])  # 调用函数/方法执行某个动作或计算
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class BarycenterEllipse(CourtMapBase):  # 定义类（封装数据与行为）
    def plot(self, x, y, cov_info: dict, title="站位质心与稳定性"):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        self.draw_court()  # 调用函数/方法执行某个动作或计算
        if len(x) >= 10:  # 条件分支判断并选择执行路径
            self.axes.scatter(x, y, s=2, color='white', alpha=0.2)  # 调用函数/方法执行某个动作或计算
        cx = cov_info.get('cx', 0.0)  # 将 cx 设为一次调用/构造的返回值
        cy = cov_info.get('cy', 0.0)  # 将 cy 设为一次调用/构造的返回值
        var_x = max(cov_info.get('var_x', 0.0), 1e-3)  # 将 var_x 设为一次调用/构造的返回值
        var_y = max(cov_info.get('var_y', 0.0), 1e-3)  # 将 var_y 设为一次调用/构造的返回值
        rx = np.sqrt(var_x)  # 将 rx 设为一次调用/构造的返回值
        ry = np.sqrt(var_y)  # 将 ry 设为一次调用/构造的返回值
        theta = 0.0  # 将表达式计算结果赋给变量 theta
        ang = np.linspace(0, 2*np.pi, 100)  # 将 ang 设为一次调用/构造的返回值
        ex = cx + rx*np.cos(ang)  # 将 ex 设为一次调用/构造的返回值
        ey = cy + ry*np.sin(ang)  # 将 ey 设为一次调用/构造的返回值
        self.axes.plot(ex, ey, color='#00ffcc', alpha=0.8)  # 调用函数/方法执行某个动作或计算
        self.axes.scatter([cx], [cy], color='#ffcc00', s=40)  # 调用函数/方法执行某个动作或计算
        self.axes.set_title(title, color='white')  # 调用函数/方法执行某个动作或计算
        self.draw()  # 调用函数/方法执行某个动作或计算

class MomentumChart(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, rallies):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        # Calculate momentum: (Hit Count * Duration) as proxy for intensity
        x = range(1, len(rallies) + 1)  # 将 x 设为一次调用/构造的返回值
        y = [r.hit_count * r.duration_sec for r in rallies] # Intensity
        
        # Color by hit count
        colors = [r.hit_count for r in rallies]  # 初始化变量 colors 为一个容器/表达式结果
        
        scatter = self.axes.scatter(x, y, c=colors, cmap='viridis', s=50, zorder=2)  # 将 scatter 设为一次调用/构造的返回值
        self.axes.plot(x, y, color='grey', alpha=0.5, zorder=1)  # 调用函数/方法执行某个动作或计算
        
        self.axes.set_xlabel("Rally Sequence", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("Intensity (Hits * Duration)", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333')  # 调用函数/方法执行某个动作或计算
        
        # Add colorbar
        # cbar = self.fig.colorbar(scatter, ax=self.axes)
        # cbar.ax.yaxis.set_tick_params(color='white')
        
        self.draw()  # 调用函数/方法执行某个动作或计算

class SpeedHeightScatter(MplCanvas):  # 定义类（封装数据与行为）
    def plot(self, df):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        # x: speed, y: y-coordinate (height proxy in 2D image)
        # In image coords, smaller y is higher. So we invert y.
        
        # Filter hits
        hits = df[df['is_hit'] == 1]  # 将表达式计算结果赋给变量 hits
        if hits.empty: return  # 条件分支判断并选择执行路径

        x = hits['ball_speed']  # 将表达式计算结果赋给变量 x
        y = hits['ball_y'] # Pixel height (0 is top)
        
        # Invert Y to show "Height" (0 at bottom) - approx
        # Assuming 720p
        y_height = 720 - y  # 将表达式计算结果赋给变量 y_height
        
        scatter = self.axes.scatter(x, y_height, c=hits['hit_player'], cmap='coolwarm', alpha=0.7)  # 将 scatter 设为一次调用/构造的返回值
        
        self.axes.set_xlabel("Ball Speed (px/s)", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel("Hit Height (px from bottom)", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.set_title("Speed vs Height Distribution", color='white')  # 调用函数/方法执行某个动作或计算
        self.axes.tick_params(colors='white')  # 调用函数/方法执行某个动作或计算
        self.axes.grid(True, color='#333')  # 调用函数/方法执行某个动作或计算
        
        self.draw()  # 调用函数/方法执行某个动作或计算
