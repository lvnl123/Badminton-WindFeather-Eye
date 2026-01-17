import numpy as np  # 导入模块，供后续使用
from .panels import MplCanvas  # 从模块导入符号，供后续调用
from mpl_toolkits.mplot3d import Axes3D  # 从模块导入符号，供后续调用

class Arena3D(MplCanvas):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent, width=6, height=5)  # 调用函数/方法执行某个动作或计算
        self.axes.remove()  # 调用函数/方法执行某个动作或计算
        self.axes = self.fig.add_subplot(111, projection='3d')  # 给对象属性 self.axes 赋值/初始化（来自当前语句右侧表达式）
        self.axes.set_facecolor('#1e1e1e')  # 调用函数/方法执行某个动作或计算
        # Remove axis backgrounds
        self.axes.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))  # 调用函数/方法执行某个动作或计算
        self.axes.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))  # 调用函数/方法执行某个动作或计算
        self.axes.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))  # 调用函数/方法执行某个动作或计算
        
    def plot_rally(self, rally_df, strokes):  # 定义函数（封装可复用逻辑）
        self.axes.cla()  # 调用函数/方法执行某个动作或计算
        
        # 1. Estimate Z (Height)
        # Heuristic: Start/End of flight (hits) have specific heights based on stroke type
        # For now, just a simple arc between hits?
        # Better: use simple parabola based on flight time and distance
        
        # Extract segments between hits
        hit_indices = rally_df[rally_df['is_hit'] == 1].index.tolist()  # 将 hit_indices 设为一次调用/构造的返回值
        
        # Add start and end of rally to indices if not present
        if not hit_indices:  # 条件分支判断并选择执行路径
            indices = [rally_df.index[0], rally_df.index[-1]]  # 初始化变量 indices 为一个容器/表达式结果
        else:  # 条件分支的否则路径
            indices = hit_indices  # 将表达式计算结果赋给变量 indices
            if indices[0] != rally_df.index[0]: indices.insert(0, rally_df.index[0])  # 条件分支判断并选择执行路径
            if indices[-1] != rally_df.index[-1]: indices.append(rally_df.index[-1])  # 条件分支判断并选择执行路径
            
        xs = rally_df['ball_x'].values  # 将表达式计算结果赋给变量 xs
        ys = rally_df['ball_y'].values  # 将表达式计算结果赋给变量 ys
        # Invert Y for visualization (0 at bottom)
        ys = 720 - ys  # 将表达式计算结果赋给变量 ys
        
        zs = np.zeros_like(xs, dtype=float)  # 将 zs 设为一次调用/构造的返回值
        
        # Simple gravity model simulation or interpolation
        for i in range(len(indices) - 1):  # 循环遍历序列/迭代器
            start = indices[i] - rally_df.index[0]  # 将表达式计算结果赋给变量 start
            end = indices[i+1] - rally_df.index[0]  # 将表达式计算结果赋给变量 end
            if end <= start: continue  # 条件分支判断并选择执行路径
            
            segment_len = end - start  # 将表达式计算结果赋给变量 segment_len
            # Parabola: z = 4 * h * (x)(1-x) where x is 0..1
            # Height depends on distance
            dist = np.sqrt((xs[start]-xs[end])**2 + (ys[start]-ys[end])**2)  # 将 dist 设为一次调用/构造的返回值
            peak_height = 100 + dist * 0.5 # Heuristic px height
            
            t = np.linspace(0, 1, segment_len)  # 将 t 设为一次调用/构造的返回值
            z_arc = 4 * peak_height * t * (1 - t)  # 将 z_arc 设为一次调用/构造的返回值
            
            # Add base height (e.g. hit point height)
            base_h = 100 # 1 meter approx
            zs[start:end] = z_arc + base_h  # 执行当前语句（保持与上文逻辑一致）
            
        # Plot Trajectory
        self.axes.plot(xs, ys, zs, color='#00ffcc', linewidth=2, label='球路轨迹')  # 调用函数/方法执行某个动作或计算
        
        # Plot Projection (Shadow)
        self.axes.plot(xs, ys, np.zeros_like(zs), color='#00ffcc', linewidth=1, alpha=0.3, linestyle='--')  # 调用函数/方法执行某个动作或计算
        
        # Plot Hits
        hit_rows = rally_df[rally_df['is_hit'] == 1]  # 将表达式计算结果赋给变量 hit_rows
        for _, row in hit_rows.iterrows():  # 循环遍历序列/迭代器
            idx = int(row.name - rally_df.index[0])  # 将 idx 设为一次调用/构造的返回值
            if idx < len(xs):  # 条件分支判断并选择执行路径
                self.axes.scatter(xs[idx], ys[idx], zs[idx], color='red', s=50, marker='x')  # 调用函数/方法执行某个动作或计算
                
        # Set limits
        self.axes.set_xlim(0, 1280)  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylim(0, 720)  # 调用函数/方法执行某个动作或计算
        self.axes.set_zlim(0, 600)  # 调用函数/方法执行某个动作或计算
        
        self.axes.set_xlabel('宽度 (X)')  # 调用函数/方法执行某个动作或计算
        self.axes.set_ylabel('深度 (Y)')  # 调用函数/方法执行某个动作或计算
        self.axes.set_zlabel('高度 (Z)')  # 调用函数/方法执行某个动作或计算
        
        # View angle
        self.axes.view_init(elev=20, azim=-60)  # 调用函数/方法执行某个动作或计算
        
        self.draw()  # 调用函数/方法执行某个动作或计算
