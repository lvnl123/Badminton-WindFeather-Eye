import numpy as np  # 导入模块，供后续使用
from collections import deque  # 从模块导入符号，供后续调用

class KalmanFilter:  # 定义类（封装数据与行为）
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):  # 定义函数（封装可复用逻辑）
        self.dt = dt  # 给对象属性 self.dt 赋值/初始化（来自当前语句右侧表达式）
        
        self.process_noise = process_noise  # 给对象属性 self.process_noise 赋值/初始化（来自当前语句右侧表达式）
        self.measurement_noise = measurement_noise  # 给对象属性 self.measurement_noise 赋值/初始化（来自当前语句右侧表达式）
        
        self.x = None  # 给对象属性 self.x 赋值/初始化（来自当前语句右侧表达式）
        self.P = None  # 给对象属性 self.P 赋值/初始化（来自当前语句右侧表达式）
        self.F = None  # 给对象属性 self.F 赋值/初始化（来自当前语句右侧表达式）
        self.H = None  # 给对象属性 self.H 赋值/初始化（来自当前语句右侧表达式）
        self.Q = None  # 给对象属性 self.Q 赋值/初始化（来自当前语句右侧表达式）
        self.R = None  # 给对象属性 self.R 赋值/初始化（来自当前语句右侧表达式）
        
        self.initialized = False  # 给对象属性 self.initialized 赋值/初始化（来自当前语句右侧表达式）
        
    def init(self, x, y):  # 定义函数（封装可复用逻辑）
        self.x = np.array([x, y, 0, 0], dtype=np.float64)  # 给对象属性 self.x 赋值/初始化（来自当前语句右侧表达式）
        self.P = np.eye(4) * 1000  # 给对象属性 self.P 赋值/初始化（来自当前语句右侧表达式）
        
        self.F = np.array([  # 给对象属性 self.F 赋值/初始化（来自当前语句右侧表达式）
            [1, 0, self.dt, 0],  # 执行当前语句（保持与上文逻辑一致）
            [0, 1, 0, self.dt],  # 执行当前语句（保持与上文逻辑一致）
            [0, 0, 1, 0],  # 执行当前语句（保持与上文逻辑一致）
            [0, 0, 0, 1]  # 执行当前语句（保持与上文逻辑一致）
        ], dtype=np.float64)  # 执行当前语句（保持与上文逻辑一致）
        
        self.H = np.array([  # 给对象属性 self.H 赋值/初始化（来自当前语句右侧表达式）
            [1, 0, 0, 0],  # 执行当前语句（保持与上文逻辑一致）
            [0, 1, 0, 0]  # 执行当前语句（保持与上文逻辑一致）
        ], dtype=np.float64)  # 执行当前语句（保持与上文逻辑一致）
        
        self.Q = np.eye(4) * self.process_noise  # 给对象属性 self.Q 赋值/初始化（来自当前语句右侧表达式）
        self.R = np.eye(2) * self.measurement_noise  # 给对象属性 self.R 赋值/初始化（来自当前语句右侧表达式）
        
        self.initialized = True  # 给对象属性 self.initialized 赋值/初始化（来自当前语句右侧表达式）
        
    def predict(self):  # 定义函数（封装可复用逻辑）
        if not self.initialized:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
            
        self.x = self.F @ self.x  # 给对象属性 self.x 赋值/初始化（来自当前语句右侧表达式）
        self.P = self.F @ self.P @ self.F.T + self.Q  # 给对象属性 self.P 赋值/初始化（来自当前语句右侧表达式）
        return self.x[:2]  # 从函数返回结果
    
    def update(self, measurement):  # 定义函数（封装可复用逻辑）
        if not self.initialized:  # 条件分支判断并选择执行路径
            self.init(measurement[0], measurement[1])  # 调用函数/方法执行某个动作或计算
            return measurement  # 从函数返回结果
        
        z = np.array(measurement, dtype=np.float64)  # 将 z 设为一次调用/构造的返回值
        y = z - self.H @ self.x  # 将表达式计算结果赋给变量 y
        S = self.H @ self.P @ self.H.T + self.R  # 将表达式计算结果赋给变量 S
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 将 K 设为一次调用/构造的返回值
        
        self.x = self.x + K @ y  # 给对象属性 self.x 赋值/初始化（来自当前语句右侧表达式）
        self.P = (np.eye(4) - K @ self.H) @ self.P  # 给对象属性 self.P 赋值/初始化（来自当前语句右侧表达式）
        
        return self.x[:2]  # 从函数返回结果

class KalmanTrajectorySmoother:  # 定义类（封装数据与行为）
    def __init__(self, max_gap=10, process_noise=1.0, measurement_noise=10.0):  # 定义函数（封装可复用逻辑）
        self.max_gap = max_gap  # 给对象属性 self.max_gap 赋值/初始化（来自当前语句右侧表达式）
        self.process_noise = process_noise  # 给对象属性 self.process_noise 赋值/初始化（来自当前语句右侧表达式）
        self.measurement_noise = measurement_noise  # 给对象属性 self.measurement_noise 赋值/初始化（来自当前语句右侧表达式）
        
    def smooth(self, x_list, y_list, vis_list):  # 定义函数（封装可复用逻辑）
        if len(x_list) == 0:  # 条件分支判断并选择执行路径
            return x_list, y_list, vis_list  # 从函数返回结果
        
        smoothed_x = []  # 初始化变量 smoothed_x 为一个容器/表达式结果
        smoothed_y = []  # 初始化变量 smoothed_y 为一个容器/表达式结果
        smoothed_vis = []  # 初始化变量 smoothed_vis 为一个容器/表达式结果
        
        kf = KalmanFilter(  # 将表达式计算结果赋给变量 kf
            dt=1.0,  # 将表达式计算结果赋给变量 dt
            process_noise=self.process_noise,  # 将表达式计算结果赋给变量 process_noise
            measurement_noise=self.measurement_noise  # 将表达式计算结果赋给变量 measurement_noise
        )  # 执行当前语句（保持与上文逻辑一致）
        
        gap_count = 0  # 将表达式计算结果赋给变量 gap_count
        last_valid_x = None  # 将表达式计算结果赋给变量 last_valid_x
        last_valid_y = None  # 将表达式计算结果赋给变量 last_valid_y
        
        for i in range(len(x_list)):  # 循环遍历序列/迭代器
            if vis_list[i] == 1:  # 条件分支判断并选择执行路径
                if not kf.initialized:  # 条件分支判断并选择执行路径
                    kf.init(x_list[i], y_list[i])  # 调用函数/方法执行某个动作或计算
                    smoothed_x.append(x_list[i])  # 调用函数/方法执行某个动作或计算
                    smoothed_y.append(y_list[i])  # 调用函数/方法执行某个动作或计算
                    smoothed_vis.append(1)  # 调用函数/方法执行某个动作或计算
                    last_valid_x = x_list[i]  # 将表达式计算结果赋给变量 last_valid_x
                    last_valid_y = y_list[i]  # 将表达式计算结果赋给变量 last_valid_y
                else:  # 条件分支的否则路径
                    predicted = kf.predict()  # 将 predicted 设为一次调用/构造的返回值
                    updated = kf.update([x_list[i], y_list[i]])  # 将 updated 设为一次调用/构造的返回值
                    
                    smoothed_x.append(updated[0])  # 调用函数/方法执行某个动作或计算
                    smoothed_y.append(updated[1])  # 调用函数/方法执行某个动作或计算
                    smoothed_vis.append(1)  # 调用函数/方法执行某个动作或计算
                    last_valid_x = x_list[i]  # 将表达式计算结果赋给变量 last_valid_x
                    last_valid_y = y_list[i]  # 将表达式计算结果赋给变量 last_valid_y
                
                gap_count = 0  # 将表达式计算结果赋给变量 gap_count
            else:  # 条件分支的否则路径
                gap_count += 1  # 执行当前语句（保持与上文逻辑一致）
                
                if gap_count <= self.max_gap and kf.initialized:  # 条件分支判断并选择执行路径
                    predicted = kf.predict()  # 将 predicted 设为一次调用/构造的返回值
                    
                    if last_valid_x is not None:  # 条件分支判断并选择执行路径
                        dx = predicted[0] - last_valid_x  # 将表达式计算结果赋给变量 dx
                        dy = predicted[1] - last_valid_y  # 将表达式计算结果赋给变量 dy
                        dist = np.sqrt(dx*dx + dy*dy)  # 将 dist 设为一次调用/构造的返回值
                        
                        if dist < 200:  # 条件分支判断并选择执行路径
                            smoothed_x.append(predicted[0])  # 调用函数/方法执行某个动作或计算
                            smoothed_y.append(predicted[1])  # 调用函数/方法执行某个动作或计算
                            smoothed_vis.append(1)  # 调用函数/方法执行某个动作或计算
                        else:  # 条件分支的否则路径
                            smoothed_x.append(0)  # 调用函数/方法执行某个动作或计算
                            smoothed_y.append(0)  # 调用函数/方法执行某个动作或计算
                            smoothed_vis.append(0)  # 调用函数/方法执行某个动作或计算
                            kf = KalmanFilter(  # 将表达式计算结果赋给变量 kf
                                dt=1.0,  # 将表达式计算结果赋给变量 dt
                                process_noise=self.process_noise,  # 将表达式计算结果赋给变量 process_noise
                                measurement_noise=self.measurement_noise  # 将表达式计算结果赋给变量 measurement_noise
                            )  # 执行当前语句（保持与上文逻辑一致）
                    else:  # 条件分支的否则路径
                        smoothed_x.append(predicted[0])  # 调用函数/方法执行某个动作或计算
                        smoothed_y.append(predicted[1])  # 调用函数/方法执行某个动作或计算
                        smoothed_vis.append(1)  # 调用函数/方法执行某个动作或计算
                else:  # 条件分支的否则路径
                    smoothed_x.append(0)  # 调用函数/方法执行某个动作或计算
                    smoothed_y.append(0)  # 调用函数/方法执行某个动作或计算
                    smoothed_vis.append(0)  # 调用函数/方法执行某个动作或计算
                    kf = KalmanFilter(  # 将表达式计算结果赋给变量 kf
                        dt=1.0,  # 将表达式计算结果赋给变量 dt
                        process_noise=self.process_noise,  # 将表达式计算结果赋给变量 process_noise
                        measurement_noise=self.measurement_noise  # 将表达式计算结果赋给变量 measurement_noise
                    )  # 执行当前语句（保持与上文逻辑一致）
        
        return smoothed_x, smoothed_y, smoothed_vis  # 从函数返回结果
