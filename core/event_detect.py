import numpy as np  # 导入模块，供后续使用
from scipy.signal import find_peaks  # 从模块导入符号，供后续调用
import json  # 导入模块，供后续使用
import os  # 导入模块，供后续使用


class EventDetector:  # 定义类（封装数据与行为）
    def __init__(self, trajectory_data, poses=None):  # 定义函数（封装可复用逻辑）
        self.trajectory_data = trajectory_data  # 给对象属性 self.trajectory_data 赋值/初始化（来自当前语句右侧表达式）
        self.poses = poses  # 给对象属性 self.poses 赋值/初始化（来自当前语句右侧表达式）
        self.hit_frames = []  # 给对象属性 self.hit_frames 赋值/初始化（来自当前语句右侧表达式）
        self.hit_players = []  # 给对象属性 self.hit_players 赋值/初始化（来自当前语句右侧表达式）

    def detect_hits(self, fps=25, prominence=2, angle_threshold=30, velocity_threshold=3, min_frame_gap=13, min_continuation_frames=5, min_movement_threshold=20):  # 定义函数（封装可复用逻辑）
        frames = []  # 初始化变量 frames 为一个容器/表达式结果
        realx = []  # 初始化变量 realx 为一个容器/表达式结果
        realy = []  # 初始化变量 realy 为一个容器/表达式结果

        for frame_idx, data in enumerate(self.trajectory_data):  # 循环遍历序列/迭代器
            if data is not None and len(data) >= 2:  # 条件分支判断并选择执行路径
                x, y = data  # 执行当前语句（保持与上文逻辑一致）
                if x > 0 and y > 0:  # 条件分支判断并选择执行路径
                    frames.append(frame_idx)  # 调用函数/方法执行某个动作或计算
                    realx.append(x)  # 调用函数/方法执行某个动作或计算
                    realy.append(y)  # 调用函数/方法执行某个动作或计算

        if len(frames) == 0:  # 条件分支判断并选择执行路径
            print("No valid trajectory points found!")  # 调用函数/方法执行某个动作或计算
            return [], []  # 从函数返回结果

        frames = np.array(frames)  # 将 frames 设为一次调用/构造的返回值
        realx = np.array(realx)  # 将 realx 设为一次调用/构造的返回值
        realy = np.array(realy)  # 将 realy 设为一次调用/构造的返回值

        points = np.column_stack([realx, realy, frames])  # 将 points 设为一次调用/构造的返回值
        x, y, z = points.T  # 执行当前语句（保持与上文逻辑一致）

        hit_indices = []  # 初始化变量 hit_indices 为一个容器/表达式结果

        peaks, properties = find_peaks(y, prominence=prominence)  # 调用函数/方法执行某个动作或计算
        valleys, _ = find_peaks(-y, prominence=prominence)  # 调用函数/方法执行某个动作或计算
        
        print(f"Detected {len(peaks)} peaks and {len(valleys)} valleys with prominence={prominence}")  # 调用函数/方法执行某个动作或计算

        for peak_idx in peaks:  # 循环遍历序列/迭代器
            if peak_idx < 2 or peak_idx >= len(y) - 2:  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式

            prev_slope = y[peak_idx] - y[peak_idx - 1]  # 将表达式计算结果赋给变量 prev_slope
            next_slope = y[peak_idx + 1] - y[peak_idx]  # 将表达式计算结果赋给变量 next_slope
            
            if prev_slope > 0 and next_slope < 0:  # 条件分支判断并选择执行路径
                angle = self._calculate_angle(  # 将表达式计算结果赋给变量 angle
                    [x[peak_idx - 1], y[peak_idx - 1], x[peak_idx], y[peak_idx]],  # 执行当前语句（保持与上文逻辑一致）
                    [x[peak_idx], y[peak_idx], x[peak_idx + 1], y[peak_idx + 1]]  # 执行当前语句（保持与上文逻辑一致）
                )  # 执行当前语句（保持与上文逻辑一致）
                
                if angle > angle_threshold:  # 条件分支判断并选择执行路径
                    hit_indices.append(peak_idx)  # 调用函数/方法执行某个动作或计算

        for valley_idx in valleys:  # 循环遍历序列/迭代器
            if valley_idx < 2 or valley_idx >= len(y) - 2:  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式

            prev_slope = y[valley_idx] - y[valley_idx - 1]  # 将表达式计算结果赋给变量 prev_slope
            next_slope = y[valley_idx + 1] - y[valley_idx]  # 将表达式计算结果赋给变量 next_slope
            
            if prev_slope < 0 and next_slope > 0:  # 条件分支判断并选择执行路径
                angle = self._calculate_angle(  # 将表达式计算结果赋给变量 angle
                    [x[valley_idx - 1], y[valley_idx - 1], x[valley_idx], y[valley_idx]],  # 执行当前语句（保持与上文逻辑一致）
                    [x[valley_idx], y[valley_idx], x[valley_idx + 1], y[valley_idx + 1]]  # 执行当前语句（保持与上文逻辑一致）
                )  # 执行当前语句（保持与上文逻辑一致）
                
                if angle > angle_threshold:  # 条件分支判断并选择执行路径
                    hit_indices.append(valley_idx)  # 调用函数/方法执行某个动作或计算

        hit_indices = sorted(list(set(hit_indices)))  # 将 hit_indices 设为一次调用/构造的返回值
        hit_frames = [int(frames[i]) for i in hit_indices]  # 初始化变量 hit_frames 为一个容器/表达式结果

        hit_frames = self._merge_consecutive_hits(hit_frames, min_frame_gap=min_frame_gap)  # 将 hit_frames 设为一次调用/构造的返回值

        hit_frames = self._validate_hit_continuation(hit_frames, min_continuation_frames=min_continuation_frames, min_movement_threshold=min_movement_threshold)  # 将 hit_frames 设为一次调用/构造的返回值

        landing_frame = self._detect_landing_frame()  # 将 landing_frame 设为一次调用/构造的返回值
        if landing_frame is not None:  # 条件分支判断并选择执行路径
            hit_frames = [f for f in hit_frames if f < landing_frame]  # 初始化变量 hit_frames 为一个容器/表达式结果
            print(f"Filtered out hits after landing frame {landing_frame}")  # 调用函数/方法执行某个动作或计算

        if self.poses is not None:  # 条件分支判断并选择执行路径
            hit_players = self._filter_hits_by_pose(hit_frames)  # 将 hit_players 设为一次调用/构造的返回值
        else:  # 条件分支的否则路径
            hit_players = [1] * len(hit_frames)  # 初始化变量 hit_players 为一个容器/表达式结果

        self.hit_frames = hit_frames  # 给对象属性 self.hit_frames 赋值/初始化（来自当前语句右侧表达式）
        self.hit_players = hit_players  # 给对象属性 self.hit_players 赋值/初始化（来自当前语句右侧表达式）

        return hit_frames, hit_players  # 从函数返回结果

    def _calculate_angle(self, line1, line2):  # 定义函数（封装可复用逻辑）
        x1, y1, x2, y2 = line1  # 执行当前语句（保持与上文逻辑一致）
        x3, y3, x4, y4 = line2  # 执行当前语句（保持与上文逻辑一致）

        vec1 = np.array([x2 - x1, y2 - y1])  # 将 vec1 设为一次调用/构造的返回值
        vec2 = np.array([x4 - x3, y4 - y3])  # 将 vec2 设为一次调用/构造的返回值

        unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)  # 将 unit_vec1 设为一次调用/构造的返回值
        unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)  # 将 unit_vec2 设为一次调用/构造的返回值

        dot_product = np.dot(unit_vec1, unit_vec2)  # 将 dot_product 设为一次调用/构造的返回值
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 将 dot_product 设为一次调用/构造的返回值

        angle = np.degrees(np.arccos(dot_product))  # 将 angle 设为一次调用/构造的返回值
        return angle  # 从函数返回结果

    def _filter_hits_by_pose(self, hit_frames):  # 定义函数（封装可复用逻辑）
        hit_players = []  # 初始化变量 hit_players 为一个容器/表达式结果

        for frame_idx in hit_frames:  # 循环遍历序列/迭代器
            if frame_idx >= len(self.trajectory_data):  # 条件分支判断并选择执行路径
                hit_players.append(0)  # 调用函数/方法执行某个动作或计算
                continue  # 控制流语句：改变当前代码块的执行方式

            trajectory_point = self.trajectory_data[frame_idx]  # 将表达式计算结果赋给变量 trajectory_point
            if trajectory_point is None or len(trajectory_point) < 2:  # 条件分支判断并选择执行路径
                hit_players.append(0)  # 调用函数/方法执行某个动作或计算
                continue  # 控制流语句：改变当前代码块的执行方式

            ball_pos = np.array(trajectory_point[:2])  # 将 ball_pos 设为一次调用/构造的返回值

            reached_by = 0  # 将表达式计算结果赋给变量 reached_by
            dist_reached = 1e99  # 将表达式计算结果赋给变量 dist_reached

            if self.poses is not None and frame_idx < len(self.poses):  # 条件分支判断并选择执行路径
                for player_idx in range(min(2, self.poses.shape[1])):  # 循环遍历序列/迭代器
                    pose_data = self.poses[frame_idx, player_idx]  # 将表达式计算结果赋给变量 pose_data

                    if pose_data is None:  # 条件分支判断并选择执行路径
                        continue  # 控制流语句：改变当前代码块的执行方式

                    pose_centroid = self._get_pose_centroid(pose_data)  # 将 pose_centroid 设为一次调用/构造的返回值

                    if pose_centroid is not None:  # 条件分支判断并选择执行路径
                        dist = np.linalg.norm(ball_pos - pose_centroid)  # 将 dist 设为一次调用/构造的返回值
                        if dist < dist_reached:  # 条件分支判断并选择执行路径
                            dist_reached = dist  # 将表达式计算结果赋给变量 dist_reached
                            reached_by = player_idx + 1  # 将表达式计算结果赋给变量 reached_by

            hit_players.append(reached_by)  # 调用函数/方法执行某个动作或计算

        return hit_players  # 从函数返回结果

    def _get_pose_centroid(self, pose_data):  # 定义函数（封装可复用逻辑）
        valid_points = []  # 初始化变量 valid_points 为一个容器/表达式结果

        for i in range(pose_data.shape[0]):  # 循环遍历序列/迭代器
            x, y = pose_data[i, 0], pose_data[i, 1]  # 执行当前语句（保持与上文逻辑一致）
            if x > 0 and y > 0:  # 条件分支判断并选择执行路径
                valid_points.append([x, y])  # 调用函数/方法执行某个动作或计算

        if len(valid_points) > 0:  # 条件分支判断并选择执行路径
            return np.mean(valid_points, axis=0)  # 从函数返回结果
        return None  # 从函数返回结果

    def _validate_hit_continuation(self, hit_frames, min_continuation_frames=5, min_movement_threshold=20):  # 定义函数（封装可复用逻辑）
        validated_hits = []  # 初始化变量 validated_hits 为一个容器/表达式结果
        
        for hit_frame in hit_frames:  # 循环遍历序列/迭代器
            if hit_frame >= len(self.trajectory_data):  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式
            
            hit_data = self.trajectory_data[hit_frame]  # 将表达式计算结果赋给变量 hit_data
            if hit_data is None or len(hit_data) < 2:  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式
            
            hit_x = hit_data[0]  # 将表达式计算结果赋给变量 hit_x
            hit_y = hit_data[1]  # 将表达式计算结果赋给变量 hit_y
            
            if hit_x <= 0 or hit_y <= 0:  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式
            
            movement_count = 0  # 将表达式计算结果赋给变量 movement_count
            
            for i in range(1, min_continuation_frames + 1):  # 循环遍历序列/迭代器
                if hit_frame + i >= len(self.trajectory_data):  # 条件分支判断并选择执行路径
                    break  # 控制流语句：改变当前代码块的执行方式
                
                next_data = self.trajectory_data[hit_frame + i]  # 将表达式计算结果赋给变量 next_data
                if next_data is None or len(next_data) < 2:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                
                next_x = next_data[0]  # 将表达式计算结果赋给变量 next_x
                next_y = next_data[1]  # 将表达式计算结果赋给变量 next_y
                
                if next_x <= 0 or next_y <= 0:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                
                distance = np.sqrt((next_x - hit_x)**2 + (next_y - hit_y)**2)  # 将 distance 设为一次调用/构造的返回值
                
                if distance >= min_movement_threshold:  # 条件分支判断并选择执行路径
                    movement_count += 1  # 执行当前语句（保持与上文逻辑一致）
            
            if movement_count >= 1:  # 条件分支判断并选择执行路径
                validated_hits.append(hit_frame)  # 调用函数/方法执行某个动作或计算
            else:  # 条件分支的否则路径
                print(f"  Frame {hit_frame}: Invalid hit - ball does not continue moving (movement_count={movement_count})")  # 调用函数/方法执行某个动作或计算
        
        print(f"Validated {len(validated_hits)}/{len(hit_frames)} hits after continuation check")  # 调用函数/方法执行某个动作或计算
        
        return validated_hits  # 从函数返回结果

    def _merge_consecutive_hits(self, hit_frames, min_frame_gap=10):  # 定义函数（封装可复用逻辑）
        if len(hit_frames) == 0:  # 条件分支判断并选择执行路径
            return hit_frames  # 从函数返回结果

        merged_hits = [hit_frames[0]]  # 初始化变量 merged_hits 为一个容器/表达式结果

        for i in range(1, len(hit_frames)):  # 循环遍历序列/迭代器
            if hit_frames[i] - merged_hits[-1] >= min_frame_gap:  # 条件分支判断并选择执行路径
                merged_hits.append(hit_frames[i])  # 调用函数/方法执行某个动作或计算

        return merged_hits  # 从函数返回结果

    def _detect_landing_frame(self):  # 定义函数（封装可复用逻辑）
        if len(self.trajectory_data) == 0:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        
        valid_frames = []  # 初始化变量 valid_frames 为一个容器/表达式结果
        valid_y = []  # 初始化变量 valid_y 为一个容器/表达式结果
        
        for frame_idx, data in enumerate(self.trajectory_data):  # 循环遍历序列/迭代器
            if data is not None and len(data) >= 2 and data[0] > 0 and data[1] > 0:  # 条件分支判断并选择执行路径
                valid_frames.append(frame_idx)  # 调用函数/方法执行某个动作或计算
                valid_y.append(data[1])  # 调用函数/方法执行某个动作或计算
        
        if len(valid_y) == 0:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        
        valid_y = np.array(valid_y)  # 将 valid_y 设为一次调用/构造的返回值
        
        if len(valid_y) < 10:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        
        ground_y = np.percentile(valid_y, 90)  # 将 ground_y 设为一次调用/构造的返回值
        
        for i in range(len(valid_frames) - 1, max(0, len(valid_frames) - 50), -1):  # 循环遍历序列/迭代器
            frame_idx = valid_frames[i]  # 将表达式计算结果赋给变量 frame_idx
            y = valid_y[i]  # 将表达式计算结果赋给变量 y
            
            if y >= ground_y - 20:  # 条件分支判断并选择执行路径
                return frame_idx  # 从函数返回结果
        
        return None  # 从函数返回结果

    def save_hit_events(self, output_path):  # 定义函数（封装可复用逻辑）
        hit_events = []  # 初始化变量 hit_events 为一个容器/表达式结果

        for frame_idx, player_idx in zip(self.hit_frames, self.hit_players):  # 循环遍历序列/迭代器
            hit_events.append({  # 执行当前语句（保持与上文逻辑一致）
                'frame': frame_idx,  # 执行当前语句（保持与上文逻辑一致）
                'player': player_idx  # 执行当前语句（保持与上文逻辑一致）
            })  # 执行当前语句（保持与上文逻辑一致）

        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 调用函数/方法执行某个动作或计算
        with open(output_path, 'w') as f:  # 上下文管理：确保资源正确释放
            json.dump(hit_events, f, indent=2)  # 调用函数/方法执行某个动作或计算

        print(f"Hit events saved to {output_path}")  # 调用函数/方法执行某个动作或计算
        return hit_events  # 从函数返回结果

    @staticmethod  # 装饰器：修改/包装下方函数或类的行为
    def load_hit_events(json_path):  # 定义函数（封装可复用逻辑）
        with open(json_path, 'r') as f:  # 上下文管理：确保资源正确释放
            hit_events = json.load(f)  # 将 hit_events 设为一次调用/构造的返回值

        hit_frames = [event['frame'] for event in hit_events]  # 初始化变量 hit_frames 为一个容器/表达式结果
        hit_players = [event['player'] for event in hit_events]  # 初始化变量 hit_players 为一个容器/表达式结果

        return hit_frames, hit_players  # 从函数返回结果
