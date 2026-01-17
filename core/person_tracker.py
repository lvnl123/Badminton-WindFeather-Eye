import numpy as np  # 导入模块，供后续使用
from typing import List, Tuple, Optional  # 从模块导入符号，供后续调用
from collections import deque  # 从模块导入符号，供后续调用


class KalmanFilter:  # 定义类（封装数据与行为）
    def __init__(self, initial_state: np.ndarray, process_noise: float = 1.0, measurement_noise: float = 1.0):  # 定义函数（封装可复用逻辑）
        self.state = initial_state.copy()  # 给对象属性 self.state 赋值/初始化（来自当前语句右侧表达式）
        self.covariance = np.eye(4) * 10.0  # 给对象属性 self.covariance 赋值/初始化（来自当前语句右侧表达式）
        
        self.F = np.eye(4)  # 给对象属性 self.F 赋值/初始化（来自当前语句右侧表达式）
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 给对象属性 self.H 赋值/初始化（来自当前语句右侧表达式）
        
        self.Q = np.eye(4) * process_noise  # 给对象属性 self.Q 赋值/初始化（来自当前语句右侧表达式）
        self.R = np.eye(2) * measurement_noise  # 给对象属性 self.R 赋值/初始化（来自当前语句右侧表达式）

    def predict(self) -> np.ndarray:  # 定义函数（封装可复用逻辑）
        self.state = self.F @ self.state  # 给对象属性 self.state 赋值/初始化（来自当前语句右侧表达式）
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q  # 给对象属性 self.covariance 赋值/初始化（来自当前语句右侧表达式）
        return self.state[:2]  # 从函数返回结果

    def update(self, measurement: np.ndarray):  # 定义函数（封装可复用逻辑）
        z = measurement  # 将表达式计算结果赋给变量 z
        y = z - self.H @ self.state  # 将表达式计算结果赋给变量 y
        S = self.H @ self.covariance @ self.H.T + self.R  # 将表达式计算结果赋给变量 S
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # 将 K 设为一次调用/构造的返回值
        self.state = self.state + K @ y  # 给对象属性 self.state 赋值/初始化（来自当前语句右侧表达式）
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance  # 给对象属性 self.covariance 赋值/初始化（来自当前语句右侧表达式）


class PersonTracker:  # 定义类（封装数据与行为）
    def __init__(self, max_persons: int = 2):  # 定义函数（封装可复用逻辑）
        self.max_persons = max_persons  # 给对象属性 self.max_persons 赋值/初始化（来自当前语句右侧表达式）
        self.kalman_filters = [None] * max_persons  # 给对象属性 self.kalman_filters 赋值/初始化（来自当前语句右侧表达式）
        self.position_history = [deque(maxlen=10) for _ in range(max_persons)]  # 给对象属性 self.position_history 赋值/初始化（来自当前语句右侧表达式）
        self.velocity_history = [deque(maxlen=5) for _ in range(max_persons)]  # 给对象属性 self.velocity_history 赋值/初始化（来自当前语句右侧表达式）
        self.confidence_scores = [0.0] * max_persons  # 给对象属性 self.confidence_scores 赋值/初始化（来自当前语句右侧表达式）
        self.track_lengths = [0] * max_persons  # 给对象属性 self.track_lengths 赋值/初始化（来自当前语句右侧表达式）
        self.last_seen_frames = [-1] * max_persons  # 给对象属性 self.last_seen_frames 赋值/初始化（来自当前语句右侧表达式）
        self.current_frame = 0  # 给对象属性 self.current_frame 赋值/初始化（来自当前语句右侧表达式）

    def get_person_center(self, keypoints: np.ndarray) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
        valid_keypoints = keypoints[keypoints[:, 0] > 0]  # 将表达式计算结果赋给变量 valid_keypoints
        if len(valid_keypoints) > 0:  # 条件分支判断并选择执行路径
            return np.mean(valid_keypoints, axis=0)  # 从函数返回结果
        return (0, 0)  # 从函数返回结果

    def get_head_position(self, keypoints: np.ndarray) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
        nose = keypoints[0]  # 将表达式计算结果赋给变量 nose
        left_eye = keypoints[1]  # 将表达式计算结果赋给变量 left_eye
        right_eye = keypoints[2]  # 将表达式计算结果赋给变量 right_eye
        
        head_points = []  # 初始化变量 head_points 为一个容器/表达式结果
        if nose[0] > 0 and nose[1] > 0:  # 条件分支判断并选择执行路径
            head_points.append(nose)  # 调用函数/方法执行某个动作或计算
        if left_eye[0] > 0 and left_eye[1] > 0:  # 条件分支判断并选择执行路径
            head_points.append(left_eye)  # 调用函数/方法执行某个动作或计算
        if right_eye[0] > 0 and right_eye[1] > 0:  # 条件分支判断并选择执行路径
            head_points.append(right_eye)  # 调用函数/方法执行某个动作或计算
        
        if len(head_points) > 0:  # 条件分支判断并选择执行路径
            return np.mean(head_points, axis=0)  # 从函数返回结果
        return (0, 0)  # 从函数返回结果

    def get_foot_position(self, keypoints: np.ndarray) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
        left_ankle = keypoints[15]  # 将表达式计算结果赋给变量 left_ankle
        right_ankle = keypoints[16]  # 将表达式计算结果赋给变量 right_ankle
        
        foot_points = []  # 初始化变量 foot_points 为一个容器/表达式结果
        if left_ankle[0] > 0 and left_ankle[1] > 0:  # 条件分支判断并选择执行路径
            foot_points.append(left_ankle)  # 调用函数/方法执行某个动作或计算
        if right_ankle[0] > 0 and right_ankle[1] > 0:  # 条件分支判断并选择执行路径
            foot_points.append(right_ankle)  # 调用函数/方法执行某个动作或计算
        
        if len(foot_points) > 0:  # 条件分支判断并选择执行路径
            return np.mean(foot_points, axis=0)  # 从函数返回结果
        return (0, 0)  # 从函数返回结果

    def get_shoulder_position(self, keypoints: np.ndarray) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
        left_shoulder = keypoints[5]  # 将表达式计算结果赋给变量 left_shoulder
        right_shoulder = keypoints[6]  # 将表达式计算结果赋给变量 right_shoulder
        
        shoulder_points = []  # 初始化变量 shoulder_points 为一个容器/表达式结果
        if left_shoulder[0] > 0 and left_shoulder[1] > 0:  # 条件分支判断并选择执行路径
            shoulder_points.append(left_shoulder)  # 调用函数/方法执行某个动作或计算
        if right_shoulder[0] > 0 and right_shoulder[1] > 0:  # 条件分支判断并选择执行路径
            shoulder_points.append(right_shoulder)  # 调用函数/方法执行某个动作或计算
        
        if len(shoulder_points) > 0:  # 条件分支判断并选择执行路径
            return np.mean(shoulder_points, axis=0)  # 从函数返回结果
        return (0, 0)  # 从函数返回结果

    def get_keypoint_confidence(self, keypoints: np.ndarray) -> float:  # 定义函数（封装可复用逻辑）
        valid_count = np.sum(keypoints[:, 0] > 0)  # 将 valid_count 设为一次调用/构造的返回值
        return valid_count / 17.0  # 从函数返回结果

    def calculate_velocity(self, current_pos: Tuple[float, float], track_idx: int) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
        if len(self.position_history[track_idx]) > 0:  # 条件分支判断并选择执行路径
            last_pos = self.position_history[track_idx][-1]  # 将表达式计算结果赋给变量 last_pos
            velocity = (current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])  # 初始化变量 velocity 为一个容器/表达式结果
            return velocity  # 从函数返回结果
        return (0, 0)  # 从函数返回结果

    def predict_position(self, track_idx: int) -> Optional[Tuple[float, float]]:  # 定义函数（封装可复用逻辑）
        if self.kalman_filters[track_idx] is not None:  # 条件分支判断并选择执行路径
            predicted = self.kalman_filters[track_idx].predict()  # 将 predicted 设为一次调用/构造的返回值
            return (predicted[0], predicted[1])  # 从函数返回结果
        return None  # 从函数返回结果

    def update_kalman_filter(self, track_idx: int, position: Tuple[float, float]):  # 定义函数（封装可复用逻辑）
        if self.kalman_filters[track_idx] is None:  # 条件分支判断并选择执行路径
            state = np.array([position[0], position[1], 0, 0])  # 将 state 设为一次调用/构造的返回值
            self.kalman_filters[track_idx] = KalmanFilter(state)  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self.kalman_filters[track_idx].update(np.array(position))  # 调用函数/方法执行某个动作或计算

    def calculate_match_cost(self, current_keypoints: np.ndarray, track_idx: int,   # 定义函数（封装可复用逻辑）
                            predicted_pos: Optional[Tuple[float, float]] = None) -> float:  # 执行当前语句（保持与上文逻辑一致）
        current_head = self.get_head_position(current_keypoints)  # 将 current_head 设为一次调用/构造的返回值
        current_foot = self.get_foot_position(current_keypoints)  # 将 current_foot 设为一次调用/构造的返回值
        current_shoulder = self.get_shoulder_position(current_keypoints)  # 将 current_shoulder 设为一次调用/构造的返回值
        current_center = self.get_person_center(current_keypoints)  # 将 current_center 设为一次调用/构造的返回值
        
        if len(self.position_history[track_idx]) == 0:  # 条件分支判断并选择执行路径
            return 0.0  # 从函数返回结果

        last_head = self.position_history[track_idx][-1].get('head', (0, 0))  # 将 last_head 设为一次调用/构造的返回值
        last_foot = self.position_history[track_idx][-1].get('foot', (0, 0))  # 将 last_foot 设为一次调用/构造的返回值
        last_shoulder = self.position_history[track_idx][-1].get('shoulder', (0, 0))  # 将 last_shoulder 设为一次调用/构造的返回值
        last_center = self.position_history[track_idx][-1].get('center', (0, 0))  # 将 last_center 设为一次调用/构造的返回值
        
        head_dist = np.linalg.norm(np.array(current_head) - np.array(last_head))  # 将 head_dist 设为一次调用/构造的返回值
        foot_dist = np.linalg.norm(np.array(current_foot) - np.array(last_foot))  # 将 foot_dist 设为一次调用/构造的返回值
        shoulder_dist = np.linalg.norm(np.array(current_shoulder) - np.array(last_shoulder))  # 将 shoulder_dist 设为一次调用/构造的返回值
        center_dist = np.linalg.norm(np.array(current_center) - np.array(last_center))  # 将 center_dist 设为一次调用/构造的返回值
        
        cost = 0.4 * head_dist + 0.3 * shoulder_dist + 0.2 * foot_dist + 0.1 * center_dist  # 将表达式计算结果赋给变量 cost
        
        if predicted_pos is not None:  # 条件分支判断并选择执行路径
            pred_dist = np.linalg.norm(np.array(current_center) - np.array(predicted_pos))  # 将 pred_dist 设为一次调用/构造的返回值
            cost = 0.7 * cost + 0.3 * pred_dist  # 将表达式计算结果赋给变量 cost
        
        confidence = self.get_keypoint_confidence(current_keypoints)  # 将 confidence 设为一次调用/构造的返回值
        cost = cost / (confidence + 0.01)  # 将 cost 设为一次调用/构造的返回值
        
        return cost  # 从函数返回结果

    def match_persons(self, current_keypoints: List[np.ndarray]) -> List[int]:  # 定义函数（封装可复用逻辑）
        self.current_frame += 1  # 执行当前语句（保持与上文逻辑一致）
        
        if len(current_keypoints) == 0:  # 条件分支判断并选择执行路径
            for i in range(self.max_persons):  # 循环遍历序列/迭代器
                self.last_seen_frames[i] = self.current_frame  # 执行当前语句（保持与上文逻辑一致）
            return []  # 从函数返回结果

        if self.current_frame == 1 or all(kf is None for kf in self.kalman_filters):  # 条件分支判断并选择执行路径
            for i, kp in enumerate(current_keypoints[:self.max_persons]):  # 循环遍历序列/迭代器
                center = self.get_person_center(kp)  # 将 center 设为一次调用/构造的返回值
                self.update_kalman_filter(i, center)  # 调用函数/方法执行某个动作或计算
                self.position_history[i].append({  # 执行当前语句（保持与上文逻辑一致）
                    'head': self.get_head_position(kp),  # 执行当前语句（保持与上文逻辑一致）
                    'foot': self.get_foot_position(kp),  # 执行当前语句（保持与上文逻辑一致）
                    'shoulder': self.get_shoulder_position(kp),  # 执行当前语句（保持与上文逻辑一致）
                    'center': center  # 执行当前语句（保持与上文逻辑一致）
                })  # 执行当前语句（保持与上文逻辑一致）
                self.confidence_scores[i] = self.get_keypoint_confidence(kp)  # 调用函数/方法执行某个动作或计算
                self.track_lengths[i] = 1  # 执行当前语句（保持与上文逻辑一致）
                self.last_seen_frames[i] = self.current_frame  # 执行当前语句（保持与上文逻辑一致）
            return list(range(min(len(current_keypoints), self.max_persons)))  # 从函数返回结果

        num_current = len(current_keypoints)  # 将 num_current 设为一次调用/构造的返回值
        
        if num_current == 1:  # 条件分支判断并选择执行路径
            current_center = self.get_person_center(current_keypoints[0])  # 将 current_center 设为一次调用/构造的返回值
            current_head = self.get_head_position(current_keypoints[0])  # 将 current_head 设为一次调用/构造的返回值
            
            costs = []  # 初始化变量 costs 为一个容器/表达式结果
            for i in range(self.max_persons):  # 循环遍历序列/迭代器
                if self.kalman_filters[i] is not None:  # 条件分支判断并选择执行路径
                    predicted = self.predict_position(i)  # 将 predicted 设为一次调用/构造的返回值
                    cost = self.calculate_match_cost(current_keypoints[0], i, predicted)  # 将 cost 设为一次调用/构造的返回值
                    
                    last_center = self.position_history[i][-1].get('center', (0, 0))  # 将 last_center 设为一次调用/构造的返回值
                    x_distance = abs(current_center[0] - last_center[0])  # 将 x_distance 设为一次调用/构造的返回值
                    
                    if x_distance > 100:  # 条件分支判断并选择执行路径
                        cost *= 2.0  # 执行当前语句（保持与上文逻辑一致）
                    
                    costs.append((cost, i))  # 调用函数/方法执行某个动作或计算
                else:  # 条件分支的否则路径
                    costs.append((float('inf'), i))  # 调用函数/方法执行某个动作或计算
            
            costs.sort(key=lambda x: x[0])  # 调用函数/方法执行某个动作或计算
            best_track = costs[0][1]  # 将表达式计算结果赋给变量 best_track
            
            self.update_kalman_filter(best_track, current_center)  # 调用函数/方法执行某个动作或计算
            self.position_history[best_track].append({  # 执行当前语句（保持与上文逻辑一致）
                'head': current_head,  # 执行当前语句（保持与上文逻辑一致）
                'foot': self.get_foot_position(current_keypoints[0]),  # 执行当前语句（保持与上文逻辑一致）
                'shoulder': self.get_shoulder_position(current_keypoints[0]),  # 执行当前语句（保持与上文逻辑一致）
                'center': current_center  # 执行当前语句（保持与上文逻辑一致）
            })  # 执行当前语句（保持与上文逻辑一致）
            self.confidence_scores[best_track] = self.get_keypoint_confidence(current_keypoints[0])  # 调用函数/方法执行某个动作或计算
            self.track_lengths[best_track] += 1  # 执行当前语句（保持与上文逻辑一致）
            self.last_seen_frames[best_track] = self.current_frame  # 执行当前语句（保持与上文逻辑一致）
            
            return [best_track]  # 从函数返回结果

        if num_current == 2:  # 条件分支判断并选择执行路径
            current_centers = [self.get_person_center(kp) for kp in current_keypoints]  # 初始化变量 current_centers 为一个容器/表达式结果
            current_heads = [self.get_head_position(kp) for kp in current_keypoints]  # 初始化变量 current_heads 为一个容器/表达式结果
            
            cost_matrix = np.full((2, self.max_persons), float('inf'))  # 将 cost_matrix 设为一次调用/构造的返回值
            
            for i in range(2):  # 循环遍历序列/迭代器
                for j in range(self.max_persons):  # 循环遍历序列/迭代器
                    if self.kalman_filters[j] is not None:  # 条件分支判断并选择执行路径
                        predicted = self.predict_position(j)  # 将 predicted 设为一次调用/构造的返回值
                        cost = self.calculate_match_cost(current_keypoints[i], j, predicted)  # 将 cost 设为一次调用/构造的返回值
                        
                        last_center = self.position_history[j][-1].get('center', (0, 0))  # 将 last_center 设为一次调用/构造的返回值
                        x_distance = abs(current_centers[i][0] - last_center[0])  # 将 x_distance 设为一次调用/构造的返回值
                        
                        if x_distance > 100:  # 条件分支判断并选择执行路径
                            cost *= 2.0  # 执行当前语句（保持与上文逻辑一致）
                        
                        cost_matrix[i, j] = cost  # 执行当前语句（保持与上文逻辑一致）
            
            if self.max_persons == 2:  # 条件分支判断并选择执行路径
                cost00 = cost_matrix[0, 0]  # 将表达式计算结果赋给变量 cost00
                cost01 = cost_matrix[0, 1]  # 将表达式计算结果赋给变量 cost01
                cost10 = cost_matrix[1, 0]  # 将表达式计算结果赋给变量 cost10
                cost11 = cost_matrix[1, 1]  # 将表达式计算结果赋给变量 cost11
                
                if cost00 + cost11 < cost01 + cost10:  # 条件分支判断并选择执行路径
                    assignment = [0, 1]  # 初始化变量 assignment 为一个容器/表达式结果
                else:  # 条件分支的否则路径
                    assignment = [1, 0]  # 初始化变量 assignment 为一个容器/表达式结果
            else:  # 条件分支的否则路径
                assignment = [0, 1]  # 初始化变量 assignment 为一个容器/表达式结果
            
            for i, track_idx in enumerate(assignment):  # 循环遍历序列/迭代器
                if track_idx < self.max_persons:  # 条件分支判断并选择执行路径
                    self.update_kalman_filter(track_idx, current_centers[i])  # 调用函数/方法执行某个动作或计算
                    self.position_history[track_idx].append({  # 执行当前语句（保持与上文逻辑一致）
                        'head': current_heads[i],  # 执行当前语句（保持与上文逻辑一致）
                        'foot': self.get_foot_position(current_keypoints[i]),  # 执行当前语句（保持与上文逻辑一致）
                        'shoulder': self.get_shoulder_position(current_keypoints[i]),  # 执行当前语句（保持与上文逻辑一致）
                        'center': current_centers[i]  # 执行当前语句（保持与上文逻辑一致）
                    })  # 执行当前语句（保持与上文逻辑一致）
                    self.confidence_scores[track_idx] = self.get_keypoint_confidence(current_keypoints[i])  # 调用函数/方法执行某个动作或计算
                    self.track_lengths[track_idx] += 1  # 执行当前语句（保持与上文逻辑一致）
                    self.last_seen_frames[track_idx] = self.current_frame  # 执行当前语句（保持与上文逻辑一致）
            
            return assignment  # 从函数返回结果

        return list(range(min(num_current, self.max_persons)))  # 从函数返回结果


def track_poses(poses_list: List[List[np.ndarray]], max_persons: int = 2) -> np.ndarray:  # 定义函数（封装可复用逻辑）
    tracker = PersonTracker(max_persons=max_persons)  # 将 tracker 设为一次调用/构造的返回值
    total_frames = len(poses_list)  # 将 total_frames 设为一次调用/构造的返回值
    poses_array = np.zeros((total_frames, max_persons, 17, 2), dtype=np.float32)  # 将 poses_array 设为一次调用/构造的返回值

    for frame_idx, frame_poses in enumerate(poses_list):  # 循环遍历序列/迭代器
        if len(frame_poses) == 0:  # 条件分支判断并选择执行路径
            continue  # 控制流语句：改变当前代码块的执行方式

        assignment = tracker.match_persons(frame_poses)  # 将 assignment 设为一次调用/构造的返回值

        for person_idx, original_idx in enumerate(assignment):  # 循环遍历序列/迭代器
            if person_idx < max_persons and original_idx < len(frame_poses):  # 条件分支判断并选择执行路径
                poses_array[frame_idx, person_idx] = frame_poses[original_idx][:17, :2]  # 执行当前语句（保持与上文逻辑一致）

    return poses_array  # 从函数返回结果
