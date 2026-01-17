import numpy as np  # 导入模块，供后续使用
from typing import List, Tuple, Optional  # 从模块导入符号，供后续调用


class CourtBasedPlayerAssigner:  # 定义类（封装数据与行为）
    def __init__(self, frame_height: int, frame_width: int, court_info: Optional[List[float]] = None,   # 定义函数（封装可复用逻辑）
                 extended_court_points: Optional[np.ndarray] = None):  # 执行当前语句（保持与上文逻辑一致）
        self.frame_height = frame_height  # 给对象属性 self.frame_height 赋值/初始化（来自当前语句右侧表达式）
        self.frame_width = frame_width  # 给对象属性 self.frame_width 赋值/初始化（来自当前语句右侧表达式）
        self.court_info = court_info  # 给对象属性 self.court_info 赋值/初始化（来自当前语句右侧表达式）
        self.extended_court_points = extended_court_points  # 给对象属性 self.extended_court_points 赋值/初始化（来自当前语句右侧表达式）
        
        if court_info is not None and len(court_info) >= 5:  # 条件分支判断并选择执行路径
            self.net_y = court_info[4]  # 给对象属性 self.net_y 赋值/初始化（来自当前语句右侧表达式）
        else:  # 条件分支的否则路径
            self.net_y = frame_height / 2  # 给对象属性 self.net_y 赋值/初始化（来自当前语句右侧表达式）

    def set_court_info(self, court_info: List[float], extended_court_points: Optional[np.ndarray] = None):  # 定义函数（封装可复用逻辑）
        self.court_info = court_info  # 给对象属性 self.court_info 赋值/初始化（来自当前语句右侧表达式）
        if len(court_info) >= 5:  # 条件分支判断并选择执行路径
            self.net_y = court_info[4]  # 给对象属性 self.net_y 赋值/初始化（来自当前语句右侧表达式）
        if extended_court_points is not None:  # 条件分支判断并选择执行路径
            self.extended_court_points = extended_court_points  # 给对象属性 self.extended_court_points 赋值/初始化（来自当前语句右侧表达式）

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

    def get_person_center(self, keypoints: np.ndarray) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
        valid_keypoints = keypoints[keypoints[:, 0] > 0]  # 将表达式计算结果赋给变量 valid_keypoints
        if len(valid_keypoints) > 0:  # 条件分支判断并选择执行路径
            return np.mean(valid_keypoints, axis=0)  # 从函数返回结果
        return (0, 0)  # 从函数返回结果

    def is_in_top_half(self, keypoints: np.ndarray) -> bool:  # 定义函数（封装可复用逻辑）
        foot_pos = self.get_foot_position(keypoints)  # 将 foot_pos 设为一次调用/构造的返回值
        if foot_pos[0] == 0 and foot_pos[1] == 0:  # 条件分支判断并选择执行路径
            center_pos = self.get_person_center(keypoints)  # 将 center_pos 设为一次调用/构造的返回值
            if center_pos[0] == 0 and center_pos[1] == 0:  # 条件分支判断并选择执行路径
                return False  # 从函数返回结果
            return center_pos[1] < self.net_y  # 从函数返回结果
        return foot_pos[1] < self.net_y  # 从函数返回结果

    def is_in_bottom_half(self, keypoints: np.ndarray) -> bool:  # 定义函数（封装可复用逻辑）
        foot_pos = self.get_foot_position(keypoints)  # 将 foot_pos 设为一次调用/构造的返回值
        if foot_pos[0] == 0 and foot_pos[1] == 0:  # 条件分支判断并选择执行路径
            center_pos = self.get_person_center(keypoints)  # 将 center_pos 设为一次调用/构造的返回值
            if center_pos[0] == 0 and center_pos[1] == 0:  # 条件分支判断并选择执行路径
                return False  # 从函数返回结果
            return center_pos[1] > self.net_y  # 从函数返回结果
        return foot_pos[1] > self.net_y  # 从函数返回结果

    def is_in_court(self, keypoints: np.ndarray) -> bool:  # 定义函数（封装可复用逻辑）
        if self.court_info is None or self.extended_court_points is None:  # 条件分支判断并选择执行路径
            return True  # 从函数返回结果

        l_a = self.court_info[0]  # 将表达式计算结果赋给变量 l_a
        l_b = self.court_info[1]  # 将表达式计算结果赋给变量 l_b
        r_a = self.court_info[2]  # 将表达式计算结果赋给变量 r_a
        r_b = self.court_info[3]  # 将表达式计算结果赋给变量 r_b

        ankle_x = (keypoints[15][0] + keypoints[16][0]) / 2  # 初始化变量 ankle_x 为一个容器/表达式结果
        ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2  # 初始化变量 ankle_y 为一个容器/表达式结果

        top = ankle_y > self.extended_court_points[0][1]  # 将表达式计算结果赋给变量 top
        bottom = ankle_y < self.extended_court_points[5][1]  # 将表达式计算结果赋给变量 bottom

        lmp_x = (ankle_y - l_b) / l_a  # 初始化变量 lmp_x 为一个容器/表达式结果
        rmp_x = (ankle_y - r_b) / r_a  # 初始化变量 rmp_x 为一个容器/表达式结果
        left = ankle_x > lmp_x  # 将表达式计算结果赋给变量 left
        right = ankle_x < rmp_x  # 将表达式计算结果赋给变量 right

        if left and right and top and bottom:  # 条件分支判断并选择执行路径
            return True  # 从函数返回结果
        else:  # 条件分支的否则路径
            return False  # 从函数返回结果

    def assign_players(self, keypoints_list: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:  # 定义函数（封装可复用逻辑）
        if len(keypoints_list) < 2:  # 条件分支判断并选择执行路径
            return None, None  # 从函数返回结果

        valid_indices = []  # 初始化变量 valid_indices 为一个容器/表达式结果
        for i, keypoints in enumerate(keypoints_list):  # 循环遍历序列/迭代器
            if self.is_in_court(keypoints):  # 条件分支判断并选择执行路径
                valid_indices.append(i)  # 调用函数/方法执行某个动作或计算

        if len(valid_indices) < 2:  # 条件分支判断并选择执行路径
            return None, None  # 从函数返回结果

        valid_keypoints = [keypoints_list[i] for i in valid_indices]  # 初始化变量 valid_keypoints 为一个容器/表达式结果

        top_player = None  # 将表达式计算结果赋给变量 top_player
        bottom_player = None  # 将表达式计算结果赋给变量 bottom_player

        for keypoints in valid_keypoints:  # 循环遍历序列/迭代器
            if self.is_in_top_half(keypoints):  # 条件分支判断并选择执行路径
                if top_player is None:  # 条件分支判断并选择执行路径
                    top_player = keypoints  # 将表达式计算结果赋给变量 top_player
            elif self.is_in_bottom_half(keypoints):  # 条件分支判断并选择执行路径
                if bottom_player is None:  # 条件分支判断并选择执行路径
                    bottom_player = keypoints  # 将表达式计算结果赋给变量 bottom_player

        if top_player is None and bottom_player is None:  # 条件分支判断并选择执行路径
            centers = [self.get_person_center(kp) for kp in valid_keypoints]  # 初始化变量 centers 为一个容器/表达式结果
            centers = [(i, c) for i, c in enumerate(centers) if c[0] > 0 or c[1] > 0]  # 初始化变量 centers 为一个容器/表达式结果
            centers.sort(key=lambda x: x[1][1])  # 调用函数/方法执行某个动作或计算
            
            if len(centers) >= 2:  # 条件分支判断并选择执行路径
                top_player = valid_keypoints[centers[0][0]]  # 将表达式计算结果赋给变量 top_player
                bottom_player = valid_keypoints[centers[1][0]]  # 将表达式计算结果赋给变量 bottom_player

        return top_player, bottom_player  # 从函数返回结果

    def assign_players_with_indices(self, keypoints_list: List[np.ndarray]) -> Tuple[Optional[int], Optional[int]]:  # 定义函数（封装可复用逻辑）
        if len(keypoints_list) < 2:  # 条件分支判断并选择执行路径
            return None, None  # 从函数返回结果

        top_idx = None  # 将表达式计算结果赋给变量 top_idx
        bottom_idx = None  # 将表达式计算结果赋给变量 bottom_idx

        for i, keypoints in enumerate(keypoints_list):  # 循环遍历序列/迭代器
            if self.is_in_top_half(keypoints):  # 条件分支判断并选择执行路径
                if top_idx is None:  # 条件分支判断并选择执行路径
                    top_idx = i  # 将表达式计算结果赋给变量 top_idx
            elif self.is_in_bottom_half(keypoints):  # 条件分支判断并选择执行路径
                if bottom_idx is None:  # 条件分支判断并选择执行路径
                    bottom_idx = i  # 将表达式计算结果赋给变量 bottom_idx

        if top_idx is None and bottom_idx is None:  # 条件分支判断并选择执行路径
            centers = [(i, self.get_person_center(kp)) for i, kp in enumerate(keypoints_list)]  # 初始化变量 centers 为一个容器/表达式结果
            centers = [(i, c) for i, c in centers if c[0] > 0 or c[1] > 0]  # 初始化变量 centers 为一个容器/表达式结果
            centers.sort(key=lambda x: x[1][1])  # 调用函数/方法执行某个动作或计算
            
            if len(centers) >= 2:  # 条件分支判断并选择执行路径
                top_idx = centers[0][0]  # 将表达式计算结果赋给变量 top_idx
                bottom_idx = centers[1][0]  # 将表达式计算结果赋给变量 bottom_idx

        return top_idx, bottom_idx  # 从函数返回结果


def assign_players_court_based(poses_list: List[List[np.ndarray]],   # 定义函数（封装可复用逻辑）
                                frame_height: int,   # 执行当前语句（保持与上文逻辑一致）
                                frame_width: int) -> np.ndarray:  # 执行当前语句（保持与上文逻辑一致）
    assigner = CourtBasedPlayerAssigner(frame_height, frame_width)  # 将 assigner 设为一次调用/构造的返回值
    total_frames = len(poses_list)  # 将 total_frames 设为一次调用/构造的返回值
    poses_array = np.zeros((total_frames, 2, 17, 2), dtype=np.float32)  # 将 poses_array 设为一次调用/构造的返回值

    for frame_idx, frame_poses in enumerate(poses_list):  # 循环遍历序列/迭代器
        if len(frame_poses) < 2:  # 条件分支判断并选择执行路径
            continue  # 控制流语句：改变当前代码块的执行方式

        top_player, bottom_player = assigner.assign_players(frame_poses)  # 调用函数/方法执行某个动作或计算

        if top_player is not None:  # 条件分支判断并选择执行路径
            poses_array[frame_idx, 0] = top_player[:17, :2]  # 执行当前语句（保持与上文逻辑一致）
        if bottom_player is not None:  # 条件分支判断并选择执行路径
            poses_array[frame_idx, 1] = bottom_player[:17, :2]  # 执行当前语句（保持与上文逻辑一致）

    return poses_array  # 从函数返回结果
