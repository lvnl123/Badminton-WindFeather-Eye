import cv2  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
from pathlib import Path  # 从模块导入符号，供后续调用
from typing import List, Tuple, Optional  # 从模块导入符号，供后续调用
from tqdm import tqdm  # 从模块导入符号，供后续调用
from PIL import Image, ImageDraw, ImageFont  # 从模块导入符号，供后续调用


def _put_chinese_text(frame: np.ndarray, text: str, position: Tuple[int, int],   # 定义函数（封装可复用逻辑）
                      font_size: int = 40, color: Tuple[int, int, int] = (0, 255, 0),   # 执行当前语句（保持与上文逻辑一致）
                      font_path: str = None) -> np.ndarray:  # 执行当前语句（保持与上文逻辑一致）
    if font_path is None:  # 条件分支判断并选择执行路径
        font_path = r"C:\Windows\Fonts\msyh.ttc"  # 将表达式计算结果赋给变量 font_path
    
    try:  # 开始异常捕获保护块
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将 frame_rgb 设为一次调用/构造的返回值
        pil_image = Image.fromarray(frame_rgb)  # 将 pil_image 设为一次调用/构造的返回值
        draw = ImageDraw.Draw(pil_image)  # 将 draw 设为一次调用/构造的返回值
        
        try:  # 开始异常捕获保护块
            font = ImageFont.truetype(font_path, font_size)  # 将 font 设为一次调用/构造的返回值
        except:  # 捕获异常并进行处理
            font = ImageFont.load_default()  # 将 font 设为一次调用/构造的返回值
        
        draw.text(position, text, font=font, fill=color)  # 调用函数/方法执行某个动作或计算
        
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # 将 frame_bgr 设为一次调用/构造的返回值
        return frame_bgr  # 从函数返回结果
    except Exception as e:  # 捕获异常并进行处理
        font = cv2.FONT_HERSHEY_SIMPLEX  # 将表达式计算结果赋给变量 font
        cv2.putText(frame, text, position, font, font_size / 30, color, 3, cv2.LINE_AA)  # 调用函数/方法执行某个动作或计算
        return frame  # 从函数返回结果


def visualize_combined(  # 定义函数（封装可复用逻辑）
    video_path: str,  # 执行当前语句（保持与上文逻辑一致）
    ball_positions: List[Tuple[int, int]],  # 执行当前语句（保持与上文逻辑一致）
    poses: np.ndarray,  # 执行当前语句（保持与上文逻辑一致）
    output_path: str,  # 执行当前语句（保持与上文逻辑一致）
    traj_len: int = 10,  # 执行当前语句（保持与上文逻辑一致）
    skeleton_pairs=None,  # 将表达式计算结果赋给变量 skeleton_pairs
    court_keypoints=None,  # 将表达式计算结果赋给变量 court_keypoints
    partitioned_keypoints=None,  # 将表达式计算结果赋给变量 partitioned_keypoints
    net_keypoints: Optional[List[List[int]]] = None,  # 执行当前语句（保持与上文逻辑一致）
    hit_frames: Optional[List[int]] = None,  # 执行当前语句（保持与上文逻辑一致）
    per_frame_court_keypoints: Optional[List[Optional[List[List[int]]]]] = None,  # 执行当前语句（保持与上文逻辑一致）
    per_frame_net_keypoints: Optional[List[Optional[List[List[int]]]]] = None,  # 执行当前语句（保持与上文逻辑一致）
    stroke_types: Optional[List[str]] = None,  # 执行当前语句（保持与上文逻辑一致）
    keep_player_skeleton: bool = True,  # 执行当前语句（保持与上文逻辑一致）
    keep_ball_trajectory: bool = True,  # 执行当前语句（保持与上文逻辑一致）
    keep_stroke_type_hint: bool = True,  # 执行当前语句（保持与上文逻辑一致）
    frame_callback=None,  # 将表达式计算结果赋给变量 frame_callback
    progress_callback=None,  # 将表达式计算结果赋给变量 progress_callback
    emit_every_n_frames: int = 1  # 执行当前语句（保持与上文逻辑一致）
):  # 执行当前语句（保持与上文逻辑一致）
    if skeleton_pairs is None:  # 条件分支判断并选择执行路径
        skeleton_pairs = [  # 初始化变量 skeleton_pairs 为一个容器/表达式结果
            (0, 1), (0, 2), (1, 3), (2, 4),  # 执行当前语句（保持与上文逻辑一致）
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 执行当前语句（保持与上文逻辑一致）
            (5, 11), (6, 12), (11, 12),  # 执行当前语句（保持与上文逻辑一致）
            (11, 13), (13, 15), (12, 14), (14, 16)  # 调用函数/方法执行某个动作或计算
        ]  # 执行当前语句（保持与上文逻辑一致）

    cap = cv2.VideoCapture(video_path)  # 将 cap 设为一次调用/构造的返回值
    if not cap.isOpened():  # 条件分支判断并选择执行路径
        raise ValueError(f"Cannot open video: {video_path}")  # 调用函数/方法执行某个动作或计算

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 将 width 设为一次调用/构造的返回值
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 将 height 设为一次调用/构造的返回值
    fps = cap.get(cv2.CAP_PROP_FPS)  # 将 fps 设为一次调用/构造的返回值

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 将 fourcc 设为一次调用/构造的返回值
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # 将 out 设为一次调用/构造的返回值

    court_zones = None  # 将表达式计算结果赋给变量 court_zones
    
    current_partitioned_keypoints = partitioned_keypoints  # 将表达式计算结果赋给变量 current_partitioned_keypoints
    current_net_keypoints = net_keypoints  # 将表达式计算结果赋给变量 current_net_keypoints
    
    if per_frame_court_keypoints is not None and len(per_frame_court_keypoints) > 0:  # 条件分支判断并选择执行路径
        if per_frame_court_keypoints[0] is not None:  # 条件分支判断并选择执行路径
            court_zones = _extract_court_zones(per_frame_court_keypoints[0])  # 将 court_zones 设为一次调用/构造的返回值
    elif partitioned_keypoints is not None:  # 条件分支判断并选择执行路径
        court_zones = _extract_court_zones(partitioned_keypoints)  # 将 court_zones 设为一次调用/构造的返回值
    
    ball_speeds = _calculate_ball_speeds(ball_positions, fps)  # 将 ball_speeds 设为一次调用/构造的返回值
    speed_thresholds = _calculate_speed_thresholds(ball_speeds)  # 将 speed_thresholds 设为一次调用/构造的返回值

    frame_idx = 0  # 将表达式计算结果赋给变量 frame_idx
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 将 total_frames 设为一次调用/构造的返回值

    with tqdm(total=total_frames, desc="Creating combined visualization") as pbar:  # 上下文管理：确保资源正确释放
        while True:  # 条件循环，直到条件不满足
            ret, frame = cap.read()  # 调用函数/方法执行某个动作或计算
            if not ret:  # 条件分支判断并选择执行路径
                break  # 控制流语句：改变当前代码块的执行方式
            
            if per_frame_court_keypoints is not None and frame_idx < len(per_frame_court_keypoints):  # 条件分支判断并选择执行路径
                current_partitioned_keypoints = per_frame_court_keypoints[frame_idx]  # 将表达式计算结果赋给变量 current_partitioned_keypoints
            else:  # 条件分支的否则路径
                current_partitioned_keypoints = partitioned_keypoints  # 将表达式计算结果赋给变量 current_partitioned_keypoints
            
            if per_frame_net_keypoints is not None and frame_idx < len(per_frame_net_keypoints):  # 条件分支判断并选择执行路径
                current_net_keypoints = per_frame_net_keypoints[frame_idx]  # 将表达式计算结果赋给变量 current_net_keypoints
            else:  # 条件分支的否则路径
                current_net_keypoints = net_keypoints  # 将表达式计算结果赋给变量 current_net_keypoints
            
            if current_partitioned_keypoints is not None:  # 条件分支判断并选择执行路径
                court_zones = _extract_court_zones(current_partitioned_keypoints)  # 将 court_zones 设为一次调用/构造的返回值
            else:  # 条件分支的否则路径
                court_zones = None  # 将表达式计算结果赋给变量 court_zones

            if current_partitioned_keypoints is not None or current_net_keypoints is not None:  # 条件分支判断并选择执行路径
                if court_zones is not None and frame_idx < poses.shape[0]:  # 条件分支判断并选择执行路径
                    _highlight_player_zones(frame, poses[frame_idx], court_zones)  # 调用函数/方法执行某个动作或计算

                if current_partitioned_keypoints is not None:  # 条件分支判断并选择执行路径
                    c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4],  # 初始化变量 c_edges 为一个容器/表达式结果
                               [3, 8], [4, 9], [5, 6], [5, 10], [6, 7], [6, 11], [7, 8],  # 执行当前语句（保持与上文逻辑一致）
                               [7, 12], [8, 9], [8, 13], [9, 14], [10, 11], [10, 15],  # 执行当前语句（保持与上文逻辑一致）
                               [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],  # 执行当前语句（保持与上文逻辑一致）
                               [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18],  # 执行当前语句（保持与上文逻辑一致）
                               [17, 22], [18, 19], [18, 23], [19, 24], [20, 21], [20, 25],  # 执行当前语句（保持与上文逻辑一致）
                               [21, 22], [21, 26], [22, 23], [22, 27], [23, 24], [23, 28],  # 执行当前语句（保持与上文逻辑一致）
                               [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],  # 执行当前语句（保持与上文逻辑一致）
                               [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32],  # 执行当前语句（保持与上文逻辑一致）
                               [32, 33], [33, 34]]  # 执行当前语句（保持与上文逻辑一致）
                    court_color_edge = (53, 195, 242)  # 初始化变量 court_color_edge 为一个容器/表达式结果
                    court_color_kps = (5, 135, 242)  # 初始化变量 court_color_kps 为一个容器/表达式结果

                    for e in c_edges:  # 循环遍历序列/迭代器
                        cv2.line(frame, (int(current_partitioned_keypoints[e[0]][0]),  # 执行当前语句（保持与上文逻辑一致）
                                         int(current_partitioned_keypoints[e[0]][1])),  # 执行当前语句（保持与上文逻辑一致）
                                 (int(current_partitioned_keypoints[e[1]][0]),  # 执行当前语句（保持与上文逻辑一致）
                                  int(current_partitioned_keypoints[e[1]][1])),  # 执行当前语句（保持与上文逻辑一致）
                                 court_color_edge, 2, lineType=cv2.LINE_AA)  # 执行当前语句（保持与上文逻辑一致）

                    for kp in current_partitioned_keypoints:  # 循环遍历序列/迭代器
                        cv2.circle(frame, tuple(kp), 1, court_color_kps, 5)  # 调用函数/方法执行某个动作或计算

                if current_net_keypoints is not None:  # 条件分支判断并选择执行路径
                    net_edges = [[0, 1], [2, 3], [0, 4], [1, 5]]  # 初始化变量 net_edges 为一个容器/表达式结果
                    net_color_edge = (255, 165, 0)  # 初始化变量 net_color_edge 为一个容器/表达式结果
                    net_color_kps = (255, 140, 0)  # 初始化变量 net_color_kps 为一个容器/表达式结果

                    for e in net_edges:  # 循环遍历序列/迭代器
                        cv2.line(frame, (int(current_net_keypoints[e[0]][0]),  # 执行当前语句（保持与上文逻辑一致）
                                         int(current_net_keypoints[e[0]][1])),  # 执行当前语句（保持与上文逻辑一致）
                                 (int(current_net_keypoints[e[1]][0]),  # 执行当前语句（保持与上文逻辑一致）
                                  int(current_net_keypoints[e[1]][1])),  # 执行当前语句（保持与上文逻辑一致）
                                 net_color_edge, 2, lineType=cv2.LINE_AA)  # 执行当前语句（保持与上文逻辑一致）

                    for kp in current_net_keypoints:  # 循环遍历序列/迭代器
                        cv2.circle(frame, tuple(kp), 1, net_color_kps, 5)  # 调用函数/方法执行某个动作或计算

            if frame_idx < len(ball_positions) and frame_idx < poses.shape[0]:  # 条件分支判断并选择执行路径
                ball_pos = ball_positions[frame_idx]  # 将表达式计算结果赋给变量 ball_pos
                
                if keep_ball_trajectory:  # 条件分支判断并选择执行路径
                    if ball_pos is not None and ball_pos[0] > 0 and ball_pos[1] > 0:  # 条件分支判断并选择执行路径
                        start_idx = max(0, frame_idx - traj_len)  # 将 start_idx 设为一次调用/构造的返回值
                        for j in range(start_idx, frame_idx):  # 循环遍历序列/迭代器
                            prev_ball = ball_positions[j]  # 将表达式计算结果赋给变量 prev_ball
                            if prev_ball is not None and prev_ball[0] > 0 and prev_ball[1] > 0:  # 条件分支判断并选择执行路径
                                alpha = 1.0 - ((frame_idx - j) / traj_len)  # 将 alpha 设为一次调用/构造的返回值
                                speed = ball_speeds[j] if j < len(ball_speeds) else 0  # 将 speed 设为一次调用/构造的返回值
                                color = _get_speed_color(speed, speed_thresholds, alpha)  # 将 color 设为一次调用/构造的返回值
                                cv2.circle(frame, tuple(map(int, prev_ball)), 4, color, -1)  # 调用函数/方法执行某个动作或计算
                        
                        current_speed = ball_speeds[frame_idx] if frame_idx < len(ball_speeds) else 0  # 将 current_speed 设为一次调用/构造的返回值
                        current_color = _get_speed_color(current_speed, speed_thresholds, 1.0)  # 将 current_color 设为一次调用/构造的返回值
                        cv2.circle(frame, tuple(map(int, ball_pos)), 8, current_color, -1)  # 调用函数/方法执行某个动作或计算
                        cv2.circle(frame, tuple(map(int, ball_pos)), 12, (255, 255, 255), 2)  # 调用函数/方法执行某个动作或计算
                
                if keep_player_skeleton:  # 条件分支判断并选择执行路径
                    for person_idx in range(2):  # 循环遍历序列/迭代器
                        person_poses = poses[frame_idx, person_idx]  # 将表达式计算结果赋给变量 person_poses
                        
                        for start_idx, end_idx in skeleton_pairs:  # 循环遍历序列/迭代器
                            pt1 = person_poses[start_idx]  # 将表达式计算结果赋给变量 pt1
                            pt2 = person_poses[end_idx]  # 将表达式计算结果赋给变量 pt2
                            
                            if np.any(pt1) and np.any(pt2):  # 条件分支判断并选择执行路径
                                color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)  # 初始化变量 color 为一个容器/表达式结果
                                pt1 = tuple(map(int, pt1))  # 将 pt1 设为一次调用/构造的返回值
                                pt2 = tuple(map(int, pt2))  # 将 pt2 设为一次调用/构造的返回值
                                cv2.line(frame, pt1, pt2, color, 4)  # 调用函数/方法执行某个动作或计算

                        for joint_idx in range(17):  # 循环遍历序列/迭代器
                            joint = person_poses[joint_idx]  # 将表达式计算结果赋给变量 joint
                            if np.any(joint):  # 条件分支判断并选择执行路径
                                color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)  # 初始化变量 color 为一个容器/表达式结果
                                joint = tuple(map(int, joint))  # 将 joint 设为一次调用/构造的返回值
                                cv2.circle(frame, joint, 5, color, -1)  # 调用函数/方法执行某个动作或计算

            if hit_frames is not None:  # 条件分支判断并选择执行路径
                current_hit_count = sum(1 for hit_frame in hit_frames if hit_frame <= frame_idx)  # 将 current_hit_count 设为一次调用/构造的返回值
                text = f"Hits: {current_hit_count}"  # 将表达式计算结果赋给变量 text
                font = cv2.FONT_HERSHEY_SIMPLEX  # 将表达式计算结果赋给变量 font
                font_scale = 1.5  # 将表达式计算结果赋给变量 font_scale
                thickness = 3  # 将表达式计算结果赋给变量 thickness
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]  # 将 text_size 设为一次调用/构造的返回值
                text_x = width - text_size[0] - 20  # 将表达式计算结果赋给变量 text_x
                text_y = 60  # 将表达式计算结果赋给变量 text_y
                
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)  # 调用函数/方法执行某个动作或计算

            if keep_stroke_type_hint and stroke_types is not None and hit_frames is not None:  # 条件分支判断并选择执行路径
                current_stroke_type = None  # 将表达式计算结果赋给变量 current_stroke_type
                for hit_frame, stroke_type in zip(hit_frames, stroke_types):  # 循环遍历序列/迭代器
                    if hit_frame <= frame_idx:  # 条件分支判断并选择执行路径
                        current_stroke_type = stroke_type  # 将表达式计算结果赋给变量 current_stroke_type
                
                if current_stroke_type is not None:  # 条件分支判断并选择执行路径
                    stroke_text = f"击球类型: {current_stroke_type}"  # 将表达式计算结果赋给变量 stroke_text
                    frame = _put_chinese_text(frame, stroke_text, (20, 20), font_size=40, color=(0, 255, 0))  # 将 frame 设为一次调用/构造的返回值

            if frame_callback is not None and emit_every_n_frames > 0 and (frame_idx % emit_every_n_frames == 0):  # 条件分支判断并选择执行路径
                frame_callback(frame_idx, frame)  # 调用函数/方法执行某个动作或计算

            out.write(frame)  # 调用函数/方法执行某个动作或计算
            frame_idx += 1  # 执行当前语句（保持与上文逻辑一致）
            pbar.update(1)  # 调用函数/方法执行某个动作或计算
            if progress_callback is not None:  # 条件分支判断并选择执行路径
                progress_callback(frame_idx, total_frames)  # 调用函数/方法执行某个动作或计算

    cap.release()  # 调用函数/方法执行某个动作或计算
    out.release()  # 调用函数/方法执行某个动作或计算
    print(f"Combined visualization saved to {output_path}")  # 调用函数/方法执行某个动作或计算


def _extract_court_zones(partitioned_keypoints: List[List[int]]) -> List[List[Tuple[int, int]]]:  # 定义函数（封装可复用逻辑）
    c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4],  # 初始化变量 c_edges 为一个容器/表达式结果
               [3, 8], [4, 9], [5, 6], [5, 10], [6, 7], [6, 11], [7, 8],  # 执行当前语句（保持与上文逻辑一致）
               [7, 12], [8, 9], [8, 13], [9, 14], [10, 11], [10, 15],  # 执行当前语句（保持与上文逻辑一致）
               [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],  # 执行当前语句（保持与上文逻辑一致）
               [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18],  # 执行当前语句（保持与上文逻辑一致）
               [17, 22], [18, 19], [18, 23], [19, 24], [20, 21], [20, 25],  # 执行当前语句（保持与上文逻辑一致）
               [21, 22], [21, 26], [22, 23], [22, 27], [23, 24], [23, 28],  # 执行当前语句（保持与上文逻辑一致）
               [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],  # 执行当前语句（保持与上文逻辑一致）
               [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32],  # 执行当前语句（保持与上文逻辑一致）
               [32, 33], [33, 34]]  # 执行当前语句（保持与上文逻辑一致）

    zones = []  # 初始化变量 zones 为一个容器/表达式结果
    zone_edges = [  # 初始化变量 zone_edges 为一个容器/表达式结果
        [0, 5, 6, 1],  # 执行当前语句（保持与上文逻辑一致）
        [5, 10, 11, 6],  # 执行当前语句（保持与上文逻辑一致）
        [10, 15, 16, 11],  # 执行当前语句（保持与上文逻辑一致）
        [15, 20, 21, 16],  # 执行当前语句（保持与上文逻辑一致）
        [20, 25, 26, 21],  # 执行当前语句（保持与上文逻辑一致）
        [25, 30, 31, 26],  # 执行当前语句（保持与上文逻辑一致）
        [30, 31, 32, 33],  # 执行当前语句（保持与上文逻辑一致）
        [31, 26, 27, 32],  # 执行当前语句（保持与上文逻辑一致）
        [26, 21, 22, 27],  # 执行当前语句（保持与上文逻辑一致）
        [21, 16, 17, 22],  # 执行当前语句（保持与上文逻辑一致）
        [16, 11, 12, 17],  # 执行当前语句（保持与上文逻辑一致）
        [11, 6, 7, 12],  # 执行当前语句（保持与上文逻辑一致）
        [6, 1, 2, 7],  # 执行当前语句（保持与上文逻辑一致）
        [1, 2, 3, 8],  # 执行当前语句（保持与上文逻辑一致）
        [2, 3, 4, 9],  # 执行当前语句（保持与上文逻辑一致）
        [3, 8, 9, 4],  # 执行当前语句（保持与上文逻辑一致）
        [8, 13, 14, 9],  # 执行当前语句（保持与上文逻辑一致）
        [13, 18, 19, 14],  # 执行当前语句（保持与上文逻辑一致）
        [18, 23, 24, 19],  # 执行当前语句（保持与上文逻辑一致）
        [23, 28, 29, 24],  # 执行当前语句（保持与上文逻辑一致）
        [28, 33, 34, 29],  # 执行当前语句（保持与上文逻辑一致）
        [33, 32, 27, 28],  # 执行当前语句（保持与上文逻辑一致）
        [32, 27, 22, 23],  # 执行当前语句（保持与上文逻辑一致）
        [27, 22, 17, 18],  # 执行当前语句（保持与上文逻辑一致）
        [22, 17, 12, 13],  # 执行当前语句（保持与上文逻辑一致）
        [17, 12, 7, 8]  # 执行当前语句（保持与上文逻辑一致）
    ]  # 执行当前语句（保持与上文逻辑一致）

    for zone in zone_edges:  # 循环遍历序列/迭代器
        zone_points = []  # 初始化变量 zone_points 为一个容器/表达式结果
        for idx in zone:  # 循环遍历序列/迭代器
            zone_points.append((partitioned_keypoints[idx][0], partitioned_keypoints[idx][1]))  # 调用函数/方法执行某个动作或计算
        zones.append(zone_points)  # 调用函数/方法执行某个动作或计算

    return zones  # 从函数返回结果


def _point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:  # 定义函数（封装可复用逻辑）
    x, y = point  # 执行当前语句（保持与上文逻辑一致）
    n = len(polygon)  # 将 n 设为一次调用/构造的返回值
    inside = False  # 将表达式计算结果赋给变量 inside

    p1x, p1y = polygon[0]  # 执行当前语句（保持与上文逻辑一致）
    for i in range(n + 1):  # 循环遍历序列/迭代器
        p2x, p2y = polygon[i % n]  # 执行当前语句（保持与上文逻辑一致）
        if y > min(p1y, p2y):  # 条件分支判断并选择执行路径
            if y <= max(p1y, p2y):  # 条件分支判断并选择执行路径
                if x <= max(p1x, p2x):  # 条件分支判断并选择执行路径
                    if p1y != p2y:  # 条件分支判断并选择执行路径
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x  # 初始化变量 xinters 为一个容器/表达式结果
                    if p1x == p2x or x <= xinters:  # 条件分支判断并选择执行路径
                        inside = not inside  # 将表达式计算结果赋给变量 inside
        p1x, p1y = p2x, p2y  # 执行当前语句（保持与上文逻辑一致）

    return inside  # 从函数返回结果


def _highlight_player_zones(frame: np.ndarray, poses: np.ndarray, court_zones: List[List[Tuple[int, int]]]):  # 定义函数（封装可复用逻辑）
    for person_idx in range(2):  # 循环遍历序列/迭代器
        person_poses = poses[person_idx]  # 将表达式计算结果赋给变量 person_poses
        
        ankle_left = person_poses[15]  # 将表达式计算结果赋给变量 ankle_left
        ankle_right = person_poses[16]  # 将表达式计算结果赋给变量 ankle_right
        
        color = (0, 255, 0) if person_idx == 0 else (255, 165, 0)  # 初始化变量 color 为一个容器/表达式结果
        
        if np.any(ankle_left):  # 条件分支判断并选择执行路径
            ankle_left_point = (int(ankle_left[0]), int(ankle_left[1]))  # 初始化变量 ankle_left_point 为一个容器/表达式结果
            
            for zone in court_zones:  # 循环遍历序列/迭代器
                if _point_in_polygon(ankle_left_point, zone):  # 条件分支判断并选择执行路径
                    overlay = frame.copy()  # 将 overlay 设为一次调用/构造的返回值
                    pts = np.array(zone, np.int32)  # 将 pts 设为一次调用/构造的返回值
                    pts = pts.reshape((-1, 1, 2))  # 将 pts 设为一次调用/构造的返回值
                    cv2.fillPoly(overlay, [pts], color)  # 调用函数/方法执行某个动作或计算
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # 调用函数/方法执行某个动作或计算
        
        if np.any(ankle_right):  # 条件分支判断并选择执行路径
            ankle_right_point = (int(ankle_right[0]), int(ankle_right[1]))  # 初始化变量 ankle_right_point 为一个容器/表达式结果
            
            for zone in court_zones:  # 循环遍历序列/迭代器
                if _point_in_polygon(ankle_right_point, zone):  # 条件分支判断并选择执行路径
                    overlay = frame.copy()  # 将 overlay 设为一次调用/构造的返回值
                    pts = np.array(zone, np.int32)  # 将 pts 设为一次调用/构造的返回值
                    pts = pts.reshape((-1, 1, 2))  # 将 pts 设为一次调用/构造的返回值
                    cv2.fillPoly(overlay, [pts], color)  # 调用函数/方法执行某个动作或计算
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # 调用函数/方法执行某个动作或计算


def _calculate_ball_speeds(ball_positions: List[Tuple[int, int]], fps: float) -> List[float]:  # 定义函数（封装可复用逻辑）
    speeds = []  # 初始化变量 speeds 为一个容器/表达式结果
    
    for i in range(len(ball_positions)):  # 循环遍历序列/迭代器
        if i == 0:  # 条件分支判断并选择执行路径
            speeds.append(0)  # 调用函数/方法执行某个动作或计算
            continue  # 控制流语句：改变当前代码块的执行方式
        
        curr_pos = ball_positions[i]  # 将表达式计算结果赋给变量 curr_pos
        prev_pos = ball_positions[i - 1]  # 将表达式计算结果赋给变量 prev_pos
        
        if curr_pos is None or prev_pos is None:  # 条件分支判断并选择执行路径
            speeds.append(0)  # 调用函数/方法执行某个动作或计算
            continue  # 控制流语句：改变当前代码块的执行方式
        
        if curr_pos[0] <= 0 or curr_pos[1] <= 0 or prev_pos[0] <= 0 or prev_pos[1] <= 0:  # 条件分支判断并选择执行路径
            speeds.append(0)  # 调用函数/方法执行某个动作或计算
            continue  # 控制流语句：改变当前代码块的执行方式
        
        dx = curr_pos[0] - prev_pos[0]  # 将表达式计算结果赋给变量 dx
        dy = curr_pos[1] - prev_pos[1]  # 将表达式计算结果赋给变量 dy
        distance = np.sqrt(dx**2 + dy**2)  # 将 distance 设为一次调用/构造的返回值
        
        speed = distance * fps  # 将表达式计算结果赋给变量 speed
        speeds.append(speed)  # 调用函数/方法执行某个动作或计算
    
    return speeds  # 从函数返回结果


def _calculate_speed_thresholds(speeds: List[float]) -> Tuple[float, float, float]:  # 定义函数（封装可复用逻辑）
    valid_speeds = [s for s in speeds if s > 0]  # 初始化变量 valid_speeds 为一个容器/表达式结果
    
    if len(valid_speeds) == 0:  # 条件分支判断并选择执行路径
        return (100, 300, 600)  # 从函数返回结果
    
    q25 = np.percentile(valid_speeds, 25)  # 将 q25 设为一次调用/构造的返回值
    q50 = np.percentile(valid_speeds, 50)  # 将 q50 设为一次调用/构造的返回值
    q75 = np.percentile(valid_speeds, 75)  # 将 q75 设为一次调用/构造的返回值
    
    slow_threshold = q25  # 将表达式计算结果赋给变量 slow_threshold
    medium_threshold = q50  # 将表达式计算结果赋给变量 medium_threshold
    fast_threshold = q75  # 将表达式计算结果赋给变量 fast_threshold
    
    return (slow_threshold, medium_threshold, fast_threshold)  # 从函数返回结果


def _get_speed_color(speed: float, thresholds: Tuple[float, float, float], alpha: float = 1.0) -> Tuple[int, int, int]:  # 定义函数（封装可复用逻辑）
    slow_threshold, medium_threshold, fast_threshold = thresholds  # 执行当前语句（保持与上文逻辑一致）
    
    if speed < slow_threshold:  # 条件分支判断并选择执行路径
        base_color = (0, 255, 0)  # 初始化变量 base_color 为一个容器/表达式结果
    elif speed < medium_threshold:  # 条件分支判断并选择执行路径
        base_color = (0, 255, 255)  # 初始化变量 base_color 为一个容器/表达式结果
    elif speed < fast_threshold:  # 条件分支判断并选择执行路径
        base_color = (0, 0, 255)  # 初始化变量 base_color 为一个容器/表达式结果
    else:  # 条件分支的否则路径
        base_color = (255, 0, 255)  # 初始化变量 base_color 为一个容器/表达式结果
    
    color = (int(base_color[0] * alpha), int(base_color[1] * alpha), int(base_color[2] * alpha))  # 初始化变量 color 为一个容器/表达式结果
    
    return color  # 从函数返回结果


def load_ball_positions(json_path: str) -> List[Tuple[int, int]]:  # 定义函数（封装可复用逻辑）
    import json  # 导入模块，供后续使用
    
    with open(json_path, 'r') as f:  # 上下文管理：确保资源正确释放
        data = json.load(f)  # 将 data 设为一次调用/构造的返回值
    
    ball_positions = []  # 初始化变量 ball_positions 为一个容器/表达式结果
    
    for frame_idx in sorted(data.keys(), key=lambda x: int(x)):  # 循环遍历序列/迭代器
        frame_data = data[frame_idx]  # 将表达式计算结果赋给变量 frame_data
        x = frame_data.get('x', None)  # 将 x 设为一次调用/构造的返回值
        y = frame_data.get('y', None)  # 将 y 设为一次调用/构造的返回值
        visible = frame_data.get('visible', 0)  # 将 visible 设为一次调用/构造的返回值
        
        if x is not None and y is not None and visible == 1:  # 条件分支判断并选择执行路径
            ball_positions.append((x, y))  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            ball_positions.append(None)  # 调用函数/方法执行某个动作或计算
    
    return ball_positions  # 从函数返回结果


def create_combined_visualization(  # 定义函数（封装可复用逻辑）
    video_path: str,  # 执行当前语句（保持与上文逻辑一致）
    ball_json_path: str,  # 执行当前语句（保持与上文逻辑一致）
    poses_path: str,  # 执行当前语句（保持与上文逻辑一致）
    output_path: str,  # 执行当前语句（保持与上文逻辑一致）
    traj_len: int = 10,  # 执行当前语句（保持与上文逻辑一致）
    court_keypoints=None,  # 将表达式计算结果赋给变量 court_keypoints
    partitioned_keypoints=None,  # 将表达式计算结果赋给变量 partitioned_keypoints
    net_keypoints: Optional[List[List[int]]] = None,  # 执行当前语句（保持与上文逻辑一致）
    hit_frames: Optional[List[int]] = None,  # 执行当前语句（保持与上文逻辑一致）
    per_frame_court_keypoints: Optional[List[Optional[List[List[int]]]]] = None,  # 执行当前语句（保持与上文逻辑一致）
    per_frame_net_keypoints: Optional[List[Optional[List[List[int]]]]] = None,  # 执行当前语句（保持与上文逻辑一致）
    stroke_types: Optional[List[str]] = None  # 执行当前语句（保持与上文逻辑一致）
):  # 执行当前语句（保持与上文逻辑一致）
    ball_positions = load_ball_positions(ball_json_path)  # 将 ball_positions 设为一次调用/构造的返回值
    poses = np.load(poses_path)  # 将 poses 设为一次调用/构造的返回值
    
    print(f"球网关键点数量: {len(net_keypoints) if net_keypoints is not None else 0}")  # 调用函数/方法执行某个动作或计算
    print(f"羽毛球场关键点数量: {len(partitioned_keypoints) if partitioned_keypoints is not None else 0}")  # 调用函数/方法执行某个动作或计算
    print(f"每帧球场关键点: {len(per_frame_court_keypoints) if per_frame_court_keypoints is not None else 0} 帧")  # 调用函数/方法执行某个动作或计算
    print(f"每帧球网关键点: {len(per_frame_net_keypoints) if per_frame_net_keypoints is not None else 0} 帧")  # 调用函数/方法执行某个动作或计算
    
    visualize_combined(  # 执行当前语句（保持与上文逻辑一致）
        video_path=video_path,  # 将表达式计算结果赋给变量 video_path
        ball_positions=ball_positions,  # 将表达式计算结果赋给变量 ball_positions
        poses=poses,  # 将表达式计算结果赋给变量 poses
        output_path=output_path,  # 将表达式计算结果赋给变量 output_path
        traj_len=traj_len,  # 将表达式计算结果赋给变量 traj_len
        court_keypoints=court_keypoints,  # 将表达式计算结果赋给变量 court_keypoints
        partitioned_keypoints=partitioned_keypoints,  # 将表达式计算结果赋给变量 partitioned_keypoints
        net_keypoints=net_keypoints,  # 将表达式计算结果赋给变量 net_keypoints
        hit_frames=hit_frames,  # 将表达式计算结果赋给变量 hit_frames
        per_frame_court_keypoints=per_frame_court_keypoints,  # 将表达式计算结果赋给变量 per_frame_court_keypoints
        per_frame_net_keypoints=per_frame_net_keypoints,  # 将表达式计算结果赋给变量 per_frame_net_keypoints
        stroke_types=stroke_types  # 将表达式计算结果赋给变量 stroke_types
    )  # 执行当前语句（保持与上文逻辑一致）


def load_court_keypoints(json_path: str) -> Optional[List[List[int]]]:  # 定义函数（封装可复用逻辑）
    import json  # 导入模块，供后续使用
    
    try:  # 开始异常捕获保护块
        with open(json_path, 'r') as f:  # 上下文管理：确保资源正确释放
            data = json.load(f)  # 将 data 设为一次调用/构造的返回值
        
        if 'court_keypoints' in data:  # 条件分支判断并选择执行路径
            return data['court_keypoints']  # 从函数返回结果
        elif 'partitioned_keypoints' in data:  # 条件分支判断并选择执行路径
            return data['partitioned_keypoints']  # 从函数返回结果
        return None  # 从函数返回结果
    except Exception as e:  # 捕获异常并进行处理
        print(f"Warning: Could not load court keypoints from {json_path}: {e}")  # 调用函数/方法执行某个动作或计算
        return None  # 从函数返回结果


def load_net_keypoints(json_path: str) -> Optional[List[List[int]]]:  # 定义函数（封装可复用逻辑）
    import json  # 导入模块，供后续使用
    
    try:  # 开始异常捕获保护块
        with open(json_path, 'r') as f:  # 上下文管理：确保资源正确释放
            data = json.load(f)  # 将 data 设为一次调用/构造的返回值
        
        if 'net_keypoints' in data:  # 条件分支判断并选择执行路径
            return data['net_keypoints']  # 从函数返回结果
        return None  # 从函数返回结果
    except Exception as e:  # 捕获异常并进行处理
        print(f"Warning: Could not load net keypoints from {json_path}: {e}")  # 调用函数/方法执行某个动作或计算
        return None  # 从函数返回结果


if __name__ == "__main__":  # 条件分支判断并选择执行路径
    import argparse  # 导入模块，供后续使用
    
    parser = argparse.ArgumentParser(description='Create combined visualization of ball trajectory and player poses')  # 将 parser 设为一次调用/构造的返回值
    parser.add_argument('--video', type=str, required=True, help='Input video path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--ball_json', type=str, required=True, help='Ball detection JSON path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--poses', type=str, required=True, help='Poses numpy array path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--output', type=str, required=True, help='Output video path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--traj_len', type=int, default=10, help='Trajectory length to display')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--court_json', type=str, default=None, help='Court keypoints JSON path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--net_json', type=str, default=None, help='Net keypoints JSON path')  # 调用函数/方法执行某个动作或计算
    
    args = parser.parse_args()  # 将 args 设为一次调用/构造的返回值
    
    court_keypoints = None  # 将表达式计算结果赋给变量 court_keypoints
    partitioned_keypoints = None  # 将表达式计算结果赋给变量 partitioned_keypoints
    net_keypoints = None  # 将表达式计算结果赋给变量 net_keypoints
    
    if args.court_json:  # 条件分支判断并选择执行路径
        partitioned_keypoints = load_court_keypoints(args.court_json)  # 将 partitioned_keypoints 设为一次调用/构造的返回值
    
    if args.net_json:  # 条件分支判断并选择执行路径
        net_keypoints = load_net_keypoints(args.net_json)  # 将 net_keypoints 设为一次调用/构造的返回值
    
    create_combined_visualization(  # 执行当前语句（保持与上文逻辑一致）
        args.video,  # 执行当前语句（保持与上文逻辑一致）
        args.ball_json,  # 执行当前语句（保持与上文逻辑一致）
        args.poses,  # 执行当前语句（保持与上文逻辑一致）
        args.output,  # 执行当前语句（保持与上文逻辑一致）
        args.traj_len,  # 执行当前语句（保持与上文逻辑一致）
        court_keypoints,  # 执行当前语句（保持与上文逻辑一致）
        partitioned_keypoints,  # 执行当前语句（保持与上文逻辑一致）
        net_keypoints  # 执行当前语句（保持与上文逻辑一致）
    )  # 执行当前语句（保持与上文逻辑一致）
