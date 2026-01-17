import cv2  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
from pathlib import Path  # 从模块导入符号，供后续调用
from typing import List, Tuple, Dict, Optional  # 从模块导入符号，供后续调用
from tqdm import tqdm  # 从模块导入符号，供后续调用
from .person_tracker import track_poses  # 从模块导入符号，供后续调用
from .court_based_assigner import CourtBasedPlayerAssigner  # 从模块导入符号，供后续调用


class PoseDetector:  # 定义类（封装数据与行为）
    def __init__(self, device='cuda', model='rtmpose-m', use_court_based=False):  # 定义函数（封装可复用逻辑）
        self.device = device  # 给对象属性 self.device 赋值/初始化（来自当前语句右侧表达式）
        self.model = model  # 给对象属性 self.model 赋值/初始化（来自当前语句右侧表达式）
        self.use_court_based = use_court_based  # 给对象属性 self.use_court_based 赋值/初始化（来自当前语句右侧表达式）
        self.inferencer = None  # 给对象属性 self.inferencer 赋值/初始化（来自当前语句右侧表达式）
        self.court_assigner = None  # 给对象属性 self.court_assigner 赋值/初始化（来自当前语句右侧表达式）
        self._init_inferencer()  # 调用函数/方法执行某个动作或计算

    def _init_inferencer(self):  # 定义函数（封装可复用逻辑）
        try:  # 开始异常捕获保护块
            from mmpose.apis import MMPoseInferencer  # 从模块导入符号，供后续调用
            
            model_map = {  # 初始化变量 model_map 为一个容器/表达式结果
                'rtmpose-t': 'rtmpose-t_8xb256-420e_coco-256x192',  # 执行当前语句（保持与上文逻辑一致）
                'rtmpose-s': 'rtmpose-s_8xb256-420e_coco-256x192',  # 执行当前语句（保持与上文逻辑一致）
                'rtmpose-m': 'rtmpose-m_8xb256-420e_coco-256x192',  # 执行当前语句（保持与上文逻辑一致）
                'rtmpose-l': 'rtmpose-l_8xb256-420e_coco-256x192',  # 执行当前语句（保持与上文逻辑一致）
            }  # 执行当前语句（保持与上文逻辑一致）
            
            pose2d_model = model_map.get(self.model, 'rtmpose-m_8xb256-420e_coco-256x192')  # 将 pose2d_model 设为一次调用/构造的返回值
            
            self.inferencer = MMPoseInferencer(  # 给对象属性 self.inferencer 赋值/初始化（来自当前语句右侧表达式）
                pose2d=pose2d_model,  # 将表达式计算结果赋给变量 pose2d
                device=self.device,  # 将表达式计算结果赋给变量 device
                show_progress=False  # 将表达式计算结果赋给变量 show_progress
            )  # 执行当前语句（保持与上文逻辑一致）
            print(f"MMPose inferencer initialized successfully with model: {pose2d_model}")  # 调用函数/方法执行某个动作或计算
        except Exception as e:  # 捕获异常并进行处理
            print(f"Failed to initialize MMPose inferencer: {e}")  # 调用函数/方法执行某个动作或计算
            self.inferencer = None  # 给对象属性 self.inferencer 赋值/初始化（来自当前语句右侧表达式）

    def set_court_info(self, court_info: List[float], extended_court_points: Optional[np.ndarray] = None):  # 定义函数（封装可复用逻辑）
        if self.use_court_based:  # 条件分支判断并选择执行路径
            if self.court_assigner is None:  # 条件分支判断并选择执行路径
                self.court_assigner = CourtBasedPlayerAssigner(720, 1280)  # 给对象属性 self.court_assigner 赋值/初始化（来自当前语句右侧表达式）
            self.court_assigner.set_court_info(court_info, extended_court_points)  # 调用函数/方法执行某个动作或计算
            print(f"Court info set for court-based assignment: net_y={court_info[4] if len(court_info) >= 5 else 'N/A'}")  # 调用函数/方法执行某个动作或计算

    def detect_video(  # 定义函数（封装可复用逻辑）
        self,  # 执行当前语句（保持与上文逻辑一致）
        video_path: str,  # 执行当前语句（保持与上文逻辑一致）
        frame_callback=None,  # 将表达式计算结果赋给变量 frame_callback
        progress_callback=None,  # 将表达式计算结果赋给变量 progress_callback
        emit_every_n_frames: int = 1  # 执行当前语句（保持与上文逻辑一致）
    ) -> Tuple[np.ndarray, List[Dict]]:  # 执行当前语句（保持与上文逻辑一致）
        cap = cv2.VideoCapture(video_path)  # 将 cap 设为一次调用/构造的返回值
        if not cap.isOpened():  # 条件分支判断并选择执行路径
            raise ValueError(f"Cannot open video: {video_path}")  # 调用函数/方法执行某个动作或计算

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 将 total_frames 设为一次调用/构造的返回值
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 将 width 设为一次调用/构造的返回值
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 将 height 设为一次调用/构造的返回值
        fps = cap.get(cv2.CAP_PROP_FPS)  # 将 fps 设为一次调用/构造的返回值

        all_poses = []  # 初始化变量 all_poses 为一个容器/表达式结果
        frame_indices = []  # 初始化变量 frame_indices 为一个容器/表达式结果

        if self.inferencer is not None:  # 条件分支判断并选择执行路径
            try:  # 开始异常捕获保护块
                results_generator = self.inferencer(video_path, show=False)  # 将 results_generator 设为一次调用/构造的返回值
                cap_preview = None  # 将表达式计算结果赋给变量 cap_preview
                if frame_callback is not None:  # 条件分支判断并选择执行路径
                    cap_preview = cv2.VideoCapture(video_path)  # 将 cap_preview 设为一次调用/构造的返回值
                
                skeleton_pairs = [  # 初始化变量 skeleton_pairs 为一个容器/表达式结果
                    (0, 1), (0, 2), (1, 3), (2, 4),  # 执行当前语句（保持与上文逻辑一致）
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 执行当前语句（保持与上文逻辑一致）
                    (5, 11), (6, 12), (11, 12),  # 执行当前语句（保持与上文逻辑一致）
                    (11, 13), (13, 15), (12, 14), (14, 16)  # 调用函数/方法执行某个动作或计算
                ]  # 执行当前语句（保持与上文逻辑一致）
                
                for frame_idx, result in enumerate(tqdm(results_generator, total=total_frames, desc="Detecting poses")):  # 循环遍历序列/迭代器
                    predictions = result.get('predictions', [])  # 将 predictions 设为一次调用/构造的返回值
                    frame_poses = None  # 将表达式计算结果赋给变量 frame_poses
                    if predictions and len(predictions) > 0:  # 条件分支判断并选择执行路径
                        frame_poses = predictions[0]  # 将表达式计算结果赋给变量 frame_poses
                        all_poses.append(frame_poses)  # 调用函数/方法执行某个动作或计算
                        frame_indices.append(frame_idx)  # 调用函数/方法执行某个动作或计算
                    
                    if cap_preview is not None:  # 条件分支判断并选择执行路径
                        ret_preview, frame_preview = cap_preview.read()  # 调用函数/方法执行某个动作或计算
                        if ret_preview and emit_every_n_frames > 0 and (frame_idx % emit_every_n_frames == 0):  # 条件分支判断并选择执行路径
                            persons = frame_poses if isinstance(frame_poses, list) else ([frame_poses] if frame_poses is not None else [])  # 将 persons 设为一次调用/构造的返回值
                            for person_idx, person in enumerate(persons[:2]):  # 循环遍历序列/迭代器
                                keypoints = person.get('keypoints', None) if isinstance(person, dict) else None  # 将 keypoints 设为一次调用/构造的返回值
                                if keypoints is None:  # 条件分支判断并选择执行路径
                                    continue  # 控制流语句：改变当前代码块的执行方式
                                kp = np.array(keypoints)[:17, :2]  # 将 kp 设为一次调用/构造的返回值
                                for start_idx, end_idx in skeleton_pairs:  # 循环遍历序列/迭代器
                                    pt1 = kp[start_idx]  # 将表达式计算结果赋给变量 pt1
                                    pt2 = kp[end_idx]  # 将表达式计算结果赋给变量 pt2
                                    if np.any(pt1) and np.any(pt2):  # 条件分支判断并选择执行路径
                                        color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)  # 初始化变量 color 为一个容器/表达式结果
                                        cv2.line(frame_preview, tuple(map(int, pt1)), tuple(map(int, pt2)), color, 3)  # 调用函数/方法执行某个动作或计算
                                for joint in kp:  # 循环遍历序列/迭代器
                                    if np.any(joint):  # 条件分支判断并选择执行路径
                                        color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)  # 初始化变量 color 为一个容器/表达式结果
                                        cv2.circle(frame_preview, tuple(map(int, joint)), 4, color, -1)  # 调用函数/方法执行某个动作或计算
                            frame_callback(frame_idx, frame_preview, frame_poses)  # 调用函数/方法执行某个动作或计算
                    elif frame_callback is not None and emit_every_n_frames > 0 and (frame_idx % emit_every_n_frames == 0):  # 条件分支判断并选择执行路径
                        frame_callback(frame_idx, None, frame_poses)  # 调用函数/方法执行某个动作或计算

                    if progress_callback is not None:  # 条件分支判断并选择执行路径
                        progress_callback(frame_idx + 1, total_frames)  # 调用函数/方法执行某个动作或计算
                
                if cap_preview is not None:  # 条件分支判断并选择执行路径
                    cap_preview.release()  # 调用函数/方法执行某个动作或计算
            except Exception as e:  # 捕获异常并进行处理
                print(f"Error during pose detection: {e}")  # 调用函数/方法执行某个动作或计算
                all_poses = []  # 初始化变量 all_poses 为一个容器/表达式结果
                frame_indices = []  # 初始化变量 frame_indices 为一个容器/表达式结果
        else:  # 条件分支的否则路径
            print("MMPose inferencer not available, skipping pose detection")  # 调用函数/方法执行某个动作或计算

        cap.release()  # 调用函数/方法执行某个动作或计算

        poses_array = self._poses_to_array(all_poses, total_frames)  # 将 poses_array 设为一次调用/构造的返回值
        
        return poses_array, {  # 从函数返回结果
            'width': width,  # 执行当前语句（保持与上文逻辑一致）
            'height': height,  # 执行当前语句（保持与上文逻辑一致）
            'fps': fps,  # 执行当前语句（保持与上文逻辑一致）
            'total_frames': total_frames,  # 执行当前语句（保持与上文逻辑一致）
            'frame_indices': frame_indices  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）

    def _poses_to_array(self, poses_list: List[Dict], total_frames: int) -> np.ndarray:  # 定义函数（封装可复用逻辑）
        all_frame_poses = []  # 初始化变量 all_frame_poses 为一个容器/表达式结果

        for frame_poses in poses_list:  # 循环遍历序列/迭代器
            persons = frame_poses if isinstance(frame_poses, list) else [frame_poses]  # 将 persons 设为一次调用/构造的返回值
            
            valid_persons = []  # 初始化变量 valid_persons 为一个容器/表达式结果
            for person in persons:  # 循环遍历序列/迭代器
                keypoints = person.get('keypoints', None)  # 将 keypoints 设为一次调用/构造的返回值
                if keypoints is not None and len(keypoints) >= 17:  # 条件分支判断并选择执行路径
                    valid_persons.append(np.array(keypoints)[:17, :2])  # 调用函数/方法执行某个动作或计算
            
            if self.use_court_based and self.court_assigner is not None and len(valid_persons) >= 2:  # 条件分支判断并选择执行路径
                top_player, bottom_player = self.court_assigner.assign_players(valid_persons)  # 调用函数/方法执行某个动作或计算
                if top_player is not None and bottom_player is not None:  # 条件分支判断并选择执行路径
                    all_frame_poses.append([top_player, bottom_player])  # 调用函数/方法执行某个动作或计算
                else:  # 条件分支的否则路径
                    court_persons = []  # 初始化变量 court_persons 为一个容器/表达式结果
                    for person in valid_persons:  # 循环遍历序列/迭代器
                        if self.court_assigner.is_in_court(person):  # 条件分支判断并选择执行路径
                            court_persons.append(person)  # 调用函数/方法执行某个动作或计算
                    all_frame_poses.append(court_persons)  # 调用函数/方法执行某个动作或计算
            elif self.use_court_based and self.court_assigner is not None:  # 条件分支判断并选择执行路径
                court_persons = []  # 初始化变量 court_persons 为一个容器/表达式结果
                for person in valid_persons:  # 循环遍历序列/迭代器
                    if self.court_assigner.is_in_court(person):  # 条件分支判断并选择执行路径
                        court_persons.append(person)  # 调用函数/方法执行某个动作或计算
                all_frame_poses.append(court_persons)  # 调用函数/方法执行某个动作或计算
            else:  # 条件分支的否则路径
                all_frame_poses.append(valid_persons)  # 调用函数/方法执行某个动作或计算

        poses_array = track_poses(all_frame_poses, max_persons=2)  # 将 poses_array 设为一次调用/构造的返回值

        return poses_array  # 从函数返回结果

    def save_poses(self, poses: np.ndarray, output_path: str):  # 定义函数（封装可复用逻辑）
        np.save(output_path, poses)  # 调用函数/方法执行某个动作或计算
        print(f"Poses saved to {output_path}")  # 调用函数/方法执行某个动作或计算

    def load_poses(self, poses_path: str) -> np.ndarray:  # 定义函数（封装可复用逻辑）
        poses = np.load(poses_path)  # 将 poses 设为一次调用/构造的返回值
        print(f"Poses loaded from {poses_path}")  # 调用函数/方法执行某个动作或计算
        return poses  # 从函数返回结果

    def visualize_poses(self, video_path: str, poses: np.ndarray, output_path: str, skeleton_pairs=None):  # 定义函数（封装可复用逻辑）
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

        frame_idx = 0  # 将表达式计算结果赋给变量 frame_idx
        total_frames = poses.shape[0]  # 将表达式计算结果赋给变量 total_frames

        with tqdm(total=total_frames, desc="Visualizing poses") as pbar:  # 上下文管理：确保资源正确释放
            while True:  # 条件循环，直到条件不满足
                ret, frame = cap.read()  # 调用函数/方法执行某个动作或计算
                if not ret:  # 条件分支判断并选择执行路径
                    break  # 控制流语句：改变当前代码块的执行方式

                if frame_idx < total_frames:  # 条件分支判断并选择执行路径
                    for person_idx in range(2):  # 循环遍历序列/迭代器
                        person_poses = poses[frame_idx, person_idx]  # 将表达式计算结果赋给变量 person_poses
                        
                        for start_idx, end_idx in skeleton_pairs:  # 循环遍历序列/迭代器
                            pt1 = person_poses[start_idx]  # 将表达式计算结果赋给变量 pt1
                            pt2 = person_poses[end_idx]  # 将表达式计算结果赋给变量 pt2
                            
                            if np.any(pt1) and np.any(pt2):  # 条件分支判断并选择执行路径
                                color = (0, 255, 0) if person_idx == 0 else (0, 0, 255)  # 初始化变量 color 为一个容器/表达式结果
                                pt1 = tuple(map(int, pt1))  # 将 pt1 设为一次调用/构造的返回值
                                pt2 = tuple(map(int, pt2))  # 将 pt2 设为一次调用/构造的返回值
                                cv2.line(frame, pt1, pt2, color, 2)  # 调用函数/方法执行某个动作或计算

                        for joint_idx in range(17):  # 循环遍历序列/迭代器
                            joint = person_poses[joint_idx]  # 将表达式计算结果赋给变量 joint
                            if np.any(joint):  # 条件分支判断并选择执行路径
                                color = (0, 255, 0) if person_idx == 0 else (0, 0, 255)  # 初始化变量 color 为一个容器/表达式结果
                                joint = tuple(map(int, joint))  # 将 joint 设为一次调用/构造的返回值
                                cv2.circle(frame, joint, 4, color, -1)  # 调用函数/方法执行某个动作或计算

                out.write(frame)  # 调用函数/方法执行某个动作或计算
                frame_idx += 1  # 执行当前语句（保持与上文逻辑一致）
                pbar.update(1)  # 调用函数/方法执行某个动作或计算

        cap.release()  # 调用函数/方法执行某个动作或计算
        out.release()  # 调用函数/方法执行某个动作或计算
        print(f"Pose visualization saved to {output_path}")  # 调用函数/方法执行某个动作或计算


def detect_poses_video(video_path: str, output_dir: str, device='cuda'):  # 定义函数（封装可复用逻辑）
    output_dir = Path(output_dir)  # 将 output_dir 设为一次调用/构造的返回值
    output_dir.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算

    video_name = Path(video_path).stem  # 将 video_name 设为一次调用/构造的返回值
    poses_path = output_dir / f"{video_name}_poses.npy"  # 将表达式计算结果赋给变量 poses_path
    vis_path = output_dir / f"{video_name}_with_poses.mp4"  # 将表达式计算结果赋给变量 vis_path

    detector = PoseDetector(device=device)  # 将 detector 设为一次调用/构造的返回值
    
    print(f"Detecting poses in {video_path}...")  # 调用函数/方法执行某个动作或计算
    poses, video_info = detector.detect_video(video_path)  # 调用函数/方法执行某个动作或计算
    
    detector.save_poses(poses, str(poses_path))  # 调用函数/方法执行某个动作或计算
    
    print(f"Visualizing poses...")  # 调用函数/方法执行某个动作或计算
    detector.visualize_poses(video_path, poses, str(vis_path))  # 调用函数/方法执行某个动作或计算
    
    print(f"Pose detection complete!")  # 调用函数/方法执行某个动作或计算
    print(f"Video info: {video_info}")  # 调用函数/方法执行某个动作或计算
    print(f"Poses shape: {poses.shape}")  # 调用函数/方法执行某个动作或计算
    
    return poses, video_info  # 从函数返回结果


if __name__ == "__main__":  # 条件分支判断并选择执行路径
    import argparse  # 导入模块，供后续使用
    
    parser = argparse.ArgumentParser(description='Detect poses in badminton video')  # 将 parser 设为一次调用/构造的返回值
    parser.add_argument('--video', type=str, required=True, help='Input video path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')  # 调用函数/方法执行某个动作或计算
    
    args = parser.parse_args()  # 将 args 设为一次调用/构造的返回值
    
    detect_poses_video(args.video, args.output_dir, args.device)  # 调用函数/方法执行某个动作或计算
