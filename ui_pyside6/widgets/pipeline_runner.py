from __future__ import annotations  # 从模块导入符号，供后续调用

from dataclasses import dataclass  # 从模块导入符号，供后续调用
from pathlib import Path  # 从模块导入符号，供后续调用
from typing import Callable, Optional  # 从模块导入符号，供后续调用

import cv2  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用

from core.ball_detect import ball_detect  # 从模块导入符号，供后续调用
from core.court_detect import CourtDetector  # 从模块导入符号，供后续调用
from core.event_detect import EventDetector  # 从模块导入符号，供后续调用
from core.export_to_csv import export_to_csv  # 从模块导入符号，供后续调用
from core.net_detect import NetDetector  # 从模块导入符号，供后续调用
from core.pose_detect import PoseDetector  # 从模块导入符号，供后续调用
from core.stroke_classify import create_classifier  # 从模块导入符号，供后续调用
from core.utils import read_json  # 从模块导入符号，供后续调用
from core.visualize_combined import load_ball_positions, visualize_combined  # 从模块导入符号，供后续调用


@dataclass(frozen=True)  # 装饰器：修改/包装下方函数或类的行为
class PipelineConfig:  # 定义类（封装数据与行为）
    model_path: str  # 执行当前语句（保持与上文逻辑一致）
    num_frames: int  # 执行当前语句（保持与上文逻辑一致）
    threshold: float  # 执行当前语句（保持与上文逻辑一致）
    traj_len: int  # 执行当前语句（保持与上文逻辑一致）
    device: str  # 执行当前语句（保持与上文逻辑一致）
    pose_model: str  # 执行当前语句（保持与上文逻辑一致）
    use_court_detection: bool  # 执行当前语句（保持与上文逻辑一致）
    court_model_path: str  # 执行当前语句（保持与上文逻辑一致）
    net_model_path: str  # 执行当前语句（保持与上文逻辑一致）
    court_detection_interval: int  # 执行当前语句（保持与上文逻辑一致）
    pose_emit_every_n: int  # 执行当前语句（保持与上文逻辑一致）
    viz_emit_every_n: int  # 执行当前语句（保持与上文逻辑一致）
    keep_player_skeleton: bool  # 执行当前语句（保持与上文逻辑一致）
    keep_ball_trajectory: bool  # 执行当前语句（保持与上文逻辑一致）
    keep_stroke_type_hint: bool  # 执行当前语句（保持与上文逻辑一致）
    stroke_model: str  # 执行当前语句（保持与上文逻辑一致）
    dataset: str  # 执行当前语句（保持与上文逻辑一致）
    stroke_seq_len: int  # 执行当前语句（保持与上文逻辑一致）


@dataclass(frozen=True)  # 装饰器：修改/包装下方函数或类的行为
class PipelineOutputs:  # 定义类（封装数据与行为）
    video_name: str  # 执行当前语句（保持与上文逻辑一致）
    video_result_dir: str  # 执行当前语句（保持与上文逻辑一致）
    ball_json_path: str  # 执行当前语句（保持与上文逻辑一致）
    ball_denoise_json_path: str  # 执行当前语句（保持与上文逻辑一致）
    poses_path: str  # 执行当前语句（保持与上文逻辑一致）
    hit_events_path: str  # 执行当前语句（保持与上文逻辑一致）
    stroke_results_path: Optional[str]  # 执行当前语句（保持与上文逻辑一致）
    combined_video_path: str  # 执行当前语句（保持与上文逻辑一致）
    csv_path: str  # 执行当前语句（保持与上文逻辑一致）


def run_pipeline(  # 定义函数（封装可复用逻辑）
    video_path: str,  # 执行当前语句（保持与上文逻辑一致）
    result_dir: str,  # 执行当前语句（保持与上文逻辑一致）
    config: PipelineConfig,  # 执行当前语句（保持与上文逻辑一致）
    *,  # 执行当前语句（保持与上文逻辑一致）
    log: Optional[Callable[[str], None]] = None,  # 执行当前语句（保持与上文逻辑一致）
    step: Optional[Callable[[str], None]] = None,  # 执行当前语句（保持与上文逻辑一致）
    overall_progress: Optional[Callable[[int], None]] = None,  # 执行当前语句（保持与上文逻辑一致）
    preview_frame: Optional[Callable[[np.ndarray], None]] = None,  # 执行当前语句（保持与上文逻辑一致）
    stop_requested: Optional[Callable[[], bool]] = None,  # 执行当前语句（保持与上文逻辑一致）
) -> PipelineOutputs:  # 执行当前语句（保持与上文逻辑一致）
    def _log(message: str):  # 定义函数（封装可复用逻辑）
        if log is not None:  # 条件分支判断并选择执行路径
            log(message)  # 调用函数/方法执行某个动作或计算

    def _step(name: str):  # 定义函数（封装可复用逻辑）
        if step is not None:  # 条件分支判断并选择执行路径
            step(name)  # 调用函数/方法执行某个动作或计算
        if log is not None:  # 条件分支判断并选择执行路径
            separator = "=" * 60  # 将 separator 设为一次调用/构造的返回值
            log(separator)  # 调用函数/方法执行某个动作或计算

    def _set_overall(p: int):  # 定义函数（封装可复用逻辑）
        if overall_progress is not None:  # 条件分支判断并选择执行路径
            overall_progress(int(max(0, min(100, p))))  # 调用函数/方法执行某个动作或计算

    def _stopping() -> bool:  # 定义函数（封装可复用逻辑）
        return bool(stop_requested and stop_requested())  # 从函数返回结果

    result_dir_path = Path(result_dir).expanduser()  # 将 result_dir_path 设为一次调用/构造的返回值
    if not result_dir_path.is_absolute():  # 条件分支判断并选择执行路径
        project_root = Path(__file__).resolve().parents[2]  # 将 project_root 设为一次调用/构造的返回值
        result_dir_path = project_root / result_dir_path  # 将表达式计算结果赋给变量 result_dir_path
    result_dir_path = result_dir_path.resolve(strict=False)  # 将 result_dir_path 设为一次调用/构造的返回值
    result_dir_path.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
    video_name = Path(video_path).stem  # 将 video_name 设为一次调用/构造的返回值
    video_result_dir = result_dir_path / video_name  # 将表达式计算结果赋给变量 video_result_dir
    video_result_dir.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算

    ball_json_path = video_result_dir / "loca_info" / f"{video_name}.json"  # 将表达式计算结果赋给变量 ball_json_path
    ball_denoise_json_path = video_result_dir / "loca_info_denoise" / f"{video_name}.json"  # 将表达式计算结果赋给变量 ball_denoise_json_path
    poses_path = video_result_dir / f"{video_name}_poses.npy"  # 将表达式计算结果赋给变量 poses_path
    combined_video_path = video_result_dir / f"{video_name}_combined.mp4"  # 将表达式计算结果赋给变量 combined_video_path
    hit_events_path = video_result_dir / f"{video_name}_hit_events.json"  # 将表达式计算结果赋给变量 hit_events_path
    stroke_results_path = video_result_dir / f"{video_name}_stroke_types.json"  # 将表达式计算结果赋给变量 stroke_results_path
    csv_path = video_result_dir / f"{video_name}_data.csv"  # 将表达式计算结果赋给变量 csv_path

    court_info = None  # 将表达式计算结果赋给变量 court_info
    extended_court_points = None  # 将表达式计算结果赋给变量 extended_court_points
    court_boundary_params = None  # 将表达式计算结果赋给变量 court_boundary_params
    partitioned_keypoints = None  # 将表达式计算结果赋给变量 partitioned_keypoints
    net_keypoints = None  # 将表达式计算结果赋给变量 net_keypoints
    per_frame_court_keypoints = []  # 初始化变量 per_frame_court_keypoints 为一个容器/表达式结果
    per_frame_net_keypoints = []  # 初始化变量 per_frame_net_keypoints 为一个容器/表达式结果

    if config.use_court_detection:  # 条件分支判断并选择执行路径
        _step("球场/球网检测")  # 调用函数/方法执行某个动作或计算
        _log("开始逐帧球场/球网检测…")  # 调用函数/方法执行某个动作或计算
        _log(f"球场模型路径: {config.court_model_path}")  # 调用函数/方法执行某个动作或计算
        _log(f"球网模型路径: {config.net_model_path}")  # 调用函数/方法执行某个动作或计算
        _log(f"检测间隔: {config.court_detection_interval} 帧")  # 调用函数/方法执行某个动作或计算

        court_detector = CourtDetector(model_path=config.court_model_path, device=config.device)  # 将 court_detector 设为一次调用/构造的返回值
        net_detector = NetDetector(model_path=config.net_model_path, device=config.device)  # 将 net_detector 设为一次调用/构造的返回值

        cap = cv2.VideoCapture(video_path)  # 将 cap 设为一次调用/构造的返回值
        if not cap.isOpened():  # 条件分支判断并选择执行路径
            raise ValueError(f"Cannot open video: {video_path}")  # 调用函数/方法执行某个动作或计算
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 将 total_frames 设为一次调用/构造的返回值
        _log(f"视频总帧数: {total_frames}")  # 调用函数/方法执行某个动作或计算
        _log(f"视频FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")  # 调用函数/方法执行某个动作或计算
        _log(f"使用设备: {config.device}")  # 调用函数/方法执行某个动作或计算

        current_court = None  # 将表达式计算结果赋给变量 current_court
        current_net = None  # 将表达式计算结果赋给变量 current_net

        for frame_idx in range(total_frames):  # 循环遍历序列/迭代器
            if _stopping():  # 条件分支判断并选择执行路径
                cap.release()  # 调用函数/方法执行某个动作或计算
                raise RuntimeError("STOP_REQUESTED")  # 调用函数/方法执行某个动作或计算

            ret, frame = cap.read()  # 调用函数/方法执行某个动作或计算
            if not ret:  # 条件分支判断并选择执行路径
                per_frame_court_keypoints.append(None)  # 调用函数/方法执行某个动作或计算
                per_frame_net_keypoints.append(None)  # 调用函数/方法执行某个动作或计算
                continue  # 控制流语句：改变当前代码块的执行方式

            if frame_idx % max(1, config.court_detection_interval) == 0:  # 条件分支判断并选择执行路径
                court_detector.reset()  # 调用函数/方法执行某个动作或计算
                court_info_result, have_court = court_detector.get_court_info(frame)  # 调用函数/方法执行某个动作或计算
                if have_court:  # 条件分支判断并选择执行路径
                    court_info = court_info_result  # 将表达式计算结果赋给变量 court_info
                    extended_court_points = court_detector._CourtDetector__extended_court_points  # 将表达式计算结果赋给变量 extended_court_points
                    court_boundary_params = court_detector.get_court_boundary_params()  # 将 court_boundary_params 设为一次调用/构造的返回值
                    partitioned_keypoints = court_detector.get_partitioned_keypoints()  # 将 partitioned_keypoints 设为一次调用/构造的返回值
                    current_court = partitioned_keypoints  # 将表达式计算结果赋给变量 current_court
                else:  # 条件分支的否则路径
                    current_court = None  # 将表达式计算结果赋给变量 current_court
                    partitioned_keypoints = None  # 将表达式计算结果赋给变量 partitioned_keypoints

                net_info_result, have_net = net_detector.get_net_info(frame)  # 调用函数/方法执行某个动作或计算
                if have_net:  # 条件分支判断并选择执行路径
                    net_keypoints = net_detector.get_partitioned_keypoints()  # 将 net_keypoints 设为一次调用/构造的返回值
                    current_net = net_keypoints  # 将表达式计算结果赋给变量 current_net
                else:  # 条件分支的否则路径
                    current_net = None  # 将表达式计算结果赋给变量 current_net
                    net_keypoints = None  # 将表达式计算结果赋给变量 net_keypoints

            per_frame_court_keypoints.append(current_court)  # 调用函数/方法执行某个动作或计算
            per_frame_net_keypoints.append(current_net)  # 调用函数/方法执行某个动作或计算

            if preview_frame is not None and config.pose_emit_every_n > 0 and frame_idx % config.pose_emit_every_n == 0:  # 条件分支判断并选择执行路径
                preview = frame.copy()  # 将 preview 设为一次调用/构造的返回值
                if current_court is not None:  # 条件分支判断并选择执行路径
                    preview = court_detector.draw_court(preview, mode="frame_select")  # 将 preview 设为一次调用/构造的返回值
                if current_net is not None:  # 条件分支判断并选择执行路径
                    preview = net_detector.draw_net(preview, mode="frame_select")  # 将 preview 设为一次调用/构造的返回值
                preview_frame(preview)  # 调用函数/方法执行某个动作或计算

            _set_overall(int(frame_idx / max(1, total_frames) * 10))  # 调用函数/方法执行某个动作或计算

        cap.release()  # 调用函数/方法执行某个动作或计算
        _log("球场/球网检测完成")  # 调用函数/方法执行某个动作或计算
        if log is not None:  # 条件分支判断并选择执行路径
            log("=" * 60)  # 调用函数/方法执行某个动作或计算

    _step("羽毛球检测")  # 调用函数/方法执行某个动作或计算
    _log("开始 TrackNetV3 羽毛球检测…")  # 调用函数/方法执行某个动作或计算
    _log(f"模型路径: {config.model_path}")  # 调用函数/方法执行某个动作或计算
    _log(f"处理帧数: {config.num_frames}")  # 调用函数/方法执行某个动作或计算
    _log(f"阈值: {config.threshold}")  # 调用函数/方法执行某个动作或计算
    _log(f"使用设备: {config.device}")  # 调用函数/方法执行某个动作或计算

    def _ball_frame_cb(frame_idx: int, frame_bgr: np.ndarray, ball_pos, visible: int):  # 定义函数（封装可复用逻辑）
        if preview_frame is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if config.pose_emit_every_n <= 0 or frame_idx % config.pose_emit_every_n != 0:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        img = frame_bgr.copy()  # 将 img 设为一次调用/构造的返回值
        if visible == 1 and ball_pos is not None:  # 条件分支判断并选择执行路径
            cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), 10, (0, 255, 255), -1)  # 调用函数/方法执行某个动作或计算
            cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), 16, (255, 255, 255), 2)  # 调用函数/方法执行某个动作或计算
        preview_frame(img)  # 调用函数/方法执行某个动作或计算

    def _ball_progress(processed: int, total: int):  # 定义函数（封装可复用逻辑）
        base = 10  # 将表达式计算结果赋给变量 base
        span = 35  # 将表达式计算结果赋给变量 span
        _set_overall(base + int(processed / max(1, total) * span))  # 调用函数/方法执行某个动作或计算

    ball_detect(  # 执行当前语句（保持与上文逻辑一致）
        video_path,  # 执行当前语句（保持与上文逻辑一致）
        str(video_result_dir),  # 执行当前语句（保持与上文逻辑一致）
        config.model_path,  # 执行当前语句（保持与上文逻辑一致）
        config.num_frames,  # 执行当前语句（保持与上文逻辑一致）
        config.threshold,  # 执行当前语句（保持与上文逻辑一致）
        frame_callback=_ball_frame_cb,  # 将表达式计算结果赋给变量 frame_callback
        progress_callback=_ball_progress,  # 将表达式计算结果赋给变量 progress_callback
    )  # 执行当前语句（保持与上文逻辑一致）

    if _stopping():  # 条件分支判断并选择执行路径
        raise RuntimeError("STOP_REQUESTED")  # 调用函数/方法执行某个动作或计算
    _log("羽毛球检测完成")  # 调用函数/方法执行某个动作或计算
    if log is not None:  # 条件分支判断并选择执行路径
        log("=" * 60)  # 调用函数/方法执行某个动作或计算

    _step("姿态检测")  # 调用函数/方法执行某个动作或计算
    _log("开始 MMPose 姿态检测…")  # 调用函数/方法执行某个动作或计算
    _log(f"姿态模型: {config.pose_model}")  # 调用函数/方法执行某个动作或计算
    _log(f"使用设备: {config.device}")  # 调用函数/方法执行某个动作或计算
    _log(f"球场检测: {'启用' if config.use_court_detection else '禁用'}")  # 调用函数/方法执行某个动作或计算

    detector = PoseDetector(device=config.device, model=config.pose_model, use_court_based=config.use_court_detection)  # 将 detector 设为一次调用/构造的返回值
    if config.use_court_detection and court_boundary_params is not None:  # 条件分支判断并选择执行路径
        detector.set_court_info(court_boundary_params, extended_court_points)  # 调用函数/方法执行某个动作或计算

    def _pose_preview(frame_idx: int, frame_bgr: Optional[np.ndarray], frame_poses):  # 定义函数（封装可复用逻辑）
        if preview_frame is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if frame_bgr is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        preview_frame(frame_bgr)  # 调用函数/方法执行某个动作或计算

    def _pose_progress(processed: int, total: int):  # 定义函数（封装可复用逻辑）
        base = 45  # 将表达式计算结果赋给变量 base
        span = 35  # 将表达式计算结果赋给变量 span
        _set_overall(base + int(processed / max(1, total) * span))  # 调用函数/方法执行某个动作或计算

    poses, video_info = detector.detect_video(  # 执行当前语句（保持与上文逻辑一致）
        video_path,  # 执行当前语句（保持与上文逻辑一致）
        frame_callback=_pose_preview,  # 将表达式计算结果赋给变量 frame_callback
        progress_callback=_pose_progress,  # 将表达式计算结果赋给变量 progress_callback
        emit_every_n_frames=max(1, config.pose_emit_every_n),  # 将 emit_every_n_frames 设为一次调用/构造的返回值
    )  # 执行当前语句（保持与上文逻辑一致）
    detector.save_poses(poses, str(poses_path))  # 调用函数/方法执行某个动作或计算
    _log(f"姿态数组: {poses.shape}")  # 调用函数/方法执行某个动作或计算
    if log is not None:  # 条件分支判断并选择执行路径
        log("=" * 60)  # 调用函数/方法执行某个动作或计算

    if _stopping():  # 条件分支判断并选择执行路径
        raise RuntimeError("STOP_REQUESTED")  # 调用函数/方法执行某个动作或计算

    _step("事件检测")  # 调用函数/方法执行某个动作或计算
    _log("开始击球事件检测…")  # 调用函数/方法执行某个动作或计算
    _log(f"轨迹长度: {config.traj_len}")  # 调用函数/方法执行某个动作或计算
    _log(f"使用设备: {config.device}")  # 调用函数/方法执行某个动作或计算

    ball_data = read_json(str(ball_denoise_json_path))  # 将 ball_data 设为一次调用/构造的返回值
    trajectory_data = []  # 初始化变量 trajectory_data 为一个容器/表达式结果
    for frame_idx in range(len(ball_data)):  # 循环遍历序列/迭代器
        frame_key = str(frame_idx)  # 将 frame_key 设为一次调用/构造的返回值
        frame_data = ball_data.get(frame_key, None)  # 将 frame_data 设为一次调用/构造的返回值
        if frame_data is None:  # 条件分支判断并选择执行路径
            trajectory_data.append(None)  # 调用函数/方法执行某个动作或计算
            continue  # 控制流语句：改变当前代码块的执行方式
        x = frame_data.get("x", 0)  # 将 x 设为一次调用/构造的返回值
        y = frame_data.get("y", 0)  # 将 y 设为一次调用/构造的返回值
        visible = frame_data.get("visible", 0)  # 将 visible 设为一次调用/构造的返回值
        if visible == 1 and x > 0 and y > 0:  # 条件分支判断并选择执行路径
            trajectory_data.append([x, y])  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            trajectory_data.append(None)  # 调用函数/方法执行某个动作或计算

    event_detector = EventDetector(trajectory_data, poses)  # 将 event_detector 设为一次调用/构造的返回值
    hit_frames, hit_players = event_detector.detect_hits(  # 执行当前语句（保持与上文逻辑一致）
        fps=video_info["fps"],  # 将表达式计算结果赋给变量 fps
        prominence=1.0,  # 将表达式计算结果赋给变量 prominence
        angle_threshold=15,  # 将表达式计算结果赋给变量 angle_threshold
        min_frame_gap=5,  # 将表达式计算结果赋给变量 min_frame_gap
        min_continuation_frames=2,  # 将表达式计算结果赋给变量 min_continuation_frames
        min_movement_threshold=5,  # 将表达式计算结果赋给变量 min_movement_threshold
    )  # 执行当前语句（保持与上文逻辑一致）
    event_detector.save_hit_events(str(hit_events_path))  # 调用函数/方法执行某个动作或计算
    _log(f"击球次数: {len(hit_frames)}")  # 调用函数/方法执行某个动作或计算
    _set_overall(82)  # 调用函数/方法执行某个动作或计算
    if log is not None:  # 条件分支判断并选择执行路径
        log("=" * 60)  # 调用函数/方法执行某个动作或计算

    if _stopping():  # 条件分支判断并选择执行路径
        raise RuntimeError("STOP_REQUESTED")  # 调用函数/方法执行某个动作或计算

    _step("击球类型识别")  # 调用函数/方法执行某个动作或计算
    _log("开始击球类型识别…")  # 调用函数/方法执行某个动作或计算
    stroke_type_names = None  # 将表达式计算结果赋给变量 stroke_type_names
    stroke_results_written = False  # 将表达式计算结果赋给变量 stroke_results_written
    try:  # 开始异常捕获保护块
        _log(f"使用数据集: {config.stroke_model}")  # 调用函数/方法执行某个动作或计算
        _log(f"序列长度: {config.stroke_seq_len}")  # 调用函数/方法执行某个动作或计算
        _log(f"使用设备: {config.device}")  # 调用函数/方法执行某个动作或计算
        _log(f"击球事件数: {len(hit_frames)}")  # 调用函数/方法执行某个动作或计算
        _log(f"姿态数组形状: {poses.shape}")  # 调用函数/方法执行某个动作或计算
        _log(f"轨迹数据长度: {len(trajectory_data)}")  # 调用函数/方法执行某个动作或计算
        
        classifier = create_classifier(dataset=config.stroke_model, seq_len=config.stroke_seq_len)  # 将 classifier 设为一次调用/构造的返回值
        _log(f"分类器模型类型: {classifier.model_type}")  # 调用函数/方法执行某个动作或计算
        _log(f"分类器类别数: {classifier.n_classes}")  # 调用函数/方法执行某个动作或计算
        
        stroke_types = classifier.classify_hits(trajectory_data, poses, hit_frames)  # 将 stroke_types 设为一次调用/构造的返回值
        _log(f"识别完成，共识别 {len(stroke_types)} 个击球类型")  # 调用函数/方法执行某个动作或计算
        
        classifier.save_stroke_results(hit_frames, hit_players, stroke_types, str(stroke_results_path))  # 调用函数/方法执行某个动作或计算
        stroke_type_names = [classifier.get_stroke_type_name(st) for st in stroke_types]  # 初始化变量 stroke_type_names 为一个容器/表达式结果
        stroke_results_written = True  # 将表达式计算结果赋给变量 stroke_results_written
        
        from collections import Counter  # 导入模块，供后续使用
        stroke_counter = Counter(stroke_types)  # 将 stroke_counter 设为一次调用/构造的返回值
        _log("击球类型统计:")  # 调用函数/方法执行某个动作或计算
        for stroke_type_id, count in sorted(stroke_counter.items()):  # 循环遍历序列/迭代器
            stroke_name = classifier.get_stroke_type_name(stroke_type_id)  # 将 stroke_name 设为一次调用/构造的返回值
            percentage = (count / len(stroke_types)) * 100  # 将 percentage 设为一次调用/构造的返回值
            _log(f"  {stroke_name}: {count} 次 ({percentage:.1f}%)")  # 调用函数/方法执行某个动作或计算
        
        _log("击球类型识别完成")  # 调用函数/方法执行某个动作或计算
    except Exception as e:  # 捕获异常并进行处理
        _log(f"击球类型识别失败: {e}")  # 调用函数/方法执行某个动作或计算
        stroke_type_names = None  # 将表达式计算结果赋给变量 stroke_type_names
        stroke_results_written = False  # 将表达式计算结果赋给变量 stroke_results_written
    _set_overall(86)  # 调用函数/方法执行某个动作或计算
    if log is not None:  # 条件分支判断并选择执行路径
        log("=" * 60)  # 调用函数/方法执行某个动作或计算

    if _stopping():  # 条件分支判断并选择执行路径
        raise RuntimeError("STOP_REQUESTED")  # 调用函数/方法执行某个动作或计算

    _step("合成可视化")  # 调用函数/方法执行某个动作或计算
    _log("开始生成合成可视化视频…")  # 调用函数/方法执行某个动作或计算
    _log(f"保留球员骨架: {'是' if config.keep_player_skeleton else '否'}")  # 调用函数/方法执行某个动作或计算
    _log(f"保留羽毛球轨迹: {'是' if config.keep_ball_trajectory else '否'}")  # 调用函数/方法执行某个动作或计算
    _log(f"保留击球类别提示: {'是' if config.keep_stroke_type_hint else '否'}")  # 调用函数/方法执行某个动作或计算
    _log(f"轨迹长度: {config.traj_len}")  # 调用函数/方法执行某个动作或计算
    _log(f"击球事件数: {len(hit_frames) if hit_frames else 0}")  # 调用函数/方法执行某个动作或计算

    ball_positions = load_ball_positions(str(ball_denoise_json_path))  # 将 ball_positions 设为一次调用/构造的返回值

    def _viz_frame_cb(frame_idx: int, frame_bgr: np.ndarray):  # 定义函数（封装可复用逻辑）
        if preview_frame is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        preview_frame(frame_bgr)  # 调用函数/方法执行某个动作或计算

    def _viz_progress(processed: int, total: int):  # 定义函数（封装可复用逻辑）
        base = 86  # 将表达式计算结果赋给变量 base
        span = 12  # 将表达式计算结果赋给变量 span
        _set_overall(base + int(processed / max(1, total) * span))  # 调用函数/方法执行某个动作或计算

    visualize_combined(  # 执行当前语句（保持与上文逻辑一致）
        video_path=video_path,  # 将表达式计算结果赋给变量 video_path
        ball_positions=ball_positions,  # 将表达式计算结果赋给变量 ball_positions
        poses=poses,  # 将表达式计算结果赋给变量 poses
        output_path=str(combined_video_path),  # 将 output_path 设为一次调用/构造的返回值
        traj_len=config.traj_len,  # 将表达式计算结果赋给变量 traj_len
        court_keypoints=court_info,  # 将表达式计算结果赋给变量 court_keypoints
        partitioned_keypoints=partitioned_keypoints,  # 将表达式计算结果赋给变量 partitioned_keypoints
        net_keypoints=net_keypoints,  # 将表达式计算结果赋给变量 net_keypoints
        hit_frames=hit_frames,  # 将表达式计算结果赋给变量 hit_frames
        per_frame_court_keypoints=per_frame_court_keypoints if config.use_court_detection else None,  # 将表达式计算结果赋给变量 per_frame_court_keypoints
        per_frame_net_keypoints=per_frame_net_keypoints if config.use_court_detection else None,  # 将表达式计算结果赋给变量 per_frame_net_keypoints
        stroke_types=stroke_type_names,  # 将表达式计算结果赋给变量 stroke_types
        keep_player_skeleton=config.keep_player_skeleton,  # 将表达式计算结果赋给变量 keep_player_skeleton
        keep_ball_trajectory=config.keep_ball_trajectory,  # 将表达式计算结果赋给变量 keep_ball_trajectory
        keep_stroke_type_hint=config.keep_stroke_type_hint,  # 将表达式计算结果赋给变量 keep_stroke_type_hint
        frame_callback=_viz_frame_cb,  # 将表达式计算结果赋给变量 frame_callback
        progress_callback=_viz_progress,  # 将表达式计算结果赋给变量 progress_callback
        emit_every_n_frames=max(1, config.viz_emit_every_n),  # 将 emit_every_n_frames 设为一次调用/构造的返回值
    )  # 执行当前语句（保持与上文逻辑一致）

    if _stopping():  # 条件分支判断并选择执行路径
        raise RuntimeError("STOP_REQUESTED")  # 调用函数/方法执行某个动作或计算

    _step("导出数据")  # 调用函数/方法执行某个动作或计算
    _log("开始导出 CSV…")  # 调用函数/方法执行某个动作或计算
    _log(f"输出目录: {video_result_dir}")  # 调用函数/方法执行某个动作或计算
    _log(f"视频FPS: {video_info['fps']:.2f}")  # 调用函数/方法执行某个动作或计算
    _log(f"总帧数: {video_info['total_frames']}")  # 调用函数/方法执行某个动作或计算
    _log(f"击球事件数: {len(hit_frames) if hit_frames else 0}")  # 调用函数/方法执行某个动作或计算
    _log(f"击球类型识别: {'成功' if stroke_results_written else '失败'}")  # 调用函数/方法执行某个动作或计算

    export_to_csv(  # 执行当前语句（保持与上文逻辑一致）
        hit_events_path=str(hit_events_path),  # 将 hit_events_path 设为一次调用/构造的返回值
        poses_path=str(poses_path),  # 将 poses_path 设为一次调用/构造的返回值
        ball_json_path=str(ball_json_path),  # 将 ball_json_path 设为一次调用/构造的返回值
        ball_denoise_json_path=str(ball_denoise_json_path),  # 将 ball_denoise_json_path 设为一次调用/构造的返回值
        output_csv_path=str(csv_path),  # 将 output_csv_path 设为一次调用/构造的返回值
        fps=video_info["fps"],  # 将表达式计算结果赋给变量 fps
        stroke_types_path=str(stroke_results_path) if stroke_results_written else None,  # 将 stroke_types_path 设为一次调用/构造的返回值
    )  # 执行当前语句（保持与上文逻辑一致）
    _set_overall(100)  # 调用函数/方法执行某个动作或计算
    if log is not None:  # 条件分支判断并选择执行路径
        log("=" * 60)  # 调用函数/方法执行某个动作或计算
        log("分析完成")  # 调用函数/方法执行某个动作或计算

    return PipelineOutputs(  # 从函数返回结果
        video_name=video_name,  # 将表达式计算结果赋给变量 video_name
        video_result_dir=str(video_result_dir),  # 将 video_result_dir 设为一次调用/构造的返回值
        ball_json_path=str(ball_json_path),  # 将 ball_json_path 设为一次调用/构造的返回值
        ball_denoise_json_path=str(ball_denoise_json_path),  # 将 ball_denoise_json_path 设为一次调用/构造的返回值
        poses_path=str(poses_path),  # 将 poses_path 设为一次调用/构造的返回值
        hit_events_path=str(hit_events_path),  # 将 hit_events_path 设为一次调用/构造的返回值
        stroke_results_path=str(stroke_results_path) if stroke_results_written else None,  # 将 stroke_results_path 设为一次调用/构造的返回值
        combined_video_path=str(combined_video_path),  # 将 combined_video_path 设为一次调用/构造的返回值
        csv_path=str(csv_path),  # 将 csv_path 设为一次调用/构造的返回值
    )  # 执行当前语句（保持与上文逻辑一致）
