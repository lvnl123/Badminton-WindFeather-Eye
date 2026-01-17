"""
将 TrackNetV3（羽毛球检测）与 MMPose（人体姿态）串成一个端到端分析管线。

这个脚本的定位更偏“命令行入口/离线批处理”：
- 输入：单个比赛/训练视频
- 输出：球轨迹 JSON + 姿态 numpy + 击球事件 JSON +（可选）击球类型 JSON + 合成可视化视频 + CSV
- 可选：逐帧球场/球网关键点检测，用于更稳定的选手分配与可视化

注意：
- run_combined_pipeline 是一个“把各模块依次调用起来”的编排函数，本身不训练模型。
- 大模型权重、视频路径、输出目录等都通过参数/命令行传入。
"""

import os  # 标准库：提供操作系统相关接口（当前文件中未直接使用，但可能用于扩展）
import sys  # 标准库：解释器与运行环境相关接口（当前文件中未直接使用，但可能用于扩展）
import argparse  # 标准库：命令行参数解析（脚本入口处使用）
import cv2  # OpenCV：视频读取/写入与图像处理（用于球场/球网逐帧检测）
from pathlib import Path  # 标准库：面向对象的路径处理（统一拼路径与创建目录）

from core.ball_detect import ball_detect  # TrackNetV3 推理入口：输出球轨迹 JSON 与（可选）中间可视化
from core.pose_detect import PoseDetector  # 姿态检测封装：负责调用姿态模型并保存 npy
from core.visualize_combined import create_combined_visualization  # 合成可视化：叠加球轨迹/骨架/提示并输出视频
from core.utils import read_json  # 工具函数：读取 JSON（用于读取去噪球轨迹）
from core.court_based_assigner import assign_players_court_based  # 球场约束的选手分配（当前脚本未直接调用）
from core.court_detect import CourtDetector  # 球场关键点检测器：输出球场信息与分区关键点
from core.net_detect import NetDetector  # 球网关键点检测器：输出球网信息与分区关键点
from core.event_detect import EventDetector  # 事件检测器：从轨迹+姿态中检测击球帧/击球方
from core.export_to_csv import export_to_csv  # 导出模块：把 JSON/npy 汇总输出 CSV
from core.stroke_classify import create_classifier  # 击球类型分类器工厂：构建并加载模型权重


def run_combined_pipeline(  # 定义函数（封装可复用逻辑）
    video_path: str,  # 输入视频路径（文件系统路径字符串）
    result_dir: str,  # 输出根目录（会在其下创建 <video_name>/ 子目录）
    model_path: str = "e:\\learn\\TrackNetV3_migrated\\model_best.pth",  # TrackNetV3 权重路径（默认指向本机路径）
    num_frames: int = 3,  # TrackNetV3 使用的帧堆叠数量（常见为 3）
    threshold: float = 0.5,  # 羽毛球检测阈值（越大越严格，误检少但漏检可能增多）
    traj_len: int = 10,  # 合成视频中可视化显示的轨迹长度（显示最近 N 帧的轨迹）
    device: str = 'cuda',  # 推理设备（'cuda' 或 'cpu'），传递给部分模型
    pose_model: str = 'rtmpose-m',  # 姿态模型规格（影响精度与速度）
    use_court_detection: bool = True,  # 是否启用球场/球网逐帧检测（用于选手分配与可视化）
    court_model_path: str = "models/court_kpRCNN.pth",  # 球场检测模型权重路径（相对路径）
    court_detection_interval: int = 30  # 球场/球网检测间隔（每 N 帧重新检测一次）
):  # 执行当前语句（保持与上文逻辑一致）
    """
    运行“球检测 + 姿态检测 + 事件检测 +（可选）击球类型 + 合成视频 + CSV 导出”的完整流程。

    参数说明（只描述关键语义，细节由各子模块定义）：
    - video_path: 输入视频文件路径。
    - result_dir: 输出根目录；实际会在其下创建子目录 <video_name>/。
    - model_path/num_frames/threshold: TrackNetV3 的权重与推理参数。
    - traj_len: 可视化叠加的轨迹长度（显示最近 N 帧轨迹）。
    - device: 推理设备字符串（例如 'cuda' 或 'cpu'），传递给部分子模块。
    - pose_model: MMPose 模型尺度选择（如 rtmpose-m）。
    - use_court_detection: 是否启用逐帧球场/球网关键点检测。
    - court_model_path/court_detection_interval: 球场检测权重路径与检测间隔。

    产物（文件名约定）：
    - loca_info/<video>.json: 原始球位置（可能包含不可见/缺失）
    - loca_info_denoise/<video>.json: 去噪后的球位置（供事件检测/可视化优先使用）
    - <video>_poses.npy: 姿态序列
    - <video>_hit_events.json: 击球事件（击球帧、击球方等）
    - <video>_stroke_types.json: 击球类型（如果分类成功）
    - <video>_combined.mp4: 合成可视化视频（球+骨架+提示）
    - <video>_data.csv: 汇总表
    """
    # 将输出目录转换为 Path，统一后续路径拼接方式
    result_dir = Path(result_dir)  # 将字符串路径转换为 Path，后续用 / 拼接子路径更安全
    # 确保输出根目录存在；parents=True 允许递归创建多级目录
    result_dir.mkdir(parents=True, exist_ok=True)  # 递归创建输出目录；存在则不报错

    # 视频名用于组织输出目录与输出文件名前缀（不带扩展名）
    video_name = Path(video_path).stem  # 取视频文件名（不含扩展名），作为输出子目录与文件前缀
    
    # 控制台打印一些关键参数，便于定位本次分析运行的配置
    print("=" * 60)  # 分隔线：增强终端输出可读性
    print("TrackNetV3 + MMPose Combined Pipeline")  # 打印管线名称，便于确认运行的脚本
    print("=" * 60)  # 分隔线：增强终端输出可读性
    print(f"Video: {video_path}")  # 打印输入视频路径，便于核对输入
    print(f"Result directory: {result_dir}")  # 打印输出根目录，便于定位产物
    print(f"Model: {model_path}")  # 打印 TrackNet 权重路径，便于排查权重是否正确
    print(f"Number of frames: {num_frames}")  # 打印 TrackNet 帧堆叠参数，便于复现实验配置
    print(f"Detection threshold: {threshold}")  # 打印阈值参数，便于理解检测严格程度
    print(f"Trajectory length: {traj_len}")  # 打印轨迹显示长度，影响可视化效果
    print(f"Device: {device}")  # 打印推理设备，便于判断是否走 GPU
    print(f"Pose model: {pose_model}")  # 打印姿态模型规格，便于判断速度/精度
    print(f"Use court detection: {use_court_detection}")  # 打印是否开启球场/球网检测
    print("=" * 60)  # 分隔线：增强终端输出可读性

    # 每个视频单独一个子目录，避免多次运行相互覆盖
    video_result_dir = result_dir / video_name  # 每个视频单独一个输出子目录，避免不同视频产物混在一起
    video_result_dir.mkdir(parents=True, exist_ok=True)  # 确保视频输出子目录存在
    
    # TrackNetV3 输出的 JSON 位置：原始与去噪（去噪更适合事件检测）
    ball_json_path = video_result_dir / "loca_info" / f"{video_name}.json"  # TrackNet 输出的原始球位置 JSON 路径
    ball_denoise_json_path = video_result_dir / "loca_info_denoise" / f"{video_name}.json"  # 去噪球位置 JSON 路径（更适合事件检测）
    
    # 姿态输出是一个 numpy 数组文件（每帧一组关键点）
    poses_path = video_result_dir / f"{video_name}_poses.npy"  # 姿态检测输出的 npy 路径（每帧关键点）
    
    # 三段视频：球轨迹可视化、姿态可视化、最终合成可视化
    ball_video_path = video_result_dir / f"{video_name}_with_trajectory_attention.mp4"  # TrackNet 轨迹可视化视频路径
    poses_video_path = video_result_dir / f"{video_name}_with_poses.mp4"  # 姿态可视化视频路径（当前脚本未生成该文件）
    combined_video_path = video_result_dir / f"{video_name}_combined.mp4"  # 合成可视化视频路径（最终输出）

    # 球场/球网检测相关变量：
    # - court_info: 某些可视化函数需要的“球场信息”结构
    # - extended_court_points/court_boundary_params: 用于后续“基于球场区域的选手分配/约束”
    # - partitioned_keypoints/net_keypoints: 归类后的关键点（例如上/下半场等）
    court_info = None  # 球场检测结果（供可视化叠加/辅助逻辑使用）
    extended_court_points = None  # 球场扩展关键点（供基于球场的约束/推断使用）
    court_boundary_params = None  # 球场边界参数（供姿态检测的选手分配等使用）
    partitioned_keypoints = None  # 球场分区关键点（例如分上/下半场）
    net_keypoints = None  # 球网分区关键点（用于可视化/区域划分）
    
    # 逐帧保存球场/球网关键点：可用于后续逐帧叠加可视化（避免只用某一帧的检测结果）
    per_frame_court_keypoints = []  # 逐帧缓存球场关键点（长度应与视频帧数一致）
    per_frame_net_keypoints = []  # 逐帧缓存球网关键点（长度应与视频帧数一致）
    
    if use_court_detection:  # 只有开启球场检测时才执行逐帧检测逻辑（耗时较大）
        # Step 0：逐帧球场/球网检测（通常每 N 帧跑一次检测，其他帧复用最近结果）
        print("\nStep 0: Court and net detection (per-frame)...")  # 提示当前步骤：球场/球网检测
        court_detector = CourtDetector(model_path=court_model_path, device=device)  # 初始化球场检测器（加载权重并选择设备）
        net_detector = NetDetector(model_path="models/net_kpRCNN.pth", device=device)  # 初始化球网检测器（权重路径在此处写死）
        
        # 打开视频以遍历帧；这里使用 OpenCV 的 VideoCapture
        cap = cv2.VideoCapture(video_path)  # 打开视频文件，准备逐帧读取
        # CAP_PROP_FRAME_COUNT 是容器层面的帧数估计，少数编码格式可能不准
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧总数（部分编码可能返回估计值）
        
        print(f"Total frames: {total_frames}")  # 打印帧数，便于估算运行时间
        print(f"Detection interval: every {court_detection_interval} frames")  # 打印检测间隔，决定检测频率与速度
        
        for frame_idx in range(total_frames):  # 遍历每一帧，按间隔刷新检测结果并缓存到逐帧数组
            # 通过随机访问方式跳到指定帧；实现简单，但可能比顺序 read 更慢
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # 设置读取位置到指定帧索引（随机访问）
            ret, frame = cap.read()  # 读取一帧：ret 表示是否成功，frame 为 BGR 图像
            
            if not ret:  # 读取失败：可能是视频损坏、索引越界或解码问题
                # 读帧失败时用 None 占位，保证数组索引与帧号一一对应
                per_frame_court_keypoints.append(None)  # 记录该帧球场关键点缺失
                per_frame_net_keypoints.append(None)  # 记录该帧球网关键点缺失
                continue  # 跳过本帧的检测与缓存逻辑，进入下一帧
            
            if frame_idx % court_detection_interval == 0:  # 到达检测间隔：重新跑一次球场/球网检测并刷新当前结果
                # reset() 通常用于清掉上一帧的缓存状态，确保检测不被历史影响
                court_detector.reset()  # 重置检测器内部状态（例如跟踪缓存），让本次检测更干净
                # get_court_info 返回（结果, 是否检测到球场）
                court_info_result, have_court = court_detector.get_court_info(frame)  # 执行球场检测并返回结果与是否检测到
                
                if have_court:  # 如果检测到球场，则更新当前可用的球场信息
                    # 一旦检测到球场，保存当前帧的球场信息供后续模块使用
                    court_info = court_info_result  # 保存球场信息结构（用于可视化等）
                    extended_court_points = court_detector._CourtDetector__extended_court_points  # 读取内部扩展关键点（私有字段访问）
                    court_boundary_params = court_detector.get_court_boundary_params()  # 计算球场边界参数（用于约束/分配）
                    partitioned_keypoints = court_detector.get_partitioned_keypoints()  # 获取分区后的关键点（用于定位上/下半场）
                else:  # 未检测到球场：清空当前 interval 的球场关键点，避免沿用旧的错误结果
                    # 未检测到则清空本次 interval 的球场关键点
                    partitioned_keypoints = None  # 标记当前没有可用球场关键点
                
                # 球网检测与球场检测类似：返回（结果, 是否检测到球网）
                net_info_result, have_net = net_detector.get_net_info(frame)  # 执行球网检测并返回结果与是否检测到
                if have_net:  # 检测到球网：更新当前可用的球网关键点
                    net_keypoints = net_detector.get_partitioned_keypoints()  # 获取分区后的球网关键点
                else:  # 未检测到球网：清空当前球网关键点
                    net_keypoints = None  # 标记当前没有可用球网关键点
            
            # 将“当前 interval 最近一次的检测结果”写入逐帧列表
            per_frame_court_keypoints.append(partitioned_keypoints)  # 缓存该帧球场关键点（可能为 None 或复用的上次结果）
            per_frame_net_keypoints.append(net_keypoints)  # 缓存该帧球网关键点（可能为 None 或复用的上次结果）
            
            if frame_idx % 100 == 0:  # 每 100 帧打印一次进度，减少终端输出量
                # 控制台进度提示，避免运行时看起来“卡死”
                print(f"  Processed {frame_idx}/{total_frames} frames...")  # 打印检测进度
        
        # 释放视频句柄
        cap.release()  # 释放视频资源，避免文件句柄占用与内存泄漏
        
        # 统计检测成功率，用于评估球场模型在该视频上的适配程度
        successful_detections = sum(1 for kp in per_frame_court_keypoints if kp is not None)  # 统计球场检测成功的帧数
        print(f"Court detection completed!")  # 提示球场检测阶段结束
        print(f"  - Successful detections: {successful_detections}/{total_frames} frames")  # 打印成功检测帧数
        print(f"  - Detection rate: {successful_detections/total_frames*100:.1f}%")  # 打印检测成功率百分比

    # Step 1：运行 TrackNetV3 做羽毛球检测，并生成球轨迹 JSON 与球轨迹可视化视频
    print("\nStep 1: Ball detection with TrackNetV3...")  # 提示当前步骤：TrackNet 羽毛球检测
    ball_detect(video_path, str(video_result_dir), model_path, num_frames, threshold)  # 调用球检测：输出 JSON 与轨迹视频到 video_result_dir
    print(f"Ball detection completed!")  # 提示球检测完成
    print(f"  - Ball JSON: {ball_json_path}")  # 打印球 JSON 路径，便于定位文件
    print(f"  - Ball video: {ball_video_path}")  # 打印球轨迹视频路径，便于播放检查

    # Step 2：运行 MMPose 做姿态检测，返回 poses（每帧关键点）与视频信息（fps 等）
    print("\nStep 2: Pose detection with MMPose...")  # 提示当前步骤：姿态检测
    detector = PoseDetector(device=device, model=pose_model, use_court_based=use_court_detection)  # 初始化姿态检测器（选择设备与模型规格）
    
    if use_court_detection and court_boundary_params is not None:  # 只有在球场检测有效时才给姿态检测器注入球场边界
        # 将球场边界等信息交给 PoseDetector，使其可以进行“基于球场的选手分配/筛选”
        detector.set_court_info(court_boundary_params, extended_court_points)  # 设置球场信息，用于后续基于场地的选手分配/约束
    
    poses, video_info = detector.detect_video(video_path)  # 对整段视频做姿态推理，返回姿态数组与视频信息（fps 等）
    # 保存姿态结果到 npy，供后续复盘与可视化读取
    detector.save_poses(poses, str(poses_path))  # 将姿态数组保存为 .npy 文件，供后续模块读取
    print(f"Pose detection completed!")  # 提示姿态检测完成
    print(f"  - Poses array: {poses_path}")  # 打印姿态文件路径
    print(f"  - Poses shape: {poses.shape}")  # 打印姿态数组形状，便于核对帧数与关键点维度

    # Step 2.5：事件检测（击球帧捕捉）
    # 思路：从去噪球轨迹 JSON 取出每帧 (x, y) 或 None，结合姿态序列检测击球帧
    print("\nStep 2.5: Event detection (hitting frame capture)...")  # 提示当前步骤：击球事件检测
    # ball_denoise_json_path 由 ball_detect 产出；这里用 read_json 读取成 dict
    ball_data = read_json(str(ball_denoise_json_path))  # 读取去噪球轨迹 JSON（key 为帧号字符串）
    trajectory_data = []  # 初始化轨迹列表：每一帧对应 [x,y] 或 None
    for frame_idx in range(len(ball_data)):  # 按帧遍历 JSON（假设 key 数量与帧数一致）
        # JSON 的 key 是字符串帧号，因此这里把 int 转成 str
        frame_key = str(frame_idx)  # 将帧号转字符串，用于访问 JSON 字典的 key
        if frame_key in ball_data:  # 正常情况：该帧在 JSON 中有记录（可能 visible=0）
            frame_data = ball_data[frame_key]  # 取出该帧记录（通常包含 x、y、visible 等字段）
            # 缺失字段时用 0 兜底，避免 KeyError
            x = frame_data.get('x', 0)  # 读取 x 坐标（缺失则用 0 兜底）
            y = frame_data.get('y', 0)  # 读取 y 坐标（缺失则用 0 兜底）
            visible = frame_data.get('visible', 0)  # 读取可见性标志（1=可见，0=不可见）
            if visible == 1 and x > 0 and y > 0:  # 仅当可见且坐标为正时，认为该帧球位置有效
                # visible==1 且坐标为正：认为该帧球位置可信
                trajectory_data.append([x, y])  # 记录有效球坐标，供事件检测与分类器使用
            else:  # 不可见或坐标异常：用 None 表示缺失
                # 不可见或异常坐标：用 None 占位（下游算法需要处理缺失）
                trajectory_data.append(None)  # 记录缺失，EventDetector 需要对缺失做鲁棒处理
        else:  # 异常情况：该帧号不存在于 JSON（用 None 占位以对齐索引）
            # 若该帧完全不存在（理论上不应发生），也用 None 占位
            trajectory_data.append(None)  # 记录缺失，避免索引错位

    # EventDetector 会把轨迹与姿态结合，找出击球的峰值/转折等特征帧
    event_detector = EventDetector(trajectory_data, poses)  # 初始化事件检测器（需要轨迹与姿态共同判断击球）
    hit_frames, hit_players = event_detector.detect_hits(  # 运行击球检测，输出击球帧列表与击球方/选手标识
        fps=video_info['fps'],  # 使用视频帧率把“帧间距”等阈值与时间尺度对应起来
        prominence=1.0,  # 峰值显著性阈值（用于检测轨迹变化的强度）
        angle_threshold=15,  # 角度变化阈值（用于识别急转/反弹等击球特征）
        min_frame_gap=5,  # 两次击球之间的最小帧间隔（抑制重复检测）
        min_continuation_frames=2,  # 轨迹持续性要求（过滤短噪声片段）
        min_movement_threshold=5  # 最小位移阈值（过滤几乎不动的噪声）
    )  # detect_hits 调用结束：得到击球帧与击球方结果
    
    # 保存击球事件，供 UI 复盘与 CSV 导出使用
    hit_events_path = video_result_dir / f"{video_name}_hit_events.json"  # 事件 JSON 输出路径
    event_detector.save_hit_events(str(hit_events_path))  # 保存事件结果到 JSON，供 UI/导出使用
    
    print(f"Event detection completed!")  # 提示事件检测完成
    print(f"  - Hit frames detected: {len(hit_frames)}")  # 打印检测到的击球次数
    print(f"  - Hit events JSON: {hit_events_path}")  # 打印事件 JSON 路径
    if len(hit_frames) > 0:  # 仅在确实检测到击球事件时才打印示例内容
        print(f"  - First 5 hit frames: {hit_frames[:5]}")  # 打印前 5 个击球帧索引，便于抽查
        print(f"  - First 5 hit players: {hit_players[:5]}")  # 打印前 5 个击球方结果，便于抽查

    # Step 2.6：击球类型识别（可选）
    # 这个步骤通常依赖额外的分类器权重；失败时允许继续跑后续步骤
    print("\nStep 2.6: Stroke type classification...")  # 提示当前步骤：击球类型分类（可选）
    try:  # 用 try 包住分类步骤：分类失败不应阻断整个管线
        classifier = create_classifier(dataset='shuttleset', seq_len=100)  # 创建并加载分类器（指定数据集与序列长度）
        stroke_types = classifier.classify_hits(trajectory_data, poses, hit_frames)  # 对每个击球事件输出一个类别 ID
        
        stroke_results_path = video_result_dir / f"{video_name}_stroke_types.json"  # 分类结果 JSON 输出路径
        classifier.save_stroke_results(hit_frames, hit_players, stroke_types, str(stroke_results_path))  # 保存分类结果（含击球帧、击球方、类别）
        
        print(f"Stroke classification completed!")  # 提示击球类型分类完成
        print(f"  - Stroke types classified: {len(stroke_types)}")  # 打印分类数量（应与 hit_frames 数一致）
        print(f"  - Stroke results JSON: {stroke_results_path}")  # 打印分类结果 JSON 路径
        if len(stroke_types) > 0:  # 仅在有分类结果时打印示例
            print(f"  - First 5 stroke types: {[classifier.get_stroke_type_name(st) for st in stroke_types[:5]]}")  # 打印前 5 个类别名称
    except Exception as e:  # 捕获分类阶段异常（例如缺少权重、shape 不匹配、模型加载失败等）
        # 分类失败时：用 -1 填充，避免后续可视化/导出依赖 stroke_types 时报错
        print(f"Stroke classification failed: {e}")  # 打印失败原因，便于排查
        print("Continuing without stroke classification...")  # 明确继续后续步骤（不让用户误以为整体失败）
        stroke_types = [-1] * len(hit_frames)  # 用 -1 填充表示“未知/未分类”，保证下游逻辑仍有对齐长度

    # Step 3：合成可视化视频（球轨迹 + 骨架 + 击球提示）
    print("\nStep 3: Creating combined visualization...")  # 提示当前步骤：生成合成可视化视频
    stroke_type_names = None  # 初始化“类别名称”列表；如果分类成功则填充，否则保持 None
    if 'stroke_types' in locals() and len(stroke_types) > 0:  # 确认 stroke_types 存在且非空（避免未定义）
        # 可视化通常显示可读名称而不是数字类别
        stroke_type_names = [classifier.get_stroke_type_name(st) for st in stroke_types]  # 将类别 ID 映射为可读中文/英文名称
    
    create_combined_visualization(  # 调用合成可视化：叠加球轨迹、骨架、球场/球网与击球提示
        video_path=video_path,  # 输入原视频路径
        ball_json_path=str(ball_denoise_json_path),  # 使用去噪球轨迹 JSON（更稳定）
        poses_path=str(poses_path),  # 姿态 npy 路径（供可视化读取）
        output_path=str(combined_video_path),  # 合成视频输出路径
        traj_len=traj_len,  # 轨迹显示长度
        court_keypoints=court_info,  # 球场信息（用于叠加绘制）
        partitioned_keypoints=partitioned_keypoints,  # 球场分区关键点（用于绘制/定位）
        net_keypoints=net_keypoints,  # 球网关键点（用于绘制）
        hit_frames=hit_frames,  # 击球帧列表（用于在击球时刻加提示）
        per_frame_court_keypoints=per_frame_court_keypoints,  # 逐帧球场关键点（用于更连贯的绘制）
        per_frame_net_keypoints=per_frame_net_keypoints,  # 逐帧球网关键点（用于更连贯的绘制）
        stroke_types=stroke_type_names  # 击球类型名称列表（用于在击球提示中显示）
    )  # 合成可视化结束：输出 combined_video_path 文件
    print(f"Combined visualization completed!")  # 提示合成可视化完成
    print(f"  - Combined video: {combined_video_path}")  # 打印合成视频路径

    # Step 4：将关键结果汇总导出为 CSV（便于下游统计分析/标注）
    print("\nStep 4: Exporting data to CSV...")  # 提示当前步骤：导出 CSV
    csv_path = video_result_dir / f"{video_name}_data.csv"  # CSV 输出路径（汇总表）
    stroke_types_path = video_result_dir / f"{video_name}_stroke_types.json"  # 分类结果 JSON 路径（若存在则用于导出）
    if not stroke_types_path.exists():  # 如果分类结果文件不存在，说明分类失败或未运行
        # 没有击球类型文件时传 None，导出逻辑会跳过相关列
        stroke_types_path = None  # 用 None 表示没有击球类型数据可供导出
    export_to_csv(  # 调用导出模块：把事件/姿态/球轨迹汇总成结构化表格
        hit_events_path=str(hit_events_path),  # 输入：击球事件 JSON
        poses_path=str(poses_path),  # 输入：姿态 npy
        ball_json_path=str(ball_json_path),  # 输入：原始球轨迹 JSON
        ball_denoise_json_path=str(ball_denoise_json_path),  # 输入：去噪球轨迹 JSON
        output_csv_path=str(csv_path),  # 输出：CSV 文件路径
        fps=video_info['fps'],  # 传入 fps，用于把帧号换算成时间等字段
        stroke_types_path=str(stroke_types_path) if stroke_types_path else None  # 可选：击球类型 JSON 路径（不存在则传 None）
    )  # CSV 导出结束：输出 csv_path 文件
    print(f"CSV export completed!")  # 提示 CSV 导出完成
    print(f"  - CSV file: {csv_path}")  # 打印 CSV 路径

    # 汇总输出路径，方便直接在终端复制/定位产物
    print("\n" + "=" * 60)  # 分隔线：输出结束区块更醒目
    print("Pipeline completed successfully!")  # 打印整体完成提示
    print("=" * 60)  # 分隔线：结束区块边界
    print("\nOutput files:")  # 打印产物列表标题
    print(f"1. Ball trajectory video: {ball_video_path}")  # 产物 1：球轨迹可视化视频
    print(f"2. Poses array: {poses_path}")  # 产物 2：姿态数组文件
    print(f"3. Hit events JSON: {hit_events_path}")  # 产物 3：击球事件 JSON
    print(f"4. Stroke types JSON: {stroke_results_path if 'stroke_results_path' in locals() else 'N/A'}")  # 产物 4：击球类型 JSON（可能不存在）
    print(f"5. Combined video (ball + poses): {combined_video_path}")  # 产物 5：合成可视化视频
    print(f"6. Data CSV: {csv_path}")  # 产物 6：CSV 汇总表
    print("=" * 60)  # 分隔线：结束区块边界


if __name__ == "__main__":  # 脚本作为主程序运行时才执行命令行解析与管线调用（被 import 时不执行）
    # 命令行参数解析：将用户输入映射到 run_combined_pipeline 的参数
    parser = argparse.ArgumentParser(description='Run combined TrackNetV3 and MMPose pipeline for badminton video analysis')  # 创建参数解析器并设置说明文本
    
    parser.add_argument('--video', type=str, required=True, help='Input video path')  # 必填参数：输入视频路径
    parser.add_argument('--result_dir', type=str, default='./results', help='Result directory')  # 可选参数：输出目录（相对或绝对）
    parser.add_argument('--model', type=str, default='e:\\learn\\TrackNetV3_migrated\\model_best.pth', help='Model path')  # 可选参数：TrackNet 权重路径
    parser.add_argument('--num_frames', type=int, default=3, help='Number of frames for TrackNetV3')  # 可选参数：TrackNet 帧堆叠数
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (0.0-1.0)')  # 可选参数：检测阈值
    parser.add_argument('--traj_len', type=int, default=10, help='Trajectory length to display')  # 可选参数：轨迹显示长度
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')  # 可选参数：推理设备
    parser.add_argument('--pose_model', type=str, default='rtmpose-m',  # 可选参数：姿态模型规格（决定速度/精度）
                        choices=['rtmpose-t', 'rtmpose-s', 'rtmpose-m', 'rtmpose-l', 'rtmpose-x'],  # 限制可选值，避免拼写错误
                        help='MMPose model size (t=tiny, s=small, m=medium, l=large, x=xlarge)')  # 帮助文本：解释规格含义
    parser.add_argument('--use_court_detection', action='store_true', default=True,  # 开关参数：命令行存在则为 True（此处默认也为 True）
                        help='Use court detection for player assignment')  # 帮助文本：是否启用球场检测辅助分配
    parser.add_argument('--court_model', type=str, default='models/court_kpRCNN.pth',  # 可选参数：球场模型权重路径
                        help='Court detection model path')  # 帮助文本：球场模型路径
    parser.add_argument('--court_detection_interval', type=int, default=30,  # 可选参数：球场检测间隔（帧）
                        help='Court detection interval in frames (default: 30)')  # 帮助文本：间隔含义
    
    # 真正解析命令行
    args = parser.parse_args()  # 解析命令行参数并生成 Namespace 对象
    
    # 将解析到的参数传入管线编排函数
    run_combined_pipeline(  # 调用管线编排函数，将命令行参数映射到函数参数
        video_path=args.video,  # 输入视频路径
        result_dir=args.result_dir,  # 输出目录
        model_path=args.model,  # TrackNet 权重路径
        num_frames=args.num_frames,  # TrackNet 帧堆叠数
        threshold=args.threshold,  # 检测阈值
        traj_len=args.traj_len,  # 轨迹显示长度
        device=args.device,  # 推理设备
        pose_model=args.pose_model,  # 姿态模型规格
        use_court_detection=args.use_court_detection,  # 是否启用球场检测
        court_model_path=args.court_model,  # 球场模型路径
        court_detection_interval=args.court_detection_interval  # 球场检测间隔
    )  # 运行结束后，产物写入 results/<video_name>/ 下
