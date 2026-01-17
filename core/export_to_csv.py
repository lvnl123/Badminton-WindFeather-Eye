import json  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
import pandas as pd  # 导入模块，供后续使用
from pathlib import Path  # 从模块导入符号，供后续调用


def export_to_csv(  # 定义函数（封装可复用逻辑）
    hit_events_path: str,  # 执行当前语句（保持与上文逻辑一致）
    poses_path: str,  # 执行当前语句（保持与上文逻辑一致）
    ball_json_path: str,  # 执行当前语句（保持与上文逻辑一致）
    ball_denoise_json_path: str,  # 执行当前语句（保持与上文逻辑一致）
    output_csv_path: str,  # 执行当前语句（保持与上文逻辑一致）
    fps: float = 25.0,  # 执行当前语句（保持与上文逻辑一致）
    stroke_types_path: str = None  # 执行当前语句（保持与上文逻辑一致）
):  # 执行当前语句（保持与上文逻辑一致）
    hit_events = []  # 初始化变量 hit_events 为一个容器/表达式结果
    with open(hit_events_path, 'r') as f:  # 上下文管理：确保资源正确释放
        hit_events = json.load(f)  # 将 hit_events 设为一次调用/构造的返回值
    
    hit_frames_set = set(event['frame'] for event in hit_events)  # 将 hit_frames_set 设为一次调用/构造的返回值
    hit_frame_to_player = {event['frame']: event['player'] for event in hit_events}  # 初始化变量 hit_frame_to_player 为一个容器/表达式结果
    
    stroke_types = {}  # 初始化变量 stroke_types 为一个容器/表达式结果
    if stroke_types_path and Path(stroke_types_path).exists():  # 条件分支判断并选择执行路径
        with open(stroke_types_path, 'r') as f:  # 上下文管理：确保资源正确释放
            stroke_data = json.load(f)  # 将 stroke_data 设为一次调用/构造的返回值
            stroke_types = {event['frame']: event for event in stroke_data}  # 初始化变量 stroke_types 为一个容器/表达式结果
    
    with open(ball_json_path, 'r') as f:  # 上下文管理：确保资源正确释放
        ball_data = json.load(f)  # 将 ball_data 设为一次调用/构造的返回值
    
    with open(ball_denoise_json_path, 'r') as f:  # 上下文管理：确保资源正确释放
        ball_denoise_data = json.load(f)  # 将 ball_denoise_data 设为一次调用/构造的返回值
    
    poses = np.load(poses_path)  # 将 poses 设为一次调用/构造的返回值
    total_frames = poses.shape[0]  # 将表达式计算结果赋给变量 total_frames
    
    rows = []  # 初始化变量 rows 为一个容器/表达式结果
    cumulative_hit_count = 0  # 将表达式计算结果赋给变量 cumulative_hit_count
    
    for frame_idx in range(total_frames):  # 循环遍历序列/迭代器
        frame_str = str(frame_idx)  # 将 frame_str 设为一次调用/构造的返回值
        
        ball_info = ball_data.get(frame_str, {'visible': 0, 'x': 0, 'y': 0})  # 将 ball_info 设为一次调用/构造的返回值
        ball_denoise_info = ball_denoise_data.get(frame_str, {'visible': 0, 'x': 0, 'y': 0})  # 将 ball_denoise_info 设为一次调用/构造的返回值
        
        ball_x = ball_info.get('x', 0)  # 将 ball_x 设为一次调用/构造的返回值
        ball_y = ball_info.get('y', 0)  # 将 ball_y 设为一次调用/构造的返回值
        ball_visible = ball_info.get('visible', 0)  # 将 ball_visible 设为一次调用/构造的返回值
        
        ball_denoise_x = ball_denoise_info.get('x', 0)  # 将 ball_denoise_x 设为一次调用/构造的返回值
        ball_denoise_y = ball_denoise_info.get('y', 0)  # 将 ball_denoise_y 设为一次调用/构造的返回值
        ball_denoise_visible = ball_denoise_info.get('visible', 0)  # 将 ball_denoise_visible 设为一次调用/构造的返回值
        
        is_hit = frame_idx in hit_frames_set  # 将表达式计算结果赋给变量 is_hit
        hit_player = hit_frame_to_player.get(frame_idx, 0)  # 将 hit_player 设为一次调用/构造的返回值
        
        stroke_type_id = -1  # 将表达式计算结果赋给变量 stroke_type_id
        stroke_type_name = ''  # 将表达式计算结果赋给变量 stroke_type_name
        stroke_type_name_en = ''  # 将表达式计算结果赋给变量 stroke_type_name_en
        if frame_idx in stroke_types:  # 条件分支判断并选择执行路径
            stroke_info = stroke_types[frame_idx]  # 将表达式计算结果赋给变量 stroke_info
            stroke_type_id = stroke_info.get('stroke_type_id', -1)  # 将 stroke_type_id 设为一次调用/构造的返回值
            stroke_type_name = stroke_info.get('stroke_type_name', '')  # 将 stroke_type_name 设为一次调用/构造的返回值
            stroke_type_name_en = stroke_info.get('stroke_type_name_en', '')  # 将 stroke_type_name_en 设为一次调用/构造的返回值
        
        if is_hit:  # 条件分支判断并选择执行路径
            cumulative_hit_count += 1  # 执行当前语句（保持与上文逻辑一致）
        
        ball_speed = 0  # 将表达式计算结果赋给变量 ball_speed
        if frame_idx > 0:  # 条件分支判断并选择执行路径
            prev_frame_str = str(frame_idx - 1)  # 将 prev_frame_str 设为一次调用/构造的返回值
            prev_ball = ball_denoise_data.get(prev_frame_str, {'visible': 0, 'x': 0, 'y': 0})  # 将 prev_ball 设为一次调用/构造的返回值
            if prev_ball['visible'] == 1 and ball_denoise_visible == 1:  # 条件分支判断并选择执行路径
                dx = ball_denoise_x - prev_ball['x']  # 将表达式计算结果赋给变量 dx
                dy = ball_denoise_y - prev_ball['y']  # 将表达式计算结果赋给变量 dy
                distance = np.sqrt(dx**2 + dy**2)  # 将 distance 设为一次调用/构造的返回值
                ball_speed = distance * fps  # 将表达式计算结果赋给变量 ball_speed
        
        row = {  # 初始化变量 row 为一个容器/表达式结果
            'frame': frame_idx,  # 执行当前语句（保持与上文逻辑一致）
            'time_seconds': frame_idx / fps,  # 执行当前语句（保持与上文逻辑一致）
            'ball_x': ball_x,  # 执行当前语句（保持与上文逻辑一致）
            'ball_y': ball_y,  # 执行当前语句（保持与上文逻辑一致）
            'ball_visible': ball_visible,  # 执行当前语句（保持与上文逻辑一致）
            'ball_denoise_x': ball_denoise_x,  # 执行当前语句（保持与上文逻辑一致）
            'ball_denoise_y': ball_denoise_y,  # 执行当前语句（保持与上文逻辑一致）
            'ball_denoise_visible': ball_denoise_visible,  # 执行当前语句（保持与上文逻辑一致）
            'ball_speed': ball_speed,  # 执行当前语句（保持与上文逻辑一致）
            'is_hit': 1 if is_hit else 0,  # 执行当前语句（保持与上文逻辑一致）
            'hit_player': hit_player,  # 执行当前语句（保持与上文逻辑一致）
            'cumulative_hit_count': cumulative_hit_count,  # 执行当前语句（保持与上文逻辑一致）
            'stroke_type_id': stroke_type_id,  # 执行当前语句（保持与上文逻辑一致）
            'stroke_type_name': stroke_type_name,  # 执行当前语句（保持与上文逻辑一致）
            'stroke_type_name_en': stroke_type_name_en  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        
        for player_idx in range(2):  # 循环遍历序列/迭代器
            for joint_idx in range(17):  # 循环遍历序列/迭代器
                joint = poses[frame_idx, player_idx, joint_idx]  # 将表达式计算结果赋给变量 joint
                row[f'player{player_idx+1}_joint{joint_idx}_x'] = joint[0]  # 执行当前语句（保持与上文逻辑一致）
                row[f'player{player_idx+1}_joint{joint_idx}_y'] = joint[1]  # 执行当前语句（保持与上文逻辑一致）
        
        rows.append(row)  # 调用函数/方法执行某个动作或计算
    
    df = pd.DataFrame(rows)  # 将 df 设为一次调用/构造的返回值
    df.to_csv(output_csv_path, index=False)  # 调用函数/方法执行某个动作或计算
    print(f"CSV file exported to: {output_csv_path}")  # 调用函数/方法执行某个动作或计算
    print(f"Total frames: {len(df)}")  # 调用函数/方法执行某个动作或计算
    print(f"Total hits: {cumulative_hit_count}")  # 调用函数/方法执行某个动作或计算
    print(f"Columns: {len(df.columns)}")  # 调用函数/方法执行某个动作或计算
    
    return df  # 从函数返回结果


if __name__ == "__main__":  # 条件分支判断并选择执行路径
    import argparse  # 导入模块，供后续使用
    
    parser = argparse.ArgumentParser(description='Export video analysis data to CSV')  # 将 parser 设为一次调用/构造的返回值
    parser.add_argument('--hit_events', type=str, required=True, help='Hit events JSON path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--poses', type=str, required=True, help='Poses numpy array path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--ball_json', type=str, required=True, help='Ball detection JSON path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--ball_denoise_json', type=str, required=True, help='Ball denoised JSON path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV path')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--fps', type=float, default=25.0, help='Video FPS')  # 调用函数/方法执行某个动作或计算
    
    args = parser.parse_args()  # 将 args 设为一次调用/构造的返回值
    
    export_to_csv(  # 执行当前语句（保持与上文逻辑一致）
        args.hit_events,  # 执行当前语句（保持与上文逻辑一致）
        args.poses,  # 执行当前语句（保持与上文逻辑一致）
        args.ball_json,  # 执行当前语句（保持与上文逻辑一致）
        args.ball_denoise_json,  # 执行当前语句（保持与上文逻辑一致）
        args.output_csv,  # 执行当前语句（保持与上文逻辑一致）
        args.fps  # 执行当前语句（保持与上文逻辑一致）
    )  # 执行当前语句（保持与上文逻辑一致）
