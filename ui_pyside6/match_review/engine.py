import pandas as pd  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
import json  # 导入模块，供后续使用
from pathlib import Path  # 从模块导入符号，供后续调用
from dataclasses import dataclass  # 从模块导入符号，供后续调用
from typing import List, Dict, Optional, Tuple  # 从模块导入符号，供后续调用
from scipy.spatial import ConvexHull  # 从模块导入符号，供后续调用
import logging  # 导入模块，供后续使用

# Configure logging
logging.basicConfig(level=logging.INFO)  # 调用函数/方法执行某个动作或计算
logger = logging.getLogger(__name__)  # 将 logger 设为一次调用/构造的返回值

@dataclass  # 装饰器：修改/包装下方函数或类的行为
class Rally:  # 定义类（封装数据与行为）
    id: int  # 执行当前语句（保持与上文逻辑一致）
    start_frame: int  # 执行当前语句（保持与上文逻辑一致）
    end_frame: int  # 执行当前语句（保持与上文逻辑一致）
    duration_sec: float  # 执行当前语句（保持与上文逻辑一致）
    hit_count: int  # 执行当前语句（保持与上文逻辑一致）
    strokes: List[dict]  # 执行当前语句（保持与上文逻辑一致）
    # Trajectory data slice for this rally
    trajectory: pd.DataFrame   # 执行当前语句（保持与上文逻辑一致）
    # Player stats in this rally: {player_id: {'dist': float, 'avg_speed': float, 'max_speed': float}}
    player_stats: Dict[int, dict]   # 执行当前语句（保持与上文逻辑一致）

class MatchEngine:  # 定义类（封装数据与行为）
    def __init__(self, match_dir: str):  # 定义函数（封装可复用逻辑）
        self.match_dir = Path(match_dir)  # 给对象属性 self.match_dir 赋值/初始化（来自当前语句右侧表达式）
        self.match_name = self.match_dir.name  # 给对象属性 self.match_name 赋值/初始化（来自当前语句右侧表达式）
        self.df = None  # 给对象属性 self.df 赋值/初始化（来自当前语句右侧表达式）
        self.hits = []  # 给对象属性 self.hits 赋值/初始化（来自当前语句右侧表达式）
        self.strokes = []  # 给对象属性 self.strokes 赋值/初始化（来自当前语句右侧表达式）
        self.rallies: List[Rally] = []  # 执行当前语句（保持与上文逻辑一致）
        # Global stats
        self.global_player_stats = {  # 给对象属性 self.global_player_stats 赋值/初始化（来自当前语句右侧表达式）
            1: {'dist': 0.0, 'speed_dist': [], 'coverage': 0.0, 'type_counts': {}},  # 执行当前语句（保持与上文逻辑一致）
            2: {'dist': 0.0, 'speed_dist': [], 'coverage': 0.0, 'type_counts': {}}  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        self.poses = None # npy array

    def load_data(self):  # 定义函数（封装可复用逻辑）
        """Load all data files from the directory."""  # 执行当前语句（保持与上文逻辑一致）
        try:  # 开始异常捕获保护块
            # 1. Load CSV
            csv_path = self.match_dir / f"{self.match_name}_data.csv"  # 将表达式计算结果赋给变量 csv_path
            if not csv_path.exists():  # 条件分支判断并选择执行路径
                raise FileNotFoundError(f"CSV not found: {csv_path}")  # 调用函数/方法执行某个动作或计算
            self.df = pd.read_csv(csv_path)  # 给对象属性 self.df 赋值/初始化（来自当前语句右侧表达式）
            
            # Ensure frame is index for faster lookup
            # self.df.set_index('frame', drop=False, inplace=True)

            # 2. Load Hits
            hits_path = self.match_dir / f"{self.match_name}_hit_events.json"  # 将表达式计算结果赋给变量 hits_path
            if hits_path.exists():  # 条件分支判断并选择执行路径
                with open(hits_path, 'r', encoding='utf-8') as f:  # 上下文管理：确保资源正确释放
                    self.hits = json.load(f)  # 给对象属性 self.hits 赋值/初始化（来自当前语句右侧表达式）
            
            # 3. Load Strokes
            strokes_path = self.match_dir / f"{self.match_name}_stroke_types.json"  # 将表达式计算结果赋给变量 strokes_path
            if strokes_path.exists():  # 条件分支判断并选择执行路径
                with open(strokes_path, 'r', encoding='utf-8') as f:  # 上下文管理：确保资源正确释放
                    self.strokes = json.load(f)  # 给对象属性 self.strokes 赋值/初始化（来自当前语句右侧表达式）

            # 4. Load Poses (optional)
            poses_path = self.match_dir / f"{self.match_name}_poses.npy"  # 将表达式计算结果赋给变量 poses_path
            if poses_path.exists():  # 条件分支判断并选择执行路径
                self.poses = np.load(poses_path)  # 给对象属性 self.poses 赋值/初始化（来自当前语句右侧表达式）

            logger.info(f"Loaded match data: {len(self.df)} frames, {len(self.hits)} hits")  # 调用函数/方法执行某个动作或计算
            
            self._process_data()  # 调用函数/方法执行某个动作或计算
            
        except Exception as e:  # 捕获异常并进行处理
            logger.error(f"Error loading data: {e}")  # 调用函数/方法执行某个动作或计算
            raise  # 执行当前语句（保持与上文逻辑一致）

    def _process_data(self):  # 定义函数（封装可复用逻辑）
        """Process raw data into rallies and advanced stats."""  # 执行当前语句（保持与上文逻辑一致）
        self._calculate_velocities()  # 调用函数/方法执行某个动作或计算
        self._segment_rallies()  # 调用函数/方法执行某个动作或计算
        self._calculate_global_stats()  # 调用函数/方法执行某个动作或计算

    def _calculate_velocities(self):  # 定义函数（封装可复用逻辑）
        """Calculate ball and player velocities if not present."""  # 执行当前语句（保持与上文逻辑一致）
        # Calculate player speed (px/frame -> px/sec assuming 30fps if not specified, 
        # but better use time_seconds diff)
        
        dt = self.df['time_seconds'].diff().fillna(0.04) # Default to 0.04s (25fps) if diff is 0
        dt[dt == 0] = 0.04  # 执行当前语句（保持与上文逻辑一致）
        
        # Player 1 (Bottom?) - usually player1_joint0_x/y is root (hips)
        p1_dx = self.df['player1_joint0_x'].diff().fillna(0)  # 将 p1_dx 设为一次调用/构造的返回值
        p1_dy = self.df['player1_joint0_y'].diff().fillna(0)  # 将 p1_dy 设为一次调用/构造的返回值
        self.df['p1_speed'] = np.sqrt(p1_dx**2 + p1_dy**2) / dt  # 执行当前语句（保持与上文逻辑一致）
        
        # Player 2 (Top?)
        p2_dx = self.df['player2_joint0_x'].diff().fillna(0)  # 将 p2_dx 设为一次调用/构造的返回值
        p2_dy = self.df['player2_joint0_y'].diff().fillna(0)  # 将 p2_dy 设为一次调用/构造的返回值
        self.df['p2_speed'] = np.sqrt(p2_dx**2 + p2_dy**2) / dt  # 执行当前语句（保持与上文逻辑一致）
        
        # Smooth speeds
        self.df['p1_speed'] = self.df['p1_speed'].rolling(window=5, min_periods=1).mean()  # 调用函数/方法执行某个动作或计算
        self.df['p2_speed'] = self.df['p2_speed'].rolling(window=5, min_periods=1).mean()  # 调用函数/方法执行某个动作或计算

    def _segment_rallies(self):  # 定义函数（封装可复用逻辑）
        """Segment the match into rallies based on hit events and time gaps."""  # 执行当前语句（保持与上文逻辑一致）
        if not self.hits:  # 条件分支判断并选择执行路径
            # Fallback: treat whole match as one rally if no hits
            return  # 从函数返回结果

        # Sort hits by frame
        sorted_hits = sorted(self.hits, key=lambda x: x['frame'])  # 将 sorted_hits 设为一次调用/构造的返回值
        
        current_rally_hits = []  # 初始化变量 current_rally_hits 为一个容器/表达式结果
        rally_id = 1  # 将表达式计算结果赋给变量 rally_id
        
        # Threshold for new rally: > 4 seconds gap between hits
        NEW_RALLY_THRESHOLD_FRAMES = 30 * 4   # 将表达式计算结果赋给变量 NEW_RALLY_THRESHOLD_FRAMES
        
        last_hit_frame = -999  # 将表达式计算结果赋给变量 last_hit_frame
        
        for hit in sorted_hits:  # 循环遍历序列/迭代器
            frame = hit['frame']  # 将表达式计算结果赋给变量 frame
            
            # Check if this hit starts a new rally
            if frame - last_hit_frame > NEW_RALLY_THRESHOLD_FRAMES and current_rally_hits:  # 条件分支判断并选择执行路径
                # Finish previous rally
                self._create_rally(rally_id, current_rally_hits)  # 调用函数/方法执行某个动作或计算
                rally_id += 1  # 执行当前语句（保持与上文逻辑一致）
                current_rally_hits = []  # 初始化变量 current_rally_hits 为一个容器/表达式结果
            
            # Add stroke info to hit if available
            stroke_info = next((s for s in self.strokes if s['frame'] == frame), None)  # 将 stroke_info 设为一次调用/构造的返回值
            if stroke_info:  # 条件分支判断并选择执行路径
                hit.update(stroke_info)  # 调用函数/方法执行某个动作或计算
                
            current_rally_hits.append(hit)  # 调用函数/方法执行某个动作或计算
            last_hit_frame = frame  # 将表达式计算结果赋给变量 last_hit_frame
            
        # Add last rally
        if current_rally_hits:  # 条件分支判断并选择执行路径
            self._create_rally(rally_id, current_rally_hits)  # 调用函数/方法执行某个动作或计算

    def _create_rally(self, rally_id: int, hits: List[dict]):  # 定义函数（封装可复用逻辑）
        """Create a Rally object from a list of hits."""  # 执行当前语句（保持与上文逻辑一致）
        start_frame = max(0, hits[0]['frame'] - 30) # Start 1 sec before first hit
        end_frame = min(len(self.df)-1, hits[-1]['frame'] + 60) # End 2 sec after last hit
        
        rally_df = self.df.iloc[start_frame:end_frame+1].copy()  # 将 rally_df 设为一次调用/构造的返回值
        duration = rally_df['time_seconds'].max() - rally_df['time_seconds'].min()  # 将 duration 设为一次调用/构造的返回值
        
        # Calculate stats for this rally
        p1_dist = (rally_df['p1_speed'] * rally_df['time_seconds'].diff().fillna(0)).sum()  # 初始化变量 p1_dist 为一个容器/表达式结果
        p2_dist = (rally_df['p2_speed'] * rally_df['time_seconds'].diff().fillna(0)).sum()  # 初始化变量 p2_dist 为一个容器/表达式结果
        
        r = Rally(  # 将表达式计算结果赋给变量 r
            id=rally_id,  # 将表达式计算结果赋给变量 id
            start_frame=start_frame,  # 将表达式计算结果赋给变量 start_frame
            end_frame=end_frame,  # 将表达式计算结果赋给变量 end_frame
            duration_sec=duration,  # 将表达式计算结果赋给变量 duration_sec
            hit_count=len(hits),  # 将 hit_count 设为一次调用/构造的返回值
            strokes=hits,  # 将表达式计算结果赋给变量 strokes
            trajectory=rally_df,  # 将表达式计算结果赋给变量 trajectory
            player_stats={  # 初始化变量 player_stats 为一个容器/表达式结果
                1: {'dist': p1_dist, 'avg_speed': rally_df['p1_speed'].mean(), 'max_speed': rally_df['p1_speed'].max()},  # 执行当前语句（保持与上文逻辑一致）
                2: {'dist': p2_dist, 'avg_speed': rally_df['p2_speed'].mean(), 'max_speed': rally_df['p2_speed'].max()}  # 执行当前语句（保持与上文逻辑一致）
            }  # 执行当前语句（保持与上文逻辑一致）
        )  # 执行当前语句（保持与上文逻辑一致）
        self.rallies.append(r)  # 调用函数/方法执行某个动作或计算

    def _calculate_global_stats(self):  # 定义函数（封装可复用逻辑）
        """Calculate aggregate stats for the whole match."""  # 执行当前语句（保持与上文逻辑一致）
        # Distance
        # We sum distances from all rallies to avoid counting dead time walking
        self.global_player_stats[1]['dist'] = sum(r.player_stats[1]['dist'] for r in self.rallies)  # 调用函数/方法执行某个动作或计算
        self.global_player_stats[2]['dist'] = sum(r.player_stats[2]['dist'] for r in self.rallies)  # 调用函数/方法执行某个动作或计算
        
        # Speed Distribution (sample from all rally frames)
        all_p1_speeds = []  # 初始化变量 all_p1_speeds 为一个容器/表达式结果
        all_p2_speeds = []  # 初始化变量 all_p2_speeds 为一个容器/表达式结果
        for r in self.rallies:  # 循环遍历序列/迭代器
            all_p1_speeds.extend(r.trajectory['p1_speed'].dropna().tolist())  # 调用函数/方法执行某个动作或计算
            all_p2_speeds.extend(r.trajectory['p2_speed'].dropna().tolist())  # 调用函数/方法执行某个动作或计算
            
        self.global_player_stats[1]['speed_dist'] = all_p1_speeds  # 执行当前语句（保持与上文逻辑一致）
        self.global_player_stats[2]['speed_dist'] = all_p2_speeds  # 执行当前语句（保持与上文逻辑一致）
        
        # Stroke Counts
        for r in self.rallies:  # 循环遍历序列/迭代器
            for s in r.strokes:  # 循环遍历序列/迭代器
                p = s.get('player', 0)  # 将 p 设为一次调用/构造的返回值
                st_name = s.get('stroke_type_name', 'Unknown')  # 将 st_name 设为一次调用/构造的返回值
                if p in [1, 2]:  # 条件分支判断并选择执行路径
                    self.global_player_stats[p]['type_counts'][st_name] = (  # 执行当前语句（保持与上文逻辑一致）
                        self.global_player_stats[p]['type_counts'].get(st_name, 0) + 1  # 执行当前语句（保持与上文逻辑一致）
                    )  # 执行当前语句（保持与上文逻辑一致）

        # Court Coverage (Convex Hull Area)
        # Filter valid positions (not 0,0)
        for p in [1, 2]:  # 循环遍历序列/迭代器
            x_col = f'player{p}_joint0_x'  # 将表达式计算结果赋给变量 x_col
            y_col = f'player{p}_joint0_y'  # 将表达式计算结果赋给变量 y_col
            points = self.df[[x_col, y_col]].values  # 将表达式计算结果赋给变量 points
            # Filter out (0,0) or NaN
            mask = (points[:,0] > 10) & (points[:,1] > 10) & ~np.isnan(points[:,0])  # 初始化变量 mask 为一个容器/表达式结果
            valid_points = points[mask]  # 将表达式计算结果赋给变量 valid_points
            
            if len(valid_points) > 3:  # 条件分支判断并选择执行路径
                try:  # 开始异常捕获保护块
                    hull = ConvexHull(valid_points)  # 将 hull 设为一次调用/构造的返回值
                    self.global_player_stats[p]['coverage'] = hull.area  # 执行当前语句（保持与上文逻辑一致）
                except:  # 捕获异常并进行处理
                    self.global_player_stats[p]['coverage'] = 0.0  # 执行当前语句（保持与上文逻辑一致）

    def get_transition_matrix(self, player_id: int) -> pd.DataFrame:  # 定义函数（封装可复用逻辑）
        """
        Calculate the stroke type transition matrix for a player.
        Rows: Previous Stroke Type (by Opponent or Self?)
        Let's do: My Previous Stroke -> My Current Stroke (Chain) OR Opponent Stroke -> My Response
        Professional analysis usually looks at: Opponent Shot -> My Response (Tactical Response)
        """
        transitions = {}  # 初始化变量 transitions 为一个容器/表达式结果
        
        for r in self.rallies:  # 循环遍历序列/迭代器
            strokes = r.strokes  # 将表达式计算结果赋给变量 strokes
            for i in range(1, len(strokes)):  # 循环遍历序列/迭代器
                curr = strokes[i]  # 将表达式计算结果赋给变量 curr
                prev = strokes[i-1]  # 将表达式计算结果赋给变量 prev
                
                if curr.get('player') == player_id and prev.get('player') != player_id:  # 条件分支判断并选择执行路径
                    # Opponent shot -> My response
                    prev_type = prev.get('stroke_type_name', 'Unknown')  # 将 prev_type 设为一次调用/构造的返回值
                    curr_type = curr.get('stroke_type_name', 'Unknown')  # 将 curr_type 设为一次调用/构造的返回值
                    
                    if prev_type not in transitions: transitions[prev_type] = {}  # 条件分支判断并选择执行路径
                    transitions[prev_type][curr_type] = transitions[prev_type].get(curr_type, 0) + 1  # 执行当前语句（保持与上文逻辑一致）

        # Convert to DataFrame
        if not transitions:  # 条件分支判断并选择执行路径
            return pd.DataFrame()  # 从函数返回结果
            
        df = pd.DataFrame(transitions).fillna(0).T # Rows: Opponent Shot, Cols: My Response
        # Normalize by row (probability)
        df_norm = df.div(df.sum(axis=1), axis=0)  # 将 df_norm 设为一次调用/构造的返回值
        return df_norm  # 从函数返回结果

    def get_sankey_data(self) -> dict:  # 定义函数（封装可复用逻辑）
        """
        Generate data for Sankey diagram: Serve -> ... -> End Reason
        Simplified to 3 steps: Service -> 3rd Shot -> Outcome
        """
        flows = [] # (source, target, value)
        
        for r in self.rallies:  # 循环遍历序列/迭代器
            if len(r.strokes) < 1: continue  # 条件分支判断并选择执行路径
            
            # 1. Service
            first_stroke = r.strokes[0]  # 将表达式计算结果赋给变量 first_stroke
            service_type = first_stroke.get('stroke_type_name', 'Serve')  # 将 service_type 设为一次调用/构造的返回值
            
            # 2. Outcome (Last Shot)
            last_stroke = r.strokes[-1]  # 将表达式计算结果赋给变量 last_stroke
            last_type = last_stroke.get('stroke_type_name', 'End')  # 将 last_type 设为一次调用/构造的返回值
            winner = "Rally End" # We don't have score info in json usually, unless inferred
            
            # Simple flow: Service -> Last Shot Type
            flows.append((service_type, last_type))  # 调用函数/方法执行某个动作或计算
            
        # Aggregate
        from collections import Counter  # 从模块导入符号，供后续调用
        counts = Counter(flows)  # 将 counts 设为一次调用/构造的返回值
        
        return {  # 从函数返回结果
            "sources": [k[0] for k in counts.keys()],  # 执行当前语句（保持与上文逻辑一致）
            "targets": [k[1] for k in counts.keys()],  # 执行当前语句（保持与上文逻辑一致）
            "values": list(counts.values())  # 调用函数/方法执行某个动作或计算
        }  # 执行当前语句（保持与上文逻辑一致）

    def get_player_radar_data(self, player_id: int) -> Dict[str, float]:  # 定义函数（封装可复用逻辑）
        stats = self.global_player_stats[player_id]  # 将表达式计算结果赋给变量 stats
        counts = stats['type_counts']  # 将表达式计算结果赋给变量 counts
        total_shots = sum(counts.values()) if counts else 1  # 将 total_shots 设为一次调用/构造的返回值
        
        # 1. Attack: Smashes / Drives
        attack_keywords = ['杀', 'smash', 'drive', '抽']  # 初始化变量 attack_keywords 为一个容器/表达式结果
        attack_count = sum(v for k, v in counts.items() if any(x in k.lower() for x in attack_keywords))  # 将 attack_count 设为一次调用/构造的返回值
        attack_score = min(100, (attack_count / total_shots) * 300) # Heuristic
        
        # 2. Defense: Lifts / Clears (assuming defensive)
        def_keywords = ['挑', 'lift', 'clear', '高远']  # 初始化变量 def_keywords 为一个容器/表达式结果
        def_count = sum(v for k, v in counts.items() if any(x in k.lower() for x in def_keywords))  # 将 def_count 设为一次调用/构造的返回值
        def_score = min(100, (def_count / total_shots) * 300)  # 将 def_score 设为一次调用/构造的返回值
        
        # 3. Speed: Avg speed in rallies
        avg_speed = np.mean(stats['speed_dist']) if stats['speed_dist'] else 0  # 将 avg_speed 设为一次调用/构造的返回值
        speed_score = min(100, avg_speed * 0.5) # px/frame factor
        
        # 4. Stamina: Total Distance / Rallies
        dist_score = min(100, stats['dist'] / 1000) # Normalize
        
        # 5. Control: Net shots / Drops
        ctrl_keywords = ['网', 'net', 'drop', '吊', '放']  # 初始化变量 ctrl_keywords 为一个容器/表达式结果
        ctrl_count = sum(v for k, v in counts.items() if any(x in k.lower() for x in ctrl_keywords))  # 将 ctrl_count 设为一次调用/构造的返回值
        ctrl_score = min(100, (ctrl_count / total_shots) * 400)  # 将 ctrl_score 设为一次调用/构造的返回值
        
        diversity_types = len(counts.keys())  # 将 diversity_types 设为一次调用/构造的返回值
        diversity_score = min(100, diversity_types * 12.5)  # 将 diversity_score 设为一次调用/构造的返回值
        return {  # 从函数返回结果
            "进攻": float(attack_score),  # 执行当前语句（保持与上文逻辑一致）
            "防守": float(def_score),  # 执行当前语句（保持与上文逻辑一致）
            "速度": float(speed_score),  # 执行当前语句（保持与上文逻辑一致）
            "体能": float(dist_score),  # 执行当前语句（保持与上文逻辑一致）
            "控制": float(ctrl_score),  # 执行当前语句（保持与上文逻辑一致）
            "多样性": float(diversity_score)  # 调用函数/方法执行某个动作或计算
        }  # 执行当前语句（保持与上文逻辑一致）

    def get_speed_series(self, player_id: int) -> pd.DataFrame:  # 定义函数（封装可复用逻辑）
        col = 'p1_speed' if player_id == 1 else 'p2_speed'  # 将表达式计算结果赋给变量 col
        return self.df[['time_seconds', col]].rename(columns={col: 'speed'}).dropna()  # 从函数返回结果

    def get_accel_series(self, player_id: int) -> pd.DataFrame:  # 定义函数（封装可复用逻辑）
        s = self.get_speed_series(player_id).copy()  # 将 s 设为一次调用/构造的返回值
        dt = s['time_seconds'].diff().fillna(0.04)  # 将 dt 设为一次调用/构造的返回值
        dv = s['speed'].diff().fillna(0.0)  # 将 dv 设为一次调用/构造的返回值
        accel = (dv / dt).rolling(window=3, min_periods=1).mean()  # 初始化变量 accel 为一个容器/表达式结果
        s['accel'] = accel  # 执行当前语句（保持与上文逻辑一致）
        return s  # 从函数返回结果

    def get_player_zone_ratios(self, player_id: int) -> Dict[str, float]:  # 定义函数（封装可复用逻辑）
        x_col = f'player{player_id}_joint0_x'  # 将表达式计算结果赋给变量 x_col
        y_col = f'player{player_id}_joint0_y'  # 将表达式计算结果赋给变量 y_col
        d = self.df[[x_col, y_col, 'time_seconds']].dropna()  # 将 d 设为一次调用/构造的返回值
        if d.empty:  # 条件分支判断并选择执行路径
            return {'front': 0.0, 'back': 0.0, 'left': 0.0, 'right': 0.0, 'net_aggr': 0.0}  # 从函数返回结果
        W = float(self.df['ball_x'].max() if 'ball_x' in self.df else d[x_col].max())  # 将 W 设为一次调用/构造的返回值
        H = float(self.df['ball_y'].max() if 'ball_y' in self.df else d[y_col].max())  # 将 H 设为一次调用/构造的返回值
        dt = d['time_seconds'].diff().fillna(0.04)  # 将 dt 设为一次调用/构造的返回值
        left_mask = d[x_col] < W * 0.5  # 将表达式计算结果赋给变量 left_mask
        right_mask = ~left_mask  # 将表达式计算结果赋给变量 right_mask
        front_mask = d[y_col] < H * 0.5  # 将表达式计算结果赋给变量 front_mask
        back_mask = ~front_mask  # 将表达式计算结果赋给变量 back_mask
        total = dt.sum()  # 将 total 设为一次调用/构造的返回值
        left = dt[left_mask].sum() / total if total > 0 else 0.0  # 将 left 设为一次调用/构造的返回值
        right = dt[right_mask].sum() / total if total > 0 else 0.0  # 将 right 设为一次调用/构造的返回值
        front = dt[front_mask].sum() / total if total > 0 else 0.0  # 将 front 设为一次调用/构造的返回值
        back = dt[back_mask].sum() / total if total > 0 else 0.0  # 将 back 设为一次调用/构造的返回值
        speed_col = 'p1_speed' if player_id == 1 else 'p2_speed'  # 将表达式计算结果赋给变量 speed_col
        net_speed = self.df.loc[front_mask, speed_col].dropna().mean() if front_mask.any() else 0.0  # 将 net_speed 设为一次调用/构造的返回值
        net_aggr = front * (net_speed if np.isfinite(net_speed) else 0.0)  # 将 net_aggr 设为一次调用/构造的返回值
        return {'front': float(front), 'back': float(back), 'left': float(left), 'right': float(right), 'net_aggr': float(net_aggr)}  # 从函数返回结果

    def get_barycenter_cov(self, player_id: int) -> Dict[str, float]:  # 定义函数（封装可复用逻辑）
        x_col = f'player{player_id}_joint0_x'  # 将表达式计算结果赋给变量 x_col
        y_col = f'player{player_id}_joint0_y'  # 将表达式计算结果赋给变量 y_col
        pts = self.df[[x_col, y_col]].dropna().values  # 将 pts 设为一次调用/构造的返回值
        if len(pts) < 5:  # 条件分支判断并选择执行路径
            return {'cx': 0.0, 'cy': 0.0, 'var_x': 0.0, 'var_y': 0.0, 'cov_xy': 0.0}  # 从函数返回结果
        cx = float(np.mean(pts[:,0]))  # 将 cx 设为一次调用/构造的返回值
        cy = float(np.mean(pts[:,1]))  # 将 cy 设为一次调用/构造的返回值
        cov = np.cov(pts.T)  # 将 cov 设为一次调用/构造的返回值
        return {'cx': cx, 'cy': cy, 'var_x': float(cov[0,0]), 'var_y': float(cov[1,1]), 'cov_xy': float(cov[0,1])}  # 从函数返回结果

    def get_physical_kpis(self, player_id: int) -> Dict[str, float]:  # 定义函数（封装可复用逻辑）
        s = self.get_speed_series(player_id)  # 将 s 设为一次调用/构造的返回值
        if s.empty:  # 条件分支判断并选择执行路径
            return {'avg_speed': 0.0, 'p95_speed': 0.0, 'max_speed': 0.0, 'accel_peak': 0.0, 'accel_mean': 0.0}  # 从函数返回结果
        avg_speed = float(s['speed'].mean())  # 将 avg_speed 设为一次调用/构造的返回值
        p95_speed = float(np.quantile(s['speed'], 0.95))  # 将 p95_speed 设为一次调用/构造的返回值
        max_speed = float(s['speed'].max())  # 将 max_speed 设为一次调用/构造的返回值
        a = self.get_accel_series(player_id)  # 将 a 设为一次调用/构造的返回值
        accel_peak = float(np.nanmax(np.abs(a['accel']))) if not a.empty else 0.0  # 将 accel_peak 设为一次调用/构造的返回值
        accel_mean = float(np.nanmean(np.abs(a['accel']))) if not a.empty else 0.0  # 将 accel_mean 设为一次调用/构造的返回值
        return {'avg_speed': avg_speed, 'p95_speed': p95_speed, 'max_speed': max_speed, 'accel_peak': accel_peak, 'accel_mean': accel_mean}  # 从函数返回结果

    def get_chord_data(self, player_id: Optional[int] = None, min_count: int = 2) -> Dict[str, List[dict]]:  # 定义函数（封装可复用逻辑）
        nodes = []  # 初始化变量 nodes 为一个容器/表达式结果
        links = []  # 初始化变量 links 为一个容器/表达式结果
        label_norm = {  # 初始化变量 label_norm 为一个容器/表达式结果
            '杀': '杀球', 'smash': '杀球',  # 执行当前语句（保持与上文逻辑一致）
            '抽': '抽球', 'drive': '抽球',  # 执行当前语句（保持与上文逻辑一致）
            '吊': '吊球', 'drop': '吊球',  # 执行当前语句（保持与上文逻辑一致）
            '网': '网前', 'net': '网前',  # 执行当前语句（保持与上文逻辑一致）
            '挑': '挑球', 'lift': '挑球',  # 执行当前语句（保持与上文逻辑一致）
            '高': '高远', 'clear': '高远'  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        def normalize(t: str) -> str:  # 定义函数（封装可复用逻辑）
            xl = (t or '').lower()  # 初始化变量 xl 为一个容器/表达式结果
            for k, v in label_norm.items():  # 循环遍历序列/迭代器
                if k in xl:  # 条件分支判断并选择执行路径
                    return v  # 从函数返回结果
            return t or '未知'  # 从函数返回结果
        from collections import Counter  # 从模块导入符号，供后续调用
        pair_counter = Counter()  # 将 pair_counter 设为一次调用/构造的返回值
        type_set = set()  # 将 type_set 设为一次调用/构造的返回值
        for r in self.rallies:  # 循环遍历序列/迭代器
            seq = []  # 初始化变量 seq 为一个容器/表达式结果
            for s in r.strokes:  # 循环遍历序列/迭代器
                if player_id is not None and s.get('player') != player_id:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                st = normalize(s.get('stroke_type_name', ''))  # 将 st 设为一次调用/构造的返回值
                seq.append(st)  # 调用函数/方法执行某个动作或计算
            for i in range(len(seq) - 1):  # 循环遍历序列/迭代器
                a = seq[i]; b = seq[i+1]  # 将表达式计算结果赋给变量 a
                if a and b:  # 条件分支判断并选择执行路径
                    pair_counter[(a, b)] += 1  # 执行当前语句（保持与上文逻辑一致）
                    type_set.add(a); type_set.add(b)  # 调用函数/方法执行某个动作或计算
        nodes = [{'name': t} for t in sorted(type_set)]  # 初始化变量 nodes 为一个容器/表达式结果
        if not nodes:  # 条件分支判断并选择执行路径
            return {'nodes': [], 'links': []}  # 从函数返回结果
        for (a, b), v in pair_counter.items():  # 循环遍历序列/迭代器
            if v >= max(1, min_count):  # 条件分支判断并选择执行路径
                links.append({'source': a, 'target': b, 'value': int(v)})  # 调用函数/方法执行某个动作或计算
        return {'nodes': nodes, 'links': links}  # 从函数返回结果

    def get_theme_river_data(self, window_sec: float = 2.0, player_id: Optional[int] = None) -> Dict[str, object]:  # 定义函数（封装可复用逻辑）
        label_norm = {  # 初始化变量 label_norm 为一个容器/表达式结果
            '杀': '杀球', 'smash': '杀球',  # 执行当前语句（保持与上文逻辑一致）
            '抽': '抽球', 'drive': '抽球',  # 执行当前语句（保持与上文逻辑一致）
            '吊': '吊球', 'drop': '吊球',  # 执行当前语句（保持与上文逻辑一致）
            '网': '网前', 'net': '网前',  # 执行当前语句（保持与上文逻辑一致）
            '挑': '挑球', 'lift': '挑球',  # 执行当前语句（保持与上文逻辑一致）
            '高': '高远', 'clear': '高远'  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        def normalize(t: str) -> str:  # 定义函数（封装可复用逻辑）
            xl = (t or '').lower()  # 初始化变量 xl 为一个容器/表达式结果
            for k, v in label_norm.items():  # 循环遍历序列/迭代器
                if k in xl:  # 条件分支判断并选择执行路径
                    return v  # 从函数返回结果
            return t or '未知'  # 从函数返回结果
        strokes = []  # 初始化变量 strokes 为一个容器/表达式结果
        for r in self.rallies:  # 循环遍历序列/迭代器
            for s in r.strokes:  # 循环遍历序列/迭代器
                if player_id is not None and s.get('player') != player_id:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                frame = s.get('frame', None)  # 将 frame 设为一次调用/构造的返回值
                if frame is None or frame >= len(self.df):  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                t = float(self.df.iloc[frame]['time_seconds']) if 'time_seconds' in self.df.columns else float(frame) * 0.04  # 将 t 设为一次调用/构造的返回值
                st = normalize(s.get('stroke_type_name', ''))  # 将 st 设为一次调用/构造的返回值
                strokes.append((t, st))  # 调用函数/方法执行某个动作或计算
        if not strokes:  # 条件分支判断并选择执行路径
            return {'times': [], 'series': {}}  # 从函数返回结果
        strokes.sort(key=lambda x: x[0])  # 调用函数/方法执行某个动作或计算
        t_min = strokes[0][0]  # 将表达式计算结果赋给变量 t_min
        t_max = strokes[-1][0]  # 将表达式计算结果赋给变量 t_max
        step = max(0.2, window_sec / 5.0)  # 将 step 设为一次调用/构造的返回值
        times = np.arange(t_min, t_max + step, step)  # 将 times 设为一次调用/构造的返回值
        types = sorted(list(set(st for _, st in strokes)))  # 将 types 设为一次调用/构造的返回值
        series = {tp: [] for tp in types}  # 初始化变量 series 为一个容器/表达式结果
        half = window_sec / 2.0  # 将表达式计算结果赋给变量 half
        ts_arr = np.array([t for t, _ in strokes])  # 将 ts_arr 设为一次调用/构造的返回值
        st_arr = np.array([st for _, st in strokes])  # 将 st_arr 设为一次调用/构造的返回值
        for t in times:  # 循环遍历序列/迭代器
            mask = (ts_arr >= t - half) & (ts_arr <= t + half)  # 初始化变量 mask 为一个容器/表达式结果
            total = int(mask.sum())  # 将 total 设为一次调用/构造的返回值
            if total == 0:  # 条件分支判断并选择执行路径
                for tp in types:  # 循环遍历序列/迭代器
                    series[tp].append(0.0)  # 调用函数/方法执行某个动作或计算
            else:  # 条件分支的否则路径
                for tp in types:  # 循环遍历序列/迭代器
                    series[tp].append(float(np.sum(st_arr[mask] == tp)) / float(total))  # 调用函数/方法执行某个动作或计算
        return {'times': times.tolist(), 'series': series, 'window_sec': window_sec}  # 从函数返回结果
