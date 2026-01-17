import torch  # 导入模块，供后续使用
from torch import Tensor, nn  # 从模块导入符号，供后续调用
import numpy as np  # 导入模块，供后续使用
from pathlib import Path  # 从模块导入符号，供后续调用
import json  # 导入模块，供后续使用

from .bst import BST, BST_CG, BST_AP, BST_CG_AP  # 从模块导入符号，供后续调用


def get_bone_pairs(skeleton_format='coco'):  # 定义函数（封装可复用逻辑）
    if skeleton_format == 'coco':  # 条件分支判断并选择执行路径
        pairs = [  # 初始化变量 pairs 为一个容器/表达式结果
            (0,1),(0,2),(1,2),(1,3),(2,4),   # head
            (3,5),(4,6),                     # ears to shoulders
            (5,7),(7,9),(6,8),(8,10),        # arms
            (5,6),(5,11),(6,12),(11,12),     # torso
            (11,13),(13,15),(12,14),(14,16)  # legs
        ]  # 执行当前语句（保持与上文逻辑一致）
    else:  # 条件分支的否则路径
        raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）
    return pairs  # 从函数返回结果


def create_bones(joints: np.ndarray, pairs) -> np.ndarray:  # 定义函数（封装可复用逻辑）
    bones = []  # 初始化变量 bones 为一个容器/表达式结果
    for start, end in pairs:  # 循环遍历序列/迭代器
        start_j = joints[:, :, start, :]  # 将表达式计算结果赋给变量 start_j
        end_j = joints[:, :, end, :]  # 将表达式计算结果赋给变量 end_j
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)  # 将 bone 设为一次调用/构造的返回值
        bones.append(bone)  # 调用函数/方法执行某个动作或计算
    return np.stack(bones, axis=-2)  # 从函数返回结果


class StrokeClassifier:  # 定义类（封装数据与行为）
    def __init__(self, model_path, model_type='BST_CG_AP', seq_len=100, n_classes=35, n_joints=17, dataset: str = 'shuttleset'):  # 定义函数（封装可复用逻辑）
        self.use_cuda = torch.cuda.is_available()  # 给对象属性 self.use_cuda 赋值/初始化（来自当前语句右侧表达式）
        self.device = 'cuda' if self.use_cuda else 'cpu'  # 给对象属性 self.device 赋值/初始化（来自当前语句右侧表达式）
        self.n_joints = n_joints  # 给对象属性 self.n_joints 赋值/初始化（来自当前语句右侧表达式）
        self.seq_len = seq_len  # 给对象属性 self.seq_len 赋值/初始化（来自当前语句右侧表达式）
        self.n_classes = n_classes  # 给对象属性 self.n_classes 赋值/初始化（来自当前语句右侧表达式）
        self.model_type = model_type  # 给对象属性 self.model_type 赋值/初始化（来自当前语句右侧表达式）
        self.dataset = dataset  # 给对象属性 self.dataset 赋值/初始化（来自当前语句右侧表达式）

        self.net = self._load_model(model_path, model_type, seq_len, n_classes)  # 给对象属性 self.net 赋值/初始化（来自当前语句右侧表达式）
        self.net.eval()  # 调用函数/方法执行某个动作或计算

    def _load_model(self, model_path, model_type, seq_len, n_classes):  # 定义函数（封装可复用逻辑）
        n_joints = 17  # 将表达式计算结果赋给变量 n_joints
        n_bones = 19  # 将表达式计算结果赋给变量 n_bones
        extra = 1  # 将表达式计算结果赋给变量 extra
        in_channels = 2  # 将表达式计算结果赋给变量 in_channels
        in_dim = (n_joints + n_bones * extra) * in_channels  # 初始化变量 in_dim 为一个容器/表达式结果

        if model_type == 'BST':  # 条件分支判断并选择执行路径
            net = BST(  # 将表达式计算结果赋给变量 net
                in_dim=in_dim,  # 将表达式计算结果赋给变量 in_dim
                n_class=n_classes,  # 将表达式计算结果赋给变量 n_class
                seq_len=seq_len,  # 将表达式计算结果赋给变量 seq_len
                depth_tem=2,  # 将表达式计算结果赋给变量 depth_tem
                depth_inter=1  # 将表达式计算结果赋给变量 depth_inter
            )  # 执行当前语句（保持与上文逻辑一致）
        elif model_type == 'BST_CG':  # 条件分支判断并选择执行路径
            net = BST_CG(  # 将表达式计算结果赋给变量 net
                in_dim=in_dim,  # 将表达式计算结果赋给变量 in_dim
                n_class=n_classes,  # 将表达式计算结果赋给变量 n_class
                seq_len=seq_len,  # 将表达式计算结果赋给变量 seq_len
                depth_tem=2,  # 将表达式计算结果赋给变量 depth_tem
                depth_inter=1  # 将表达式计算结果赋给变量 depth_inter
            )  # 执行当前语句（保持与上文逻辑一致）
        elif model_type == 'BST_AP':  # 条件分支判断并选择执行路径
            net = BST_AP(  # 将表达式计算结果赋给变量 net
                in_dim=in_dim,  # 将表达式计算结果赋给变量 in_dim
                n_class=n_classes,  # 将表达式计算结果赋给变量 n_class
                seq_len=seq_len,  # 将表达式计算结果赋给变量 seq_len
                depth_tem=2,  # 将表达式计算结果赋给变量 depth_tem
                depth_inter=1  # 将表达式计算结果赋给变量 depth_inter
            )  # 执行当前语句（保持与上文逻辑一致）
        elif model_type == 'BST_CG_AP':  # 条件分支判断并选择执行路径
            net = BST_CG_AP(  # 将表达式计算结果赋给变量 net
                in_dim=in_dim,  # 将表达式计算结果赋给变量 in_dim
                n_class=n_classes,  # 将表达式计算结果赋给变量 n_class
                seq_len=seq_len,  # 将表达式计算结果赋给变量 seq_len
                depth_tem=2,  # 将表达式计算结果赋给变量 depth_tem
                depth_inter=1  # 将表达式计算结果赋给变量 depth_inter
            )  # 执行当前语句（保持与上文逻辑一致）
        else:  # 条件分支的否则路径
            raise NotImplementedError(f"Model type {model_type} not supported")  # 调用函数/方法执行某个动作或计算

        net.load_state_dict(torch.load(str(model_path), map_location=self.device, weights_only=True))  # 调用函数/方法执行某个动作或计算
        return net.to(self.device)  # 从函数返回结果

    def prepare_hit_segment(self, trajectory_data, poses, hit_frame, seq_len=100):  # 定义函数（封装可复用逻辑）
        if hit_frame < seq_len // 2:  # 条件分支判断并选择执行路径
            start_frame = 0  # 将表达式计算结果赋给变量 start_frame
            end_frame = min(seq_len, len(trajectory_data))  # 将 end_frame 设为一次调用/构造的返回值
        else:  # 条件分支的否则路径
            start_frame = hit_frame - seq_len // 2  # 将表达式计算结果赋给变量 start_frame
            end_frame = min(hit_frame + seq_len // 2, len(trajectory_data))  # 将 end_frame 设为一次调用/构造的返回值

        segment_length = end_frame - start_frame  # 将表达式计算结果赋给变量 segment_length
        if segment_length < seq_len:  # 条件分支判断并选择执行路径
            pad_before = (seq_len - segment_length) // 2  # 初始化变量 pad_before 为一个容器/表达式结果
            pad_after = seq_len - segment_length - pad_before  # 将表达式计算结果赋给变量 pad_after
        else:  # 条件分支的否则路径
            pad_before = 0  # 将表达式计算结果赋给变量 pad_before
            pad_after = 0  # 将表达式计算结果赋给变量 pad_after

        n_joints = 17  # 将表达式计算结果赋给变量 n_joints
        human_pose = np.zeros((seq_len, 2, n_joints, 2))  # 将 human_pose 设为一次调用/构造的返回值
        shuttle = np.zeros((seq_len, 2))  # 将 shuttle 设为一次调用/构造的返回值
        pos = np.zeros((seq_len, 2, 2))  # 将 pos 设为一次调用/构造的返回值

        for i in range(segment_length):  # 循环遍历序列/迭代器
            frame_idx = start_frame + i  # 将表达式计算结果赋给变量 frame_idx
            output_idx = pad_before + i  # 将表达式计算结果赋给变量 output_idx

            if frame_idx < len(trajectory_data):  # 条件分支判断并选择执行路径
                traj = trajectory_data[frame_idx]  # 将表达式计算结果赋给变量 traj
                if traj is not None and len(traj) >= 2:  # 条件分支判断并选择执行路径
                    shuttle[output_idx] = [traj[0], traj[1]]  # 执行当前语句（保持与上文逻辑一致）

            if poses is not None and frame_idx < len(poses):  # 条件分支判断并选择执行路径
                for player_idx in range(min(2, poses.shape[1])):  # 循环遍历序列/迭代器
                    pose_data = poses[frame_idx, player_idx]  # 将表达式计算结果赋给变量 pose_data
                    if pose_data is not None:  # 条件分支判断并选择执行路径
                        for joint_idx in range(min(n_joints, pose_data.shape[0])):  # 循环遍历序列/迭代器
                            x, y = pose_data[joint_idx, 0], pose_data[joint_idx, 1]  # 执行当前语句（保持与上文逻辑一致）
                            if x > 0 and y > 0:  # 条件分支判断并选择执行路径
                                human_pose[output_idx, player_idx, joint_idx] = [x, y]  # 执行当前语句（保持与上文逻辑一致）
                                pos[output_idx, player_idx] = [x, y]  # 执行当前语句（保持与上文逻辑一致）

        pairs = get_bone_pairs('coco')  # 将 pairs 设为一次调用/构造的返回值
        bones = create_bones(human_pose, pairs)  # 将 bones 设为一次调用/构造的返回值
        
        mid_joints = []  # 初始化变量 mid_joints 为一个容器/表达式结果
        for start, end in pairs:  # 循环遍历序列/迭代器
            start_j = human_pose[:, :, start, :]  # 将表达式计算结果赋给变量 start_j
            end_j = human_pose[:, :, end, :]  # 将表达式计算结果赋给变量 end_j
            mid_j = np.where((start_j != 0.0) & (end_j != 0.0), (start_j + end_j) / 2, 0.0)  # 将 mid_j 设为一次调用/构造的返回值
            mid_joints.append(mid_j)  # 调用函数/方法执行某个动作或计算
        bones_center = np.stack(mid_joints, axis=-2)  # 将 bones_center 设为一次调用/构造的返回值
        
        human_pose = np.concatenate((human_pose, bones_center), axis=-2)  # 将 human_pose 设为一次调用/构造的返回值

        return human_pose, shuttle, pos  # 从函数返回结果

    def classify_hit(self, trajectory_data, poses, hit_frame):  # 定义函数（封装可复用逻辑）
        human_pose, shuttle, pos = self.prepare_hit_segment(  # 执行当前语句（保持与上文逻辑一致）
            trajectory_data, poses, hit_frame, self.seq_len  # 执行当前语句（保持与上文逻辑一致）
        )  # 执行当前语句（保持与上文逻辑一致）

        human_pose_tensor = torch.from_numpy(human_pose).float().unsqueeze(0).to(self.device)  # 将 human_pose_tensor 设为一次调用/构造的返回值
        shuttle_tensor = torch.from_numpy(shuttle).float().unsqueeze(0).to(self.device)  # 将 shuttle_tensor 设为一次调用/构造的返回值
        pos_tensor = torch.from_numpy(pos).float().unsqueeze(0).to(self.device)  # 将 pos_tensor 设为一次调用/构造的返回值
        video_len_tensor = torch.tensor([self.seq_len]).to(self.device)  # 将 video_len_tensor 设为一次调用/构造的返回值

        with torch.no_grad():  # 上下文管理：确保资源正确释放
            b, t, n, j, d = human_pose_tensor.shape  # 执行当前语句（保持与上文逻辑一致）
            human_pose_tensor = human_pose_tensor.reshape(b, t, n, -1)  # 将 human_pose_tensor 设为一次调用/构造的返回值
            logits = self.net(human_pose_tensor, shuttle_tensor, pos_tensor, video_len_tensor)  # 将 logits 设为一次调用/构造的返回值
            pred = torch.argmax(logits, dim=1).cpu().item()  # 将 pred 设为一次调用/构造的返回值

        return pred  # 从函数返回结果

    def classify_hits(self, trajectory_data, poses, hit_frames):  # 定义函数（封装可复用逻辑）
        stroke_types = []  # 初始化变量 stroke_types 为一个容器/表达式结果

        for hit_frame in hit_frames:  # 循环遍历序列/迭代器
            stroke_type = self.classify_hit(trajectory_data, poses, hit_frame)  # 将 stroke_type 设为一次调用/构造的返回值
            stroke_types.append(stroke_type)  # 调用函数/方法执行某个动作或计算

        return stroke_types  # 从函数返回结果

    def get_stroke_type_name(self, class_id, dataset=None):  # 定义函数（封装可复用逻辑）
        dataset = self.dataset if dataset is None else dataset  # 将表达式计算结果赋给变量 dataset
        stroke_types = self._get_stroke_types(dataset)  # 将 stroke_types 设为一次调用/构造的返回值
        if 0 <= class_id < len(stroke_types):  # 条件分支判断并选择执行路径
            return stroke_types[class_id]  # 从函数返回结果
        return f"Unknown_{class_id}"  # 从函数返回结果

    def get_stroke_type_name_en(self, class_id, dataset=None):  # 定义函数（封装可复用逻辑）
        dataset = self.dataset if dataset is None else dataset  # 将表达式计算结果赋给变量 dataset
        stroke_types_en = self._get_stroke_types_en(dataset)  # 将 stroke_types_en 设为一次调用/构造的返回值
        if 0 <= class_id < len(stroke_types_en):  # 条件分支判断并选择执行路径
            return stroke_types_en[class_id]  # 从函数返回结果
        return f"Unknown_{class_id}"  # 从函数返回结果

    def _get_stroke_types(self, dataset='shuttleset'):  # 定义函数（封装可复用逻辑）
        if dataset in {'shuttleset', 'shuttleset_35classes'}:  # 条件分支判断并选择执行路径
            return [  # 从函数返回结果
                '正手高远球', '反手高远球', '正手吊球', '反手吊球',  # 执行当前语句（保持与上文逻辑一致）
                '正手杀球', '反手杀球', '正手平抽', '反手平抽',  # 执行当前语句（保持与上文逻辑一致）
                '正手网前球', '反手网前球', '正手挑球', '反手挑球',  # 执行当前语句（保持与上文逻辑一致）
                '正手推球', '反手推球', '正手扑球', '反手扑球',  # 执行当前语句（保持与上文逻辑一致）
                '正手切球', '反手切球', '正手旋转球', '反手旋转球',  # 执行当前语句（保持与上文逻辑一致）
                '正手短发球', '正手长发球', '反手短发球', '反手长发球',  # 执行当前语句（保持与上文逻辑一致）
                '正手防守', '反手防守', '正手斜线球', '反手斜线球',  # 执行当前语句（保持与上文逻辑一致）
                '正手直线球', '反手直线球', '正手挑高球', '反手挑高球',  # 执行当前语句（保持与上文逻辑一致）
                '正手半杀球', '反手半杀球', '正手重杀', '反手重杀'  # 执行当前语句（保持与上文逻辑一致）
            ]  # 执行当前语句（保持与上文逻辑一致）
        elif dataset == 'shuttleset_25classes':  # 条件分支判断并选择执行路径
            return [  # 从函数返回结果
                '未知球種',  # 执行当前语句（保持与上文逻辑一致）
                'Top_放小球', 'Top_擋小球', 'Top_殺球', 'Top_挑球',  # 执行当前语句（保持与上文逻辑一致）
                'Top_長球', 'Top_平球', 'Top_切球', 'Top_推球',  # 执行当前语句（保持与上文逻辑一致）
                'Top_撲球', 'Top_勾球', 'Top_發短球', 'Top_發長球',  # 执行当前语句（保持与上文逻辑一致）
                'Bottom_放小球', 'Bottom_擋小球', 'Bottom_殺球', 'Bottom_挑球',  # 执行当前语句（保持与上文逻辑一致）
                'Bottom_長球', 'Bottom_平球', 'Bottom_切球', 'Bottom_推球',  # 执行当前语句（保持与上文逻辑一致）
                'Bottom_撲球', 'Bottom_勾球', 'Bottom_發短球', 'Bottom_發長球'  # 执行当前语句（保持与上文逻辑一致）
            ]  # 执行当前语句（保持与上文逻辑一致）
        elif dataset == 'badDB_18classes':  # 条件分支判断并选择执行路径
            return [  # 从函数返回结果
                'Bottom-Block', 'Bottom-Clear', 'Bottom-Drive', 'Bottom-Dropshot',  # 执行当前语句（保持与上文逻辑一致）
                'Bottom-Net-Kill', 'Bottom-Net-Lift', 'Bottom-Net-Shot', 'Bottom-Serve',  # 执行当前语句（保持与上文逻辑一致）
                'Bottom-Smash', 'Top-Block', 'Top-Clear', 'Top-Drive',  # 执行当前语句（保持与上文逻辑一致）
                'Top-Dropshot', 'Top-Net-Kill', 'Top-Net-Lift', 'Top-Net-Shot',  # 执行当前语句（保持与上文逻辑一致）
                'Top-Serve', 'Top-Smash'  # 执行当前语句（保持与上文逻辑一致）
            ]  # 执行当前语句（保持与上文逻辑一致）
        elif dataset in {'badDB', 'badDB_6classes', 'tenniSet'}:  # 条件分支判断并选择执行路径
            return [  # 从函数返回结果
                '正手高远球', '反手高远球', '正手吊球', '反手吊球',  # 执行当前语句（保持与上文逻辑一致）
                '正手杀球', '反手杀球'  # 执行当前语句（保持与上文逻辑一致）
            ]  # 执行当前语句（保持与上文逻辑一致）
        else:  # 条件分支的否则路径
            return [f"Class_{i}" for i in range(self.n_classes)]  # 从函数返回结果

    def _get_stroke_types_en(self, dataset='shuttleset'):  # 定义函数（封装可复用逻辑）
        if dataset in {'shuttleset', 'shuttleset_35classes'}:  # 条件分支判断并选择执行路径
            return [  # 从函数返回结果
                'forehand_clear', 'backhand_clear', 'forehand_drop', 'backhand_drop',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_smash', 'backhand_smash', 'forehand_drive', 'backhand_drive',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_net_shot', 'backhand_net_shot', 'forehand_lift', 'backhand_lift',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_push', 'backhand_push', 'forehand_flick', 'backhand_flick',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_slice', 'backhand_slice', 'forehand_spin', 'backhand_spin',  # 执行当前语句（保持与上文逻辑一致）
                'serve_forehand_short', 'serve_forehand_long', 'serve_backhand_short', 'serve_backhand_long',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_defensive', 'backhand_defensive', 'forehand_cross_court', 'backhand_cross_court',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_straight', 'backhand_straight', 'forehand_lob', 'backhand_lob',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_half_smash', 'backhand_half_smash', 'forehand_kill', 'backhand_kill'  # 执行当前语句（保持与上文逻辑一致）
            ]  # 执行当前语句（保持与上文逻辑一致）
        elif dataset == 'shuttleset_25classes':  # 条件分支判断并选择执行路径
            return self._get_stroke_types(dataset)  # 从函数返回结果
        elif dataset == 'badDB_18classes':  # 条件分支判断并选择执行路径
            return self._get_stroke_types(dataset)  # 从函数返回结果
        elif dataset in {'badDB', 'badDB_6classes', 'tenniSet'}:  # 条件分支判断并选择执行路径
            return [  # 从函数返回结果
                'forehand_clear', 'backhand_clear', 'forehand_drop', 'backhand_drop',  # 执行当前语句（保持与上文逻辑一致）
                'forehand_smash', 'backhand_smash'  # 执行当前语句（保持与上文逻辑一致）
            ]  # 执行当前语句（保持与上文逻辑一致）
        else:  # 条件分支的否则路径
            return [f"Class_{i}" for i in range(self.n_classes)]  # 从函数返回结果

    def save_stroke_results(self, hit_frames, hit_players, stroke_types, output_path):  # 定义函数（封装可复用逻辑）
        stroke_results = []  # 初始化变量 stroke_results 为一个容器/表达式结果

        for frame, player, stroke_type in zip(hit_frames, hit_players, stroke_types):  # 循环遍历序列/迭代器
            stroke_results.append({  # 执行当前语句（保持与上文逻辑一致）
                'frame': frame,  # 执行当前语句（保持与上文逻辑一致）
                'player': player,  # 执行当前语句（保持与上文逻辑一致）
                'stroke_type_id': int(stroke_type),  # 执行当前语句（保持与上文逻辑一致）
                'stroke_type_name': self.get_stroke_type_name(int(stroke_type), dataset=self.dataset),  # 执行当前语句（保持与上文逻辑一致）
                'stroke_type_name_en': self.get_stroke_type_name_en(int(stroke_type), dataset=self.dataset)  # 调用函数/方法执行某个动作或计算
            })  # 执行当前语句（保持与上文逻辑一致）

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
        with open(output_path, 'w') as f:  # 上下文管理：确保资源正确释放
            json.dump(stroke_results, f, indent=2)  # 调用函数/方法执行某个动作或计算

        print(f"Stroke classification results saved to {output_path}")  # 调用函数/方法执行某个动作或计算
        return stroke_results  # 从函数返回结果

    @staticmethod  # 装饰器：修改/包装下方函数或类的行为
    def load_stroke_results(json_path):  # 定义函数（封装可复用逻辑）
        with open(json_path, 'r') as f:  # 上下文管理：确保资源正确释放
            stroke_results = json.load(f)  # 将 stroke_results 设为一次调用/构造的返回值

        hit_frames = [result['frame'] for result in stroke_results]  # 初始化变量 hit_frames 为一个容器/表达式结果
        hit_players = [result['player'] for result in stroke_results]  # 初始化变量 hit_players 为一个容器/表达式结果
        stroke_types = [result['stroke_type_id'] for result in stroke_results]  # 初始化变量 stroke_types 为一个容器/表达式结果

        return hit_frames, hit_players, stroke_types  # 从函数返回结果


def _pick_best_weight(model_files, preferred_serial=None):  # 定义函数（封装可复用逻辑）
    model_files = sorted(model_files, key=lambda p: p.name.lower())  # 将 model_files 设为一次调用/构造的返回值
    
    if preferred_serial is not None:  # 条件分支判断并选择执行路径
        for p in model_files:  # 循环遍历序列/迭代器
            stem = p.stem  # 将表达式计算结果赋给变量 stem
            tail = stem.split('_')[-1]  # 将 tail 设为一次调用/构造的返回值
            try:  # 开始异常捕获保护块
                score = int(tail)  # 将 score 设为一次调用/构造的返回值
            except Exception:  # 捕获异常并进行处理
                score = 0  # 将表达式计算结果赋给变量 score
            if score == preferred_serial:  # 条件分支判断并选择执行路径
                return p  # 从函数返回结果
    
    best = None  # 将表达式计算结果赋给变量 best
    best_score = -1  # 将表达式计算结果赋给变量 best_score
    for p in model_files:  # 循环遍历序列/迭代器
        stem = p.stem  # 将表达式计算结果赋给变量 stem
        tail = stem.split('_')[-1]  # 将 tail 设为一次调用/构造的返回值
        try:  # 开始异常捕获保护块
            score = int(tail)  # 将 score 设为一次调用/构造的返回值
        except Exception:  # 捕获异常并进行处理
            score = 0  # 将表达式计算结果赋给变量 score
        if score >= best_score:  # 条件分支判断并选择执行路径
            best_score = score  # 将表达式计算结果赋给变量 best_score
            best = p  # 将表达式计算结果赋给变量 best
    return best or model_files[0]  # 从函数返回结果


def create_classifier(dataset='shuttleset', seq_len=100):  # 定义函数（封装可复用逻辑）
    models_dir = Path(__file__).parent.parent / 'models' / 'bst'  # 将 models_dir 设为一次调用/构造的返回值

    if dataset in {'shuttleset', 'shuttleset_35classes'}:  # 条件分支判断并选择执行路径
        model_dir = models_dir / 'shuttleset_35classes'  # 将表达式计算结果赋给变量 model_dir
        model_files = list(model_dir.glob('*.pt'))  # 将 model_files 设为一次调用/构造的返回值
        if not model_files:  # 条件分支判断并选择执行路径
            raise FileNotFoundError(f"No model files found in {model_dir}")  # 调用函数/方法执行某个动作或计算

        model_path = _pick_best_weight(model_files, preferred_serial=5)  # 将 model_path 设为一次调用/构造的返回值（指定使用Serial 5）
        n_classes = 35  # 将表达式计算结果赋给变量 n_classes
        model_type_candidates = ['BST_CG_AP', 'BST_AP', 'BST_CG', 'BST']  # 初始化变量 model_type_candidates 为一个容器/表达式结果
    elif dataset in {'shuttleset_25classes'}:  # 条件分支判断并选择执行路径
        model_dir = models_dir / 'shuttleset_25classes'  # 将表达式计算结果赋给变量 model_dir
        model_files = list(model_dir.glob('*.pt'))  # 将 model_files 设为一次调用/构造的返回值
        if not model_files:  # 条件分支判断并选择执行路径
            raise FileNotFoundError(f"No model files found in {model_dir}")  # 调用函数/方法执行某个动作或计算
        
        model_path = _pick_best_weight(model_files, preferred_serial=4)  # 将 model_path 设为一次调用/构造的返回值（指定使用Serial 7）
        n_classes = 25  # 将表达式计算结果赋给变量 n_classes
        model_type_candidates = ['BST_CG_AP', 'BST_AP', 'BST_CG', 'BST']  # 初始化变量 model_type_candidates 为一个容器/表达式结果
    elif dataset in {'badDB', 'badDB_6classes'}:  # 条件分支判断并选择执行路径
        model_dir = models_dir / 'badDB_6classes'  # 将表达式计算结果赋给变量 model_dir
        model_files = list(model_dir.glob('*.pt'))  # 将 model_files 设为一次调用/构造的返回值
        if not model_files:  # 条件分支判断并选择执行路径
            raise FileNotFoundError(f"No model files found in {model_dir}")  # 调用函数/方法执行某个动作或计算

        model_path = _pick_best_weight(model_files)  # 将 model_path 设为一次调用/构造的返回值
        n_classes = 6  # 将表达式计算结果赋给变量 n_classes
        model_type_candidates = ['BST', 'BST_AP', 'BST_CG', 'BST_CG_AP']  # 初始化变量 model_type_candidates 为一个容器/表达式结果
    elif dataset in {'badDB_18classes'}:  # 条件分支判断并选择执行路径
        model_dir = models_dir / 'badDB_18classes'  # 将表达式计算结果赋给变量 model_dir
        model_files = list(model_dir.glob('*.pt'))  # 将 model_files 设为一次调用/构造的返回值
        if not model_files:  # 条件分支判断并选择执行路径
            raise FileNotFoundError(f"No model files found in {model_dir}")  # 调用函数/方法执行某个动作或计算

        model_path = _pick_best_weight(model_files)  # 将 model_path 设为一次调用/构造的返回值
        n_classes = 18  # 将表达式计算结果赋给变量 n_classes
        model_type_candidates = ['BST_AP', 'BST', 'BST_CG_AP', 'BST_CG']  # 初始化变量 model_type_candidates 为一个容器/表达式结果
    elif dataset == 'tenniSet':  # 条件分支判断并选择执行路径
        model_dir = models_dir / 'tenniSet_6classes'  # 将表达式计算结果赋给变量 model_dir
        model_files = list(model_dir.glob('*.pt'))  # 将 model_files 设为一次调用/构造的返回值
        if not model_files:  # 条件分支判断并选择执行路径
            raise FileNotFoundError(f"No model files found in {model_dir}")  # 调用函数/方法执行某个动作或计算

        model_path = _pick_best_weight(model_files)  # 将 model_path 设为一次调用/构造的返回值
        n_classes = 6  # 将表达式计算结果赋给变量 n_classes
        model_type_candidates = ['BST', 'BST_AP', 'BST_CG', 'BST_CG_AP']  # 初始化变量 model_type_candidates 为一个容器/表达式结果
    else:  # 条件分支的否则路径
        raise ValueError(f"Unsupported dataset: {dataset}")  # 调用函数/方法执行某个动作或计算

    last_error = None  # 将表达式计算结果赋给变量 last_error
    for model_type in model_type_candidates:  # 循环遍历序列/迭代器
        try:  # 开始异常捕获保护块
            return StrokeClassifier(  # 从函数返回结果
                model_path=model_path,  # 将表达式计算结果赋给变量 model_path
                model_type=model_type,  # 将表达式计算结果赋给变量 model_type
                seq_len=seq_len,  # 将表达式计算结果赋给变量 seq_len
                n_classes=n_classes,  # 将表达式计算结果赋给变量 n_classes
                dataset=dataset,  # 将表达式计算结果赋给变量 dataset
            )  # 执行当前语句（保持与上文逻辑一致）
        except Exception as e:  # 捕获异常并进行处理
            last_error = e  # 将表达式计算结果赋给变量 last_error
            continue  # 控制流语句：改变当前代码块的执行方式

    raise RuntimeError(f"Failed to load stroke classifier weights: {model_path}") from last_error  # 执行当前语句（保持与上文逻辑一致）
