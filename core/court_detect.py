import torch  # 导入模块，供后续使用
import torchvision  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
import copy  # 导入模块，供后续使用
import cv2  # 导入模块，供后续使用
from PIL import Image  # 从模块导入符号，供后续调用
from torchvision.transforms import transforms  # 从模块导入符号，供后续调用
from torchvision.transforms import functional as F  # 从模块导入符号，供后续调用
from typing import Tuple, List, Optional  # 从模块导入符号，供后续调用


class CourtDetector:  # 定义类（封装数据与行为）
    def __init__(self, model_path: str = "models/court_kpRCNN.pth", device: str = 'cuda'):  # 定义函数（封装可复用逻辑）
        self.device = device if torch.cuda.is_available() else 'cpu'  # 给对象属性 self.device 赋值/初始化（来自当前语句右侧表达式）
        self.model_path = model_path  # 给对象属性 self.model_path 赋值/初始化（来自当前语句右侧表达式）
        self.normal_court_info = None  # 给对象属性 self.normal_court_info 赋值/初始化（来自当前语句右侧表达式）
        self.got_info = False  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
        self.mse = None  # 给对象属性 self.mse 赋值/初始化（来自当前语句右侧表达式）
        self.setup_RCNN()  # 调用函数/方法执行某个动作或计算

    def reset(self):  # 定义函数（封装可复用逻辑）
        self.got_info = False  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
        self.normal_court_info = None  # 给对象属性 self.normal_court_info 赋值/初始化（来自当前语句右侧表达式）

    def setup_RCNN(self):  # 定义函数（封装可复用逻辑）
        self.__court_kpRCNN = torch.load(self.model_path, map_location=self.device)  # 给对象属性 self.__court_kpRCNN 赋值/初始化（来自当前语句右侧表达式）
        self.__court_kpRCNN.to(self.device).eval()  # 调用函数/方法执行某个动作或计算

    def del_RCNN(self):  # 定义函数（封装可复用逻辑）
        del self.__court_kpRCNN  # 执行当前语句（保持与上文逻辑一致）

    def get_court_info(self, img: np.ndarray) -> Tuple[Optional[List], bool]:  # 定义函数（封装可复用逻辑）
        image = img.copy()  # 将 image 设为一次调用/构造的返回值
        self.mse = None  # 给对象属性 self.mse 赋值/初始化（来自当前语句右侧表达式）
        frame_height, frame_weight, _ = image.shape  # 执行当前语句（保持与上文逻辑一致）
        image = F.to_tensor(image)  # 将 image 设为一次调用/构造的返回值
        image = image.unsqueeze(0)  # 将 image 设为一次调用/构造的返回值
        image = image.to(self.device)  # 将 image 设为一次调用/构造的返回值
        output = self.__court_kpRCNN(image)  # 将 output 设为一次调用/构造的返回值
        scores = output[0]['scores'].detach().cpu().numpy()  # 将 scores 设为一次调用/构造的返回值
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()  # 将 high_scores_idxs 设为一次调用/构造的返回值
        post_nms_idxs = torchvision.ops.nms(  # 将表达式计算结果赋给变量 post_nms_idxs
            output[0]['boxes'][high_scores_idxs],  # 执行当前语句（保持与上文逻辑一致）
            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()  # 调用函数/方法执行某个动作或计算

        if len(output[0]['keypoints'][high_scores_idxs][post_nms_idxs]) == 0:  # 条件分支判断并选择执行路径
            self.got_info = False  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
            return None, self.got_info  # 从函数返回结果

        keypoints = []  # 初始化变量 keypoints 为一个容器/表达式结果
        for kps in output[0]['keypoints'][high_scores_idxs][  # 循环遍历序列/迭代器
                post_nms_idxs].detach().cpu().numpy():  # 执行当前语句（保持与上文逻辑一致）
            keypoints.append([list(map(int, kp[:2])) for kp in kps])  # 调用函数/方法执行某个动作或计算

        self.__true_court_points = copy.deepcopy(keypoints[0])  # 给对象属性 self.__true_court_points 赋值/初始化（来自当前语句右侧表达式）
        
        l_am = (self.__true_court_points[0][1] - self.__true_court_points[4][1])  # 初始化变量 l_am 为一个容器/表达式结果
        l_ad = (self.__true_court_points[0][0] - self.__true_court_points[4][0])  # 初始化变量 l_ad 为一个容器/表达式结果
        l_a = l_am / (1 if l_ad == 0 else l_ad)  # 将 l_a 设为一次调用/构造的返回值
        l_b = self.__true_court_points[0][1] - l_a * self.__true_court_points[0][0]  # 将表达式计算结果赋给变量 l_b

        r_am = (self.__true_court_points[1][1] - self.__true_court_points[5][1])  # 初始化变量 r_am 为一个容器/表达式结果
        r_ad = (self.__true_court_points[1][0] - self.__true_court_points[5][0])  # 初始化变量 r_ad 为一个容器/表达式结果
        r_a = r_am / (1 if r_ad == 0 else r_ad)  # 将 r_a 设为一次调用/构造的返回值

        r_b = self.__true_court_points[1][1] - r_a * self.__true_court_points[1][0]  # 将表达式计算结果赋给变量 r_b
        mp_y = (self.__true_court_points[2][1] + self.__true_court_points[3][1]) / 2  # 初始化变量 mp_y 为一个容器/表达式结果

        self.__court_info = [l_a, l_b, r_a, r_b, mp_y]  # 给对象属性 self.__court_info 赋值/初始化（来自当前语句右侧表达式）

        self.__correct_points = self.__correction()  # 给对象属性 self.__correct_points 赋值/初始化（来自当前语句右侧表达式）

        if self.normal_court_info is not None:  # 条件分支判断并选择执行路径
            self.got_info = self.__check_court(self.__correct_points)  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
            if not self.got_info:  # 条件分支判断并选择执行路径
                return None, self.got_info  # 从函数返回结果

        if self.normal_court_info is None:  # 条件分支判断并选择执行路径
            self.__multi_points = self.__partition(self.__correct_points).tolist()  # 给对象属性 self.__multi_points 赋值/初始化（来自当前语句右侧表达式）
        else:  # 条件分支的否则路径
            self.__multi_points = self.__partition(self.normal_court_info).tolist()  # 给对象属性 self.__multi_points 赋值/初始化（来自当前语句右侧表达式）

        keypoints[0][0][0] -= 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][0][1] -= 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][1][0] += 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][1][1] -= 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][2][0] -= 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][3][0] += 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][4][0] -= 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][4][1] = min(keypoints[0][4][1] + 80, frame_height - 40)  # 调用函数/方法执行某个动作或计算
        keypoints[0][5][0] += 80  # 执行当前语句（保持与上文逻辑一致）
        keypoints[0][5][1] = min(keypoints[0][5][1] + 80, frame_height - 40)  # 调用函数/方法执行某个动作或计算

        self.__extended_court_points = keypoints[0]  # 给对象属性 self.__extended_court_points 赋值/初始化（来自当前语句右侧表达式）

        self.got_info = True  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）

        return self.__correct_points.tolist(), self.got_info  # 从函数返回结果

    def get_court_boundary_params(self) -> Optional[List[float]]:  # 定义函数（封装可复用逻辑）
        if not self.got_info or self.__court_info is None:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        return self.__court_info  # 从函数返回结果

    def __check_court(self, court_info):  # 定义函数（封装可复用逻辑）
        vec1 = np.array(self.normal_court_info)  # 将 vec1 设为一次调用/构造的返回值
        vec2 = np.array(court_info)  # 将 vec2 设为一次调用/构造的返回值
        mse = np.square(vec1 - vec2).mean()  # 将 mse 设为一次调用/构造的返回值
        self.mse = mse  # 给对象属性 self.mse 赋值/初始化（来自当前语句右侧表达式）
        if mse > 100:  # 条件分支判断并选择执行路径
            return False  # 从函数返回结果
        return True  # 从函数返回结果

    def __correction(self):  # 定义函数（封装可复用逻辑）
        court_kp = np.array(self.__true_court_points)  # 将 court_kp 设为一次调用/构造的返回值
        ty = np.round((court_kp[0][1] + court_kp[1][1]) / 2)  # 将 ty 设为一次调用/构造的返回值
        my = (court_kp[2][1] + court_kp[3][1]) / 2  # 初始化变量 my 为一个容器/表达式结果
        by = np.round((court_kp[4][1] + court_kp[5][1]) / 2)  # 将 by 设为一次调用/构造的返回值
        court_kp[0][1] = ty  # 执行当前语句（保持与上文逻辑一致）
        court_kp[1][1] = ty  # 执行当前语句（保持与上文逻辑一致）
        court_kp[2][1] = my  # 执行当前语句（保持与上文逻辑一致）
        court_kp[3][1] = my  # 执行当前语句（保持与上文逻辑一致）
        court_kp[4][1] = by  # 执行当前语句（保持与上文逻辑一致）
        court_kp[5][1] = by  # 执行当前语句（保持与上文逻辑一致）
        return court_kp  # 从函数返回结果

    def __partition(self, court_crkp):  # 定义函数（封装可复用逻辑）
        court_kp = np.array(court_crkp)  # 将 court_kp 设为一次调用/构造的返回值
        tlspace = np.array([  # 将表达式计算结果赋给变量 tlspace
            np.round((court_kp[0][0] - court_kp[2][0]) / 3),  # 执行当前语句（保持与上文逻辑一致）
            np.round((court_kp[2][1] - court_kp[0][1]) / 3)  # 调用函数/方法执行某个动作或计算
        ], dtype=int)  # 执行当前语句（保持与上文逻辑一致）
        trspace = np.array([  # 将表达式计算结果赋给变量 trspace
            np.round((court_kp[3][0] - court_kp[1][0]) / 3),  # 执行当前语句（保持与上文逻辑一致）
            np.round((court_kp[3][1] - court_kp[1][1]) / 3)  # 调用函数/方法执行某个动作或计算
        ], dtype=int)  # 执行当前语句（保持与上文逻辑一致）
        blspace = np.array([  # 将表达式计算结果赋给变量 blspace
            np.round((court_kp[2][0] - court_kp[4][0]) / 3),  # 执行当前语句（保持与上文逻辑一致）
            np.round((court_kp[4][1] - court_kp[2][1]) / 3)  # 调用函数/方法执行某个动作或计算
        ], dtype=int)  # 执行当前语句（保持与上文逻辑一致）
        brspace = np.array([  # 将表达式计算结果赋给变量 brspace
            np.round((court_kp[5][0] - court_kp[3][0]) / 3),  # 执行当前语句（保持与上文逻辑一致）
            np.round((court_kp[5][1] - court_kp[3][1]) / 3)  # 调用函数/方法执行某个动作或计算
        ], dtype=int)  # 执行当前语句（保持与上文逻辑一致）

        p2 = np.array([court_kp[0][0] - tlspace[0], court_kp[0][1] + tlspace[1]])  # 将 p2 设为一次调用/构造的返回值
        p3 = np.array([court_kp[1][0] + trspace[0], court_kp[1][1] + trspace[1]])  # 将 p3 设为一次调用/构造的返回值
        p4 = np.array([p2[0] - tlspace[0], p2[1] + tlspace[1]])  # 将 p4 设为一次调用/构造的返回值
        p5 = np.array([p3[0] + trspace[0], p3[1] + trspace[1]])  # 将 p5 设为一次调用/构造的返回值

        p8 = np.array([court_kp[2][0] - blspace[0], court_kp[2][1] + blspace[1]])  # 将 p8 设为一次调用/构造的返回值
        p9 = np.array([court_kp[3][0] + brspace[0], court_kp[3][1] + brspace[1]])  # 将 p9 设为一次调用/构造的返回值
        p10 = np.array([p8[0] - blspace[0], p8[1] + blspace[1]])  # 将 p10 设为一次调用/构造的返回值
        p11 = np.array([p9[0] + brspace[0], p9[1] + brspace[1]])  # 将 p11 设为一次调用/构造的返回值

        kp = np.array([  # 将表达式计算结果赋给变量 kp
            court_kp[0], court_kp[1], p2, p3, p4, p5, court_kp[2], court_kp[3],  # 执行当前语句（保持与上文逻辑一致）
            p8, p9, p10, p11, court_kp[4], court_kp[5]  # 执行当前语句（保持与上文逻辑一致）
        ], dtype=int)  # 执行当前语句（保持与上文逻辑一致）

        ukp = []  # 初始化变量 ukp 为一个容器/表达式结果

        for i in range(0, 13, 2):  # 循环遍历序列/迭代器
            sub2 = np.round((kp[i] + kp[i + 1]) / 2)  # 将 sub2 设为一次调用/构造的返回值
            sub1 = np.round((kp[i] + sub2) / 2)  # 将 sub1 设为一次调用/构造的返回值
            sub3 = np.round((kp[i + 1] + sub2) / 2)  # 将 sub3 设为一次调用/构造的返回值
            ukp.append(kp[i])  # 调用函数/方法执行某个动作或计算
            ukp.append(sub1)  # 调用函数/方法执行某个动作或计算
            ukp.append(sub2)  # 调用函数/方法执行某个动作或计算
            ukp.append(sub3)  # 调用函数/方法执行某个动作或计算
            ukp.append(kp[i + 1])  # 调用函数/方法执行某个动作或计算
        ukp = np.array(ukp, dtype=int)  # 将 ukp 设为一次调用/构造的返回值
        return ukp  # 从函数返回结果

    def draw_court(self, image: np.ndarray, mode: str = "auto") -> np.ndarray:  # 定义函数（封装可复用逻辑）
        if not self.got_info and mode == "auto":  # 条件分支判断并选择执行路径
            print("There is not court in the image! So you can't draw it.")  # 调用函数/方法执行某个动作或计算
            return image  # 从函数返回结果
        elif mode == "frame_select":  # 条件分支判断并选择执行路径
            if self.__correct_points is None:  # 条件分支判断并选择执行路径
                return image  # 从函数返回结果
            self.__multi_points = self.__partition(self.__correct_points).tolist()  # 给对象属性 self.__multi_points 赋值/初始化（来自当前语句右侧表达式）

        image_copy = image.copy()  # 将 image_copy 设为一次调用/构造的返回值
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
            cv2.line(image_copy, (int(self.__multi_points[e[0]][0]),  # 执行当前语句（保持与上文逻辑一致）
                                  int(self.__multi_points[e[0]][1])),  # 执行当前语句（保持与上文逻辑一致）
                     (int(self.__multi_points[e[1]][0]),  # 执行当前语句（保持与上文逻辑一致）
                      int(self.__multi_points[e[1]][1])),  # 执行当前语句（保持与上文逻辑一致）
                     court_color_edge,  # 执行当前语句（保持与上文逻辑一致）
                     2,  # 执行当前语句（保持与上文逻辑一致）
                     lineType=cv2.LINE_AA)  # 将表达式计算结果赋给变量 lineType
        for kps in [self.__multi_points]:  # 循环遍历序列/迭代器
            for kp in kps:  # 循环遍历序列/迭代器
                cv2.circle(image_copy, tuple(kp), 1, court_color_kps, 5)  # 调用函数/方法执行某个动作或计算

        return image_copy  # 从函数返回结果

    def player_detection(self, outputs: dict) -> Tuple[bool, Optional[List[List[List[float]]]]]:  # 定义函数（封装可复用逻辑）
        boxes = outputs[0]['boxes'].cpu().detach().numpy()  # 将 boxes 设为一次调用/构造的返回值
        filtered_joint = []  # 初始化变量 filtered_joint 为一个容器/表达式结果
        joints = outputs[0]['keypoints'].cpu().detach().numpy()  # 将 joints 设为一次调用/构造的返回值
        in_court_indices = self.__check_in_court_instances(joints)  # 将 in_court_indices 设为一次调用/构造的返回值

        if in_court_indices:  # 条件分支判断并选择执行路径
            conform, combination = self.__check_top_bot_court(in_court_indices, boxes)  # 调用函数/方法执行某个动作或计算
            if conform:  # 条件分支判断并选择执行路径
                filtered_joint.append(joints[in_court_indices[combination[0]]].tolist())  # 调用函数/方法执行某个动作或计算
                filtered_joint.append(joints[in_court_indices[combination[1]]].tolist())  # 调用函数/方法执行某个动作或计算
                filtered_joint = self.__top_bottom(filtered_joint)  # 将 filtered_joint 设为一次调用/构造的返回值

                for points in filtered_joint:  # 循环遍历序列/迭代器
                    for i, joints in enumerate(points):  # 循环遍历序列/迭代器
                        points[i] = joints[0:2]  # 执行当前语句（保持与上文逻辑一致）

                return (True, filtered_joint)  # 从函数返回结果
            else:  # 条件分支的否则路径
                return (False, None)  # 从函数返回结果
        else:  # 条件分支的否则路径
            return (False, None)  # 从函数返回结果

    def __top_bottom(self, joint):  # 定义函数（封装可复用逻辑）
        a = joint[0][-1][1] + joint[0][-2][1]  # 将表达式计算结果赋给变量 a
        b = joint[1][-1][1] + joint[1][-2][1]  # 将表达式计算结果赋给变量 b
        if a >= b:  # 条件分支判断并选择执行路径
            joint[0], joint[1] = joint[1], joint[0]  # 执行当前语句（保持与上文逻辑一致）
        return joint  # 从函数返回结果

    def __check_top_bot_court(self, indices, boxes):  # 定义函数（封装可复用逻辑）
        court_mp = self.__court_info[4]  # 将表达式计算结果赋给变量 court_mp
        for i in range(len(indices)):  # 循环遍历序列/迭代器
            combination = 1  # 将表达式计算结果赋给变量 combination
            if boxes[indices[0]][1] < court_mp < boxes[indices[combination]][3]:  # 条件分支判断并选择执行路径
                return True, [0, combination]  # 从函数返回结果
            elif boxes[indices[0]][3] > court_mp > boxes[indices[combination]][1]:  # 条件分支判断并选择执行路径
                return True, [0, combination]  # 从函数返回结果
            else:  # 条件分支的否则路径
                combination += 1  # 执行当前语句（保持与上文逻辑一致）
        return False, [0, 0]  # 从函数返回结果

    def __check_in_court_instances(self, joints):  # 定义函数（封装可复用逻辑）
        indices = []  # 初始化变量 indices 为一个容器/表达式结果
        for i in range(len(joints)):  # 循环遍历序列/迭代器
            if self.__in_court(joints[i]):  # 条件分支判断并选择执行路径
                indices.append(i)  # 调用函数/方法执行某个动作或计算
        return None if len(indices) < 2 else indices  # 从函数返回结果

    def __in_court(self, joint):  # 定义函数（封装可复用逻辑）
        l_a = self.__court_info[0]  # 将表达式计算结果赋给变量 l_a
        l_b = self.__court_info[1]  # 将表达式计算结果赋给变量 l_b
        r_a = self.__court_info[2]  # 将表达式计算结果赋给变量 r_a
        r_b = self.__court_info[3]  # 将表达式计算结果赋给变量 r_b

        ankle_x = (joint[15][0] + joint[16][0]) / 2  # 初始化变量 ankle_x 为一个容器/表达式结果
        ankle_y = (joint[15][1] + joint[16][1]) / 2  # 初始化变量 ankle_y 为一个容器/表达式结果

        top = ankle_y > self.__extended_court_points[0][1]  # 将表达式计算结果赋给变量 top
        bottom = ankle_y < self.__extended_court_points[5][1]  # 将表达式计算结果赋给变量 bottom

        lmp_x = (ankle_y - l_b) / l_a  # 初始化变量 lmp_x 为一个容器/表达式结果
        rmp_x = (ankle_y - r_b) / r_a  # 初始化变量 rmp_x 为一个容器/表达式结果
        left = ankle_x > lmp_x  # 将表达式计算结果赋给变量 left
        right = ankle_x < rmp_x  # 将表达式计算结果赋给变量 right

        if left and right and top and bottom:  # 条件分支判断并选择执行路径
            return True  # 从函数返回结果
        else:  # 条件分支的否则路径
            return False  # 从函数返回结果

    def get_net_position(self) -> Optional[float]:  # 定义函数（封装可复用逻辑）
        if self.__court_info is not None:  # 条件分支判断并选择执行路径
            return self.__court_info[4]  # 从函数返回结果
        return None  # 从函数返回结果

    def get_partitioned_keypoints(self) -> Optional[List[List[int]]]:  # 定义函数（封装可复用逻辑）
        if not self.got_info or self.__multi_points is None:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        return self.__multi_points  # 从函数返回结果
