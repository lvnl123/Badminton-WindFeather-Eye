import torch  # 导入模块，供后续使用
import torchvision  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
import copy  # 导入模块，供后续使用
import cv2  # 导入模块，供后续使用
from torchvision.transforms import functional as F  # 从模块导入符号，供后续调用
from typing import Optional, List  # 从模块导入符号，供后续调用


class NetDetector:  # 定义类（封装数据与行为）
    def __init__(self, model_path: str = "models/net_kpRCNN.pth", device: str = 'cuda'):  # 定义函数（封装可复用逻辑）
        self.device = device if torch.cuda.is_available() else 'cpu'  # 给对象属性 self.device 赋值/初始化（来自当前语句右侧表达式）
        self.model_path = model_path  # 给对象属性 self.model_path 赋值/初始化（来自当前语句右侧表达式）
        self.normal_net_info = None  # 给对象属性 self.normal_net_info 赋值/初始化（来自当前语句右侧表达式）
        self.got_info = False  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
        self.mse = None  # 给对象属性 self.mse 赋值/初始化（来自当前语句右侧表达式）
        self.setup_RCNN()  # 调用函数/方法执行某个动作或计算

    def reset(self):  # 定义函数（封装可复用逻辑）
        self.got_info = False  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
        self.normal_net_info = None  # 给对象属性 self.normal_net_info 赋值/初始化（来自当前语句右侧表达式）

    def setup_RCNN(self):  # 定义函数（封装可复用逻辑）
        self.__net_kpRCNN = torch.load(self.model_path, map_location=self.device)  # 给对象属性 self.__net_kpRCNN 赋值/初始化（来自当前语句右侧表达式）
        self.__net_kpRCNN.to(self.device).eval()  # 调用函数/方法执行某个动作或计算

    def del_RCNN(self):  # 定义函数（封装可复用逻辑）
        del self.__net_kpRCNN  # 执行当前语句（保持与上文逻辑一致）

    def get_net_info(self, img: np.ndarray):  # 定义函数（封装可复用逻辑）
        self.__correct_points = None  # 给对象属性 self.__correct_points 赋值/初始化（来自当前语句右侧表达式）
        image = img.copy()  # 将 image 设为一次调用/构造的返回值
        self.mse = None  # 给对象属性 self.mse 赋值/初始化（来自当前语句右侧表达式）
        frame_height, frame_weight, _ = image.shape  # 执行当前语句（保持与上文逻辑一致）
        image = F.to_tensor(image)  # 将 image 设为一次调用/构造的返回值
        image = image.unsqueeze(0)  # 将 image 设为一次调用/构造的返回值
        image = image.to(self.device)  # 将 image 设为一次调用/构造的返回值

        output = self.__net_kpRCNN(image)  # 将 output 设为一次调用/构造的返回值
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

        self.__true_net_points = copy.deepcopy(keypoints[0])  # 给对象属性 self.__true_net_points 赋值/初始化（来自当前语句右侧表达式）

        self.__correct_points = self.__correction()  # 给对象属性 self.__correct_points 赋值/初始化（来自当前语句右侧表达式）

        if self.normal_net_info is not None:  # 条件分支判断并选择执行路径
            self.got_info = self.__check_net(self.__true_net_points)  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）
            if not self.got_info:  # 条件分支判断并选择执行路径
                return None, self.got_info  # 从函数返回结果

        if self.normal_net_info is None:  # 条件分支判断并选择执行路径
            self.__multi_points = self.__partition(  # 给对象属性 self.__multi_points 赋值/初始化（来自当前语句右侧表达式）
                self.__correct_points).tolist()  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self.__multi_points = self.__partition(  # 给对象属性 self.__multi_points 赋值/初始化（来自当前语句右侧表达式）
                self.normal_net_info).tolist()  # 调用函数/方法执行某个动作或计算

        self.got_info = True  # 给对象属性 self.got_info 赋值/初始化（来自当前语句右侧表达式）

        return self.__correct_points.tolist(), self.got_info  # 从函数返回结果

    def draw_net(self, image: np.ndarray, mode: str = "auto") -> np.ndarray:  # 定义函数（封装可复用逻辑）
        if self.normal_net_info is None and mode == "auto":  # 条件分支判断并选择执行路径
            return image  # 从函数返回结果
        elif mode == "frame_select":  # 条件分支判断并选择执行路径
            if self.__correct_points is None:  # 条件分支判断并选择执行路径
                return image  # 从函数返回结果
            self.__multi_points = self.__partition(  # 给对象属性 self.__multi_points 赋值/初始化（来自当前语句右侧表达式）
                self.__correct_points).tolist()  # 调用函数/方法执行某个动作或计算

        image_copy = image.copy()  # 将 image_copy 设为一次调用/构造的返回值
        c_edges = [[0, 1], [2, 3], [0, 4], [1, 5]]  # 初始化变量 c_edges 为一个容器/表达式结果

        net_color_edge = (53, 195, 242)  # 初始化变量 net_color_edge 为一个容器/表达式结果
        net_color_kps = (5, 135, 242)  # 初始化变量 net_color_kps 为一个容器/表达式结果

        for e in c_edges:  # 循环遍历序列/迭代器
            cv2.line(image_copy, (int(self.__multi_points[e[0]][0]),  # 执行当前语句（保持与上文逻辑一致）
                                  int(self.__multi_points[e[0]][1])),  # 执行当前语句（保持与上文逻辑一致）
                     (int(self.__multi_points[e[1]][0]),  # 执行当前语句（保持与上文逻辑一致）
                      int(self.__multi_points[e[1]][1])),  # 执行当前语句（保持与上文逻辑一致）
                     net_color_edge,  # 执行当前语句（保持与上文逻辑一致）
                     2,  # 执行当前语句（保持与上文逻辑一致）
                     lineType=cv2.LINE_AA)  # 将表达式计算结果赋给变量 lineType

        for kps in [self.__multi_points]:  # 循环遍历序列/迭代器
            for kp in kps:  # 循环遍历序列/迭代器
                cv2.circle(image_copy, tuple(kp), 1, net_color_kps, 5)  # 调用函数/方法执行某个动作或计算

        return image_copy  # 从函数返回结果

    def __check_net(self, net_info):  # 定义函数（封装可复用逻辑）
        vec1 = np.array(self.normal_net_info)  # 将 vec1 设为一次调用/构造的返回值
        vec2 = np.array(net_info)  # 将 vec2 设为一次调用/构造的返回值
        mse = np.square(vec1 - vec2).mean()  # 将 mse 设为一次调用/构造的返回值
        self.mse = mse  # 给对象属性 self.mse 赋值/初始化（来自当前语句右侧表达式）
        if mse > 100:  # 条件分支判断并选择执行路径
            return False  # 从函数返回结果
        return True  # 从函数返回结果

    def __correction(self):  # 定义函数（封装可复用逻辑）
        net_kp = np.array(self.__true_net_points)  # 将 net_kp 设为一次调用/构造的返回值

        up_y = int((np.round(net_kp[0][1] + net_kp[3][1])) / 2)  # 将 up_y 设为一次调用/构造的返回值
        down_y = int((np.round(net_kp[1][1] + net_kp[2][1]) / 2))  # 将 down_y 设为一次调用/构造的返回值

        up_x = int(np.round((net_kp[0][0] + net_kp[1][0]) / 2))  # 将 up_x 设为一次调用/构造的返回值
        down_x = int(np.round((net_kp[3][0] + net_kp[2][0]) / 2))  # 将 down_x 设为一次调用/构造的返回值

        net_kp[0][1] = up_y  # 执行当前语句（保持与上文逻辑一致）
        net_kp[3][1] = up_y  # 执行当前语句（保持与上文逻辑一致）

        net_kp[1][1] = down_y  # 执行当前语句（保持与上文逻辑一致）
        net_kp[2][1] = down_y  # 执行当前语句（保持与上文逻辑一致）

        net_kp[0][0] = up_x  # 执行当前语句（保持与上文逻辑一致）
        net_kp[1][0] = up_x  # 执行当前语句（保持与上文逻辑一致）

        net_kp[3][0] = down_x  # 执行当前语句（保持与上文逻辑一致）
        net_kp[2][0] = down_x  # 执行当前语句（保持与上文逻辑一致）
        return net_kp  # 从函数返回结果

    def __partition(self, net_crkp):  # 定义函数（封装可复用逻辑）
        net_kp = np.array(net_crkp)  # 将 net_kp 设为一次调用/构造的返回值

        p0 = net_kp[0]  # 将表达式计算结果赋给变量 p0
        p1 = net_kp[3]  # 将表达式计算结果赋给变量 p1

        p4 = net_kp[1]  # 将表达式计算结果赋给变量 p4
        p5 = net_kp[2]  # 将表达式计算结果赋给变量 p5

        p2 = np.array([p0[0], np.round((p4[1] + p0[1]) * (0.5))], dtype=int)  # 将 p2 设为一次调用/构造的返回值
        p3 = np.array([p1[0], np.round((p5[1] + p1[1]) * (0.5))], dtype=int)  # 将 p3 设为一次调用/构造的返回值

        kp = np.array([p0, p1, p2, p3, p4, p5], dtype=int)  # 将 kp 设为一次调用/构造的返回值

        return kp  # 从函数返回结果

    def get_partitioned_keypoints(self) -> Optional[List[List[int]]]:  # 定义函数（封装可复用逻辑）
        if not self.got_info or self.__multi_points is None:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        return self.__multi_points  # 从函数返回结果
