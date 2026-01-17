import pandas as pd  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
import os  # 导入模块，供后续使用
import sys  # 导入模块，供后续使用

from .utils import read_json, write_json  # 从模块导入符号，供后续调用
from .kalman_filter import KalmanTrajectorySmoother  # 从模块导入符号，供后续调用

def smooth(json_path, court=None, save_path="./loca_info_denoise"):  # 定义函数（封装可复用逻辑）
    json_name = os.path.splitext(os.path.basename(json_path))[0]  # 将 json_name 设为一次调用/构造的返回值

    df_ls = []  # 初始化变量 df_ls 为一个容器/表达式结果
    loca_dict = read_json(json_path)  # 将 loca_dict 设为一次调用/构造的返回值

    for frame, vxy_dict in loca_dict.items():  # 循环遍历序列/迭代器
        fvxy_ditc = {}  # 初始化变量 fvxy_ditc 为一个容器/表达式结果
        fvxy_ditc["frame"] = int(frame)  # 调用函数/方法执行某个动作或计算
        for key, value in vxy_dict.items():  # 循环遍历序列/迭代器
            fvxy_ditc[key] = value  # 执行当前语句（保持与上文逻辑一致）
        df_ls.append(fvxy_ditc)  # 调用函数/方法执行某个动作或计算
    df = pd.DataFrame(df_ls)  # 将 df 设为一次调用/构造的返回值
    df = df.fillna(0)  # 将 df 设为一次调用/构造的返回值

    x = df['x'].tolist()  # 将 x 设为一次调用/构造的返回值
    y = df['y'].tolist()  # 将 y 设为一次调用/构造的返回值
    vis = df['visible'].tolist()  # 将 vis 设为一次调用/构造的返回值

    pre_dif = []  # 初始化变量 pre_dif 为一个容器/表达式结果
    for i in range(0, len(x)):  # 循环遍历序列/迭代器
        if i == 0:  # 条件分支判断并选择执行路径
            pre_dif.append(0)  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            pre_dif.append(  # 执行当前语句（保持与上文逻辑一致）
                ((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)**(1 / 2))  # 调用函数/方法执行某个动作或计算

    abnormal = [0] * len(pre_dif)  # 初始化变量 abnormal 为一个容器/表达式结果
    X_abn = x  # 将表达式计算结果赋给变量 X_abn
    y_abn = y  # 将表达式计算结果赋给变量 y_abn
    dif_error = 2  # 将表达式计算结果赋给变量 dif_error
    for i in range(len(pre_dif)):  # 循环遍历序列/迭代器
        if i == len(pre_dif):  # 条件分支判断并选择执行路径
            abnormal[i] = 0  # 执行当前语句（保持与上文逻辑一致）
        elif i == len(pre_dif) - 1:  # 条件分支判断并选择执行路径
            abnormal[i] = 0  # 执行当前语句（保持与上文逻辑一致）
        elif i == len(pre_dif) - 2:  # 条件分支判断并选择执行路径
            abnormal[i] = 0  # 执行当前语句（保持与上文逻辑一致）
        elif i == len(pre_dif) - 3:  # 条件分支判断并选择执行路径
            abnormal[i] = 0  # 执行当前语句（保持与上文逻辑一致）
        elif pre_dif[i] >= 100 and pre_dif[i + 1] >= 100:  # 条件分支判断并选择执行路径
            if vis[i:i + 2] == [1, 1]:  # 条件分支判断并选择执行路径
                abnormal[i] = 'bias1'  # 执行当前语句（保持与上文逻辑一致）
                X_abn[i] = 0  # 执行当前语句（保持与上文逻辑一致）
                y_abn[i] = 0  # 执行当前语句（保持与上文逻辑一致）
        elif pre_dif[i] >= 100 and pre_dif[i + 2] >= 100:  # 条件分支判断并选择执行路径
            if pre_dif[i + 1] < dif_error:  # 条件分支判断并选择执行路径
                if vis[i:i + 3] == [1, 1, 1]:  # 条件分支判断并选择执行路径
                    abnormal[i:i + 2] = ['bias2', 'bias2']  # 执行当前语句（保持与上文逻辑一致）
                    X_abn[i:i + 2] = [0, 0]  # 执行当前语句（保持与上文逻辑一致）
                    y_abn[i:i + 2] = [0, 0]  # 执行当前语句（保持与上文逻辑一致）
        elif i + 4 < len(pre_dif) and pre_dif[i] >= 100 and pre_dif[i + 3] >= 100:  # 条件分支判断并选择执行路径
            if pre_dif[i + 1] < dif_error and pre_dif[i + 2] < dif_error:  # 条件分支判断并选择执行路径
                if vis[i:i + 4] == [1, 1, 1, 1]:  # 条件分支判断并选择执行路径
                    abnormal[i:i + 3] = ['bias3', 'bias3', 'bias3']  # 执行当前语句（保持与上文逻辑一致）
                    X_abn[i:i + 3] = [0, 0, 0]  # 执行当前语句（保持与上文逻辑一致）
                    y_abn[i:i + 3] = [0, 0, 0]  # 执行当前语句（保持与上文逻辑一致）
        elif i + 5 < len(pre_dif) and pre_dif[i] >= 100 and pre_dif[i + 4] >= 100:  # 条件分支判断并选择执行路径
            if pre_dif[i + 1] < dif_error and pre_dif[i + 2] < dif_error and pre_dif[i + 3] < dif_error:  # 条件分支判断并选择执行路径
                if vis[i:i + 5] == [1, 1, 1, 1, 1]:  # 条件分支判断并选择执行路径
                    abnormal[i:i + 4] = ['bias4', 'bias4', 'bias4', 'bias4']  # 执行当前语句（保持与上文逻辑一致）
                    X_abn[i:i + 4] = [0, 0, 0, 0]  # 执行当前语句（保持与上文逻辑一致）
                    y_abn[i:i + 4] = [0, 0, 0, 0]  # 执行当前语句（保持与上文逻辑一致）

    vis2 = [1] * len(df)  # 初始化变量 vis2 为一个容器/表达式结果
    for i in range(len(df)):  # 循环遍历序列/迭代器
        if X_abn[i] == 0 and y_abn[i] == 0:  # 条件分支判断并选择执行路径
            vis2[i] = 0  # 执行当前语句（保持与上文逻辑一致）

    smoother = KalmanTrajectorySmoother(max_gap=8, process_noise=5.0, measurement_noise=20.0)  # 将 smoother 设为一次调用/构造的返回值
    smoothed_x, smoothed_y, smoothed_vis = smoother.smooth(X_abn, y_abn, vis2)  # 调用函数/方法执行某个动作或计算

    df['X'] = smoothed_x  # 执行当前语句（保持与上文逻辑一致）
    df['Y'] = smoothed_y  # 执行当前语句（保持与上文逻辑一致）

    for index, row in df.iterrows():  # 循环遍历序列/迭代器
        frame = str(int(row["frame"]))  # 将 frame 设为一次调用/构造的返回值
        visible = int(row["visible"])  # 将 visible 设为一次调用/构造的返回值
        x = int(row["X"])  # 将 x 设为一次调用/构造的返回值
        y = int(row["Y"])  # 将 y 设为一次调用/构造的返回值

        if x == 0 or y == 0:  # 条件分支判断并选择执行路径
            visible = 0  # 将表达式计算结果赋给变量 visible
        else:  # 条件分支的否则路径
            visible = 1  # 将表达式计算结果赋给变量 visible
        
        ball_dict = {  # 初始化变量 ball_dict 为一个容器/表达式结果
            frame: {  # 执行当前语句（保持与上文逻辑一致）
                "visible": visible,  # 执行当前语句（保持与上文逻辑一致）
                "x": x,  # 执行当前语句（保持与上文逻辑一致）
                "y": y,  # 执行当前语句（保持与上文逻辑一致）
            }  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）

        write_json(ball_dict, json_name, f"{save_path}")  # 调用函数/方法执行某个动作或计算

if __name__ == "__main__":  # 条件分支判断并选择执行路径
    import argparse  # 导入模块，供后续使用
    parser = argparse.ArgumentParser()  # 将 parser 设为一次调用/构造的返回值
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")  # 调用函数/方法执行某个动作或计算
    parser.add_argument("--output", type=str, default="loca_info_denoise", help="Output directory path")  # 调用函数/方法执行某个动作或计算
    args = parser.parse_args()  # 将 args 设为一次调用/构造的返回值
    
    smooth(args.input, court=None, save_path=args.output)  # 调用函数/方法执行某个动作或计算
