import torch  # 导入模块，供后续使用
import torchvision  # 导入模块，供后续使用
import cv2  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
import os  # 导入模块，供后续使用
from tqdm import tqdm  # 从模块导入符号，供后续调用
from .TrackNetAttention import TrackNetAttention  # 从模块导入符号，供后续调用
from .denoise import smooth  # 从模块导入符号，供后续调用
from .utils import write_json  # 从模块导入符号，供后续调用


def ball_detect(  # 定义函数（封装可复用逻辑）
    video_path,  # 执行当前语句（保持与上文逻辑一致）
    result_path,  # 执行当前语句（保持与上文逻辑一致）
    model_path="e:\\learn\\TrackNetV3_migrated\\model_best.pth",  # 将表达式计算结果赋给变量 model_path
    num_frames=3,  # 将表达式计算结果赋给变量 num_frames
    threshold=0.5,  # 将表达式计算结果赋给变量 threshold
    frame_callback=None,  # 将表达式计算结果赋给变量 frame_callback
    progress_callback=None,  # 将表达式计算结果赋给变量 progress_callback
):  # 执行当前语句（保持与上文逻辑一致）
    """
    Detect shuttlecock trajectory in video using TrackNet with Attention model.
    
    Args:
        video_path: Path to input video file
        result_path: Path to save detection results
        model_path: Path to TrackNet with Attention model weights
        num_frames: Number of frames to process as input sequence (default: 3)
    """
    imgsz = [288, 512]  # 初始化变量 imgsz 为一个容器/表达式结果
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # 将 video_name 设为一次调用/构造的返回值

    d_save_dir = os.path.join(result_path, "loca_info")  # 将 d_save_dir 设为一次调用/构造的返回值
    f_source = str(video_path)  # 将 f_source 设为一次调用/构造的返回值

    if not os.path.exists(d_save_dir):  # 条件分支判断并选择执行路径
        os.makedirs(d_save_dir)  # 调用函数/方法执行某个动作或计算
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 将 device 设为一次调用/构造的返回值
    print(f"Using device: {device}")  # 调用函数/方法执行某个动作或计算

    model = TrackNetAttention().to(device)  # 将 model 设为一次调用/构造的返回值
    if os.path.exists(model_path):  # 条件分支判断并选择执行路径
        pretrained_dict = torch.load(model_path, map_location=device)  # 将 pretrained_dict 设为一次调用/构造的返回值
        model_dict = model.state_dict()  # 将 model_dict 设为一次调用/构造的返回值
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 初始化变量 pretrained_dict 为一个容器/表达式结果
        
        model_dict.update(pretrained_dict)  # 调用函数/方法执行某个动作或计算
        model.load_state_dict(model_dict)  # 调用函数/方法执行某个动作或计算
        print(f"Loaded model from {model_path} (partial loading for attention layers)")  # 调用函数/方法执行某个动作或计算
    else:  # 条件分支的否则路径
        print(f"Warning: Model file {model_path} not found. Using random weights.")  # 调用函数/方法执行某个动作或计算
    model.eval()  # 调用函数/方法执行某个动作或计算

    vid_cap = cv2.VideoCapture(f_source)  # 将 vid_cap 设为一次调用/构造的返回值
    video_end = False  # 将表达式计算结果赋给变量 video_end
    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 将 video_len 设为一次调用/构造的返回值
    fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 将 fps 设为一次调用/构造的返回值
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 将 w 设为一次调用/构造的返回值
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 将 h 设为一次调用/构造的返回值

    print(f"Video info: {video_len} frames, {fps} fps, {w}x{h}")  # 调用函数/方法执行某个动作或计算

    count = 0  # 将表达式计算结果赋给变量 count
    with tqdm(total=video_len, desc="Processing frames") as pbar:  # 上下文管理：确保资源正确释放
        while vid_cap.isOpened():  # 条件循环，直到条件不满足
            imgs = []  # 初始化变量 imgs 为一个容器/表达式结果
            for _ in range(num_frames):  # 循环遍历序列/迭代器
                ret, img = vid_cap.read()  # 调用函数/方法执行某个动作或计算
                if not ret:  # 条件分支判断并选择执行路径
                    video_end = True  # 将表达式计算结果赋给变量 video_end
                    break  # 控制流语句：改变当前代码块的执行方式
                imgs.append(img)  # 调用函数/方法执行某个动作或计算

            if video_end:  # 条件分支判断并选择执行路径
                break  # 控制流语句：改变当前代码块的执行方式

            imgs_torch = []  # 初始化变量 imgs_torch 为一个容器/表达式结果
            for img in imgs:  # 循环遍历序列/迭代器
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将 img 设为一次调用/构造的返回值
                img_torch = torchvision.transforms.ToTensor()(img).to(device)  # 将 img_torch 设为一次调用/构造的返回值
                img_torch = torchvision.transforms.functional.resize(  # 将表达式计算结果赋给变量 img_torch
                    img_torch, imgsz, antialias=True)  # 执行当前语句（保持与上文逻辑一致）
                imgs_torch.append(img_torch)  # 调用函数/方法执行某个动作或计算

            imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)  # 将 imgs_torch 设为一次调用/构造的返回值

            with torch.no_grad():  # 上下文管理：确保资源正确释放
                preds = model(imgs_torch)  # 将 preds 设为一次调用/构造的返回值
            preds = preds[0].detach().cpu().numpy()  # 将 preds 设为一次调用/构造的返回值

            y_preds = preds > threshold  # 将表达式计算结果赋给变量 y_preds
            y_preds = y_preds.astype('float32')  # 将 y_preds 设为一次调用/构造的返回值
            y_preds = y_preds * 255  # 将表达式计算结果赋给变量 y_preds
            y_preds = y_preds.astype('uint8')  # 将 y_preds 设为一次调用/构造的返回值

            for i in range(num_frames):  # 循环遍历序列/迭代器
                if np.amax(y_preds[i]) <= 0:  # 条件分支判断并选择执行路径
                    ball_dict = {  # 初始化变量 ball_dict 为一个容器/表达式结果
                        f"{count}": {  # 执行当前语句（保持与上文逻辑一致）
                            "visible": 0,  # 执行当前语句（保持与上文逻辑一致）
                            "x": 0,  # 执行当前语句（保持与上文逻辑一致）
                            "y": 0,  # 执行当前语句（保持与上文逻辑一致）
                        }  # 执行当前语句（保持与上文逻辑一致）
                    }  # 执行当前语句（保持与上文逻辑一致）
                    write_json(ball_dict, video_name, f"{d_save_dir}")  # 调用函数/方法执行某个动作或计算
                    if frame_callback is not None and i < len(imgs):  # 条件分支判断并选择执行路径
                        frame_callback(count, imgs[i], None, 0)  # 调用函数/方法执行某个动作或计算
                else:  # 条件分支的否则路径
                    pred_img = cv2.resize(y_preds[i], (w, h),  # 将 pred_img 设为一次调用/构造的返回值
                                          interpolation=cv2.INTER_AREA)  # 将表达式计算结果赋给变量 interpolation

                    (cnts, _) = cv2.findContours(pred_img, cv2.RETR_EXTERNAL,  # 执行当前语句（保持与上文逻辑一致）
                                                 cv2.CHAIN_APPROX_SIMPLE)  # 执行当前语句（保持与上文逻辑一致）
                    rects = [cv2.boundingRect(ctr) for ctr in cnts]  # 初始化变量 rects 为一个容器/表达式结果
                    
                    if len(rects) > 0:  # 条件分支判断并选择执行路径
                        max_area_idx = 0  # 将表达式计算结果赋给变量 max_area_idx
                        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]  # 将表达式计算结果赋给变量 max_area

                        for ii in range(len(rects)):  # 循环遍历序列/迭代器
                            area = rects[ii][2] * rects[ii][3]  # 将表达式计算结果赋给变量 area
                            if area > max_area:  # 条件分支判断并选择执行路径
                                max_area_idx = ii  # 将表达式计算结果赋给变量 max_area_idx
                                max_area = area  # 将表达式计算结果赋给变量 max_area

                        target = rects[max_area_idx]  # 将表达式计算结果赋给变量 target
                        (cx_pred, cy_pred) = (int((target[0] + target[2] / 2)),  # 执行当前语句（保持与上文逻辑一致）
                                              int((target[1] + target[3] / 2)))  # 调用函数/方法执行某个动作或计算
                    else:  # 条件分支的否则路径
                        cx_pred, cy_pred = 0, 0  # 执行当前语句（保持与上文逻辑一致）

                    ball_dict = {  # 初始化变量 ball_dict 为一个容器/表达式结果
                        f"{count}": {  # 执行当前语句（保持与上文逻辑一致）
                            "visible": 1 if cx_pred > 0 and cy_pred > 0 else 0,  # 执行当前语句（保持与上文逻辑一致）
                            "x": cx_pred,  # 执行当前语句（保持与上文逻辑一致）
                            "y": cy_pred,  # 执行当前语句（保持与上文逻辑一致）
                        }  # 执行当前语句（保持与上文逻辑一致）
                    }  # 执行当前语句（保持与上文逻辑一致）
                    write_json(ball_dict, video_name, f"{d_save_dir}")  # 调用函数/方法执行某个动作或计算
                    if frame_callback is not None and i < len(imgs):  # 条件分支判断并选择执行路径
                        visible = 1 if cx_pred > 0 and cy_pred > 0 else 0  # 将表达式计算结果赋给变量 visible
                        frame_callback(count, imgs[i], (cx_pred, cy_pred), visible)  # 调用函数/方法执行某个动作或计算

                count += 1  # 执行当前语句（保持与上文逻辑一致）
                pbar.update(1)  # 调用函数/方法执行某个动作或计算
                if progress_callback is not None:  # 条件分支判断并选择执行路径
                    progress_callback(count, video_len)  # 调用函数/方法执行某个动作或计算

    while count < video_len:  # 条件循环，直到条件不满足
        ball_dict = {  # 初始化变量 ball_dict 为一个容器/表达式结果
            f"{count}": {  # 执行当前语句（保持与上文逻辑一致）
                "visible": 0,  # 执行当前语句（保持与上文逻辑一致）
                "x": 0,  # 执行当前语句（保持与上文逻辑一致）
                "y": 0,  # 执行当前语句（保持与上文逻辑一致）
            }  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        write_json(ball_dict, video_name, f"{d_save_dir}")  # 调用函数/方法执行某个动作或计算
        count += 1  # 执行当前语句（保持与上文逻辑一致）
        pbar.update(1)  # 调用函数/方法执行某个动作或计算
        if progress_callback is not None:  # 条件分支判断并选择执行路径
            progress_callback(count, video_len)  # 调用函数/方法执行某个动作或计算

    vid_cap.release()  # 调用函数/方法执行某个动作或计算
    print(f"Detection completed. Results saved to {d_save_dir}")  # 调用函数/方法执行某个动作或计算

    dd_save_dir = os.path.join(result_path, "loca_info_denoise")  # 将 dd_save_dir 设为一次调用/构造的返回值
    os.makedirs(dd_save_dir, exist_ok=True)  # 调用函数/方法执行某个动作或计算

    print("Starting trajectory smoothing...")  # 调用函数/方法执行某个动作或计算
    json_path = f"{d_save_dir}/{video_name}.json"  # 将表达式计算结果赋给变量 json_path
    smooth(json_path, court=None, save_path=dd_save_dir)  # 调用函数/方法执行某个动作或计算
    print(f"Smoothed trajectory saved to {dd_save_dir}")  # 调用函数/方法执行某个动作或计算


if __name__ == "__main__":  # 条件分支判断并选择执行路径
    import argparse  # 导入模块，供后续使用
    
    parser = argparse.ArgumentParser(description='TrackNetV3 with Attention Shuttlecock Detection')  # 将 parser 设为一次调用/构造的返回值
    parser.add_argument('--video', type=str, required=True, help='Path to input video')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--result', type=str, default='./results', help='Path to save results')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--model', type=str, default='ball_track_attention.pt', help='Path to model weights')  # 调用函数/方法执行某个动作或计算
    parser.add_argument('--num_frames', type=int, default=3, help='Number of frames in input sequence')  # 调用函数/方法执行某个动作或计算
    
    args = parser.parse_args()  # 将 args 设为一次调用/构造的返回值
    
    ball_detect(args.video, args.result, args.model, args.num_frames)  # 调用函数/方法执行某个动作或计算
