from __future__ import annotations  # 允许在类型注解中直接写类名字符串前向引用（减少引号需求）

import traceback  # 用于捕获异常后的堆栈文本，方便在 UI 日志中展示详细错误
from dataclasses import asdict  # 把 dataclass 实例转换成 dict，便于跨线程信号传递/序列化
from typing import Optional  # 给可选参数/返回值标注 Optional[T] 类型

import numpy as np  # 用于标注/传递 OpenCV 帧数据的 ndarray 类型
from PySide6.QtCore import QObject, QThread, Signal  # Qt 对象、线程与信号机制（UI 主线程通信）

from ui_pyside6.widgets.pipeline_runner import PipelineConfig, PipelineOutputs, run_pipeline  # 管线配置/输出结构与实际执行函数


class PipelineWorker(QObject):  # 在后台线程里执行分析管线的“工作对象”，通过信号把结果回传给 UI
    logLine = Signal(str)  # 输出日志行（字符串），UI 可以追加到日志窗口
    stepChanged = Signal(str)  # 当前步骤名称变化（字符串），UI 用于显示阶段标签
    overallProgressChanged = Signal(int)  # 总体进度（0-100），UI 用于进度条/百分比
    previewFrame = Signal(object)  # 预览帧（通常是 BGR ndarray），UI 用于实时预览画面
    outputsReady = Signal(object)  # 管线输出对象（dict），UI 用于加载产物路径等信息
    finished = Signal(bool, str)  # 结束信号：(是否成功, 消息/原因)，UI 用于收尾状态更新

    def __init__(self, video_path: str, result_dir: str, config: PipelineConfig):  # 初始化 worker，并保存本次运行的参数
        super().__init__()  # 初始化 QObject 基类（Qt 元对象系统需要）
        self._video_path = video_path  # 输入视频路径（字符串），供 run_pipeline 使用
        self._result_dir = result_dir  # 输出根目录（字符串），供 run_pipeline 在其下创建子目录
        self._config = config  # 管线配置（模型路径、阈值、设备等），由 UI 侧组装传入
        self._stop = False  # 停止标记；UI 点击“停止”后置 True，run_pipeline 会周期性检查

    def request_stop(self):  # UI 线程调用：请求停止后台分析
        self._stop = True  # 仅设置标记，不做阻塞等待；实际停止由 run_pipeline 在合适时机处理

    def _is_stopped(self) -> bool:  # 提供给 run_pipeline 的回调：查询是否应当停止
        return self._stop  # True 表示用户请求停止；False 表示继续运行

    def run(self):  # 后台线程入口：真正执行管线并通过信号把过程/结果回传给 UI
        try:  # 捕获所有异常，避免线程静默崩溃导致 UI 无响应
            outputs: PipelineOutputs = run_pipeline(  # 调用核心管线函数，返回结构化输出（路径等）
                self._video_path,  # 传入输入视频路径
                self._result_dir,  # 传入输出目录
                self._config,  # 传入用户选择的配置
                log=self.logLine.emit,  # 把管线日志回调绑定到 Qt 信号 emit（线程安全排队到 UI 线程）
                step=self.stepChanged.emit,  # 把步骤变化回调绑定到 Qt 信号 emit
                overall_progress=self.overallProgressChanged.emit,  # 把总体进度回调绑定到 Qt 信号 emit
                preview_frame=self._emit_preview,  # 预览帧回调：在这里再转发到 previewFrame 信号
                stop_requested=self._is_stopped,  # 停止回调：让管线能中断长循环
            )  # run_pipeline 执行结束（成功或主动停止抛异常）后才会走到这里
            self.outputsReady.emit(asdict(outputs))  # 将 dataclass 输出转 dict，通过信号通知 UI（避免直接传复杂对象）
            self.finished.emit(True, "OK")  # 正常完成：通知 UI 成功与消息
        except Exception as e:  # 捕获执行过程中的异常（包括停止信号触发的异常）
            msg = str(e)  # 异常消息文本，用于判断停止与对用户提示
            if msg == "STOP_REQUESTED":  # 约定：run_pipeline 用该字符串表示“用户请求停止”
                self.finished.emit(False, "已停止")  # 通知 UI 以“非成功”结束，但消息为“已停止”
                return  # 直接返回，避免再打印堆栈（停止不算错误）
            detail = traceback.format_exc()  # 将当前异常堆栈格式化为字符串，方便复制排查
            self.logLine.emit(detail)  # 把详细堆栈输出到 UI 日志，便于定位具体报错位置
            self.finished.emit(False, msg)  # 通知 UI 失败结束，并传递简短错误消息

    def _emit_preview(self, frame_bgr: np.ndarray):  # 管线预览回调：把帧转发到 UI 侧
        self.previewFrame.emit(frame_bgr)  # 通过 Qt 信号发出预览帧（Qt 会做跨线程队列投递）


class WorkerThread(QThread):  # 独立线程容器：用于承载 PipelineWorker 的运行
    def __init__(self, worker: PipelineWorker):  # 初始化线程并保存 worker 引用
        super().__init__()  # 初始化 QThread 基类
        self.worker = worker  # 保存 worker；线程 run() 中会调用 worker.run()

    def run(self):  # 线程真正开始执行时由 Qt 调用
        self.worker.run()  # 在该线程中执行耗时管线逻辑（避免阻塞 UI 主线程）
