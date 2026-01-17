from __future__ import annotations  # 从模块导入符号，供后续调用

from dataclasses import dataclass  # 从模块导入符号，供后续调用
from typing import Optional, Tuple  # 从模块导入符号，供后续调用

import cv2  # 导入模块，供后续使用
import numpy as np  # 导入模块，供后续使用
from PySide6.QtCore import QTimer, Qt, Signal  # 从模块导入符号，供后续调用
from PySide6.QtGui import QImage, QPixmap  # 从模块导入符号，供后续调用
from PySide6.QtWidgets import (  # 从模块导入符号，供后续调用
    QHBoxLayout,  # 执行当前语句（保持与上文逻辑一致）
    QLabel,  # 执行当前语句（保持与上文逻辑一致）
    QPushButton,  # 执行当前语句（保持与上文逻辑一致）
    QSlider,  # 执行当前语句（保持与上文逻辑一致）
    QVBoxLayout,  # 执行当前语句（保持与上文逻辑一致）
    QWidget,  # 执行当前语句（保持与上文逻辑一致）
)  # 执行当前语句（保持与上文逻辑一致）


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:  # 定义函数（封装可复用逻辑）
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # 将 frame_rgb 设为一次调用/构造的返回值
    h, w, ch = frame_rgb.shape  # 执行当前语句（保持与上文逻辑一致）
    bytes_per_line = ch * w  # 将表达式计算结果赋给变量 bytes_per_line
    return QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()  # 从函数返回结果


@dataclass  # 装饰器：修改/包装下方函数或类的行为
class VideoInfo:  # 定义类（封装数据与行为）
    fps: float  # 执行当前语句（保持与上文逻辑一致）
    total_frames: int  # 执行当前语句（保持与上文逻辑一致）
    width: int  # 执行当前语句（保持与上文逻辑一致）
    height: int  # 执行当前语句（保持与上文逻辑一致）


class VideoPlayer(QWidget):  # 定义类（封装数据与行为）
    positionChanged = Signal(int)  # 将 positionChanged 设为一次调用/构造的返回值
    videoOpened = Signal(str)  # 将 videoOpened 设为一次调用/构造的返回值

    def __init__(self, parent: Optional[QWidget] = None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._cap: Optional[cv2.VideoCapture] = None  # 执行当前语句（保持与上文逻辑一致）
        self._path: Optional[str] = None  # 执行当前语句（保持与上文逻辑一致）
        self._info: Optional[VideoInfo] = None  # 执行当前语句（保持与上文逻辑一致）
        self._playing = False  # 给对象属性 self._playing 赋值/初始化（来自当前语句右侧表达式）
        self._frame_index = 0  # 给对象属性 self._frame_index 赋值/初始化（来自当前语句右侧表达式）
        self._updating_slider = False  # 给对象属性 self._updating_slider 赋值/初始化（来自当前语句右侧表达式）

        self._frame_label = QLabel()  # 给对象属性 self._frame_label 赋值/初始化（来自当前语句右侧表达式）
        self._frame_label.setAlignment(Qt.AlignCenter)  # 调用函数/方法执行某个动作或计算
        self._frame_label.setMinimumHeight(260)  # 调用函数/方法执行某个动作或计算
        self._frame_label.setStyleSheet("QLabel{background:#0f1216;border:1px solid #2a2f3a;}")  # 调用函数/方法执行某个动作或计算

        self._play_btn = QPushButton("播放")  # 给对象属性 self._play_btn 赋值/初始化（来自当前语句右侧表达式）
        self._pause_btn = QPushButton("暂停")  # 给对象属性 self._pause_btn 赋值/初始化（来自当前语句右侧表达式）
        self._pause_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        self._time_label = QLabel("-- / --")  # 给对象属性 self._time_label 赋值/初始化（来自当前语句右侧表达式）
        self._time_label.setMinimumWidth(140)  # 调用函数/方法执行某个动作或计算
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 调用函数/方法执行某个动作或计算

        self._slider = QSlider(Qt.Horizontal)  # 给对象属性 self._slider 赋值/初始化（来自当前语句右侧表达式）
        self._slider.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        self._slider.setMinimum(0)  # 调用函数/方法执行某个动作或计算
        self._slider.sliderPressed.connect(self._on_slider_pressed)  # 调用函数/方法执行某个动作或计算
        self._slider.sliderReleased.connect(self._on_slider_released)  # 调用函数/方法执行某个动作或计算
        self._slider.valueChanged.connect(self._on_slider_changed)  # 调用函数/方法执行某个动作或计算

        controls = QHBoxLayout()  # 将 controls 设为一次调用/构造的返回值
        controls.addWidget(self._play_btn)  # 调用函数/方法执行某个动作或计算
        controls.addWidget(self._pause_btn)  # 调用函数/方法执行某个动作或计算
        controls.addWidget(self._slider, 1)  # 调用函数/方法执行某个动作或计算
        controls.addWidget(self._time_label)  # 调用函数/方法执行某个动作或计算

        layout = QVBoxLayout(self)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._frame_label, 1)  # 调用函数/方法执行某个动作或计算
        layout.addLayout(controls)  # 调用函数/方法执行某个动作或计算

        self._timer = QTimer(self)  # 给对象属性 self._timer 赋值/初始化（来自当前语句右侧表达式）
        self._timer.timeout.connect(self._tick)  # 调用函数/方法执行某个动作或计算

        self._play_btn.clicked.connect(self.play)  # 调用函数/方法执行某个动作或计算
        self._pause_btn.clicked.connect(self.pause)  # 调用函数/方法执行某个动作或计算

    def open(self, path: str) -> bool:  # 定义函数（封装可复用逻辑）
        self.close_video()  # 调用函数/方法执行某个动作或计算
        cap = cv2.VideoCapture(path)  # 将 cap 设为一次调用/构造的返回值
        if not cap.isOpened():  # 条件分支判断并选择执行路径
            cap.release()  # 调用函数/方法执行某个动作或计算
            return False  # 从函数返回结果

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # 将 fps 设为一次调用/构造的返回值
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # 将 total_frames 设为一次调用/构造的返回值
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)  # 将 width 设为一次调用/构造的返回值
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)  # 将 height 设为一次调用/构造的返回值

        self._cap = cap  # 给对象属性 self._cap 赋值/初始化（来自当前语句右侧表达式）
        self._path = path  # 给对象属性 self._path 赋值/初始化（来自当前语句右侧表达式）
        self._info = VideoInfo(fps=fps, total_frames=total_frames, width=width, height=height)  # 给对象属性 self._info 赋值/初始化（来自当前语句右侧表达式）
        self._frame_index = 0  # 给对象属性 self._frame_index 赋值/初始化（来自当前语句右侧表达式）

        self._slider.setEnabled(total_frames > 0)  # 调用函数/方法执行某个动作或计算
        self._slider.setMaximum(max(0, total_frames - 1))  # 调用函数/方法执行某个动作或计算
        self._play_btn.setEnabled(True)  # 调用函数/方法执行某个动作或计算
        self._pause_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算

        self._render_frame_at(0)  # 调用函数/方法执行某个动作或计算
        self.videoOpened.emit(path)  # 调用函数/方法执行某个动作或计算
        return True  # 从函数返回结果

    def seek(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        if self._cap is None or self._info is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self.pause()  # 调用函数/方法执行某个动作或计算
        self._render_frame_at(int(frame_index))  # 调用函数/方法执行某个动作或计算

    def info(self) -> Optional[VideoInfo]:  # 定义函数（封装可复用逻辑）
        return self._info  # 从函数返回结果

    def current_frame(self) -> int:  # 定义函数（封装可复用逻辑）
        return int(self._frame_index)  # 从函数返回结果

    def close_video(self):  # 定义函数（封装可复用逻辑）
        self.pause()  # 调用函数/方法执行某个动作或计算
        if self._cap is not None:  # 条件分支判断并选择执行路径
            self._cap.release()  # 调用函数/方法执行某个动作或计算
        self._cap = None  # 给对象属性 self._cap 赋值/初始化（来自当前语句右侧表达式）
        self._path = None  # 给对象属性 self._path 赋值/初始化（来自当前语句右侧表达式）
        self._info = None  # 给对象属性 self._info 赋值/初始化（来自当前语句右侧表达式）
        self._frame_index = 0  # 给对象属性 self._frame_index 赋值/初始化（来自当前语句右侧表达式）
        self._slider.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        self._time_label.setText("-- / --")  # 调用函数/方法执行某个动作或计算
        self._frame_label.setPixmap(QPixmap())  # 调用函数/方法执行某个动作或计算

    def play(self):  # 定义函数（封装可复用逻辑）
        if self._cap is None or self._info is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if self._playing:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._playing = True  # 给对象属性 self._playing 赋值/初始化（来自当前语句右侧表达式）
        self._play_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        self._pause_btn.setEnabled(True)  # 调用函数/方法执行某个动作或计算
        interval_ms = int(1000 / max(1.0, float(self._info.fps)))  # 将 interval_ms 设为一次调用/构造的返回值
        self._timer.start(max(1, interval_ms))  # 调用函数/方法执行某个动作或计算

    def pause(self):  # 定义函数（封装可复用逻辑）
        self._playing = False  # 给对象属性 self._playing 赋值/初始化（来自当前语句右侧表达式）
        self._timer.stop()  # 调用函数/方法执行某个动作或计算
        self._play_btn.setEnabled(self._cap is not None)  # 调用函数/方法执行某个动作或计算
        self._pause_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算

    def set_preview_frame(self, frame_bgr: np.ndarray, frame_index: Optional[int] = None, total_frames: Optional[int] = None):  # 定义函数（封装可复用逻辑）
        self.pause()  # 调用函数/方法执行某个动作或计算
        qimg = _bgr_to_qimage(frame_bgr)  # 将 qimg 设为一次调用/构造的返回值
        pix = QPixmap.fromImage(qimg)  # 将 pix 设为一次调用/构造的返回值
        self._frame_label.setPixmap(pix.scaled(self._frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))  # 调用函数/方法执行某个动作或计算
        if frame_index is not None and total_frames is not None and total_frames > 0:  # 条件分支判断并选择执行路径
            self._time_label.setText(f"{frame_index+1} / {total_frames}")  # 调用函数/方法执行某个动作或计算

    def resizeEvent(self, event):  # 定义函数（封装可复用逻辑）
        if self._frame_label.pixmap() is not None and not self._frame_label.pixmap().isNull():  # 条件分支判断并选择执行路径
            pix = self._frame_label.pixmap()  # 将 pix 设为一次调用/构造的返回值
            self._frame_label.setPixmap(pix.scaled(self._frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))  # 调用函数/方法执行某个动作或计算
        super().resizeEvent(event)  # 调用函数/方法执行某个动作或计算

    def _tick(self):  # 定义函数（封装可复用逻辑）
        if self._cap is None or self._info is None:  # 条件分支判断并选择执行路径
            self.pause()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        if self._frame_index >= max(0, self._info.total_frames):  # 条件分支判断并选择执行路径
            self.pause()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        self._render_frame_at(self._frame_index + 1)  # 调用函数/方法执行某个动作或计算

    def _render_frame_at(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        if self._cap is None or self._info is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        frame_index = int(max(0, min(frame_index, max(0, self._info.total_frames - 1))))  # 将 frame_index 设为一次调用/构造的返回值
        if frame_index != self._frame_index:  # 条件分支判断并选择执行路径
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # 调用函数/方法执行某个动作或计算
        ret, frame = self._cap.read()  # 调用函数/方法执行某个动作或计算
        if not ret or frame is None:  # 条件分支判断并选择执行路径
            self.pause()  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        self._frame_index = frame_index  # 给对象属性 self._frame_index 赋值/初始化（来自当前语句右侧表达式）

        qimg = _bgr_to_qimage(frame)  # 将 qimg 设为一次调用/构造的返回值
        pix = QPixmap.fromImage(qimg)  # 将 pix 设为一次调用/构造的返回值
        self._frame_label.setPixmap(pix.scaled(self._frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))  # 调用函数/方法执行某个动作或计算
        self._update_time_label()  # 调用函数/方法执行某个动作或计算

        if self._slider.isEnabled():  # 条件分支判断并选择执行路径
            self._updating_slider = True  # 给对象属性 self._updating_slider 赋值/初始化（来自当前语句右侧表达式）
            self._slider.setValue(self._frame_index)  # 调用函数/方法执行某个动作或计算
            self._updating_slider = False  # 给对象属性 self._updating_slider 赋值/初始化（来自当前语句右侧表达式）
        self.positionChanged.emit(self._frame_index)  # 调用函数/方法执行某个动作或计算

    def _update_time_label(self):  # 定义函数（封装可复用逻辑）
        if self._info is None:  # 条件分支判断并选择执行路径
            self._time_label.setText("-- / --")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        self._time_label.setText(f"{self._frame_index+1} / {self._info.total_frames}")  # 调用函数/方法执行某个动作或计算

    def _on_slider_pressed(self):  # 定义函数（封装可复用逻辑）
        self.pause()  # 调用函数/方法执行某个动作或计算

    def _on_slider_released(self):  # 定义函数（封装可复用逻辑）
        self._render_frame_at(self._slider.value())  # 调用函数/方法执行某个动作或计算

    def _on_slider_changed(self, value: int):  # 定义函数（封装可复用逻辑）
        if self._updating_slider:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if self._cap is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._time_label.setText(f"{value+1} / {self._slider.maximum()+1}")  # 调用函数/方法执行某个动作或计算
