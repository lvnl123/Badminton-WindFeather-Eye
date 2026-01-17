from __future__ import annotations  # 从模块导入符号，供后续调用

import json  # 导入模块，供后续使用
import os  # 导入模块，供后续使用
import sys  # 导入模块，供后续使用
import time  # 导入模块，供后续使用
from pathlib import Path  # 从模块导入符号，供后续调用
from typing import Any, Dict, Optional, List  # 从模块导入符号，供后续调用

import pandas as pd  # 导入模块，供后续使用
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt, QTimer  # 从模块导入符号，供后续调用
from PySide6.QtGui import QColor  # 从模块导入符号，供后续调用
from PySide6.QtGui import QAction  # 从模块导入符号，供后续调用
from PySide6.QtWidgets import (  # 从模块导入符号，供后续调用
    QApplication,  # 执行当前语句（保持与上文逻辑一致）
    QComboBox,  # 执行当前语句（保持与上文逻辑一致）
    QFileDialog,  # 执行当前语句（保持与上文逻辑一致）
    QDialog,  # 执行当前语句（保持与上文逻辑一致）
    QFormLayout,  # 执行当前语句（保持与上文逻辑一致）
    QGroupBox,  # 执行当前语句（保持与上文逻辑一致）
    QHBoxLayout,  # 执行当前语句（保持与上文逻辑一致）
    QLabel,  # 执行当前语句（保持与上文逻辑一致）
    QLineEdit,  # 执行当前语句（保持与上文逻辑一致）
    QMainWindow,  # 执行当前语句（保持与上文逻辑一致）
    QMessageBox,  # 执行当前语句（保持与上文逻辑一致）
    QPushButton,  # 执行当前语句（保持与上文逻辑一致）
    QProgressBar,  # 执行当前语句（保持与上文逻辑一致）
    QSpinBox,  # 执行当前语句（保持与上文逻辑一致）
    QSplitter,  # 执行当前语句（保持与上文逻辑一致）
    QTabWidget,  # 执行当前语句（保持与上文逻辑一致）
    QTableView,  # 执行当前语句（保持与上文逻辑一致）
    QTextBrowser,  # 执行当前语句（保持与上文逻辑一致）
    QTextEdit,  # 执行当前语句（保持与上文逻辑一致）
    QVBoxLayout,  # 执行当前语句（保持与上文逻辑一致）
    QWidget,  # 执行当前语句（保持与上文逻辑一致）
    QCheckBox,  # 执行当前语句（保持与上文逻辑一致）
    QDoubleSpinBox,  # 执行当前语句（保持与上文逻辑一致）
    QGridLayout,  # 执行当前语句（保持与上文逻辑一致）
    QHeaderView,  # 执行当前语句（保持与上文逻辑一致）
)  # 执行当前语句（保持与上文逻辑一致）

ROOT = Path(__file__).resolve().parents[1]  # 将 ROOT 设为一次调用/构造的返回值
if str(ROOT) not in sys.path:  # 条件分支判断并选择执行路径
    sys.path.insert(0, str(ROOT))  # 调用函数/方法执行某个动作或计算

from ui_pyside6.widgets.pipeline_runner import PipelineConfig  # 从模块导入符号，供后续调用
from ui_pyside6.widgets.pipeline_worker import PipelineWorker, WorkerThread  # 从模块导入符号，供后续调用
from ui_pyside6.widgets.data_models import DataFrameModel  # 从模块导入符号，供后续调用
from ui_pyside6.widgets.simple_plot import (  # 从模块导入符号，供后续调用
    DensityBubbleMap,  # 执行当前语句（保持与上文逻辑一致）
    MetricCard,  # 执行当前语句（保持与上文逻辑一致）
    ProDistributionChart,  # 执行当前语句（保持与上文逻辑一致）
    SimpleBarChart,  # 执行当前语句（保持与上文逻辑一致）
    SimpleLinePlot,  # 执行当前语句（保持与上文逻辑一致）
    TerritoryScatterPlot,  # 执行当前语句（保持与上文逻辑一致）
    TimelineMarkers,  # 执行当前语句（保持与上文逻辑一致）
)  # 执行当前语句（保持与上文逻辑一致）
from ui_pyside6.widgets.video_player import VideoPlayer  # 从模块导入符号，供后续调用
from ui_pyside6.match_review.review_window import MatchReviewWindow  # 从模块导入符号，供后续调用


def _apply_style(app: QApplication):  # 定义函数（封装可复用逻辑）
    app.setStyleSheet(  # 执行当前语句（保持与上文逻辑一致）
        """
        QWidget{font-family:Segoe UI,Microsoft YaHei;font-size:12px;color:#e5e7eb;background:#0b0f14;}
        QMainWindow::separator{background:#111827;width:1px;height:1px;}
        QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{background:#0f1216;border:1px solid #2a2f3a;border-radius:8px;padding:8px;}
        QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus{border:1px solid #3b82f6;}
        QPushButton{background:#111827;border:1px solid #2a2f3a;border-radius:10px;padding:10px 12px;}
        QPushButton:hover{background:#0f172a;}
        QPushButton:disabled{color:#6b7280;background:#0b0f14;border:1px solid #1f2937;}
        QGroupBox{border:1px solid #1f2937;border-radius:12px;margin-top:10px;padding:10px;}
        QGroupBox:title{subcontrol-origin:margin;left:12px;top:-2px;padding:0 6px;color:#93c5fd;}
        QTabWidget::pane{border:1px solid #1f2937;border-radius:10px;padding:0px;}
        QTabBar::tab{background:#0f1216;border:1px solid #1f2937;border-bottom:none;border-top-left-radius:8px;border-top-right-radius:8px;padding:8px 12px;margin-right:4px;}
        QTabBar::tab:selected{background:#111827;border:1px solid #334155;}
        QHeaderView::section{background:#0f1216;border:1px solid #1f2937;padding:6px;}
        QTableView{gridline-color:#1f2937;selection-background-color:#1e3a8a;border:1px solid #1f2937;border-radius:10px;}
        QTextEdit{background:#0f1216;border:1px solid #1f2937;border-radius:10px;}
        QProgressBar{background:#0f1216;border:1px solid #1f2937;border-radius:10px;text-align:center;height:18px;}
        QProgressBar::chunk{background:#22c55e;border-radius:10px;}
        """
    )  # 执行当前语句（保持与上文逻辑一致）


class StepperWidget(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._steps = ["球场/球网检测", "羽毛球检测", "姿态检测", "事件检测", "击球类型识别", "合成可视化", "导出数据"]  # 给对象属性 self._steps 赋值/初始化（来自当前语句右侧表达式）
        self._current = ""  # 给对象属性 self._current 赋值/初始化（来自当前语句右侧表达式）
        self._done: set[str] = set()  # 调用函数/方法执行某个动作或计算
        self._failed = False  # 给对象属性 self._failed 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(60)  # 调用函数/方法执行某个动作或计算

    def reset(self):  # 定义函数（封装可复用逻辑）
        self._current = ""  # 给对象属性 self._current 赋值/初始化（来自当前语句右侧表达式）
        self._done.clear()  # 调用函数/方法执行某个动作或计算
        self._failed = False  # 给对象属性 self._failed 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_current(self, step_name: str):  # 定义函数（封装可复用逻辑）
        if self._current and self._current != step_name:  # 条件分支判断并选择执行路径
            self._done.add(self._current)  # 调用函数/方法执行某个动作或计算
        self._current = step_name  # 给对象属性 self._current 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_finished(self, ok: bool):  # 定义函数（封装可复用逻辑）
        self._failed = not ok  # 给对象属性 self._failed 赋值/初始化（来自当前语句右侧表达式）
        if ok:  # 条件分支判断并选择执行路径
            self._done.update(self._steps)  # 调用函数/方法执行某个动作或计算
        self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        from PySide6.QtGui import QPainter, QPen  # 从模块导入符号，供后续调用

        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        r = self.rect().adjusted(0, 0, -1, -1)  # 将 r 设为一次调用/构造的返回值
        p.fillRect(r, QColor("#0f1216"))  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#1f2937"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRoundedRect(r, 12, 12)  # 调用函数/方法执行某个动作或计算

        if not self._steps:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果

        left = r.left() + 14  # 将 left 设为一次调用/构造的返回值
        right = r.right() - 14  # 将 right 设为一次调用/构造的返回值
        y = r.center().y()  # 将 y 设为一次调用/构造的返回值
        n = len(self._steps)  # 将 n 设为一次调用/构造的返回值
        if n <= 1:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果

        p.setPen(QPen(QColor("#334155"), 2))  # 调用函数/方法执行某个动作或计算
        p.drawLine(left, y, right, y)  # 调用函数/方法执行某个动作或计算

        step_w = (right - left) / (n - 1)  # 初始化变量 step_w 为一个容器/表达式结果
        for i, name in enumerate(self._steps):  # 循环遍历序列/迭代器
            cx = left + int(i * step_w)  # 将 cx 设为一次调用/构造的返回值
            is_done = name in self._done  # 将表达式计算结果赋给变量 is_done
            is_current = name == self._current  # 将表达式计算结果赋给变量 is_current
            color = QColor("#6b7280")  # 将 color 设为一次调用/构造的返回值
            if is_done:  # 条件分支判断并选择执行路径
                color = QColor("#22c55e")  # 将 color 设为一次调用/构造的返回值
            elif is_current:  # 条件分支判断并选择执行路径
                color = QColor("#3b82f6")  # 将 color 设为一次调用/构造的返回值
            if self._failed and is_current:  # 条件分支判断并选择执行路径
                color = QColor("#ef4444")  # 将 color 设为一次调用/构造的返回值

            radius = 7 if is_current else 6  # 将表达式计算结果赋给变量 radius
            p.setBrush(color)  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#0b0f14"), 2))  # 调用函数/方法执行某个动作或计算
            p.drawEllipse(cx - radius, y - radius, radius * 2, radius * 2)  # 调用函数/方法执行某个动作或计算

        p.setPen(QPen(QColor("#9ca3af")))  # 调用函数/方法执行某个动作或计算
        p.drawText(r.adjusted(12, 6, -12, -6), Qt.AlignLeft | Qt.AlignTop, "步骤流")  # 调用函数/方法执行某个动作或计算


class MainWindow(QMainWindow):  # 定义类（封装数据与行为）
    def __init__(self):  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.setWindowTitle("TrackNetV3_Attention - 比赛视频训练分析工作台")  # 调用函数/方法执行某个动作或计算
        self.resize(1650, 820)  # 调用函数/方法执行某个动作或计算
        self._cwd = os.getcwd()  # 给对象属性 self._cwd 赋值/初始化（来自当前语句右侧表达式）
        self._review_window = None  # 给对象属性 self._review_window 赋值/初始化（来自当前语句右侧表达式）

        self._worker_thread: Optional[WorkerThread] = None  # 执行当前语句（保持与上文逻辑一致）
        self._worker: Optional[PipelineWorker] = None  # 执行当前语句（保持与上文逻辑一致）
        self._current_outputs: Optional[Dict[str, Any]] = None  # 执行当前语句（保持与上文逻辑一致）
        self._events_raw_df = pd.DataFrame()  # 给对象属性 self._events_raw_df 赋值/初始化（来自当前语句右侧表达式）
        self._current_df = pd.DataFrame()  # 给对象属性 self._current_df 赋值/初始化（来自当前语句右侧表达式）

        self._input_player = VideoPlayer()  # 给对象属性 self._input_player 赋值/初始化（来自当前语句右侧表达式）
        self._preview_player = VideoPlayer()  # 给对象属性 self._preview_player 赋值/初始化（来自当前语句右侧表达式）
        self._output_player = VideoPlayer()  # 给对象属性 self._output_player 赋值/初始化（来自当前语句右侧表达式）
        self._compare_input = VideoPlayer()  # 给对象属性 self._compare_input 赋值/初始化（来自当前语句右侧表达式）
        self._compare_output = VideoPlayer()  # 给对象属性 self._compare_output 赋值/初始化（来自当前语句右侧表达式）
        self._compare_sync_cb = QCheckBox("同步对比播放")  # 给对象属性 self._compare_sync_cb 赋值/初始化（来自当前语句右侧表达式）
        self._compare_sync_cb.setChecked(True)  # 调用函数/方法执行某个动作或计算
        self._compare_follow_combo = QComboBox()  # 给对象属性 self._compare_follow_combo 赋值/初始化（来自当前语句右侧表达式）
        self._compare_follow_combo.addItems(["输入驱动输出", "输出驱动输入"])  # 调用函数/方法执行某个动作或计算
        self._compare_follow_combo.setCurrentIndex(0)  # 调用函数/方法执行某个动作或计算
        self._compare_guard = False  # 给对象属性 self._compare_guard 赋值/初始化（来自当前语句右侧表达式）

        self._log = QTextEdit()  # 给对象属性 self._log 赋值/初始化（来自当前语句右侧表达式）
        self._log.setReadOnly(True)  # 调用函数/方法执行某个动作或计算

        self._status_step = QLabel("就绪")  # 给对象属性 self._status_step 赋值/初始化（来自当前语句右侧表达式）
        self._status_progress = QLabel("0%")  # 给对象属性 self._status_progress 赋值/初始化（来自当前语句右侧表达式）
        self._elapsed_label = QLabel("耗时: --")  # 给对象属性 self._elapsed_label 赋值/初始化（来自当前语句右侧表达式）
        self._eta_label = QLabel("预计剩余: --")  # 给对象属性 self._eta_label 赋值/初始化（来自当前语句右侧表达式）
        self._progress_bar = QProgressBar()  # 给对象属性 self._progress_bar 赋值/初始化（来自当前语句右侧表达式）
        self._progress_bar.setRange(0, 100)  # 调用函数/方法执行某个动作或计算
        self._progress_bar.setValue(0)  # 调用函数/方法执行某个动作或计算
        self._stepper = StepperWidget()  # 给对象属性 self._stepper 赋值/初始化（来自当前语句右侧表达式）

        self._overall_label = QLabel("总体进度: 0%")  # 给对象属性 self._overall_label 赋值/初始化（来自当前语句右侧表达式）
        self._overall_label.setStyleSheet("QLabel{color:#a7f3d0;}")  # 调用函数/方法执行某个动作或计算

        self._video_path_edit = QLineEdit()  # 给对象属性 self._video_path_edit 赋值/初始化（来自当前语句右侧表达式）
        self._video_path_edit.setReadOnly(True)  # 调用函数/方法执行某个动作或计算
        self._result_dir_edit = QLineEdit(str(ROOT / "results"))  # 给对象属性 self._result_dir_edit 赋值/初始化（来自当前语句右侧表达式）

        self._result_combo = QComboBox()  # 给对象属性 self._result_combo 赋值/初始化（来自当前语句右侧表达式）
        self._refresh_results_btn = QPushButton("刷新结果")  # 给对象属性 self._refresh_results_btn 赋值/初始化（来自当前语句右侧表达式）
        self._load_result_btn = QPushButton("加载结果")  # 给对象属性 self._load_result_btn 赋值/初始化（来自当前语句右侧表达式）

        self._device_combo = QComboBox()  # 给对象属性 self._device_combo 赋值/初始化（来自当前语句右侧表达式）
        self._device_combo.addItems(["cuda", "cpu"])  # 调用函数/方法执行某个动作或计算

        self._pose_model_combo = QComboBox()  # 给对象属性 self._pose_model_combo 赋值/初始化（来自当前语句右侧表达式）
        self._pose_model_combo.addItems(["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l"])  # 调用函数/方法执行某个动作或计算
        self._pose_model_combo.setCurrentText("rtmpose-m")  # 调用函数/方法执行某个动作或计算

        self._use_court_cb = QCheckBox("启用球场/球网检测与区域高亮")  # 给对象属性 self._use_court_cb 赋值/初始化（来自当前语句右侧表达式）
        self._use_court_cb.setChecked(True)  # 调用函数/方法执行某个动作或计算

        self._keep_skeleton_cb = QCheckBox("保留球员骨架")  # 给对象属性 self._keep_skeleton_cb 赋值/初始化（来自当前语句右侧表达式）
        self._keep_skeleton_cb.setChecked(True)  # 调用函数/方法执行某个动作或计算
        self._keep_traj_cb = QCheckBox("保留羽毛球轨迹")  # 给对象属性 self._keep_traj_cb 赋值/初始化（来自当前语句右侧表达式）
        self._keep_traj_cb.setChecked(True)  # 调用函数/方法执行某个动作或计算
        self._keep_stroke_hint_cb = QCheckBox("保留击球类别提示")  # 给对象属性 self._keep_stroke_hint_cb 赋值/初始化（来自当前语句右侧表达式）
        self._keep_stroke_hint_cb.setChecked(True)  # 调用函数/方法执行某个动作或计算

        self._stroke_model_combo = QComboBox()  # 给对象属性 self._stroke_model_combo 赋值/初始化（来自当前语句右侧表达式）
        self._stroke_model_combo.addItem("ShuttleSet 35类", "shuttleset_35classes")  # 调用函数/方法执行某个动作或计算
        self._stroke_model_combo.addItem("ShuttleSet 25类", "shuttleset_25classes")  # 调用函数/方法执行某个动作或计算
        self._stroke_model_combo.addItem("badDB 18类", "badDB_18classes")  # 调用函数/方法执行某个动作或计算
        self._stroke_model_combo.setCurrentIndex(0)  # 调用函数/方法执行某个动作或计算

        self._model_path_edit = QLineEdit(str(ROOT / "models" / "ball_track_attention.pt"))  # 给对象属性 self._model_path_edit 赋值/初始化（来自当前语句右侧表达式）
        self._court_model_edit = QLineEdit(str(ROOT / "models" / "court_kpRCNN.pth"))  # 给对象属性 self._court_model_edit 赋值/初始化（来自当前语句右侧表达式）
        self._net_model_edit = QLineEdit(str(ROOT / "models" / "net_kpRCNN.pth"))  # 给对象属性 self._net_model_edit 赋值/初始化（来自当前语句右侧表达式）

        self._num_frames_spin = QSpinBox()  # 给对象属性 self._num_frames_spin 赋值/初始化（来自当前语句右侧表达式）
        self._num_frames_spin.setRange(1, 9)  # 调用函数/方法执行某个动作或计算
        self._num_frames_spin.setValue(3)  # 调用函数/方法执行某个动作或计算

        self._threshold_spin = QDoubleSpinBox()  # 给对象属性 self._threshold_spin 赋值/初始化（来自当前语句右侧表达式）
        self._threshold_spin.setRange(0.0, 1.0)  # 调用函数/方法执行某个动作或计算
        self._threshold_spin.setSingleStep(0.05)  # 调用函数/方法执行某个动作或计算
        self._threshold_spin.setValue(0.5)  # 调用函数/方法执行某个动作或计算

        self._traj_len_spin = QSpinBox()  # 给对象属性 self._traj_len_spin 赋值/初始化（来自当前语句右侧表达式）
        self._traj_len_spin.setRange(1, 60)  # 调用函数/方法执行某个动作或计算
        self._traj_len_spin.setValue(10)  # 调用函数/方法执行某个动作或计算

        self._court_interval_spin = QSpinBox()  # 给对象属性 self._court_interval_spin 赋值/初始化（来自当前语句右侧表达式）
        self._court_interval_spin.setRange(1, 300)  # 调用函数/方法执行某个动作或计算
        self._court_interval_spin.setValue(30)  # 调用函数/方法执行某个动作或计算

        self._emit_every_spin = QSpinBox()  # 给对象属性 self._emit_every_spin 赋值/初始化（来自当前语句右侧表达式）
        self._emit_every_spin.setRange(1, 60)  # 调用函数/方法执行某个动作或计算
        self._emit_every_spin.setValue(10)  # 调用函数/方法执行某个动作或计算（减少预览频率，提升流畅度）
        self._viz_emit_every_spin = QSpinBox()  # 给对象属性 self._viz_emit_every_spin 赋值/初始化（来自当前语句右侧表达式）
        self._viz_emit_every_spin.setRange(1, 60)  # 调用函数/方法执行某个动作或计算
        self._viz_emit_every_spin.setValue(5)  # 调用函数/方法执行某个动作或计算（减少预览频率，提升流畅度）

        self._run_btn = QPushButton("开始训练分析")  # 给对象属性 self._run_btn 赋值/初始化（来自当前语句右侧表达式）
        self._stop_btn = QPushButton("停止")  # 给对象属性 self._stop_btn 赋值/初始化（来自当前语句右侧表达式）
        self._stop_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        self._open_video_btn = QPushButton("导入视频")  # 给对象属性 self._open_video_btn 赋值/初始化（来自当前语句右侧表达式）
        self._open_output_btn = QPushButton("打开输出目录")  # 给对象属性 self._open_output_btn 赋值/初始化（来自当前语句右侧表达式）

        self._csv_model = DataFrameModel()  # 给对象属性 self._csv_model 赋值/初始化（来自当前语句右侧表达式）
        self._csv_view = QTableView()  # 给对象属性 self._csv_view 赋值/初始化（来自当前语句右侧表达式）
        self._csv_view.setModel(self._csv_model)  # 调用函数/方法执行某个动作或计算
        self._csv_view.setSortingEnabled(True)  # 调用函数/方法执行某个动作或计算

        self._events_model = DataFrameModel()  # 给对象属性 self._events_model 赋值/初始化（来自当前语句右侧表达式）
        self._events_view = QTableView()  # 给对象属性 self._events_view 赋值/初始化（来自当前语句右侧表达式）
        self._events_view.setModel(self._events_model)  # 调用函数/方法执行某个动作或计算
        self._events_view.setSortingEnabled(True)  # 调用函数/方法执行某个动作或计算
        self._events_view.horizontalHeader().setStretchLastSection(True)  # 调用函数/方法执行某个动作或计算
        self._events_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # 调用函数/方法执行某个动作或计算
        self._events_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)  # 调用函数/方法执行某个动作或计算
        self._events_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)  # 调用函数/方法执行某个动作或计算

        self._event_search = QLineEdit()  # 给对象属性 self._event_search 赋值/初始化（来自当前语句右侧表达式）
        self._event_search.setPlaceholderText("搜索事件（frame/player/stroke/type 等）")  # 调用函数/方法执行某个动作或计算
        self._event_player_filter = QComboBox()  # 给对象属性 self._event_player_filter 赋值/初始化（来自当前语句右侧表达式）
        self._event_player_filter.addItems(["全部"])  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter = QComboBox()  # 给对象属性 self._event_stroke_filter 赋值/初始化（来自当前语句右侧表达式）
        self._event_stroke_filter.addItems(["全部"])  # 调用函数/方法执行某个动作或计算
        self._event_reset_btn = QPushButton("清空筛选")  # 给对象属性 self._event_reset_btn 赋值/初始化（来自当前语句右侧表达式）
        self._events_page = QWidget()  # 给对象属性 self._events_page 赋值/初始化（来自当前语句右侧表达式）

        self._speed_plot = SimpleLinePlot()  # 给对象属性 self._speed_plot 赋值/初始化（来自当前语句右侧表达式）
        self._ball_y_plot = SimpleLinePlot()  # 给对象属性 self._ball_y_plot 赋值/初始化（来自当前语句右侧表达式）
        self._hit_count_plot = SimpleLinePlot()  # 给对象属性 self._hit_count_plot 赋值/初始化（来自当前语句右侧表达式）
        self._speed_hist = ProDistributionChart()  # 给对象属性 self._speed_hist 赋值/初始化（来自当前语句右侧表达式）
        self._hit_interval_hist = ProDistributionChart()  # 给对象属性 self._hit_interval_hist 赋值/初始化（来自当前语句右侧表达式）
        self._hit_height_hist = ProDistributionChart()  # 给对象属性 self._hit_height_hist 赋值/初始化（来自当前语句右侧表达式）
        self._player_speed_hist = ProDistributionChart()  # 给对象属性 self._player_speed_hist 赋值/初始化（来自当前语句右侧表达式）
        self._timeline = TimelineMarkers()  # 给对象属性 self._timeline 赋值/初始化（来自当前语句右侧表达式）
        self._stroke_bar = SimpleBarChart()  # 给对象属性 self._stroke_bar 赋值/初始化（来自当前语句右侧表达式）
        self._heatmap = DensityBubbleMap()  # 给对象属性 self._heatmap 赋值/初始化（来自当前语句右侧表达式）
        self._density_source_combo = QComboBox()  # 给对象属性 self._density_source_combo 赋值/初始化（来自当前语句右侧表达式）
        self._density_source_combo.addItems(["可见帧", "全部帧", "仅击球帧", "仅选手0击球帧", "仅选手1击球帧"])  # 调用函数/方法执行某个动作或计算
        self._density_bins_combo = QComboBox()  # 给对象属性 self._density_bins_combo 赋值/初始化（来自当前语句右侧表达式）
        self._density_bins_combo.addItems(["粗(24x14)", "中(44x24)", "细(72x40)"])  # 调用函数/方法执行某个动作或计算
        self._density_bins_combo.setCurrentText("中(44x24)")  # 调用函数/方法执行某个动作或计算
        self._density_show_current_cb = QCheckBox("显示当前帧位置")  # 给对象属性 self._density_show_current_cb 赋值/初始化（来自当前语句右侧表达式）
        self._density_show_current_cb.setChecked(True)  # 调用函数/方法执行某个动作或计算
        self._density_export_btn = QPushButton("导出密度图…")  # 给对象属性 self._density_export_btn 赋值/初始化（来自当前语句右侧表达式）

        self._m_hits = MetricCard("击球次数")  # 给对象属性 self._m_hits 赋值/初始化（来自当前语句右侧表达式）
        self._m_duration = MetricCard("时长")  # 给对象属性 self._m_duration 赋值/初始化（来自当前语句右侧表达式）
        self._m_speed_avg = MetricCard("平均球速(像素/秒)")  # 给对象属性 self._m_speed_avg 赋值/初始化（来自当前语句右侧表达式）
        self._m_speed_max = MetricCard("最大球速(像素/秒)")  # 给对象属性 self._m_speed_max 赋值/初始化（来自当前语句右侧表达式）
        self._m_visible = MetricCard("可见率")  # 给对象属性 self._m_visible 赋值/初始化（来自当前语句右侧表达式）
        self._m_output = MetricCard("输出目录")  # 给对象属性 self._m_output 赋值/初始化（来自当前语句右侧表达式）

        self._p1_map = TerritoryScatterPlot()  # 给对象属性 self._p1_map 赋值/初始化（来自当前语句右侧表达式）
        self._p2_map = TerritoryScatterPlot()  # 给对象属性 self._p2_map 赋值/初始化（来自当前语句右侧表达式）
        self._p_dist_plot = SimpleLinePlot()  # 给对象属性 self._p_dist_plot 赋值/初始化（来自当前语句右侧表达式）
        self._p1_speed_plot = SimpleLinePlot()  # 给对象属性 self._p1_speed_plot 赋值/初始化（来自当前语句右侧表达式）
        self._p2_speed_plot = SimpleLinePlot()  # 给对象属性 self._p2_speed_plot 赋值/初始化（来自当前语句右侧表达式）

        self._build_ui()  # 调用函数/方法执行某个动作或计算
        self._build_actions()  # 调用函数/方法执行某个动作或计算
        self._connect()  # 调用函数/方法执行某个动作或计算

    def _build_actions(self):  # 定义函数（封装可复用逻辑）
        open_video = QAction("导入视频", self)  # 将 open_video 设为一次调用/构造的返回值
        open_video.triggered.connect(self._choose_video)  # 调用函数/方法执行某个动作或计算
        self.menuBar().addAction(open_video)  # 调用函数/方法执行某个动作或计算

        open_out = QAction("打开输出目录", self)  # 将 open_out 设为一次调用/构造的返回值
        open_out.triggered.connect(self._open_output_dir)  # 调用函数/方法执行某个动作或计算
        self.menuBar().addAction(open_out)  # 调用函数/方法执行某个动作或计算

        review_action = QAction("比赛复盘数据", self)  # 将 review_action 设为一次调用/构造的返回值
        review_action.triggered.connect(self._open_match_review)  # 调用函数/方法执行某个动作或计算
        self.menuBar().addAction(review_action)  # 调用函数/方法执行某个动作或计算

        help_menu = self.menuBar().addMenu("帮助")  # 将 help_menu 设为一次调用/构造的返回值

        about_action = QAction("关于", self)  # 将 about_action 设为一次调用/构造的返回值
        about_action.triggered.connect(self._show_about)  # 调用函数/方法执行某个动作或计算
        help_menu.addAction(about_action)  # 调用函数/方法执行某个动作或计算

        usage_action = QAction("使用说明", self)  # 将 usage_action 设为一次调用/构造的返回值
        usage_action.triggered.connect(self._show_usage)  # 调用函数/方法执行某个动作或计算
        help_menu.addAction(usage_action)  # 调用函数/方法执行某个动作或计算

        export_menu = self.menuBar().addMenu("导出")  # 将 export_menu 设为一次调用/构造的返回值
        export_overview = QAction("导出概览截图…", self)  # 将 export_overview 设为一次调用/构造的返回值
        export_overview.triggered.connect(self._export_overview_png)  # 调用函数/方法执行某个动作或计算
        export_menu.addAction(export_overview)  # 调用函数/方法执行某个动作或计算

        export_events = QAction("导出当前事件表(CSV)…", self)  # 将 export_events 设为一次调用/构造的返回值
        export_events.triggered.connect(self._export_events_csv)  # 调用函数/方法执行某个动作或计算
        export_menu.addAction(export_events)  # 调用函数/方法执行某个动作或计算

        export_csv = QAction("导出当前 CSV 数据(CSV)…", self)  # 将 export_csv 设为一次调用/构造的返回值
        export_csv.triggered.connect(self._export_csv_csv)  # 调用函数/方法执行某个动作或计算
        export_menu.addAction(export_csv)  # 调用函数/方法执行某个动作或计算

    def _build_ui(self):  # 定义函数（封装可复用逻辑）
        player_tabs = QTabWidget()  # 将 player_tabs 设为一次调用/构造的返回值
        player_tabs.addTab(self._input_player, "输入视频")  # 调用函数/方法执行某个动作或计算
        player_tabs.addTab(self._preview_player, "检测预览")  # 调用函数/方法执行某个动作或计算
        player_tabs.addTab(self._output_player, "输出视频")  # 调用函数/方法执行某个动作或计算
        player_tabs.addTab(self._build_compare_view(), "对比")  # 调用函数/方法执行某个动作或计算

        params = QGroupBox("训练分析参数")  # 将 params 设为一次调用/构造的返回值
        form = QFormLayout(params)  # 将 form 设为一次调用/构造的返回值
        form.setLabelAlignment(Qt.AlignRight)  # 调用函数/方法执行某个动作或计算
        form.addRow("视频路径", self._video_path_edit)  # 调用函数/方法执行某个动作或计算
        form.addRow("输出目录", self._result_dir_edit)  # 调用函数/方法执行某个动作或计算
        form.addRow("设备", self._device_combo)  # 调用函数/方法执行某个动作或计算
        form.addRow("Pose 模型", self._pose_model_combo)  # 调用函数/方法执行某个动作或计算
        options = QWidget()  # 将 options 设为一次调用/构造的返回值
        options_layout = QGridLayout(options)  # 将 options_layout 设为一次调用/构造的返回值
        options_layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        options_layout.setHorizontalSpacing(12)  # 调用函数/方法执行某个动作或计算
        options_layout.setVerticalSpacing(6)  # 调用函数/方法执行某个动作或计算
        options_layout.addWidget(self._use_court_cb, 0, 0)  # 调用函数/方法执行某个动作或计算
        options_layout.addWidget(self._keep_skeleton_cb, 0, 1)  # 调用函数/方法执行某个动作或计算
        options_layout.addWidget(self._keep_traj_cb, 1, 0)  # 调用函数/方法执行某个动作或计算
        options_layout.addWidget(self._keep_stroke_hint_cb, 1, 1)  # 调用函数/方法执行某个动作或计算
        options_layout.setColumnStretch(0, 1)  # 调用函数/方法执行某个动作或计算
        options_layout.setColumnStretch(1, 1)  # 调用函数/方法执行某个动作或计算
        form.addRow("", options)  # 调用函数/方法执行某个动作或计算
        form.addRow("击球类型模型", self._stroke_model_combo)  # 调用函数/方法执行某个动作或计算
        form.addRow("TrackNet 权重", self._model_path_edit)  # 调用函数/方法执行某个动作或计算
        form.addRow("球场模型", self._court_model_edit)  # 调用函数/方法执行某个动作或计算
        form.addRow("球网模型", self._net_model_edit)  # 调用函数/方法执行某个动作或计算
        form.addRow("输入帧数", self._num_frames_spin)  # 调用函数/方法执行某个动作或计算
        form.addRow("检测阈值", self._threshold_spin)  # 调用函数/方法执行某个动作或计算
        form.addRow("轨迹长度", self._traj_len_spin)  # 调用函数/方法执行某个动作或计算
        form.addRow("球场检测间隔", self._court_interval_spin)  # 调用函数/方法执行某个动作或计算
        form.addRow("预览抽样间隔", self._emit_every_spin)  # 调用函数/方法执行某个动作或计算
        form.addRow("合成预览抽样", self._viz_emit_every_spin)  # 调用函数/方法执行某个动作或计算

        buttons = QHBoxLayout()  # 将 buttons 设为一次调用/构造的返回值
        buttons.addWidget(self._open_video_btn)  # 调用函数/方法执行某个动作或计算
        buttons.addWidget(self._run_btn)  # 调用函数/方法执行某个动作或计算
        buttons.addWidget(self._stop_btn)  # 调用函数/方法执行某个动作或计算
        buttons.addWidget(self._open_output_btn)  # 调用函数/方法执行某个动作或计算

        status_box = QGroupBox("运行状态")  # 将 status_box 设为一次调用/构造的返回值
        status_layout = QVBoxLayout(status_box)  # 将 status_layout 设为一次调用/构造的返回值
        row1 = QHBoxLayout()  # 将 row1 设为一次调用/构造的返回值
        row1.addWidget(QLabel("当前步骤:"))  # 调用函数/方法执行某个动作或计算
        row1.addWidget(self._status_step, 1)  # 调用函数/方法执行某个动作或计算
        row1.addWidget(self._overall_label)  # 调用函数/方法执行某个动作或计算
        status_layout.addLayout(row1)  # 调用函数/方法执行某个动作或计算
        row2 = QHBoxLayout()  # 将 row2 设为一次调用/构造的返回值
        row2.addWidget(QLabel("进度:"))  # 调用函数/方法执行某个动作或计算
        row2.addWidget(self._status_progress, 1)  # 调用函数/方法执行某个动作或计算
        row2.addWidget(self._elapsed_label)  # 调用函数/方法执行某个动作或计算
        row2.addWidget(self._eta_label)  # 调用函数/方法执行某个动作或计算
        status_layout.addLayout(row2)  # 调用函数/方法执行某个动作或计算
        status_layout.addWidget(self._progress_bar)  # 调用函数/方法执行某个动作或计算
        status_layout.addWidget(self._stepper)  # 调用函数/方法执行某个动作或计算

        result_box = QGroupBox("结果浏览")  # 将 result_box 设为一次调用/构造的返回值
        result_form = QFormLayout(result_box)  # 将 result_form 设为一次调用/构造的返回值
        result_form.setLabelAlignment(Qt.AlignRight)  # 调用函数/方法执行某个动作或计算
        result_form.addRow("结果集", self._result_combo)  # 调用函数/方法执行某个动作或计算
        result_btns = QWidget()  # 将 result_btns 设为一次调用/构造的返回值
        result_btns_layout = QHBoxLayout(result_btns)  # 将 result_btns_layout 设为一次调用/构造的返回值
        result_btns_layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        result_btns_layout.addWidget(self._refresh_results_btn)  # 调用函数/方法执行某个动作或计算
        result_btns_layout.addWidget(self._load_result_btn)  # 调用函数/方法执行某个动作或计算
        result_form.addRow("", result_btns)  # 调用函数/方法执行某个动作或计算

        right_tabs = QTabWidget()  # 将 right_tabs 设为一次调用/构造的返回值
        right_tabs.addTab(params, "参数")  # 调用函数/方法执行某个动作或计算
        right_tabs.addTab(self._log, "日志")  # 调用函数/方法执行某个动作或计算

        right_panel = QWidget()  # 将 right_panel 设为一次调用/构造的返回值
        right_layout = QVBoxLayout(right_panel)  # 将 right_layout 设为一次调用/构造的返回值
        right_layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        right_layout.addWidget(right_tabs, 1)  # 调用函数/方法执行某个动作或计算
        right_layout.addWidget(status_box)  # 调用函数/方法执行某个动作或计算
        right_layout.addWidget(result_box)  # 调用函数/方法执行某个动作或计算
        right_layout.addLayout(buttons)  # 调用函数/方法执行某个动作或计算

        data_tabs = QTabWidget()  # 将 data_tabs 设为一次调用/构造的返回值
        data_tabs.addTab(self._build_overview(), "概览")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._events_page, "击球事件")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._csv_view, "CSV 数据")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._speed_plot, "球速曲线")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._ball_y_plot, "球高度(像素)")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._hit_count_plot, "累计击球数")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._build_distributions(), "分布")  # 调用函数/方法执行某个动作或计算
        data_tabs.addTab(self._build_players(), "选手")  # 调用函数/方法执行某个动作或计算

        left_split = QSplitter(Qt.Vertical)  # 将 left_split 设为一次调用/构造的返回值
        left_split.addWidget(player_tabs)  # 调用函数/方法执行某个动作或计算
        left_split.addWidget(data_tabs)  # 调用函数/方法执行某个动作或计算
        left_split.setStretchFactor(0, 3)  # 调用函数/方法执行某个动作或计算
        left_split.setStretchFactor(1, 2)  # 调用函数/方法执行某个动作或计算

        root_split = QSplitter(Qt.Horizontal)  # 将 root_split 设为一次调用/构造的返回值
        root_split.addWidget(left_split)  # 调用函数/方法执行某个动作或计算
        root_split.addWidget(right_panel)  # 调用函数/方法执行某个动作或计算
        root_split.setStretchFactor(0, 5)  # 调用函数/方法执行某个动作或计算
        root_split.setStretchFactor(1, 1)  # 调用函数/方法执行某个动作或计算

        container = QWidget()  # 将 container 设为一次调用/构造的返回值
        layout = QVBoxLayout(container)  # 将 layout 设为一次调用/构造的返回值
        layout.addWidget(self._build_header())  # 调用函数/方法执行某个动作或计算
        layout.addWidget(root_split, 1)  # 调用函数/方法执行某个动作或计算
        self.setCentralWidget(container)  # 调用函数/方法执行某个动作或计算

    def _build_distributions(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        root = QWidget()  # 将 root 设为一次调用/构造的返回值
        layout = QGridLayout(root)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        layout.setSpacing(12)  # 调用函数/方法执行某个动作或计算

        self._speed_hist.set_data([], title="球速分布(Pro)", x_label="ball_speed", color="#3b82f6")  # 调用函数/方法执行某个动作或计算
        self._hit_interval_hist.set_data([], title="击球间隔分布(Pro)", x_label="delta_seconds", color="#22c55e")  # 调用函数/方法执行某个动作或计算
        self._hit_height_hist.set_data([], title="击球高度分布(Pro)", x_label="hit_y_px", color="#f59e0b")  # 调用函数/方法执行某个动作或计算
        self._player_speed_hist.set_data([], title="选手瞬时速度分布(Pro)", x_label="player_speed_px_s", color="#a78bfa")  # 调用函数/方法执行某个动作或计算

        layout.addWidget(self._speed_hist, 0, 0)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._hit_interval_hist, 0, 1)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._hit_height_hist, 1, 0)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._player_speed_hist, 1, 1)  # 调用函数/方法执行某个动作或计算

        return root  # 从函数返回结果

    def _build_players(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        root = QWidget()  # 将 root 设为一次调用/构造的返回值
        layout = QGridLayout(root)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        layout.setSpacing(12)  # 调用函数/方法执行某个动作或计算

        self._p1_map.set_points([], title="选手0覆盖(凸包)", color="#3b82f6")  # 调用函数/方法执行某个动作或计算
        self._p2_map.set_points([], title="选手1覆盖(凸包)", color="#ef4444")  # 调用函数/方法执行某个动作或计算
        self._p1_map.set_current_point(None)  # 调用函数/方法执行某个动作或计算
        self._p2_map.set_current_point(None)  # 调用函数/方法执行某个动作或计算
        self._p_dist_plot.set_series([], [], x_label="time_seconds", y_label="player_distance_px", title="选手间距")  # 调用函数/方法执行某个动作或计算
        self._p1_speed_plot.set_series([], [], x_label="time_seconds", y_label="p0_speed_px_s", title="选手0速度")  # 调用函数/方法执行某个动作或计算
        self._p2_speed_plot.set_series([], [], x_label="time_seconds", y_label="p1_speed_px_s", title="选手1速度")  # 调用函数/方法执行某个动作或计算

        layout.addWidget(self._p1_map, 0, 0)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._p2_map, 0, 1)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._p_dist_plot, 1, 0, 2, 1)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._p1_speed_plot, 1, 1)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._p2_speed_plot, 2, 1)  # 调用函数/方法执行某个动作或计算

        layout.setRowStretch(0, 3)  # 调用函数/方法执行某个动作或计算
        layout.setRowStretch(1, 1)  # 调用函数/方法执行某个动作或计算
        layout.setRowStretch(2, 1)  # 调用函数/方法执行某个动作或计算
        layout.setColumnStretch(0, 1)  # 调用函数/方法执行某个动作或计算
        layout.setColumnStretch(1, 1)  # 调用函数/方法执行某个动作或计算
        
        return root  # 从函数返回结果

    def _build_compare_view(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        root = QWidget()  # 将 root 设为一次调用/构造的返回值
        layout = QVBoxLayout(root)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算

        controls = QHBoxLayout()  # 将 controls 设为一次调用/构造的返回值
        controls.addWidget(self._compare_sync_cb)  # 调用函数/方法执行某个动作或计算
        controls.addWidget(QLabel("跟随模式"))  # 调用函数/方法执行某个动作或计算
        controls.addWidget(self._compare_follow_combo)  # 调用函数/方法执行某个动作或计算
        controls.addStretch(1)  # 调用函数/方法执行某个动作或计算
        layout.addLayout(controls)  # 调用函数/方法执行某个动作或计算

        split = QSplitter(Qt.Horizontal)  # 将 split 设为一次调用/构造的返回值
        split.addWidget(self._compare_input)  # 调用函数/方法执行某个动作或计算
        split.addWidget(self._compare_output)  # 调用函数/方法执行某个动作或计算
        split.setStretchFactor(0, 1)  # 调用函数/方法执行某个动作或计算
        split.setStretchFactor(1, 1)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(split, 1)  # 调用函数/方法执行某个动作或计算
        return root  # 从函数返回结果

    def _build_header(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        w = QWidget()  # 将 w 设为一次调用/构造的返回值
        w.setFixedHeight(52)  # 调用函数/方法执行某个动作或计算
        layout = QHBoxLayout(w)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(10, 8, 10, 8)  # 调用函数/方法执行某个动作或计算
        title = QLabel("比赛视频训练分析工作台")  # 将 title 设为一次调用/构造的返回值
        title.setStyleSheet("QLabel{font-size:16px;font-weight:600;color:#e5e7eb;}")  # 调用函数/方法执行某个动作或计算
        sub = QLabel("TrackNetV3 + MMPose + Event + BST")  # 将 sub 设为一次调用/构造的返回值
        sub.setStyleSheet("QLabel{color:#9ca3af;}")  # 调用函数/方法执行某个动作或计算
        layout.addWidget(title)  # 调用函数/方法执行某个动作或计算
        layout.addSpacing(12)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(sub, 1)  # 调用函数/方法执行某个动作或计算
        file_lab = QLabel()  # 将 file_lab 设为一次调用/构造的返回值
        file_lab.setStyleSheet("QLabel{color:#93c5fd;}")  # 调用函数/方法执行某个动作或计算
        file_lab.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 调用函数/方法执行某个动作或计算
        self._header_file = file_lab  # 给对象属性 self._header_file 赋值/初始化（来自当前语句右侧表达式）
        layout.addWidget(file_lab)  # 调用函数/方法执行某个动作或计算
        return w  # 从函数返回结果

    def _build_overview(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        root = QWidget()  # 将 root 设为一次调用/构造的返回值
        self._overview_widget = root  # 给对象属性 self._overview_widget 赋值/初始化（来自当前语句右侧表达式）
        layout = QVBoxLayout(root)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算

        cards = QWidget()  # 将 cards 设为一次调用/构造的返回值
        grid = QGridLayout(cards)  # 将 grid 设为一次调用/构造的返回值
        grid.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        grid.setHorizontalSpacing(10)  # 调用函数/方法执行某个动作或计算
        grid.setVerticalSpacing(10)  # 调用函数/方法执行某个动作或计算

        self._m_hits.set_accent(QColor("#f59e0b"))  # 调用函数/方法执行某个动作或计算
        self._m_duration.set_accent(QColor("#22c55e"))  # 调用函数/方法执行某个动作或计算
        self._m_speed_avg.set_accent(QColor("#3b82f6"))  # 调用函数/方法执行某个动作或计算
        self._m_speed_max.set_accent(QColor("#ef4444"))  # 调用函数/方法执行某个动作或计算
        self._m_visible.set_accent(QColor("#a78bfa"))  # 调用函数/方法执行某个动作或计算
        self._m_output.set_accent(QColor("#93c5fd"))  # 调用函数/方法执行某个动作或计算

        grid.addWidget(self._m_hits, 0, 0)  # 调用函数/方法执行某个动作或计算
        grid.addWidget(self._m_duration, 0, 1)  # 调用函数/方法执行某个动作或计算
        grid.addWidget(self._m_visible, 0, 2)  # 调用函数/方法执行某个动作或计算
        grid.addWidget(self._m_speed_avg, 1, 0)  # 调用函数/方法执行某个动作或计算
        grid.addWidget(self._m_speed_max, 1, 1)  # 调用函数/方法执行某个动作或计算
        grid.addWidget(self._m_output, 1, 2)  # 调用函数/方法执行某个动作或计算

        self._timeline.set_markers([], [], title="击球时间轴(点击跳转)")  # 调用函数/方法执行某个动作或计算
        self._heatmap.set_points([], title="球位置密度(气泡聚合)")  # 调用函数/方法执行某个动作或计算
        self._heatmap.set_show_current_point(True)  # 调用函数/方法执行某个动作或计算
        self._stroke_bar.set_data([], [], title="击球类型分布")  # 调用函数/方法执行某个动作或计算

        bottom_split = QSplitter(Qt.Horizontal)  # 将 bottom_split 设为一次调用/构造的返回值
        bottom_split.addWidget(self._build_density_panel())  # 调用函数/方法执行某个动作或计算
        bottom_split.addWidget(self._stroke_bar)  # 调用函数/方法执行某个动作或计算
        bottom_split.setStretchFactor(0, 3)  # 调用函数/方法执行某个动作或计算
        bottom_split.setStretchFactor(1, 2)  # 调用函数/方法执行某个动作或计算

        layout.addWidget(cards)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._timeline)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(bottom_split, 1)  # 调用函数/方法执行某个动作或计算
        return root  # 从函数返回结果

    def _build_density_panel(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        root = QWidget()  # 将 root 设为一次调用/构造的返回值
        layout = QVBoxLayout(root)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算

        bar = QWidget()  # 将 bar 设为一次调用/构造的返回值
        bar_layout = QHBoxLayout(bar)  # 将 bar_layout 设为一次调用/构造的返回值
        bar_layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(QLabel("密度来源"))  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._density_source_combo)  # 调用函数/方法执行某个动作或计算
        bar_layout.addSpacing(8)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(QLabel("网格"))  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._density_bins_combo)  # 调用函数/方法执行某个动作或计算
        bar_layout.addSpacing(8)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._density_show_current_cb)  # 调用函数/方法执行某个动作或计算
        bar_layout.addStretch(1)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._density_export_btn)  # 调用函数/方法执行某个动作或计算

        layout.addWidget(bar)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._heatmap, 1)  # 调用函数/方法执行某个动作或计算
        return root  # 从函数返回结果

    def _build_events_page(self) -> QWidget:  # 定义函数（封装可复用逻辑）
        layout = QVBoxLayout(self._events_page)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算

        bar = QWidget()  # 将 bar 设为一次调用/构造的返回值
        bar_layout = QHBoxLayout(bar)  # 将 bar_layout 设为一次调用/构造的返回值
        bar_layout.setContentsMargins(0, 0, 0, 0)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._event_search, 2)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(QLabel("选手"))  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._event_player_filter)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(QLabel("击球类型"))  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._event_stroke_filter, 1)  # 调用函数/方法执行某个动作或计算
        bar_layout.addWidget(self._event_reset_btn)  # 调用函数/方法执行某个动作或计算

        layout.addWidget(bar)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self._events_view, 1)  # 调用函数/方法执行某个动作或计算
        return self._events_page  # 从函数返回结果

    def _connect(self):  # 定义函数（封装可复用逻辑）
        self._open_video_btn.clicked.connect(self._choose_video)  # 调用函数/方法执行某个动作或计算
        self._open_output_btn.clicked.connect(self._open_output_dir)  # 调用函数/方法执行某个动作或计算
        self._run_btn.clicked.connect(self._start_pipeline)  # 调用函数/方法执行某个动作或计算
        self._stop_btn.clicked.connect(self._stop_pipeline)  # 调用函数/方法执行某个动作或计算
        self._output_player.positionChanged.connect(self._on_output_position)  # 调用函数/方法执行某个动作或计算
        self._refresh_results_btn.clicked.connect(self._refresh_results_list)  # 调用函数/方法执行某个动作或计算
        self._load_result_btn.clicked.connect(self._load_selected_result)  # 调用函数/方法执行某个动作或计算
        self._timeline.markerActivated.connect(self._seek_output)  # 调用函数/方法执行某个动作或计算
        if self._events_view.selectionModel() is not None:  # 条件分支判断并选择执行路径
            self._events_view.selectionModel().selectionChanged.connect(self._on_event_selection_changed)  # 调用函数/方法执行某个动作或计算
        self._events_view.doubleClicked.connect(self._on_event_double_clicked)  # 调用函数/方法执行某个动作或计算
        self._event_search.textChanged.connect(self._apply_event_filters)  # 调用函数/方法执行某个动作或计算
        self._event_player_filter.currentIndexChanged.connect(self._apply_event_filters)  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.currentIndexChanged.connect(self._apply_event_filters)  # 调用函数/方法执行某个动作或计算
        self._event_reset_btn.clicked.connect(self._reset_event_filters)  # 调用函数/方法执行某个动作或计算

        self._compare_input.positionChanged.connect(self._on_compare_input_pos)  # 调用函数/方法执行某个动作或计算
        self._compare_output.positionChanged.connect(self._on_compare_output_pos)  # 调用函数/方法执行某个动作或计算
        self._density_source_combo.currentIndexChanged.connect(self._refresh_density_view)  # 调用函数/方法执行某个动作或计算
        self._density_bins_combo.currentIndexChanged.connect(self._refresh_density_view)  # 调用函数/方法执行某个动作或计算
        self._density_show_current_cb.toggled.connect(self._on_density_show_current_toggled)  # 调用函数/方法执行某个动作或计算
        self._density_export_btn.clicked.connect(self._export_density_png)  # 调用函数/方法执行某个动作或计算
        self._apply_window_ready()  # 调用函数/方法执行某个动作或计算
        self._refresh_results_list()  # 调用函数/方法执行某个动作或计算
        self._build_events_page()  # 调用函数/方法执行某个动作或计算

    def _apply_window_ready(self):  # 定义函数（封装可复用逻辑）
        app = QApplication.instance()  # 将 app 设为一次调用/构造的返回值
        if app is not None:  # 条件分支判断并选择执行路径
            _apply_style(app)  # 调用函数/方法执行某个动作或计算

    def _ensure_on_screen(self):  # 定义函数（封装可复用逻辑）
        screen = self.screen() or QApplication.primaryScreen()  # 将 screen 设为一次调用/构造的返回值
        if screen is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        avail = screen.availableGeometry()  # 将 avail 设为一次调用/构造的返回值
        if avail.width() <= 0 or avail.height() <= 0:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果

        target_w = min(self.width(), int(avail.width() * 0.98))  # 将 target_w 设为一次调用/构造的返回值
        target_h = min(self.height(), int(avail.height() * 0.95))  # 将 target_h 设为一次调用/构造的返回值
        target_w = max(900, min(target_w, avail.width()))  # 将 target_w 设为一次调用/构造的返回值
        target_h = max(620, min(target_h, avail.height()))  # 将 target_h 设为一次调用/构造的返回值

        if self.width() != target_w or self.height() != target_h:  # 条件分支判断并选择执行路径
            self.resize(target_w, target_h)  # 调用函数/方法执行某个动作或计算

        x = avail.x() + max(0, (avail.width() - self.width()) // 2)  # 将 x 设为一次调用/构造的返回值
        y = avail.y() + max(0, (avail.height() - self.height()) // 2)  # 将 y 设为一次调用/构造的返回值
        self.move(x, y)  # 调用函数/方法执行某个动作或计算

    def _choose_video(self):  # 定义函数（封装可复用逻辑）
        file_path, _ = QFileDialog.getOpenFileName(  # 执行当前语句（保持与上文逻辑一致）
            self, "选择比赛视频", str(ROOT / "videos"), "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"  # 执行当前语句（保持与上文逻辑一致）
        )  # 执行当前语句（保持与上文逻辑一致）
        if not file_path:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._video_path_edit.setText(file_path)  # 调用函数/方法执行某个动作或计算
        self._header_file.setText(file_path)  # 调用函数/方法执行某个动作或计算
        self._input_player.open(file_path)  # 调用函数/方法执行某个动作或计算
        self._compare_input.open(file_path)  # 调用函数/方法执行某个动作或计算
        self._preview_player.set_preview_frame(self._get_black_frame(), 0, 0)  # 调用函数/方法执行某个动作或计算

    def _open_output_dir(self):  # 定义函数（封装可复用逻辑）
        text = self._result_dir_edit.text().strip()  # 将 text 设为一次调用/构造的返回值
        if not text:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        p = Path(text).expanduser()  # 将 p 设为一次调用/构造的返回值
        if not p.is_absolute():  # 条件分支判断并选择执行路径
            p = ROOT / p  # 将表达式计算结果赋给变量 p
        p = p.resolve(strict=False)  # 将 p 设为一次调用/构造的返回值
        p.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
        self._result_dir_edit.setText(str(p))  # 调用函数/方法执行某个动作或计算
        os.startfile(str(p))  # 调用函数/方法执行某个动作或计算
        self._refresh_results_list()  # 调用函数/方法执行某个动作或计算

    def _start_pipeline(self):  # 定义函数（封装可复用逻辑）
        if self._worker_thread is not None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        video_path = self._video_path_edit.text().strip()  # 将 video_path 设为一次调用/构造的返回值
        if not video_path:  # 条件分支判断并选择执行路径
            QMessageBox.warning(self, "提示", "请先导入视频")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        result_dir_text = self._result_dir_edit.text().strip()  # 将 result_dir_text 设为一次调用/构造的返回值
        if not result_dir_text:  # 条件分支判断并选择执行路径
            QMessageBox.warning(self, "提示", "请设置输出目录")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        result_dir_path = Path(result_dir_text).expanduser()  # 将 result_dir_path 设为一次调用/构造的返回值
        if not result_dir_path.is_absolute():  # 条件分支判断并选择执行路径
            result_dir_path = ROOT / result_dir_path  # 将表达式计算结果赋给变量 result_dir_path
        result_dir_path = result_dir_path.resolve(strict=False)  # 将 result_dir_path 设为一次调用/构造的返回值
        result_dir = str(result_dir_path)  # 将 result_dir 设为一次调用/构造的返回值
        self._result_dir_edit.setText(result_dir)  # 调用函数/方法执行某个动作或计算

        config = PipelineConfig(  # 将表达式计算结果赋给变量 config
            model_path=self._model_path_edit.text().strip(),  # 将 model_path 设为一次调用/构造的返回值
            num_frames=int(self._num_frames_spin.value()),  # 将 num_frames 设为一次调用/构造的返回值
            threshold=float(self._threshold_spin.value()),  # 将 threshold 设为一次调用/构造的返回值
            traj_len=int(self._traj_len_spin.value()),  # 将 traj_len 设为一次调用/构造的返回值
            device=self._device_combo.currentText(),  # 将 device 设为一次调用/构造的返回值
            pose_model=self._pose_model_combo.currentText(),  # 将 pose_model 设为一次调用/构造的返回值
            use_court_detection=bool(self._use_court_cb.isChecked()),  # 将 use_court_detection 设为一次调用/构造的返回值
            court_model_path=self._court_model_edit.text().strip(),  # 将 court_model_path 设为一次调用/构造的返回值
            net_model_path=self._net_model_edit.text().strip(),  # 将 net_model_path 设为一次调用/构造的返回值
            court_detection_interval=int(self._court_interval_spin.value()),  # 将 court_detection_interval 设为一次调用/构造的返回值
            pose_emit_every_n=int(self._emit_every_spin.value()),  # 将 pose_emit_every_n 设为一次调用/构造的返回值
            viz_emit_every_n=int(self._viz_emit_every_spin.value()),  # 将 viz_emit_every_n 设为一次调用/构造的返回值
            keep_player_skeleton=bool(self._keep_skeleton_cb.isChecked()),  # 将 keep_player_skeleton 设为一次调用/构造的返回值
            keep_ball_trajectory=bool(self._keep_traj_cb.isChecked()),  # 将 keep_ball_trajectory 设为一次调用/构造的返回值
            keep_stroke_type_hint=bool(self._keep_stroke_hint_cb.isChecked()),  # 将 keep_stroke_type_hint 设为一次调用/构造的返回值
            stroke_model=str(self._stroke_model_combo.currentData()),  # 将 stroke_model 设为一次调用/构造的返回值
            dataset="shuttleset",  # 将表达式计算结果赋给变量 dataset
            stroke_seq_len=100,  # 将表达式计算结果赋给变量 stroke_seq_len
        )  # 执行当前语句（保持与上文逻辑一致）

        self._log.clear()  # 调用函数/方法执行某个动作或计算
        self._status_step.setText("启动中…")  # 调用函数/方法执行某个动作或计算
        self._status_progress.setText("0%")  # 调用函数/方法执行某个动作或计算
        self._overall_label.setText("总体进度: 0%")  # 调用函数/方法执行某个动作或计算
        self._elapsed_label.setText("耗时: 0s")  # 调用函数/方法执行某个动作或计算
        self._eta_label.setText("预计剩余: --")  # 调用函数/方法执行某个动作或计算
        self._progress_bar.setValue(0)  # 调用函数/方法执行某个动作或计算
        self._stepper.reset()  # 调用函数/方法执行某个动作或计算
        self._run_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        self._stop_btn.setEnabled(True)  # 调用函数/方法执行某个动作或计算
        self._run_started_ts = time.time()  # 给对象属性 self._run_started_ts 赋值/初始化（来自当前语句右侧表达式）
        self._last_progress_ts = self._run_started_ts  # 给对象属性 self._last_progress_ts 赋值/初始化（来自当前语句右侧表达式）
        self._last_progress = 0  # 给对象属性 self._last_progress 赋值/初始化（来自当前语句右侧表达式）

        self._worker = PipelineWorker(video_path, result_dir, config)  # 给对象属性 self._worker 赋值/初始化（来自当前语句右侧表达式）
        self._worker_thread = WorkerThread(self._worker)  # 给对象属性 self._worker_thread 赋值/初始化（来自当前语句右侧表达式）
        self._worker.logLine.connect(self._append_log)  # 调用函数/方法执行某个动作或计算
        self._worker.stepChanged.connect(self._on_step)  # 调用函数/方法执行某个动作或计算
        self._worker.overallProgressChanged.connect(self._on_overall_progress)  # 调用函数/方法执行某个动作或计算
        self._worker.previewFrame.connect(self._on_preview_frame)  # 调用函数/方法执行某个动作或计算
        self._worker.outputsReady.connect(self._on_outputs_ready)  # 调用函数/方法执行某个动作或计算
        self._worker.finished.connect(self._on_finished)  # 调用函数/方法执行某个动作或计算
        self._worker_thread.start()  # 调用函数/方法执行某个动作或计算

    def _stop_pipeline(self):  # 定义函数（封装可复用逻辑）
        if self._worker is not None:  # 条件分支判断并选择执行路径
            self._worker.request_stop()  # 调用函数/方法执行某个动作或计算
            self._append_log("已请求停止…")  # 调用函数/方法执行某个动作或计算

    def _append_log(self, line: str):  # 定义函数（封装可复用逻辑）
        self._log.append(line)  # 调用函数/方法执行某个动作或计算

    def _on_step(self, step_name: str):  # 定义函数（封装可复用逻辑）
        self._status_step.setText(step_name)  # 调用函数/方法执行某个动作或计算
        self._stepper.set_current(step_name)  # 调用函数/方法执行某个动作或计算

    def _on_overall_progress(self, p: int):  # 定义函数（封装可复用逻辑）
        self._overall_label.setText(f"总体进度: {p}%")  # 调用函数/方法执行某个动作或计算
        self._status_progress.setText(f"{p}%")  # 调用函数/方法执行某个动作或计算
        self._progress_bar.setValue(int(p))  # 调用函数/方法执行某个动作或计算
        now = time.time()  # 将 now 设为一次调用/构造的返回值
        start = getattr(self, "_run_started_ts", None)  # 将 start 设为一次调用/构造的返回值
        if start is not None:  # 条件分支判断并选择执行路径
            elapsed = max(0.0, now - float(start))  # 将 elapsed 设为一次调用/构造的返回值
            self._elapsed_label.setText(f"耗时: {int(elapsed)}s")  # 调用函数/方法执行某个动作或计算
        if p > 0 and start is not None:  # 条件分支判断并选择执行路径
            elapsed = max(0.001, now - float(start))  # 将 elapsed 设为一次调用/构造的返回值
            total_est = elapsed / (p / 100.0)  # 将 total_est 设为一次调用/构造的返回值
            remain = max(0.0, total_est - elapsed)  # 将 remain 设为一次调用/构造的返回值
            self._eta_label.setText(f"预计剩余: {int(remain)}s")  # 调用函数/方法执行某个动作或计算

    def _on_preview_frame(self, frame_bgr: Any):  # 定义函数（封装可复用逻辑）
        if frame_bgr is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._preview_player.set_preview_frame(frame_bgr)  # 调用函数/方法执行某个动作或计算

    def _on_outputs_ready(self, outputs: Dict[str, Any]):  # 定义函数（封装可复用逻辑）
        self._current_outputs = outputs  # 给对象属性 self._current_outputs 赋值/初始化（来自当前语句右侧表达式）
        self._refresh_results_list(select_dir=outputs.get("video_result_dir"))  # 调用函数/方法执行某个动作或计算
        self._load_outputs(outputs)  # 调用函数/方法执行某个动作或计算

    def _load_outputs(self, outputs: Dict[str, Any]):  # 定义函数（封装可复用逻辑）
        combined_video = outputs.get("combined_video_path")  # 将 combined_video 设为一次调用/构造的返回值
        if combined_video and Path(combined_video).exists():  # 条件分支判断并选择执行路径
            self._output_player.open(combined_video)  # 调用函数/方法执行某个动作或计算
            self._compare_output.open(combined_video)  # 调用函数/方法执行某个动作或计算

        csv_path = outputs.get("csv_path")  # 将 csv_path 设为一次调用/构造的返回值
        df = pd.DataFrame()  # 将 df 设为一次调用/构造的返回值
        if csv_path and Path(csv_path).exists():  # 条件分支判断并选择执行路径
            df = pd.read_csv(csv_path)  # 将 df 设为一次调用/构造的返回值
        self._csv_model.set_dataframe(df)  # 调用函数/方法执行某个动作或计算
        self._current_df = df.copy()  # 给对象属性 self._current_df 赋值/初始化（来自当前语句右侧表达式）

        if not df.empty:  # 条件分支判断并选择执行路径
            xs = df["time_seconds"].astype(float).tolist() if "time_seconds" in df.columns else list(range(len(df)))  # 将 xs 设为一次调用/构造的返回值
            speed = df["ball_speed"].astype(float).tolist() if "ball_speed" in df.columns else [0.0] * len(xs)  # 将 speed 设为一次调用/构造的返回值
            self._speed_plot.set_series(xs, speed, x_label="time_seconds", y_label="ball_speed")  # 调用函数/方法执行某个动作或计算

            by = (  # 初始化变量 by 为一个容器/表达式结果
                df["ball_denoise_y"].astype(float).tolist()  # 调用函数/方法执行某个动作或计算
                if "ball_denoise_y" in df.columns  # 条件分支判断并选择执行路径
                else (df["ball_y"].astype(float).tolist() if "ball_y" in df.columns else [0.0] * len(xs))  # 条件分支的否则路径
            )  # 执行当前语句（保持与上文逻辑一致）
            self._ball_y_plot.set_series(xs, by, x_label="time_seconds", y_label="ball_y")  # 调用函数/方法执行某个动作或计算

            hits = (  # 初始化变量 hits 为一个容器/表达式结果
                df["cumulative_hit_count"].astype(float).tolist()  # 调用函数/方法执行某个动作或计算
                if "cumulative_hit_count" in df.columns  # 条件分支判断并选择执行路径
                else [0.0] * len(xs)  # 条件分支的否则路径
            )  # 执行当前语句（保持与上文逻辑一致）
            self._hit_count_plot.set_series(xs, hits, x_label="time_seconds", y_label="cumulative_hit_count")  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self._speed_plot.set_series([], [])  # 调用函数/方法执行某个动作或计算
            self._ball_y_plot.set_series([], [])  # 调用函数/方法执行某个动作或计算
            self._hit_count_plot.set_series([], [])  # 调用函数/方法执行某个动作或计算

        events_df = self._load_events(outputs)  # 将 events_df 设为一次调用/构造的返回值
        self._events_raw_df = events_df.copy()  # 给对象属性 self._events_raw_df 赋值/初始化（来自当前语句右侧表达式）
        self._rebuild_event_filters(events_df)  # 调用函数/方法执行某个动作或计算
        self._apply_event_filters()  # 调用函数/方法执行某个动作或计算
        self._update_overview(df, events_df, outputs)  # 调用函数/方法执行某个动作或计算
        self._update_players(df)  # 调用函数/方法执行某个动作或计算
        self._update_distributions(df, events_df)  # 调用函数/方法执行某个动作或计算

    def _load_events(self, outputs: Dict[str, Any]) -> pd.DataFrame:  # 定义函数（封装可复用逻辑）
        hit_path = outputs.get("hit_events_path")  # 将 hit_path 设为一次调用/构造的返回值
        if not hit_path or not Path(hit_path).exists():  # 条件分支判断并选择执行路径
            return pd.DataFrame()  # 从函数返回结果
        with open(hit_path, "r", encoding="utf-8") as f:  # 上下文管理：确保资源正确释放
            hits = json.load(f)  # 将 hits 设为一次调用/构造的返回值
        df = pd.DataFrame(hits)  # 将 df 设为一次调用/构造的返回值

        stroke_path = outputs.get("stroke_results_path")  # 将 stroke_path 设为一次调用/构造的返回值
        if stroke_path and Path(stroke_path).exists():  # 条件分支判断并选择执行路径
            with open(stroke_path, "r", encoding="utf-8") as f:  # 上下文管理：确保资源正确释放
                strokes = json.load(f)  # 将 strokes 设为一次调用/构造的返回值
            sdf = pd.DataFrame(strokes)  # 将 sdf 设为一次调用/构造的返回值
            if "frame" in df.columns and "frame" in sdf.columns:  # 条件分支判断并选择执行路径
                df = df.merge(sdf, on="frame", how="left", suffixes=("", "_stroke"))  # 将 df 设为一次调用/构造的返回值
        return df  # 从函数返回结果

    def _update_overview(self, df: pd.DataFrame, events_df: pd.DataFrame, outputs: Dict[str, Any]):  # 定义函数（封装可复用逻辑）
        total_hits = 0  # 将表达式计算结果赋给变量 total_hits
        if "is_hit" in df.columns:  # 条件分支判断并选择执行路径
            total_hits = int(df["is_hit"].astype(int).sum())  # 将 total_hits 设为一次调用/构造的返回值
        elif "frame" in events_df.columns:  # 条件分支判断并选择执行路径
            total_hits = int(len(events_df))  # 将 total_hits 设为一次调用/构造的返回值
        self._m_hits.set_content(value=str(total_hits), subtitle="hit frames")  # 调用函数/方法执行某个动作或计算

        duration = 0.0  # 将表达式计算结果赋给变量 duration
        if "time_seconds" in df.columns and len(df) > 0:  # 条件分支判断并选择执行路径
            duration = float(df["time_seconds"].iat[-1])  # 将 duration 设为一次调用/构造的返回值
        self._m_duration.set_content(value=f"{duration:.1f}s", subtitle="from video fps")  # 调用函数/方法执行某个动作或计算

        if "ball_speed" in df.columns and len(df) > 0:  # 条件分支判断并选择执行路径
            sp = pd.to_numeric(df["ball_speed"], errors="coerce").fillna(0.0)  # 将 sp 设为一次调用/构造的返回值
            self._m_speed_avg.set_content(value=f"{float(sp.mean()):.1f}", subtitle="avg over frames")  # 调用函数/方法执行某个动作或计算
            self._m_speed_max.set_content(value=f"{float(sp.max()):.1f}", subtitle="max over frames")  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self._m_speed_avg.set_content(value="--", subtitle="")  # 调用函数/方法执行某个动作或计算
            self._m_speed_max.set_content(value="--", subtitle="")  # 调用函数/方法执行某个动作或计算

        visible_ratio = None  # 将表达式计算结果赋给变量 visible_ratio
        vis_col = "ball_denoise_visible" if "ball_denoise_visible" in df.columns else ("ball_visible" if "ball_visible" in df.columns else None)  # 将 vis_col 设为一次调用/构造的返回值
        if vis_col is not None and len(df) > 0:  # 条件分支判断并选择执行路径
            v = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)  # 将 v 设为一次调用/构造的返回值
            visible_ratio = float((v > 0).mean())  # 将 visible_ratio 设为一次调用/构造的返回值
        self._m_visible.set_content(value=f"{(visible_ratio * 100):.1f}%" if visible_ratio is not None else "--", subtitle=vis_col or "")  # 调用函数/方法执行某个动作或计算

        out_dir = outputs.get("video_result_dir", "")  # 将 out_dir 设为一次调用/构造的返回值
        self._m_output.set_content(value=Path(out_dir).name if out_dir else "--", subtitle=out_dir)  # 调用函数/方法执行某个动作或计算

        if not events_df.empty and "frame" in events_df.columns:  # 条件分支判断并选择执行路径
            frames = events_df["frame"].astype(int).tolist()  # 将 frames 设为一次调用/构造的返回值
            xs = []  # 初始化变量 xs 为一个容器/表达式结果
            if "time_seconds" in df.columns and len(df) > 0:  # 条件分支判断并选择执行路径
                ts = df["time_seconds"].astype(float).tolist()  # 将 ts 设为一次调用/构造的返回值
                for f in frames:  # 循环遍历序列/迭代器
                    if 0 <= f < len(ts):  # 条件分支判断并选择执行路径
                        xs.append(float(ts[f]))  # 调用函数/方法执行某个动作或计算
                    else:  # 条件分支的否则路径
                        xs.append(float(f))  # 调用函数/方法执行某个动作或计算
            else:  # 条件分支的否则路径
                xs = [float(f) for f in frames]  # 初始化变量 xs 为一个容器/表达式结果
            colors: List[QColor] = []  # 执行当前语句（保持与上文逻辑一致）
            if "player" in events_df.columns:  # 条件分支判断并选择执行路径
                ps = events_df["player"].astype(int).tolist()  # 将 ps 设为一次调用/构造的返回值
                for p in ps:  # 循环遍历序列/迭代器
                    colors.append(QColor("#3b82f6") if p == 0 else QColor("#ef4444"))  # 调用函数/方法执行某个动作或计算
            self._timeline.set_markers(xs, frames, colors=colors, title="击球时间轴(点击跳转)")  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self._timeline.set_markers([], [], title="击球时间轴(点击跳转)")  # 调用函数/方法执行某个动作或计算

        self._refresh_density_view()  # 调用函数/方法执行某个动作或计算
        self._update_current_ball_marker(self._output_player.current_frame())  # 调用函数/方法执行某个动作或计算

        if not events_df.empty:  # 条件分支判断并选择执行路径
            col = None  # 将表达式计算结果赋给变量 col
            for c in ["stroke_type_name", "stroke_type_name_en", "stroke_type"]:  # 循环遍历序列/迭代器
                if c in events_df.columns:  # 条件分支判断并选择执行路径
                    col = c  # 将表达式计算结果赋给变量 col
                    break  # 控制流语句：改变当前代码块的执行方式
            if col is not None:  # 条件分支判断并选择执行路径
                s = events_df[col].fillna("").astype(str)  # 将 s 设为一次调用/构造的返回值
                s = s[s != ""]  # 将表达式计算结果赋给变量 s
                vc = s.value_counts().head(10)  # 将 vc 设为一次调用/构造的返回值
                self._stroke_bar.set_data(vc.index.tolist(), vc.astype(float).tolist(), title="击球类型分布(Top10)")  # 调用函数/方法执行某个动作或计算
            else:  # 条件分支的否则路径
                self._stroke_bar.set_data([], [], title="击球类型分布")  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self._stroke_bar.set_data([], [], title="击球类型分布")  # 调用函数/方法执行某个动作或计算

    def _update_distributions(self, df: pd.DataFrame, events_df: pd.DataFrame):  # 定义函数（封装可复用逻辑）
        speed_values: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        if df is not None and not df.empty and "ball_speed" in df.columns:  # 条件分支判断并选择执行路径
            sp = pd.to_numeric(df["ball_speed"], errors="coerce").fillna(0.0).astype(float)  # 将 sp 设为一次调用/构造的返回值
            vis_col = self._visible_column(df)  # 将 vis_col 设为一次调用/构造的返回值
            if vis_col is not None:  # 条件分支判断并选择执行路径
                vis = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)  # 将 vis 设为一次调用/构造的返回值
                sp = sp[vis > 0]  # 将表达式计算结果赋给变量 sp
            speed_values = sp.tolist()  # 将 speed_values 设为一次调用/构造的返回值
        self._speed_hist.set_data(speed_values, title="球速分布(Pro)", x_label="ball_speed", color="#3b82f6")  # 调用函数/方法执行某个动作或计算

        intervals: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        fps = 30.0  # 将表达式计算结果赋给变量 fps
        if events_df is not None and not events_df.empty and "frame" in events_df.columns:  # 条件分支判断并选择执行路径
            frames = pd.to_numeric(events_df["frame"], errors="coerce").dropna().astype(int).sort_values().tolist()  # 将 frames 设为一次调用/构造的返回值
            if len(frames) >= 2:  # 条件分支判断并选择执行路径
                if df is not None and not df.empty and "time_seconds" in df.columns:  # 条件分支判断并选择执行路径
                    ts = pd.to_numeric(df["time_seconds"], errors="coerce").fillna(0.0).astype(float).tolist()  # 将 ts 设为一次调用/构造的返回值
                    try:  # 开始异常捕获保护块
                        fps = len(df) / float(df["time_seconds"].iat[-1])  # 将 fps 设为一次调用/构造的返回值
                    except ZeroDivisionError:  # 捕获异常并进行处理
                        pass  # 控制流语句：改变当前代码块的执行方式
                    for i in range(1, len(frames)):  # 循环遍历序列/迭代器
                        a = frames[i - 1]  # 将表达式计算结果赋给变量 a
                        b = frames[i]  # 将表达式计算结果赋给变量 b
                        if 0 <= a < len(ts) and 0 <= b < len(ts):  # 条件分支判断并选择执行路径
                            intervals.append(max(0.0, float(ts[b] - ts[a])))  # 调用函数/方法执行某个动作或计算
                else:  # 条件分支的否则路径
                    info = self._output_player.info()  # 将 info 设为一次调用/构造的返回值
                    fps = float(info.fps) if info is not None else 25.0  # 将 fps 设为一次调用/构造的返回值
                    for i in range(1, len(frames)):  # 循环遍历序列/迭代器
                        intervals.append(max(0.0, float(frames[i] - frames[i - 1]) / max(1.0, fps)))  # 调用函数/方法执行某个动作或计算
        self._hit_interval_hist.set_data(intervals, title="击球间隔分布(Pro)", x_label="delta_seconds", color="#22c55e")  # 调用函数/方法执行某个动作或计算

        # Hit Height
        heights: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        if events_df is not None and not events_df.empty:  # 条件分支判断并选择执行路径
            if "y" in events_df.columns:  # 条件分支判断并选择执行路径
                heights = pd.to_numeric(events_df["y"], errors="coerce").dropna().astype(float).tolist()  # 将 heights 设为一次调用/构造的返回值
            elif "frame" in events_df.columns and df is not None and not df.empty:  # 条件分支判断并选择执行路径
                frames = pd.to_numeric(events_df["frame"], errors="coerce").dropna().astype(int).tolist()  # 将 frames 设为一次调用/构造的返回值
                y_col = "ball_denoise_y" if "ball_denoise_y" in df.columns else ("ball_y" if "ball_y" in df.columns else None)  # 将 y_col 设为一次调用/构造的返回值
                if y_col:  # 条件分支判断并选择执行路径
                    y_vals = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).astype(float).tolist()  # 将 y_vals 设为一次调用/构造的返回值
                    for f in frames:  # 循环遍历序列/迭代器
                        if 0 <= f < len(y_vals):  # 条件分支判断并选择执行路径
                            heights.append(y_vals[f])  # 调用函数/方法执行某个动作或计算
        self._hit_height_hist.set_data(heights, title="击球高度分布(Pro)", x_label="hit_y_px", color="#f59e0b")  # 调用函数/方法执行某个动作或计算

        # Player Speed
        p_speeds: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        if df is not None and not df.empty:  # 条件分支判断并选择执行路径
            p1 = getattr(self, "_p1_centroids", [])  # 将 p1 设为一次调用/构造的返回值
            p2 = getattr(self, "_p2_centroids", [])  # 将 p2 设为一次调用/构造的返回值
            t = df["time_seconds"].astype(float).tolist() if "time_seconds" in df.columns else []  # 将 t 设为一次调用/构造的返回值
            
            def calc_speeds(centroids):  # 定义函数（封装可复用逻辑）
                s = []  # 初始化变量 s 为一个容器/表达式结果
                if not centroids or len(centroids) < 2 or not t:  # 条件分支判断并选择执行路径
                    return s  # 从函数返回结果
                for i in range(1, len(centroids)):  # 循环遍历序列/迭代器
                    if i >= len(t): break  # 条件分支判断并选择执行路径
                    c1, c2 = centroids[i-1], centroids[i]  # 执行当前语句（保持与上文逻辑一致）
                    dt = t[i] - t[i-1]  # 将表达式计算结果赋给变量 dt
                    if c1 and c2 and dt > 0.001:  # 条件分支判断并选择执行路径
                        dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5  # 初始化变量 dist 为一个容器/表达式结果
                        s.append(dist / dt)  # 调用函数/方法执行某个动作或计算
                return s  # 从函数返回结果

            p_speeds.extend(calc_speeds(p1))  # 调用函数/方法执行某个动作或计算
            p_speeds.extend(calc_speeds(p2))  # 调用函数/方法执行某个动作或计算
        
        self._player_speed_hist.set_data(p_speeds, title="选手瞬时速度分布(Pro)", x_label="player_speed_px_s", color="#a78bfa")  # 调用函数/方法执行某个动作或计算

    def _player_joint_columns(self, df: pd.DataFrame, player_prefix: str):  # 定义函数（封装可复用逻辑）
        xs = []  # 初始化变量 xs 为一个容器/表达式结果
        ys = []  # 初始化变量 ys 为一个容器/表达式结果
        for j in range(17):  # 循环遍历序列/迭代器
            x = f"{player_prefix}_joint{j}_x"  # 将表达式计算结果赋给变量 x
            y = f"{player_prefix}_joint{j}_y"  # 将表达式计算结果赋给变量 y
            if x in df.columns and y in df.columns:  # 条件分支判断并选择执行路径
                xs.append(x)  # 调用函数/方法执行某个动作或计算
                ys.append(y)  # 调用函数/方法执行某个动作或计算
        return xs, ys  # 从函数返回结果

    def _compute_player_centroids(self, df: pd.DataFrame, player_prefix: str) -> List[Optional[tuple[float, float]]]:  # 定义函数（封装可复用逻辑）
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            return []  # 从函数返回结果
        xs_cols, ys_cols = self._player_joint_columns(df, player_prefix)  # 调用函数/方法执行某个动作或计算
        if not xs_cols or not ys_cols:  # 条件分支判断并选择执行路径
            return [None for _ in range(len(df))]  # 从函数返回结果

        xs = df[xs_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)  # 将 xs 设为一次调用/构造的返回值
        ys = df[ys_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)  # 将 ys 设为一次调用/构造的返回值
        out: List[Optional[tuple[float, float]]] = []  # 执行当前语句（保持与上文逻辑一致）
        for i in range(len(df)):  # 循环遍历序列/迭代器
            x_row = xs.iloc[i].to_numpy()  # 将 x_row 设为一次调用/构造的返回值
            y_row = ys.iloc[i].to_numpy()  # 将 y_row 设为一次调用/构造的返回值
            mask = (x_row > 1.0) & (y_row > 1.0)  # 初始化变量 mask 为一个容器/表达式结果
            if mask.sum() < 3:  # 条件分支判断并选择执行路径
                out.append(None)  # 调用函数/方法执行某个动作或计算
                continue  # 控制流语句：改变当前代码块的执行方式
            out.append((float(x_row[mask].mean()), float(y_row[mask].mean())))  # 调用函数/方法执行某个动作或计算
        return out  # 从函数返回结果

    def _update_players(self, df: pd.DataFrame):  # 定义函数（封装可复用逻辑）
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            self._p1_map.set_points([], title="选手0覆盖(凸包)", color="#3b82f6")  # 调用函数/方法执行某个动作或计算
            self._p2_map.set_points([], title="选手1覆盖(凸包)", color="#ef4444")  # 调用函数/方法执行某个动作或计算
            self._p_dist_plot.set_series([], [], x_label="time_seconds", y_label="player_distance_px", title="选手间距")  # 调用函数/方法执行某个动作或计算
            self._p1_speed_plot.set_series([], [], x_label="time_seconds", y_label="p0_speed_px_s", title="选手0速度")  # 调用函数/方法执行某个动作或计算
            self._p2_speed_plot.set_series([], [], x_label="time_seconds", y_label="p1_speed_px_s", title="选手1速度")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        t = df["time_seconds"].astype(float).tolist() if "time_seconds" in df.columns else list(range(len(df)))  # 将 t 设为一次调用/构造的返回值
        p1 = self._compute_player_centroids(df, "player1")  # 将 p1 设为一次调用/构造的返回值
        p2 = self._compute_player_centroids(df, "player2")  # 将 p2 设为一次调用/构造的返回值
        self._p1_centroids = p1  # 给对象属性 self._p1_centroids 赋值/初始化（来自当前语句右侧表达式）
        self._p2_centroids = p2  # 给对象属性 self._p2_centroids 赋值/初始化（来自当前语句右侧表达式）

        p1_pts = [pt for pt in p1 if pt is not None]  # 初始化变量 p1_pts 为一个容器/表达式结果
        p2_pts = [pt for pt in p2 if pt is not None]  # 初始化变量 p2_pts 为一个容器/表达式结果
        self._p1_map.set_points(p1_pts, title="选手0覆盖(凸包)", color="#3b82f6")  # 调用函数/方法执行某个动作或计算
        self._p2_map.set_points(p2_pts, title="选手1覆盖(凸包)", color="#ef4444")  # 调用函数/方法执行某个动作或计算

        info = self._output_player.info()  # 将 info 设为一次调用/构造的返回值
        fps = float(info.fps) if info is not None else 25.0  # 将 fps 设为一次调用/构造的返回值

        dist_x: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        dist_y: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        for i in range(len(df)):  # 循环遍历序列/迭代器
            if i >= len(t):  # 条件分支判断并选择执行路径
                break  # 控制流语句：改变当前代码块的执行方式
            if p1[i] is None or p2[i] is None:  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式
            dx = float(p1[i][0] - p2[i][0])  # 将 dx 设为一次调用/构造的返回值
            dy = float(p1[i][1] - p2[i][1])  # 将 dy 设为一次调用/构造的返回值
            dist_x.append(float(t[i]))  # 调用函数/方法执行某个动作或计算
            dist_y.append((dx * dx + dy * dy) ** 0.5)  # 调用函数/方法执行某个动作或计算
        self._p_dist_plot.set_series(dist_x, dist_y, x_label="time_seconds", y_label="player_distance_px", title="选手间距")  # 调用函数/方法执行某个动作或计算

        s1_x: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        s1_y: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        s2_x: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        s2_y: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        for i in range(1, len(df)):  # 循环遍历序列/迭代器
            if i >= len(t):  # 条件分支判断并选择执行路径
                break  # 控制流语句：改变当前代码块的执行方式
            if p1[i - 1] is not None and p1[i] is not None:  # 条件分支判断并选择执行路径
                dx = float(p1[i][0] - p1[i - 1][0])  # 将 dx 设为一次调用/构造的返回值
                dy = float(p1[i][1] - p1[i - 1][1])  # 将 dy 设为一次调用/构造的返回值
                s1_x.append(float(t[i]))  # 调用函数/方法执行某个动作或计算
                s1_y.append(((dx * dx + dy * dy) ** 0.5) * fps)  # 调用函数/方法执行某个动作或计算
            if p2[i - 1] is not None and p2[i] is not None:  # 条件分支判断并选择执行路径
                dx = float(p2[i][0] - p2[i - 1][0])  # 将 dx 设为一次调用/构造的返回值
                dy = float(p2[i][1] - p2[i - 1][1])  # 将 dy 设为一次调用/构造的返回值
                s2_x.append(float(t[i]))  # 调用函数/方法执行某个动作或计算
                s2_y.append(((dx * dx + dy * dy) ** 0.5) * fps)  # 调用函数/方法执行某个动作或计算
        self._p1_speed_plot.set_series(s1_x, s1_y, x_label="time_seconds", y_label="p0_speed_px_s", title="选手0速度")  # 调用函数/方法执行某个动作或计算
        self._p2_speed_plot.set_series(s2_x, s2_y, x_label="time_seconds", y_label="p1_speed_px_s", title="选手1速度")  # 调用函数/方法执行某个动作或计算

        self._update_current_player_markers(self._output_player.current_frame())  # 调用函数/方法执行某个动作或计算

    def _update_current_player_markers(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        p1 = getattr(self, "_p1_centroids", None)  # 将 p1 设为一次调用/构造的返回值
        p2 = getattr(self, "_p2_centroids", None)  # 将 p2 设为一次调用/构造的返回值
        if p1 is None or p2 is None:  # 条件分支判断并选择执行路径
            self._p1_map.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            self._p2_map.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        i = int(frame_index)  # 将 i 设为一次调用/构造的返回值
        if not (0 <= i < len(p1)) or not (0 <= i < len(p2)):  # 条件分支判断并选择执行路径
            self._p1_map.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            self._p2_map.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        self._p1_map.set_current_point(p1[i] if p1[i] is not None else None)  # 调用函数/方法执行某个动作或计算
        self._p2_map.set_current_point(p2[i] if p2[i] is not None else None)  # 调用函数/方法执行某个动作或计算

    def _on_density_show_current_toggled(self, checked: bool):  # 定义函数（封装可复用逻辑）
        self._heatmap.set_show_current_point(bool(checked))  # 调用函数/方法执行某个动作或计算
        self._update_current_ball_marker(self._output_player.current_frame())  # 调用函数/方法执行某个动作或计算

    def _density_bins(self):  # 定义函数（封装可复用逻辑）
        text = self._density_bins_combo.currentText()  # 将 text 设为一次调用/构造的返回值
        if "24x14" in text:  # 条件分支判断并选择执行路径
            return (24, 14)  # 从函数返回结果
        if "72x40" in text:  # 条件分支判断并选择执行路径
            return (72, 40)  # 从函数返回结果
        return (44, 24)  # 从函数返回结果

    def _ball_xy_columns(self, df: pd.DataFrame):  # 定义函数（封装可复用逻辑）
        x_col = "ball_denoise_x" if "ball_denoise_x" in df.columns else ("ball_x" if "ball_x" in df.columns else None)  # 将 x_col 设为一次调用/构造的返回值
        y_col = "ball_denoise_y" if "ball_denoise_y" in df.columns else ("ball_y" if "ball_y" in df.columns else None)  # 将 y_col 设为一次调用/构造的返回值
        return x_col, y_col  # 从函数返回结果

    def _visible_column(self, df: pd.DataFrame):  # 定义函数（封装可复用逻辑）
        return "ball_denoise_visible" if "ball_denoise_visible" in df.columns else ("ball_visible" if "ball_visible" in df.columns else None)  # 从函数返回结果

    def _frames_for_density_mode(self) -> Optional[List[int]]:  # 定义函数（封装可复用逻辑）
        mode = self._density_source_combo.currentText()  # 将 mode 设为一次调用/构造的返回值
        df = self._current_df  # 将表达式计算结果赋给变量 df
        events_df = self._events_raw_df  # 将表达式计算结果赋给变量 events_df
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            return []  # 从函数返回结果

        if mode == "全部帧":  # 条件分支判断并选择执行路径
            return list(range(len(df)))  # 从函数返回结果

        if mode == "可见帧":  # 条件分支判断并选择执行路径
            vis_col = self._visible_column(df)  # 将 vis_col 设为一次调用/构造的返回值
            if vis_col is None:  # 条件分支判断并选择执行路径
                return list(range(len(df)))  # 从函数返回结果
            vis = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)  # 将 vis 设为一次调用/构造的返回值
            return vis.index[vis > 0].astype(int).tolist()  # 从函数返回结果

        if mode == "仅击球帧":  # 条件分支判断并选择执行路径
            if "is_hit" in df.columns:  # 条件分支判断并选择执行路径
                hit = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)  # 将 hit 设为一次调用/构造的返回值
                return hit.index[hit > 0].astype(int).tolist()  # 从函数返回结果
            if events_df is not None and not events_df.empty and "frame" in events_df.columns:  # 条件分支判断并选择执行路径
                return pd.to_numeric(events_df["frame"], errors="coerce").dropna().astype(int).tolist()  # 从函数返回结果
            return []  # 从函数返回结果

        if mode in ["仅选手0击球帧", "仅选手1击球帧"]:  # 条件分支判断并选择执行路径
            if events_df is None or events_df.empty or "frame" not in events_df.columns or "player" not in events_df.columns:  # 条件分支判断并选择执行路径
                return []  # 从函数返回结果
            want = 0 if "0" in mode else 1  # 将表达式计算结果赋给变量 want
            e = events_df.copy()  # 将 e 设为一次调用/构造的返回值
            pl = pd.to_numeric(e["player"], errors="coerce").fillna(-999).astype(int)  # 将 pl 设为一次调用/构造的返回值
            e = e[pl == want]  # 将表达式计算结果赋给变量 e
            return pd.to_numeric(e["frame"], errors="coerce").dropna().astype(int).tolist()  # 从函数返回结果

        return None  # 从函数返回结果

    def _refresh_density_view(self, checked: bool = False):  # 定义函数（封装可复用逻辑）
        df = self._current_df  # 将表达式计算结果赋给变量 df
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            self._heatmap.set_bins(self._density_bins())  # 调用函数/方法执行某个动作或计算
            self._heatmap.set_points([], title="球位置密度(气泡聚合)")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        x_col, y_col = self._ball_xy_columns(df)  # 调用函数/方法执行某个动作或计算
        if x_col is None or y_col is None:  # 条件分支判断并选择执行路径
            self._heatmap.set_bins(self._density_bins())  # 调用函数/方法执行某个动作或计算
            self._heatmap.set_points([], title="球位置密度(气泡聚合)")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        frames = self._frames_for_density_mode()  # 将 frames 设为一次调用/构造的返回值
        if frames is None:  # 条件分支判断并选择执行路径
            frames = list(range(len(df)))  # 将 frames 设为一次调用/构造的返回值

        frames = [f for f in frames if 0 <= int(f) < len(df)]  # 初始化变量 frames 为一个容器/表达式结果
        xs = pd.to_numeric(df.loc[frames, x_col], errors="coerce").fillna(0.0).astype(float)  # 将 xs 设为一次调用/构造的返回值
        ys = pd.to_numeric(df.loc[frames, y_col], errors="coerce").fillna(0.0).astype(float)  # 将 ys 设为一次调用/构造的返回值

        vis_col = self._visible_column(df)  # 将 vis_col 设为一次调用/构造的返回值
        if self._density_source_combo.currentText() != "全部帧" and vis_col is not None:  # 条件分支判断并选择执行路径
            vis = pd.to_numeric(df.loc[frames, vis_col], errors="coerce").fillna(0).astype(int)  # 将 vis 设为一次调用/构造的返回值
            mask = vis > 0  # 将表达式计算结果赋给变量 mask
            xs = xs[mask]  # 将表达式计算结果赋给变量 xs
            ys = ys[mask]  # 将表达式计算结果赋给变量 ys

        pts = list(zip(xs.tolist(), ys.tolist()))  # 将 pts 设为一次调用/构造的返回值
        title = f"球位置密度(气泡聚合) · {self._density_source_combo.currentText()}"  # 将 title 设为一次调用/构造的返回值
        self._heatmap.set_bins(self._density_bins())  # 调用函数/方法执行某个动作或计算
        self._heatmap.set_points(pts, title=title)  # 调用函数/方法执行某个动作或计算
        self._update_current_ball_marker(self._output_player.current_frame())  # 调用函数/方法执行某个动作或计算

    def _update_current_ball_marker(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        if not self._density_show_current_cb.isChecked():  # 条件分支判断并选择执行路径
            self._heatmap.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        df = self._current_df  # 将表达式计算结果赋给变量 df
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            self._heatmap.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        if not (0 <= int(frame_index) < len(df)):  # 条件分支判断并选择执行路径
            self._heatmap.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        x_col, y_col = self._ball_xy_columns(df)  # 调用函数/方法执行某个动作或计算
        if x_col is None or y_col is None:  # 条件分支判断并选择执行路径
            self._heatmap.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        x = pd.to_numeric(df[x_col].iat[int(frame_index)], errors="coerce")  # 将 x 设为一次调用/构造的返回值
        y = pd.to_numeric(df[y_col].iat[int(frame_index)], errors="coerce")  # 将 y 设为一次调用/构造的返回值
        if pd.isna(x) or pd.isna(y):  # 条件分支判断并选择执行路径
            self._heatmap.set_current_point(None)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        self._heatmap.set_current_point((float(x), float(y)))  # 调用函数/方法执行某个动作或计算

    def _export_density_png(self):  # 定义函数（封装可复用逻辑）
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT  # 将 default_dir 设为一次调用/构造的返回值
        default_dir.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
        file_path, _ = QFileDialog.getSaveFileName(self, "导出密度图", str(default_dir / "ball_density.png"), "PNG Image (*.png)")  # 调用函数/方法执行某个动作或计算
        if not file_path:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        pix = self._heatmap.grab()  # 将 pix 设为一次调用/构造的返回值
        pix.save(file_path, "PNG")  # 调用函数/方法执行某个动作或计算
        self._append_log(f"已导出密度图: {file_path}")  # 调用函数/方法执行某个动作或计算

    def _on_finished(self, ok: bool, message: str):  # 定义函数（封装可复用逻辑）
        self._run_btn.setEnabled(True)  # 调用函数/方法执行某个动作或计算
        self._stop_btn.setEnabled(False)  # 调用函数/方法执行某个动作或计算
        if self._worker_thread is not None:  # 条件分支判断并选择执行路径
            self._worker_thread.quit()  # 调用函数/方法执行某个动作或计算
            self._worker_thread.wait(2000)  # 调用函数/方法执行某个动作或计算
        self._worker_thread = None  # 给对象属性 self._worker_thread 赋值/初始化（来自当前语句右侧表达式）
        self._worker = None  # 给对象属性 self._worker 赋值/初始化（来自当前语句右侧表达式）
        self._stepper.set_finished(ok)  # 调用函数/方法执行某个动作或计算
        if ok:  # 条件分支判断并选择执行路径
            self._status_step.setText("完成")  # 调用函数/方法执行某个动作或计算
            self._append_log("分析完成")  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self._status_step.setText("停止" if message == "已停止" else "失败")  # 调用函数/方法执行某个动作或计算
            self._append_log(message)  # 调用函数/方法执行某个动作或计算

    def _on_output_position(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        self._highlight_from_frame(frame_index)  # 调用函数/方法执行某个动作或计算

    def _highlight_from_frame(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        if self._csv_model.rowCount() <= 0:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        df = self._csv_model._df  # 将表达式计算结果赋给变量 df
        if "time_seconds" not in df.columns:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if not (0 <= frame_index < len(df)):  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        t = float(df["time_seconds"].iat[frame_index])  # 将 t 设为一次调用/构造的返回值
        self._speed_plot.set_highlight_x(t)  # 调用函数/方法执行某个动作或计算
        self._ball_y_plot.set_highlight_x(t)  # 调用函数/方法执行某个动作或计算
        self._hit_count_plot.set_highlight_x(t)  # 调用函数/方法执行某个动作或计算
        self._timeline.set_selected_by_frame(int(frame_index))  # 调用函数/方法执行某个动作或计算
        self._update_current_ball_marker(int(frame_index))  # 调用函数/方法执行某个动作或计算
        self._update_current_player_markers(int(frame_index))  # 调用函数/方法执行某个动作或计算

    def _seek_output(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        self._output_player.seek(int(frame_index))  # 调用函数/方法执行某个动作或计算
        self._highlight_from_frame(int(frame_index))  # 调用函数/方法执行某个动作或计算
        self._sync_compare_from_output(int(frame_index))  # 调用函数/方法执行某个动作或计算

    def _on_event_selection_changed(self, selected: QItemSelection, deselected: QItemSelection):  # 定义函数（封装可复用逻辑）
        if selected.indexes():  # 条件分支判断并选择执行路径
            row = selected.indexes()[0].row()  # 将 row 设为一次调用/构造的返回值
            df = self._events_model._df  # 将表达式计算结果赋给变量 df
            if "frame" not in df.columns:  # 条件分支判断并选择执行路径
                return  # 从函数返回结果
            try:  # 开始异常捕获保护块
                frame = int(df["frame"].iat[row])  # 将 frame 设为一次调用/构造的返回值
            except Exception:  # 捕获异常并进行处理
                return  # 从函数返回结果
            self._seek_output(frame)  # 调用函数/方法执行某个动作或计算

    def _on_event_double_clicked(self, index):  # 定义函数（封装可复用逻辑）
        if not index.isValid():  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        df = self._events_model._df  # 将表达式计算结果赋给变量 df
        if df is None or df.empty or "frame" not in df.columns:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        try:  # 开始异常捕获保护块
            frame = int(df["frame"].iat[index.row()])  # 将 frame 设为一次调用/构造的返回值
        except Exception:  # 捕获异常并进行处理
            return  # 从函数返回结果
        self._seek_output(frame)  # 调用函数/方法执行某个动作或计算

    def _reset_event_filters(self):  # 定义函数（封装可复用逻辑）
        self._event_search.setText("")  # 调用函数/方法执行某个动作或计算
        self._event_player_filter.blockSignals(True)  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.blockSignals(True)  # 调用函数/方法执行某个动作或计算
        self._event_player_filter.setCurrentIndex(0)  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.setCurrentIndex(0)  # 调用函数/方法执行某个动作或计算
        self._event_player_filter.blockSignals(False)  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.blockSignals(False)  # 调用函数/方法执行某个动作或计算
        self._apply_event_filters()  # 调用函数/方法执行某个动作或计算

    def _rebuild_event_filters(self, events_df: pd.DataFrame):  # 定义函数（封装可复用逻辑）
        self._event_player_filter.blockSignals(True)  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.blockSignals(True)  # 调用函数/方法执行某个动作或计算
        self._event_player_filter.clear()  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.clear()  # 调用函数/方法执行某个动作或计算
        self._event_player_filter.addItem("全部")  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.addItem("全部")  # 调用函数/方法执行某个动作或计算

        if not events_df.empty:  # 条件分支判断并选择执行路径
            if "player" in events_df.columns:  # 条件分支判断并选择执行路径
                for v in sorted(set(pd.to_numeric(events_df["player"], errors="coerce").dropna().astype(int).tolist())):  # 循环遍历序列/迭代器
                    self._event_player_filter.addItem(str(v))  # 调用函数/方法执行某个动作或计算
            stroke_col = None  # 将表达式计算结果赋给变量 stroke_col
            for c in ["stroke_type_name", "stroke_type_name_en", "stroke_type"]:  # 循环遍历序列/迭代器
                if c in events_df.columns:  # 条件分支判断并选择执行路径
                    stroke_col = c  # 将表达式计算结果赋给变量 stroke_col
                    break  # 控制流语句：改变当前代码块的执行方式
            if stroke_col is not None:  # 条件分支判断并选择执行路径
                vals = events_df[stroke_col].fillna("").astype(str)  # 将 vals 设为一次调用/构造的返回值
                vals = sorted(set([s for s in vals.tolist() if s.strip()]))  # 将 vals 设为一次调用/构造的返回值
                for s in vals[:200]:  # 循环遍历序列/迭代器
                    self._event_stroke_filter.addItem(s)  # 调用函数/方法执行某个动作或计算

        self._event_player_filter.blockSignals(False)  # 调用函数/方法执行某个动作或计算
        self._event_stroke_filter.blockSignals(False)  # 调用函数/方法执行某个动作或计算

    def _apply_event_filters(self):  # 定义函数（封装可复用逻辑）
        base = getattr(self, "_events_raw_df", pd.DataFrame()).copy()  # 将 base 设为一次调用/构造的返回值
        if base.empty:  # 条件分支判断并选择执行路径
            self._events_model.set_dataframe(base)  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        player_text = self._event_player_filter.currentText().strip()  # 将 player_text 设为一次调用/构造的返回值
        if player_text and player_text != "全部" and "player" in base.columns:  # 条件分支判断并选择执行路径
            try:  # 开始异常捕获保护块
                p = int(player_text)  # 将 p 设为一次调用/构造的返回值
                base = base[pd.to_numeric(base["player"], errors="coerce").fillna(-999).astype(int) == p]  # 将 base 设为一次调用/构造的返回值
            except Exception:  # 捕获异常并进行处理
                pass  # 控制流语句：改变当前代码块的执行方式

        stroke_text = self._event_stroke_filter.currentText().strip()  # 将 stroke_text 设为一次调用/构造的返回值
        if stroke_text and stroke_text != "全部":  # 条件分支判断并选择执行路径
            stroke_col = None  # 将表达式计算结果赋给变量 stroke_col
            for c in ["stroke_type_name", "stroke_type_name_en", "stroke_type"]:  # 循环遍历序列/迭代器
                if c in base.columns:  # 条件分支判断并选择执行路径
                    stroke_col = c  # 将表达式计算结果赋给变量 stroke_col
                    break  # 控制流语句：改变当前代码块的执行方式
            if stroke_col is not None:  # 条件分支判断并选择执行路径
                base = base[base[stroke_col].fillna("").astype(str) == stroke_text]  # 将 base 设为一次调用/构造的返回值

        q = self._event_search.text().strip()  # 将 q 设为一次调用/构造的返回值
        if q:  # 条件分支判断并选择执行路径
            q_lower = q.lower()  # 将 q_lower 设为一次调用/构造的返回值
            cols = [c for c in ["frame", "player", "stroke_type_name", "stroke_type_name_en", "stroke_type"] if c in base.columns]  # 初始化变量 cols 为一个容器/表达式结果
            if not cols:  # 条件分支判断并选择执行路径
                cols = list(base.columns)[:20]  # 将 cols 设为一次调用/构造的返回值
            mask = None  # 将表达式计算结果赋给变量 mask
            for c in cols:  # 循环遍历序列/迭代器
                s = base[c].fillna("").astype(str).str.lower().str.contains(q_lower, regex=False)  # 将 s 设为一次调用/构造的返回值
                mask = s if mask is None else (mask | s)  # 将 mask 设为一次调用/构造的返回值
            if mask is not None:  # 条件分支判断并选择执行路径
                base = base[mask]  # 将表达式计算结果赋给变量 base

        self._events_model.set_dataframe(base.reset_index(drop=True))  # 调用函数/方法执行某个动作或计算

    def _on_compare_input_pos(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        if not self._compare_sync_cb.isChecked():  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if self._compare_follow_combo.currentText() != "输入驱动输出":  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._sync_compare_from_input(frame_index)  # 调用函数/方法执行某个动作或计算

    def _on_compare_output_pos(self, frame_index: int):  # 定义函数（封装可复用逻辑）
        if not self._compare_sync_cb.isChecked():  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        if self._compare_follow_combo.currentText() != "输出驱动输入":  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._sync_compare_from_output(frame_index)  # 调用函数/方法执行某个动作或计算

    def _sync_compare_from_input(self, input_frame: int):  # 定义函数（封装可复用逻辑）
        if self._compare_guard:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        in_info = self._compare_input.info()  # 将 in_info 设为一次调用/构造的返回值
        out_info = self._compare_output.info()  # 将 out_info 设为一次调用/构造的返回值
        if in_info is None or out_info is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        t = float(input_frame) / max(1.0, float(in_info.fps))  # 将 t 设为一次调用/构造的返回值
        out_frame = int(t * float(out_info.fps))  # 将 out_frame 设为一次调用/构造的返回值
        out_frame = max(0, min(out_frame, max(0, out_info.total_frames - 1)))  # 将 out_frame 设为一次调用/构造的返回值
        if abs(out_frame - self._compare_output.current_frame()) <= 1:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._compare_guard = True  # 给对象属性 self._compare_guard 赋值/初始化（来自当前语句右侧表达式）
        try:  # 开始异常捕获保护块
            self._compare_output.seek(out_frame)  # 调用函数/方法执行某个动作或计算
        finally:  # 无论是否异常都执行的收尾逻辑
            self._compare_guard = False  # 给对象属性 self._compare_guard 赋值/初始化（来自当前语句右侧表达式）

    def _sync_compare_from_output(self, output_frame: int):  # 定义函数（封装可复用逻辑）
        if self._compare_guard:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        in_info = self._compare_input.info()  # 将 in_info 设为一次调用/构造的返回值
        out_info = self._compare_output.info()  # 将 out_info 设为一次调用/构造的返回值
        if in_info is None or out_info is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        t = float(output_frame) / max(1.0, float(out_info.fps))  # 将 t 设为一次调用/构造的返回值
        in_frame = int(t * float(in_info.fps))  # 将 in_frame 设为一次调用/构造的返回值
        in_frame = max(0, min(in_frame, max(0, in_info.total_frames - 1)))  # 将 in_frame 设为一次调用/构造的返回值
        if abs(in_frame - self._compare_input.current_frame()) <= 1:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._compare_guard = True  # 给对象属性 self._compare_guard 赋值/初始化（来自当前语句右侧表达式）
        try:  # 开始异常捕获保护块
            self._compare_input.seek(in_frame)  # 调用函数/方法执行某个动作或计算
        finally:  # 无论是否异常都执行的收尾逻辑
            self._compare_guard = False  # 给对象属性 self._compare_guard 赋值/初始化（来自当前语句右侧表达式）

    def _export_overview_png(self):  # 定义函数（封装可复用逻辑）
        widget = getattr(self, "_overview_widget", None)  # 将 widget 设为一次调用/构造的返回值
        if widget is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT  # 将 default_dir 设为一次调用/构造的返回值
        default_dir.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
        file_path, _ = QFileDialog.getSaveFileName(self, "导出概览截图", str(default_dir / "overview.png"), "PNG Image (*.png)")  # 调用函数/方法执行某个动作或计算
        if not file_path:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        pix = widget.grab()  # 将 pix 设为一次调用/构造的返回值
        pix.save(file_path, "PNG")  # 调用函数/方法执行某个动作或计算
        self._append_log(f"已导出概览截图: {file_path}")  # 调用函数/方法执行某个动作或计算

    def _export_events_csv(self):  # 定义函数（封装可复用逻辑）
        df = self._events_model._df  # 将表达式计算结果赋给变量 df
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            QMessageBox.information(self, "提示", "当前没有事件数据可导出")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT  # 将 default_dir 设为一次调用/构造的返回值
        default_dir.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
        file_path, _ = QFileDialog.getSaveFileName(self, "导出事件表 CSV", str(default_dir / "events_filtered.csv"), "CSV Files (*.csv)")  # 调用函数/方法执行某个动作或计算
        if not file_path:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        df.to_csv(file_path, index=False, encoding="utf-8-sig")  # 调用函数/方法执行某个动作或计算
        self._append_log(f"已导出事件 CSV: {file_path}")  # 调用函数/方法执行某个动作或计算

    def _export_csv_csv(self):  # 定义函数（封装可复用逻辑）
        df = self._csv_model._df  # 将表达式计算结果赋给变量 df
        if df is None or df.empty:  # 条件分支判断并选择执行路径
            QMessageBox.information(self, "提示", "当前没有 CSV 数据可导出")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT  # 将 default_dir 设为一次调用/构造的返回值
        default_dir.mkdir(parents=True, exist_ok=True)  # 调用函数/方法执行某个动作或计算
        file_path, _ = QFileDialog.getSaveFileName(self, "导出 CSV 数据", str(default_dir / "track_data.csv"), "CSV Files (*.csv)")  # 调用函数/方法执行某个动作或计算
        if not file_path:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        df.to_csv(file_path, index=False, encoding="utf-8-sig")  # 调用函数/方法执行某个动作或计算
        self._append_log(f"已导出 CSV 数据: {file_path}")  # 调用函数/方法执行某个动作或计算

    def _scan_results(self) -> List[Dict[str, Any]]:  # 定义函数（封装可复用逻辑）
        text = self._result_dir_edit.text().strip()  # 将 text 设为一次调用/构造的返回值
        if not text:  # 条件分支判断并选择执行路径
            return []  # 从函数返回结果
        root = Path(text).expanduser()  # 将 root 设为一次调用/构造的返回值
        if not root.is_absolute():  # 条件分支判断并选择执行路径
            root = ROOT / root  # 将表达式计算结果赋给变量 root
        root = root.resolve(strict=False)  # 将 root 设为一次调用/构造的返回值
        if not root.exists():  # 条件分支判断并选择执行路径
            return []  # 从函数返回结果
        results: List[Dict[str, Any]] = []  # 执行当前语句（保持与上文逻辑一致）
        for sub in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):  # 循环遍历序列/迭代器
            combined = next(iter(sorted(sub.glob("*_combined.mp4"))), None)  # 将 combined 设为一次调用/构造的返回值
            csv = next(iter(sorted(sub.glob("*_data.csv"))), None)  # 将 csv 设为一次调用/构造的返回值
            hit = next(iter(sorted(sub.glob("*_hit_events.json"))), None)  # 将 hit 设为一次调用/构造的返回值
            stroke = next(iter(sorted(sub.glob("*_stroke_types.json"))), None)  # 将 stroke 设为一次调用/构造的返回值
            if combined is None and csv is None and hit is None:  # 条件分支判断并选择执行路径
                continue  # 控制流语句：改变当前代码块的执行方式
            results.append(  # 执行当前语句（保持与上文逻辑一致）
                {  # 执行当前语句（保持与上文逻辑一致）
                    "video_result_dir": str(sub),  # 执行当前语句（保持与上文逻辑一致）
                    "combined_video_path": str(combined) if combined else "",  # 执行当前语句（保持与上文逻辑一致）
                    "csv_path": str(csv) if csv else "",  # 执行当前语句（保持与上文逻辑一致）
                    "hit_events_path": str(hit) if hit else "",  # 执行当前语句（保持与上文逻辑一致）
                    "stroke_results_path": str(stroke) if stroke else "",  # 执行当前语句（保持与上文逻辑一致）
                }  # 执行当前语句（保持与上文逻辑一致）
            )  # 执行当前语句（保持与上文逻辑一致）
        return results  # 从函数返回结果

    def _refresh_results_list(self, checked: bool = False, select_dir: Optional[str] = None):  # 定义函数（封装可复用逻辑）
        current_dir = select_dir  # 将表达式计算结果赋给变量 current_dir
        if current_dir is None:  # 条件分支判断并选择执行路径
            cur_data = self._result_combo.currentData()  # 将 cur_data 设为一次调用/构造的返回值
            if isinstance(cur_data, dict):  # 条件分支判断并选择执行路径
                current_dir = cur_data.get("video_result_dir")  # 将 current_dir 设为一次调用/构造的返回值

        items = self._scan_results()  # 将 items 设为一次调用/构造的返回值
        self._result_combo.blockSignals(True)  # 调用函数/方法执行某个动作或计算
        self._result_combo.clear()  # 调用函数/方法执行某个动作或计算
        select_index = -1  # 将表达式计算结果赋给变量 select_index
        for idx, item in enumerate(items):  # 循环遍历序列/迭代器
            name = Path(item["video_result_dir"]).name  # 将 name 设为一次调用/构造的返回值
            self._result_combo.addItem(name, item)  # 调用函数/方法执行某个动作或计算
            if current_dir and Path(item["video_result_dir"]) == Path(current_dir):  # 条件分支判断并选择执行路径
                select_index = idx  # 将表达式计算结果赋给变量 select_index
        if select_index >= 0:  # 条件分支判断并选择执行路径
            self._result_combo.setCurrentIndex(select_index)  # 调用函数/方法执行某个动作或计算
        self._result_combo.blockSignals(False)  # 调用函数/方法执行某个动作或计算

    def _load_selected_result(self, checked: bool = False):  # 定义函数（封装可复用逻辑）
        data = self._result_combo.currentData()  # 将 data 设为一次调用/构造的返回值
        if not isinstance(data, dict):  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._current_outputs = data  # 给对象属性 self._current_outputs 赋值/初始化（来自当前语句右侧表达式）
        self._load_outputs(data)  # 调用函数/方法执行某个动作或计算

    def _get_black_frame(self):  # 定义函数（封装可复用逻辑）
        import numpy as np  # 导入模块，供后续使用
        return np.zeros((360, 640, 3), dtype=np.uint8)  # 从函数返回结果

    def _open_match_review(self):  # 定义函数（封装可复用逻辑）
        if not self._review_window:  # 条件分支判断并选择执行路径
            self._review_window = MatchReviewWindow(self)  # 给对象属性 self._review_window 赋值/初始化（来自当前语句右侧表达式）
        self._review_window.show()  # 调用函数/方法执行某个动作或计算

    def _show_about(self):  # 定义函数（封装可复用逻辑）
        about_text = """
        <h2>TrackNetV3_Attention</h2>
        <p><b>版本:</b> 1.1</p>
        <p><b>定位:</b> 端到端的羽毛球视频智能分析与专业复盘系统</p>
        <hr>
        <h3>系统概览</h3>
        <p>本系统面向教练与运动员，提供从视频输入到数据洞察与战术复盘的完整流程：目标检测、姿态估计、事件识别、击球类型识别、结果合成与多维可视化。</p>
        <h3>核心功能</h3>
        <ul>
            <li><b>检测与识别:</b> TrackNetV3 羽毛球检测、MMPose 姿态检测、击球事件检测、BST 击球类型识别、球场/球网检测与透视映射</li>
            <li><b>结果管理:</b> 历史结果自动索引与加载，支持数据集下拉选择与一键刷新（results/目录）</li>
            <li><b>专业复盘:</b> 
                <ul>
                    <li>概览：六维能力雷达 + 三维战术图（滚轮缩放、按球速/类型着色切换、连线/包络开关、透明背景）</li>
                    <li>球员表现：KPI 指标（均速/95%分位/峰值等）、速度直方图+KDE、加速度时间序列（冲刺标记）、覆盖分位等值线、站位质心与稳定性椭圆</li>
                    <li>技术统计：双侧击球类型分布与速度–高度散点</li>
                    <li>深度战术：战术类型流图（ThemeRiver，时间维度的击球战术结构）、转移热力图、空间控制(Voronoi)</li>
                    <li>体能负荷：累计距离与三维击球分析</li>
                </ul>
            </li>
            <li><b>交互增强:</b> 悬停放大预览（所有图表）、复盘窗口右上角数据集刷新保留当前选择</li>
            <li><b>导出:</b> CSV / 截图 / 可视化视频</li>
        </ul>
        <h3>项目结构 (e:\\learn\\TrackNetV3_Attention)</h3>
        <ul>
            <li><b>core/</b> 数据管线与算法模块（ball_detect、pose_detect、event_detect、stroke_classify、visualize_combined、export_to_csv 等）</li>
            <li><b>models/</b> 模型权重（TrackNet、BST、球场/球网）</li>
            <li><b>videos/</b> 示例输入视频（test2.mp4、test6.mp4）</li>
            <li><b>results/</b> 输出目录（每次分析生成 <i>视频名/</i> 子目录，含 *_data.csv、*_hit_events.json、*_stroke_types.json、*_poses.npy、*_combined.mp4、loca_info/ 等）</li>
            <li><b>ui_pyside6/</b> 图形界面与复盘实现（main.py、match_review/*）</li>
            <li><b>run_combined.py</b> 可视化合成运行脚本</li>
            <li><b>docs/README.md</b> 项目文档</li>
        </ul>
        <h3>数据规范</h3>
        <ul>
            <li><b>CSV(*_data.csv):</b> 主要字段包括 time_seconds、ball_x/ball_y/ball_speed、p1_speed/p2_speed、player关节坐标等</li>
            <li><b>事件JSON(*_hit_events.json):</b> 每次击球的帧索引与选手</li>
            <li><b>类型JSON(*_stroke_types.json):</b> 每次击球的类型与帧索引</li>
            <li><b>姿态(*_poses.npy):</b> 姿态关键点数组</li>
            <li><b>视频(*_combined.mp4):</b> 合成可视化输出</li>
        </ul>
        <h3>技术栈</h3>
        <p>PySide6 + PyTorch + MMPose + Matplotlib/Seaborn</p>
        """
        QMessageBox.about(self, "关于", about_text)  # 调用函数/方法执行某个动作或计算

    def _show_usage(self):  # 定义函数（封装可复用逻辑）
        usage_text = """
        <h2>使用说明</h2>
        <h3>1. 准备与参数</h3>
        <ul>
            <li>将待分析视频放置到 <b>videos/</b> 或任意路径</li>
            <li>在“参数”页选择 <b>设备</b>(cuda/cpu)、<b>姿态模型</b>(rtmpose-t/s/m/l)、是否启用<b>球场/球网检测</b></li>
            <li>设置 <b>TrackNet 权重</b>、<b>输入帧数</b>(1–9)、<b>检测阈值</b>(0–1)、<b>轨迹长度</b>、<b>检测/预览间隔</b> 与 <b>输出目录</b></li>
        </ul>
        <h3>2. 执行分析</h3>
        <p>点击“开始训练分析”，系统自动执行：球场/球网检测 → 羽毛球检测 → 姿态检测 → 击球事件 → 击球类型 → 合成可视化 → 数据导出。过程中可随时点击“停止”。</p>
        <h3>3. 复盘窗口（比赛复盘数据）</h3>
        <ul>
            <li>右上角<b>数据集下拉</b>列出 <b>results/</b> 下历史结果；旁侧<b>刷新</b>用于新增数据后快速更新列表（保留当前选择）</li>
            <li><b>概览 (Dashboard):</b> 六维雷达 + 三维战术图。三维图支持滚轮缩放、按球速/类型着色切换、开/关包络与连线、透明背景；悬停可放大预览</li>
            <li><b>战术复盘 (Tactical):</b> 回合列表选择 → 三维球路沙盘与详情信息（时长、拍数、选手跑动与均速等）</li>
            <li><b>球员表现 (Physical):</b> 
                <ul>
                    <li>KPI：平均速度、95%分位、最大速度、加速度峰值/均值、前/后场占比、左/右占比、近网攻势指数</li>
                    <li>速度分布：直方图+KDE；加速度时间序列：高加速度点标注；覆盖分位等值线：50/80/95%；站位质心与稳定性：质心+椭圆</li>
                </ul>
            </li>
            <li><b>技术统计 (Technical):</b> 双侧击球类型饼图、速度–高度散点</li>
            <li><b>深度战术 (Deep Tactics):</b> 战术类型流图（ThemeRiver，展示各类型随时间的占比趋势）、战术转移热力图、空间控制 (Voronoi)</li>
            <li><b>体能负荷 (Load):</b> 累计距离曲线与三维击球分析</li>
        </ul>
        <h3>4. 输出目录结构</h3>
        <ul>
            <li>路径：<b>e:\\learn\\TrackNetV3_Attention\\results\\&lt;视频名&gt;</b></li>
            <li><b>*_data.csv:</b> 时间戳、球坐标/球速、选手速度与关键点等帧级数据</li>
            <li><b>*_hit_events.json:</b> 每次击球的帧与选手</li>
            <li><b>*_stroke_types.json:</b> 每次击球的类型与帧</li>
            <li><b>*_poses.npy:</b> 姿态关键点数组</li>
            <li><b>*_combined.mp4:</b> 合成可视化视频</li>
            <li><b>loca_info/、loca_info_denoise/:</b> 球场位置信息与去噪版本</li>
        </ul>
        <h3>5. 模型与示例</h3>
        <ul>
            <li><b>models/:</b> TrackNet/球场/球网/BST 权重</li>
            <li><b>videos/:</b> 示例 test2.mp4、test6.mp4</li>
            <li><b>run_combined.py:</b> 合成可视化运行脚本</li>
        </ul>
        <h3>6. 常见问题</h3>
        <ul>
            <li>复盘列表未出现新数据：点击复盘窗口右上角“刷新”按钮</li>
            <li>三维图过小或偏移：使用滚轮缩放与视角按钮，或点击“重置缩放”</li>
            <li>中文显示异常：确保系统已安装中文字体（微软雅黑/黑体）</li>
        </ul>
        """
        dlg = QDialog(self)  # 将 dlg 设为一次调用/构造的返回值
        dlg.setWindowTitle("使用说明")  # 调用函数/方法执行某个动作或计算
        dlg.resize(1050, 780)  # 调用函数/方法执行某个动作或计算
        layout = QVBoxLayout(dlg)  # 将 layout 设为一次调用/构造的返回值
        layout.setContentsMargins(14, 14, 14, 14)  # 调用函数/方法执行某个动作或计算
        layout.setSpacing(10)  # 调用函数/方法执行某个动作或计算

        view = QTextBrowser(dlg)  # 将 view 设为一次调用/构造的返回值
        view.setOpenExternalLinks(True)  # 调用函数/方法执行某个动作或计算
        view.setHtml(usage_text)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(view, 1)  # 调用函数/方法执行某个动作或计算

        btns = QHBoxLayout()  # 将 btns 设为一次调用/构造的返回值
        btns.addStretch(1)  # 调用函数/方法执行某个动作或计算
        close_btn = QPushButton("关闭", dlg)  # 将 close_btn 设为一次调用/构造的返回值
        close_btn.clicked.connect(dlg.close)  # 调用函数/方法执行某个动作或计算
        btns.addWidget(close_btn)  # 调用函数/方法执行某个动作或计算
        layout.addLayout(btns)  # 调用函数/方法执行某个动作或计算

        dlg.exec()  # 调用函数/方法执行某个动作或计算


def main():  # 定义函数（封装可复用逻辑）
    app = QApplication(sys.argv)  # 将 app 设为一次调用/构造的返回值
    _apply_style(app)  # 调用函数/方法执行某个动作或计算
    w = MainWindow()  # 将 w 设为一次调用/构造的返回值
    w.show()  # 调用函数/方法执行某个动作或计算
    QTimer.singleShot(0, w._ensure_on_screen)  # 调用函数/方法执行某个动作或计算
    sys.exit(app.exec())  # 调用函数/方法执行某个动作或计算


if __name__ == "__main__":  # 条件分支判断并选择执行路径
    main()  # 调用函数/方法执行某个动作或计算
