from __future__ import annotations  # 从模块导入符号，供后续调用

from math import ceil, sqrt, pi, exp  # 从模块导入符号，供后续调用
from typing import List, Optional, Sequence, Tuple  # 从模块导入符号，供后续调用

import numpy as np  # 导入模块，供后续使用
from PySide6.QtCore import Qt, Signal, QPointF  # 从模块导入符号，供后续调用
from PySide6.QtWidgets import QToolTip, QWidget  # 从模块导入符号，供后续调用
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QBrush, QPainterPath, QPolygonF  # 从模块导入符号，供后续调用

try:  # 开始异常捕获保护块
    from scipy.stats import gaussian_kde  # 从模块导入符号，供后续调用
    from scipy.spatial import ConvexHull  # 从模块导入符号，供后续调用
    SCIPY_AVAILABLE = True  # 将表达式计算结果赋给变量 SCIPY_AVAILABLE
except ImportError:  # 捕获异常并进行处理
    SCIPY_AVAILABLE = False  # 将表达式计算结果赋给变量 SCIPY_AVAILABLE


class SimpleLinePlot(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._xs: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        self._ys: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        self._x_label = "x"  # 给对象属性 self._x_label 赋值/初始化（来自当前语句右侧表达式）
        self._y_label = "y"  # 给对象属性 self._y_label 赋值/初始化（来自当前语句右侧表达式）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._highlight_x: Optional[float] = None  # 执行当前语句（保持与上文逻辑一致）
        self.setMinimumHeight(160)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_series(self, xs: List[float], ys: List[float], *, x_label: str = "x", y_label: str = "y", title: str = ""):  # 定义函数（封装可复用逻辑）
        self._xs = xs  # 给对象属性 self._xs 赋值/初始化（来自当前语句右侧表达式）
        self._ys = ys  # 给对象属性 self._ys 赋值/初始化（来自当前语句右侧表达式）
        self._x_label = x_label  # 给对象属性 self._x_label 赋值/初始化（来自当前语句右侧表达式）
        self._y_label = y_label  # 给对象属性 self._y_label 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_highlight_x(self, x: Optional[float]):  # 定义函数（封装可复用逻辑）
        self._highlight_x = x  # 给对象属性 self._highlight_x 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算

        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        # No background fill

        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin - 18)  # 将 plot 设为一次调用/构造的返回值

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if len(self._xs) < 2 or len(self._ys) < 2:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        x_min = min(self._xs)  # 将 x_min 设为一次调用/构造的返回值
        x_max = max(self._xs)  # 将 x_max 设为一次调用/构造的返回值
        y_min = min(self._ys)  # 将 y_min 设为一次调用/构造的返回值
        y_max = max(self._ys)  # 将 y_max 设为一次调用/构造的返回值
        if x_max <= x_min:  # 条件分支判断并选择执行路径
            x_max = x_min + 1.0  # 将表达式计算结果赋给变量 x_max
        if y_max <= y_min:  # 条件分支判断并选择执行路径
            y_max = y_min + 1.0  # 将表达式计算结果赋给变量 y_max

        p.setPen(QPen(QColor("#2a2f3a"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRect(plot)  # 调用函数/方法执行某个动作或计算

        def to_px(x: float, y: float) -> Tuple[float, float]:  # 定义函数（封装可复用逻辑）
            px = plot.left() + (x - x_min) / (x_max - x_min) * plot.width()  # 将 px 设为一次调用/构造的返回值
            py = plot.bottom() - (y - y_min) / (y_max - y_min) * plot.height()  # 将 py 设为一次调用/构造的返回值
            return px, py  # 从函数返回结果

        p.setPen(QPen(QColor("#22c55e"), 2))  # 调用函数/方法执行某个动作或计算
        last = to_px(self._xs[0], self._ys[0])  # 将 last 设为一次调用/构造的返回值
        for i in range(1, min(len(self._xs), len(self._ys))):  # 循环遍历序列/迭代器
            cur = to_px(self._xs[i], self._ys[i])  # 将 cur 设为一次调用/构造的返回值
            p.drawLine(int(last[0]), int(last[1]), int(cur[0]), int(cur[1]))  # 调用函数/方法执行某个动作或计算
            last = cur  # 将表达式计算结果赋给变量 last

        if self._highlight_x is not None:  # 条件分支判断并选择执行路径
            hx = max(x_min, min(x_max, self._highlight_x))  # 将 hx 设为一次调用/构造的返回值
            px, _ = to_px(hx, y_min)  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#f59e0b"), 1))  # 调用函数/方法执行某个动作或计算
            p.drawLine(int(px), plot.top(), int(px), plot.bottom())  # 调用函数/方法执行某个动作或计算

        p.setPen(QPen(QColor("#9ca3af")))  # 调用函数/方法执行某个动作或计算
        p.drawText(rect.adjusted(margin, rect.height() - 18, -margin, -2), Qt.AlignLeft | Qt.AlignVCenter, self._x_label)  # 调用函数/方法执行某个动作或计算
        p.drawText(rect.adjusted(margin, rect.height() - 18, -margin, -2), Qt.AlignRight | Qt.AlignVCenter, self._y_label)  # 调用函数/方法执行某个动作或计算


class MetricCard(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, title: str = "", parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._value = "--"  # 给对象属性 self._value 赋值/初始化（来自当前语句右侧表达式）
        self._subtitle = ""  # 给对象属性 self._subtitle 赋值/初始化（来自当前语句右侧表达式）
        self._accent = QColor("#3b82f6")  # 给对象属性 self._accent 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(74)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_content(self, *, title: Optional[str] = None, value: Optional[str] = None, subtitle: Optional[str] = None):  # 定义函数（封装可复用逻辑）
        if title is not None:  # 条件分支判断并选择执行路径
            self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        if value is not None:  # 条件分支判断并选择执行路径
            self._value = value  # 给对象属性 self._value 赋值/初始化（来自当前语句右侧表达式）
        if subtitle is not None:  # 条件分支判断并选择执行路径
            self._subtitle = subtitle  # 给对象属性 self._subtitle 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_accent(self, color: QColor):  # 定义函数（封装可复用逻辑）
        self._accent = color  # 给对象属性 self._accent 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        r = self.rect().adjusted(0, 0, -1, -1)  # 将 r 设为一次调用/构造的返回值
        # No background fill
        
        # Draw translucent background for readability
        p.setBrush(QColor(15, 18, 22, 180))  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#1f2937"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRoundedRect(r, 12, 12)  # 调用函数/方法执行某个动作或计算

        p.setPen(QPen(self._accent, 4))  # 调用函数/方法执行某个动作或计算
        p.drawLine(r.left() + 12, r.top() + 14, r.left() + 12, r.bottom() - 14)  # 调用函数/方法执行某个动作或计算

        title_font = QFont("Segoe UI", 9)  # 将 title_font 设为一次调用/构造的返回值
        value_font = QFont("Segoe UI", 16, QFont.Weight.DemiBold)  # 将 value_font 设为一次调用/构造的返回值
        sub_font = QFont("Segoe UI", 9)  # 将 sub_font 设为一次调用/构造的返回值

        p.setFont(title_font)  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
        p.drawText(r.adjusted(22, 10, -10, -10), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        p.setFont(value_font)  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#e5e7eb")))  # 调用函数/方法执行某个动作或计算
        p.drawText(r.adjusted(22, 22, -10, -22), Qt.AlignLeft | Qt.AlignVCenter, self._value)  # 调用函数/方法执行某个动作或计算

        p.setFont(sub_font)  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#9ca3af")))  # 调用函数/方法执行某个动作或计算
        p.drawText(r.adjusted(22, 0, -10, 10), Qt.AlignLeft | Qt.AlignBottom, self._subtitle)  # 调用函数/方法执行某个动作或计算


class SimpleBarChart(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._labels: List[str] = []  # 执行当前语句（保持与上文逻辑一致）
        self._values: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(180)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_data(self, labels: Sequence[str], values: Sequence[float], *, title: str = ""):  # 定义函数（封装可复用逻辑）
        self._labels = list(labels)  # 给对象属性 self._labels 赋值/初始化（来自当前语句右侧表达式）
        self._values = [float(v) for v in values]  # 给对象属性 self._values 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        # No background fill

        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin - 18)  # 将 plot 设为一次调用/构造的返回值

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if not self._labels or not self._values:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        n = min(len(self._labels), len(self._values))  # 将 n 设为一次调用/构造的返回值
        max_v = max(self._values[:n]) if n > 0 else 1.0  # 将 max_v 设为一次调用/构造的返回值
        if max_v <= 0:  # 条件分支判断并选择执行路径
            max_v = 1.0  # 将表达式计算结果赋给变量 max_v

        p.setPen(QPen(QColor("#1f2937"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)  # 调用函数/方法执行某个动作或计算

        gap = 8  # 将表达式计算结果赋给变量 gap
        bar_w = max(8, int((plot.width() - gap * (n + 1)) / max(1, n)))  # 将 bar_w 设为一次调用/构造的返回值
        x = plot.left() + gap  # 将 x 设为一次调用/构造的返回值

        p.setFont(QFont("Segoe UI", 8))  # 调用函数/方法执行某个动作或计算
        for i in range(n):  # 循环遍历序列/迭代器
            v = max(0.0, float(self._values[i]))  # 将 v 设为一次调用/构造的返回值
            h = int((v / max_v) * max(1, plot.height() - 18))  # 将 h 设为一次调用/构造的返回值
            bar = (x, plot.bottom() - 18 - h, bar_w, h)  # 初始化变量 bar 为一个容器/表达式结果
            p.fillRect(*bar, QColor("#22c55e") if i % 2 == 0 else QColor("#3b82f6"))  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#111827"), 1))  # 调用函数/方法执行某个动作或计算
            p.drawRect(*bar)  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#9ca3af")))  # 调用函数/方法执行某个动作或计算
            label = self._labels[i]  # 将表达式计算结果赋给变量 label
            p.drawText(x - 4, plot.bottom() - 16, bar_w + 8, 16, Qt.AlignHCenter | Qt.AlignVCenter, label[:6])  # 调用函数/方法执行某个动作或计算
            x += bar_w + gap  # 执行当前语句（保持与上文逻辑一致）


class TimelineMarkers(QWidget):  # 定义类（封装数据与行为）
    markerActivated = Signal(int)  # 将 markerActivated 设为一次调用/构造的返回值

    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._xs: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        self._frames: List[int] = []  # 执行当前语句（保持与上文逻辑一致）
        self._colors: List[QColor] = []  # 执行当前语句（保持与上文逻辑一致）
        self._selected = -1  # 给对象属性 self._selected 赋值/初始化（来自当前语句右侧表达式）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(86)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_markers(self, xs: Sequence[float], frames: Sequence[int], *, colors: Optional[Sequence[QColor]] = None, title: str = ""):  # 定义函数（封装可复用逻辑）
        self._xs = [float(x) for x in xs]  # 给对象属性 self._xs 赋值/初始化（来自当前语句右侧表达式）
        self._frames = [int(f) for f in frames]  # 给对象属性 self._frames 赋值/初始化（来自当前语句右侧表达式）
        self._colors = list(colors) if colors is not None else []  # 给对象属性 self._colors 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._selected = -1  # 给对象属性 self._selected 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_selected_by_frame(self, frame: int):  # 定义函数（封装可复用逻辑）
        if not self._frames:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        try:  # 开始异常捕获保护块
            idx = self._frames.index(int(frame))  # 将 idx 设为一次调用/构造的返回值
        except ValueError:  # 捕获异常并进行处理
            return  # 从函数返回结果
        self._selected = idx  # 给对象属性 self._selected 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def mousePressEvent(self, event):  # 定义函数（封装可复用逻辑）
        if not self._xs or not self._frames:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        rect = self.rect().adjusted(12, 24, -12, -18)  # 将 rect 设为一次调用/构造的返回值
        if rect.width() <= 0:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        x_min = min(self._xs)  # 将 x_min 设为一次调用/构造的返回值
        x_max = max(self._xs)  # 将 x_max 设为一次调用/构造的返回值
        if x_max <= x_min:  # 条件分支判断并选择执行路径
            x_max = x_min + 1.0  # 将表达式计算结果赋给变量 x_max
        px = float(event.position().x())  # 将 px 设为一次调用/构造的返回值
        best = -1  # 将表达式计算结果赋给变量 best
        best_dist = 1e18  # 将表达式计算结果赋给变量 best_dist
        for i, x in enumerate(self._xs):  # 循环遍历序列/迭代器
            t = (x - x_min) / (x_max - x_min)  # 初始化变量 t 为一个容器/表达式结果
            mx = rect.left() + t * rect.width()  # 将 mx 设为一次调用/构造的返回值
            d = abs(mx - px)  # 将 d 设为一次调用/构造的返回值
            if d < best_dist:  # 条件分支判断并选择执行路径
                best_dist = d  # 将表达式计算结果赋给变量 best_dist
                best = i  # 将表达式计算结果赋给变量 best
        if best >= 0:  # 条件分支判断并选择执行路径
            self._selected = best  # 给对象属性 self._selected 赋值/初始化（来自当前语句右侧表达式）
            self.markerActivated.emit(int(self._frames[best]))  # 调用函数/方法执行某个动作或计算
            self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        # No background fill

        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin - 8)  # 将 plot 设为一次调用/构造的返回值

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if not self._xs or not self._frames:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无事件")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        x_min = min(self._xs)  # 将 x_min 设为一次调用/构造的返回值
        x_max = max(self._xs)  # 将 x_max 设为一次调用/构造的返回值
        if x_max <= x_min:  # 条件分支判断并选择执行路径
            x_max = x_min + 1.0  # 将表达式计算结果赋给变量 x_max

        y = (plot.top() + plot.bottom()) / 2  # 初始化变量 y 为一个容器/表达式结果
        p.setPen(QPen(QColor("#334155"), 2))  # 调用函数/方法执行某个动作或计算
        p.drawLine(plot.left(), int(y), plot.right(), int(y))  # 调用函数/方法执行某个动作或计算

        for i, x in enumerate(self._xs):  # 循环遍历序列/迭代器
            t = (x - x_min) / (x_max - x_min)  # 初始化变量 t 为一个容器/表达式结果
            mx = plot.left() + t * plot.width()  # 将 mx 设为一次调用/构造的返回值
            color = self._colors[i] if i < len(self._colors) else QColor("#f59e0b")  # 将 color 设为一次调用/构造的返回值
            r = 6  # 将表达式计算结果赋给变量 r
            if i == self._selected:  # 条件分支判断并选择执行路径
                r = 9  # 将表达式计算结果赋给变量 r
            p.setBrush(color)  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#0b0f14"), 2))  # 调用函数/方法执行某个动作或计算
            p.drawEllipse(int(mx - r), int(y - r), int(r * 2), int(r * 2))  # 调用函数/方法执行某个动作或计算


class SimpleHeatmap(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._points: List[Tuple[float, float]] = []  # 执行当前语句（保持与上文逻辑一致）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._bins = (72, 40)  # 给对象属性 self._bins 赋值/初始化（来自当前语句右侧表达式）
        self._image: Optional[QImage] = None  # 执行当前语句（保持与上文逻辑一致）
        self._w = 0  # 给对象属性 self._w 赋值/初始化（来自当前语句右侧表达式）
        self._h = 0  # 给对象属性 self._h 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(220)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_points(self, points: Sequence[Tuple[float, float]], *, title: str = "", canvas_size: Optional[Tuple[int, int]] = None):  # 定义函数（封装可复用逻辑）
        self._points = [(float(x), float(y)) for x, y in points]  # 给对象属性 self._points 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        if canvas_size is not None:  # 条件分支判断并选择执行路径
            self._w, self._h = int(canvas_size[0]), int(canvas_size[1])  # 调用函数/方法执行某个动作或计算
        self._rebuild_image()  # 调用函数/方法执行某个动作或计算
        self.update()  # 调用函数/方法执行某个动作或计算

    def _rebuild_image(self):  # 定义函数（封装可复用逻辑）
        if not self._points:  # 条件分支判断并选择执行路径
            self._image = None  # 给对象属性 self._image 赋值/初始化（来自当前语句右侧表达式）
            return  # 从函数返回结果
        xs = [p[0] for p in self._points]  # 初始化变量 xs 为一个容器/表达式结果
        ys = [p[1] for p in self._points]  # 初始化变量 ys 为一个容器/表达式结果
        x_min, x_max = min(xs), max(xs)  # 调用函数/方法执行某个动作或计算
        y_min, y_max = min(ys), max(ys)  # 调用函数/方法执行某个动作或计算
        if x_max <= x_min:  # 条件分支判断并选择执行路径
            x_max = x_min + 1.0  # 将表达式计算结果赋给变量 x_max
        if y_max <= y_min:  # 条件分支判断并选择执行路径
            y_max = y_min + 1.0  # 将表达式计算结果赋给变量 y_max

        bw, bh = self._bins  # 执行当前语句（保持与上文逻辑一致）
        grid = [[0 for _ in range(bw)] for _ in range(bh)]  # 初始化变量 grid 为一个容器/表达式结果
        for x, y in self._points:  # 循环遍历序列/迭代器
            gx = int((x - x_min) / (x_max - x_min) * (bw - 1))  # 将 gx 设为一次调用/构造的返回值
            gy = int((y - y_min) / (y_max - y_min) * (bh - 1))  # 将 gy 设为一次调用/构造的返回值
            gx = max(0, min(bw - 1, gx))  # 将 gx 设为一次调用/构造的返回值
            gy = max(0, min(bh - 1, gy))  # 将 gy 设为一次调用/构造的返回值
            grid[gy][gx] += 1  # 执行当前语句（保持与上文逻辑一致）

        max_c = max(max(row) for row in grid) if grid else 1  # 将 max_c 设为一次调用/构造的返回值
        if max_c <= 0:  # 条件分支判断并选择执行路径
            max_c = 1  # 将表达式计算结果赋给变量 max_c

        img = QImage(bw, bh, QImage.Format_RGB32)  # 将 img 设为一次调用/构造的返回值
        for y in range(bh):  # 循环遍历序列/迭代器
            for x in range(bw):  # 循环遍历序列/迭代器
                v = grid[y][x] / max_c  # 将表达式计算结果赋给变量 v
                r = int(255 * v)  # 将 r 设为一次调用/构造的返回值
                g = int(80 + 80 * (1.0 - v))  # 将 g 设为一次调用/构造的返回值
                b = int(40 + 140 * (1.0 - v))  # 将 b 设为一次调用/构造的返回值
                img.setPixel(x, y, QColor(r, g, b).rgb())  # 调用函数/方法执行某个动作或计算
        self._image = img  # 给对象属性 self._image 赋值/初始化（来自当前语句右侧表达式）

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        # No background fill

        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)  # 将 plot 设为一次调用/构造的返回值

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if self._image is None:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        target = plot  # 将表达式计算结果赋给变量 target
        p.drawImage(target, self._image.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#1f2937"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)  # 调用函数/方法执行某个动作或计算


class DensityBubbleMap(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._points: List[Tuple[float, float]] = []  # 执行当前语句（保持与上文逻辑一致）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._bins = (44, 24)  # 给对象属性 self._bins 赋值/初始化（来自当前语句右侧表达式）
        self._grid: List[List[int]] = []  # 执行当前语句（保持与上文逻辑一致）
        self._bounds: Optional[Tuple[float, float, float, float]] = None  # 执行当前语句（保持与上文逻辑一致）
        self._current_point: Optional[Tuple[float, float]] = None  # 执行当前语句（保持与上文逻辑一致）
        self._show_current_point = True  # 给对象属性 self._show_current_point 赋值/初始化（来自当前语句右侧表达式）
        self._last_hover_cell: Optional[Tuple[int, int]] = None  # 执行当前语句（保持与上文逻辑一致）
        self.setMinimumHeight(220)  # 调用函数/方法执行某个动作或计算
        self.setMouseTracking(True)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_points(self, points: Sequence[Tuple[float, float]], *, title: str = ""):  # 定义函数（封装可复用逻辑）
        self._points = [(float(x), float(y)) for x, y in points]  # 给对象属性 self._points 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._rebuild_bins()  # 调用函数/方法执行某个动作或计算
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_bins(self, bins: Tuple[int, int]):  # 定义函数（封装可复用逻辑）
        bw, bh = int(bins[0]), int(bins[1])  # 调用函数/方法执行某个动作或计算
        bw = max(8, min(160, bw))  # 将 bw 设为一次调用/构造的返回值
        bh = max(6, min(120, bh))  # 将 bh 设为一次调用/构造的返回值
        self._bins = (bw, bh)  # 给对象属性 self._bins 赋值/初始化（来自当前语句右侧表达式）
        self._rebuild_bins()  # 调用函数/方法执行某个动作或计算
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_current_point(self, point: Optional[Tuple[float, float]]):  # 定义函数（封装可复用逻辑）
        self._current_point = (float(point[0]), float(point[1])) if point is not None else None  # 给对象属性 self._current_point 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_show_current_point(self, show: bool):  # 定义函数（封装可复用逻辑）
        self._show_current_point = bool(show)  # 给对象属性 self._show_current_point 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def _rebuild_bins(self):  # 定义函数（封装可复用逻辑）
        if not self._points:  # 条件分支判断并选择执行路径
            self._grid = []  # 给对象属性 self._grid 赋值/初始化（来自当前语句右侧表达式）
            self._bounds = None  # 给对象属性 self._bounds 赋值/初始化（来自当前语句右侧表达式）
            return  # 从函数返回结果

        xs = [p[0] for p in self._points]  # 初始化变量 xs 为一个容器/表达式结果
        ys = [p[1] for p in self._points]  # 初始化变量 ys 为一个容器/表达式结果
        x_min, x_max = min(xs), max(xs)  # 调用函数/方法执行某个动作或计算
        y_min, y_max = min(ys), max(ys)  # 调用函数/方法执行某个动作或计算
        if x_max <= x_min:  # 条件分支判断并选择执行路径
            x_max = x_min + 1.0  # 将表达式计算结果赋给变量 x_max
        if y_max <= y_min:  # 条件分支判断并选择执行路径
            y_max = y_min + 1.0  # 将表达式计算结果赋给变量 y_max
        self._bounds = (x_min, x_max, y_min, y_max)  # 给对象属性 self._bounds 赋值/初始化（来自当前语句右侧表达式）

        bw, bh = self._bins  # 执行当前语句（保持与上文逻辑一致）
        grid = [[0 for _ in range(bw)] for _ in range(bh)]  # 初始化变量 grid 为一个容器/表达式结果
        for x, y in self._points:  # 循环遍历序列/迭代器
            gx = int((x - x_min) / (x_max - x_min) * (bw - 1))  # 将 gx 设为一次调用/构造的返回值
            gy = int((y - y_min) / (y_max - y_min) * (bh - 1))  # 将 gy 设为一次调用/构造的返回值
            gx = max(0, min(bw - 1, gx))  # 将 gx 设为一次调用/构造的返回值
            gy = max(0, min(bh - 1, gy))  # 将 gy 设为一次调用/构造的返回值
            grid[gy][gx] += 1  # 执行当前语句（保持与上文逻辑一致）
        self._grid = grid  # 给对象属性 self._grid 赋值/初始化（来自当前语句右侧表达式）

    def leaveEvent(self, event):  # 定义函数（封装可复用逻辑）
        self._last_hover_cell = None  # 给对象属性 self._last_hover_cell 赋值/初始化（来自当前语句右侧表达式）
        super().leaveEvent(event)  # 调用函数/方法执行某个动作或计算

    def mouseMoveEvent(self, event):  # 定义函数（封装可复用逻辑）
        if not self._grid or self._bounds is None:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)  # 将 plot 设为一次调用/构造的返回值
        if not plot.contains(int(event.position().x()), int(event.position().y())):  # 条件分支判断并选择执行路径
            self._last_hover_cell = None  # 给对象属性 self._last_hover_cell 赋值/初始化（来自当前语句右侧表达式）
            return  # 从函数返回结果

        bw, bh = self._bins  # 执行当前语句（保持与上文逻辑一致）
        x_step = plot.width() / max(1, bw)  # 将 x_step 设为一次调用/构造的返回值
        y_step = plot.height() / max(1, bh)  # 将 y_step 设为一次调用/构造的返回值
        gx = int((event.position().x() - plot.left()) / max(1e-6, x_step))  # 将 gx 设为一次调用/构造的返回值
        gy = int((event.position().y() - plot.top()) / max(1e-6, y_step))  # 将 gy 设为一次调用/构造的返回值
        gx = max(0, min(bw - 1, gx))  # 将 gx 设为一次调用/构造的返回值
        gy = max(0, min(bh - 1, gy))  # 将 gy 设为一次调用/构造的返回值
        cell = (gx, gy)  # 初始化变量 cell 为一个容器/表达式结果
        if cell == self._last_hover_cell:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        self._last_hover_cell = cell  # 给对象属性 self._last_hover_cell 赋值/初始化（来自当前语句右侧表达式）

        c = self._grid[gy][gx]  # 将表达式计算结果赋给变量 c
        max_c = max(max(row) for row in self._grid) if self._grid else 1  # 将 max_c 设为一次调用/构造的返回值
        if max_c <= 0:  # 条件分支判断并选择执行路径
            max_c = 1  # 将表达式计算结果赋给变量 max_c
        density = c / max_c  # 将表达式计算结果赋给变量 density
        QToolTip.showText(  # 执行当前语句（保持与上文逻辑一致）
            event.globalPosition().toPoint(),  # 执行当前语句（保持与上文逻辑一致）
            f"cell=({gx},{gy})\ncount={c}\ndensity={density:.2f}",  # 执行当前语句（保持与上文逻辑一致）
            self,  # 执行当前语句（保持与上文逻辑一致）
        )  # 执行当前语句（保持与上文逻辑一致）

    @staticmethod  # 装饰器：修改/包装下方函数或类的行为
    def _mix(a: QColor, b: QColor, t: float) -> QColor:  # 定义函数（封装可复用逻辑）
        t = max(0.0, min(1.0, float(t)))  # 将 t 设为一次调用/构造的返回值
        return QColor(  # 从函数返回结果
            int(a.red() + (b.red() - a.red()) * t),  # 执行当前语句（保持与上文逻辑一致）
            int(a.green() + (b.green() - a.green()) * t),  # 执行当前语句（保持与上文逻辑一致）
            int(a.blue() + (b.blue() - a.blue()) * t),  # 执行当前语句（保持与上文逻辑一致）
        )  # 执行当前语句（保持与上文逻辑一致）

    def _color_for(self, t: float) -> QColor:  # 定义函数（封装可复用逻辑）
        c1 = QColor("#2563eb")  # 将 c1 设为一次调用/构造的返回值
        c2 = QColor("#22c55e")  # 将 c2 设为一次调用/构造的返回值
        c3 = QColor("#f59e0b")  # 将 c3 设为一次调用/构造的返回值
        c4 = QColor("#ef4444")  # 将 c4 设为一次调用/构造的返回值
        if t < 0.35:  # 条件分支判断并选择执行路径
            return self._mix(c1, c2, t / 0.35)  # 从函数返回结果
        if t < 0.7:  # 条件分支判断并选择执行路径
            return self._mix(c2, c3, (t - 0.35) / 0.35)  # 从函数返回结果
        return self._mix(c3, c4, (t - 0.7) / 0.3)  # 从函数返回结果

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        # No background fill

        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)  # 将 plot 设为一次调用/构造的返回值

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if not self._grid or self._bounds is None:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        p.setPen(QPen(QColor("#1f2937"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)  # 调用函数/方法执行某个动作或计算

        bw, bh = self._bins  # 执行当前语句（保持与上文逻辑一致）
        max_c = max(max(row) for row in self._grid) if self._grid else 1  # 将 max_c 设为一次调用/构造的返回值
        if max_c <= 0:  # 条件分支判断并选择执行路径
            max_c = 1  # 将表达式计算结果赋给变量 max_c

        x_step = plot.width() / max(1, bw)  # 将 x_step 设为一次调用/构造的返回值
        y_step = plot.height() / max(1, bh)  # 将 y_step 设为一次调用/构造的返回值

        p.setPen(QPen(QColor("#111827"), 1))  # 调用函数/方法执行某个动作或计算
        for i in range(1, 6):  # 循环遍历序列/迭代器
            x = plot.left() + int(i / 6 * plot.width())  # 将 x 设为一次调用/构造的返回值
            p.drawLine(x, plot.top(), x, plot.bottom())  # 调用函数/方法执行某个动作或计算
        for i in range(1, 4):  # 循环遍历序列/迭代器
            y = plot.top() + int(i / 4 * plot.height())  # 将 y 设为一次调用/构造的返回值
            p.drawLine(plot.left(), y, plot.right(), y)  # 调用函数/方法执行某个动作或计算

        for gy in range(bh):  # 循环遍历序列/迭代器
            row = self._grid[gy]  # 将表达式计算结果赋给变量 row
            for gx in range(bw):  # 循环遍历序列/迭代器
                c = row[gx]  # 将表达式计算结果赋给变量 c
                if c <= 0:  # 条件分支判断并选择执行路径
                    continue  # 控制流语句：改变当前代码块的执行方式
                t = c / max_c  # 将表达式计算结果赋给变量 t
                cx = plot.left() + (gx + 0.5) * x_step  # 将 cx 设为一次调用/构造的返回值
                cy = plot.top() + (gy + 0.5) * y_step  # 将 cy 设为一次调用/构造的返回值
                radius = 2.0 + 14.0 * sqrt(t)  # 将 radius 设为一次调用/构造的返回值
                color = self._color_for(t)  # 将 color 设为一次调用/构造的返回值
                color.setAlpha(int(120 + 110 * t))  # 调用函数/方法执行某个动作或计算
                p.setBrush(color)  # 调用函数/方法执行某个动作或计算
                p.setPen(QPen(QColor("#0b0f14"), 1))  # 调用函数/方法执行某个动作或计算
                p.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))  # 调用函数/方法执行某个动作或计算

        if self._show_current_point and self._current_point is not None and self._bounds is not None:  # 条件分支判断并选择执行路径
            x_min, x_max, y_min, y_max = self._bounds  # 执行当前语句（保持与上文逻辑一致）
            px = (self._current_point[0] - x_min) / (x_max - x_min)  # 初始化变量 px 为一个容器/表达式结果
            py = (self._current_point[1] - y_min) / (y_max - y_min)  # 初始化变量 py 为一个容器/表达式结果
            px = max(0.0, min(1.0, px))  # 将 px 设为一次调用/构造的返回值
            py = max(0.0, min(1.0, py))  # 将 py 设为一次调用/构造的返回值
            cx = plot.left() + px * plot.width()  # 将 cx 设为一次调用/构造的返回值
            cy = plot.top() + py * plot.height()  # 将 cy 设为一次调用/构造的返回值
            p.setBrush(QColor(0, 0, 0, 0))  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#e5e7eb"), 2))  # 调用函数/方法执行某个动作或计算
            p.drawEllipse(int(cx - 7), int(cy - 7), 14, 14)  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#f59e0b"), 2))  # 调用函数/方法执行某个动作或计算
            p.drawLine(int(cx - 10), int(cy), int(cx + 10), int(cy))  # 调用函数/方法执行某个动作或计算
            p.drawLine(int(cx), int(cy - 10), int(cx), int(cy + 10))  # 调用函数/方法执行某个动作或计算

        legend_w = 120  # 将表达式计算结果赋给变量 legend_w
        legend_h = 10  # 将表达式计算结果赋给变量 legend_h
        legend = plot.adjusted(plot.width() - legend_w - 8, 8, -8, 0)  # 将 legend 设为一次调用/构造的返回值
        legend.setHeight(legend_h)  # 调用函数/方法执行某个动作或计算
        for i in range(legend_w):  # 循环遍历序列/迭代器
            t = i / max(1, legend_w - 1)  # 将 t 设为一次调用/构造的返回值
            p.setPen(QPen(self._color_for(t), 1))  # 调用函数/方法执行某个动作或计算
            p.drawLine(legend.left() + i, legend.top(), legend.left() + i, legend.bottom())  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(QColor("#9ca3af")))  # 调用函数/方法执行某个动作或计算
        p.setFont(QFont("Segoe UI", 8))  # 调用函数/方法执行某个动作或计算
        p.drawText(legend.adjusted(0, legend_h + 2, 0, legend_h + 16), Qt.AlignLeft | Qt.AlignTop, "低")  # 调用函数/方法执行某个动作或计算
        p.drawText(legend.adjusted(0, legend_h + 2, 0, legend_h + 16), Qt.AlignRight | Qt.AlignTop, "高")  # 调用函数/方法执行某个动作或计算


class ProDistributionChart(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._values: List[float] = []  # 执行当前语句（保持与上文逻辑一致）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._x_label = ""  # 给对象属性 self._x_label 赋值/初始化（来自当前语句右侧表达式）
        self._color = QColor("#3b82f6")  # 给对象属性 self._color 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(220)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算
        self.setMouseTracking(True)  # 调用函数/方法执行某个动作或计算

    def set_data(self, values: Sequence[float], *, title: str = "", x_label: str = "", color: str = "#3b82f6"):  # 定义函数（封装可复用逻辑）
        self._values = sorted([float(v) for v in values if v is not None])  # 给对象属性 self._values 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._x_label = x_label  # 给对象属性 self._x_label 赋值/初始化（来自当前语句右侧表达式）
        self._color = QColor(color)  # 给对象属性 self._color 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin - 40, -margin - 20)  # Reserve right space for ECDF axis

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if len(self._values) < 2:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        data = np.array(self._values)  # 将 data 设为一次调用/构造的返回值
        v_min, v_max = data[0], data[-1]  # 执行当前语句（保持与上文逻辑一致）
        if v_max <= v_min: v_max = v_min + 1.0  # 条件分支判断并选择执行路径
        
        # Grid
        p.setPen(QPen(QColor("#1f2937"), 1, Qt.DotLine))  # 调用函数/方法执行某个动作或计算
        for i in range(5):  # 循环遍历序列/迭代器
            y = plot.bottom() - i * plot.height() / 4  # 将 y 设为一次调用/构造的返回值
            p.drawLine(plot.left(), int(y), plot.right(), int(y))  # 调用函数/方法执行某个动作或计算
        
        # 1. Histogram (Background)
        hist, bin_edges = np.histogram(data, bins=30, density=True)  # 调用函数/方法执行某个动作或计算
        max_h = hist.max() if hist.max() > 0 else 1.0  # 将 max_h 设为一次调用/构造的返回值
        bin_w = plot.width() / 30  # 将 bin_w 设为一次调用/构造的返回值
        
        p.setPen(Qt.NoPen)  # 调用函数/方法执行某个动作或计算
        c_hist = QColor(self._color)  # 将 c_hist 设为一次调用/构造的返回值
        c_hist.setAlpha(30)  # 调用函数/方法执行某个动作或计算
        p.setBrush(c_hist)  # 调用函数/方法执行某个动作或计算
        
        for i, h in enumerate(hist):  # 循环遍历序列/迭代器
            bh = h / max_h * plot.height()  # 将 bh 设为一次调用/构造的返回值
            bx = plot.left() + i * bin_w  # 将 bx 设为一次调用/构造的返回值
            p.drawRect(int(bx), int(plot.bottom() - bh), int(bin_w) + 1, int(bh))  # 调用函数/方法执行某个动作或计算

        # 2. KDE Curve
        xs = np.linspace(v_min, v_max, 100)  # 将 xs 设为一次调用/构造的返回值
        ys = np.zeros_like(xs)  # 将 ys 设为一次调用/构造的返回值
        if SCIPY_AVAILABLE:  # 条件分支判断并选择执行路径
            try:  # 开始异常捕获保护块
                kde = gaussian_kde(data)  # 将 kde 设为一次调用/构造的返回值
                ys = kde(xs)  # 将 ys 设为一次调用/构造的返回值
            except: pass  # 捕获异常并进行处理
        else:  # 条件分支的否则路径
             # simple smooth
             ys = np.interp(xs, (bin_edges[:-1]+bin_edges[1:])/2, hist)  # 将 ys 设为一次调用/构造的返回值

        max_y = ys.max() if ys.max() > 0 else 1.0  # 将 max_y 设为一次调用/构造的返回值
        path = QPainterPath()  # 将 path 设为一次调用/构造的返回值
        path.moveTo(plot.left(), plot.bottom())  # 调用函数/方法执行某个动作或计算
        for i, x in enumerate(xs):  # 循环遍历序列/迭代器
            px = plot.left() + (x - v_min)/(v_max - v_min) * plot.width()  # 将 px 设为一次调用/构造的返回值
            py = plot.bottom() - (ys[i]/max_y) * plot.height()  # 将 py 设为一次调用/构造的返回值
            path.lineTo(px, py)  # 调用函数/方法执行某个动作或计算
        path.lineTo(plot.right(), plot.bottom())  # 调用函数/方法执行某个动作或计算
        path.closeSubpath()  # 调用函数/方法执行某个动作或计算
        
        c_kde = QColor(self._color)  # 将 c_kde 设为一次调用/构造的返回值
        c_kde.setAlpha(60)  # 调用函数/方法执行某个动作或计算
        p.setBrush(c_kde)  # 调用函数/方法执行某个动作或计算
        p.setPen(QPen(self._color, 2))  # 调用函数/方法执行某个动作或计算
        p.drawPath(path)  # 调用函数/方法执行某个动作或计算

        # 3. ECDF Curve (Right Axis)
        p.setPen(QPen(QColor("#f59e0b"), 2, Qt.DashLine))  # 调用函数/方法执行某个动作或计算
        p.setBrush(Qt.NoBrush)  # 调用函数/方法执行某个动作或计算
        path_ecdf = QPainterPath()  # 将 path_ecdf 设为一次调用/构造的返回值
        path_ecdf.moveTo(plot.left(), plot.bottom())  # 调用函数/方法执行某个动作或计算
        # Downsample for drawing
        step = max(1, len(data) // 100)  # 将 step 设为一次调用/构造的返回值
        for i in range(0, len(data), step):  # 循环遍历序列/迭代器
            px = plot.left() + (data[i] - v_min)/(v_max - v_min) * plot.width()  # 将 px 设为一次调用/构造的返回值
            py = plot.bottom() - (i / len(data)) * plot.height()  # 将 py 设为一次调用/构造的返回值
            path_ecdf.lineTo(px, py)  # 调用函数/方法执行某个动作或计算
        path_ecdf.lineTo(plot.right(), plot.top())  # 调用函数/方法执行某个动作或计算
        p.drawPath(path_ecdf)  # 调用函数/方法执行某个动作或计算

        # 4. Rug Plot (Bottom)
        p.setPen(QPen(QColor("#e5e7eb"), 1))  # 调用函数/方法执行某个动作或计算
        p.setBrush(Qt.NoBrush)  # 调用函数/方法执行某个动作或计算
        rug_y = plot.bottom() + 4  # 将 rug_y 设为一次调用/构造的返回值
        for v in data:  # 循环遍历序列/迭代器
            px = plot.left() + (v - v_min)/(v_max - v_min) * plot.width()  # 将 px 设为一次调用/构造的返回值
            p.drawLine(int(px), int(rug_y), int(px), int(rug_y + 4))  # 调用函数/方法执行某个动作或计算

        # 5. Stats Markers
        stats = {  # 初始化变量 stats 为一个容器/表达式结果
            "Mean": np.mean(data),  # 执行当前语句（保持与上文逻辑一致）
            "Max": v_max,  # 执行当前语句（保持与上文逻辑一致）
        }  # 执行当前语句（保持与上文逻辑一致）
        p.setFont(QFont("Segoe UI", 8))  # 调用函数/方法执行某个动作或计算
        for k, v in stats.items():  # 循环遍历序列/迭代器
            px = plot.left() + (v - v_min)/(v_max - v_min) * plot.width()  # 将 px 设为一次调用/构造的返回值
            p.setPen(QPen(QColor("#e5e7eb"), 1, Qt.DashLine))  # 调用函数/方法执行某个动作或计算
            p.drawLine(int(px), plot.top(), int(px), plot.bottom())  # 调用函数/方法执行某个动作或计算
            p.setPen(QColor("#e5e7eb"))  # 调用函数/方法执行某个动作或计算
            p.drawText(int(px) + 4, plot.top() + 10 if k == "Mean" else plot.top() + 24, f"{k}:{v:.1f}")  # 调用函数/方法执行某个动作或计算

        # Axes
        p.setPen(QColor("#9ca3af"))  # 调用函数/方法执行某个动作或计算
        p.drawText(rect.adjusted(margin, rect.height()-18, -margin, -2), Qt.AlignLeft, f"{v_min:.1f}")  # 调用函数/方法执行某个动作或计算
        p.drawText(plot.adjusted(0, plot.height()+2, 0, 18), Qt.AlignRight, f"{v_max:.1f}")  # 调用函数/方法执行某个动作或计算
        
        # Right Axis Labels (0% - 100%)
        p.setPen(QColor("#f59e0b"))  # 调用函数/方法执行某个动作或计算
        p.drawText(rect.adjusted(rect.width()-36, margin+title_h, -2, 0), Qt.AlignRight|Qt.AlignTop, "100%")  # 调用函数/方法执行某个动作或计算
        p.drawText(rect.adjusted(rect.width()-36, rect.height()-margin-20, -2, -margin), Qt.AlignRight|Qt.AlignBottom, "0%")  # 调用函数/方法执行某个动作或计算



class TerritoryScatterPlot(QWidget):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self._points: List[Tuple[float, float]] = []  # 执行当前语句（保持与上文逻辑一致）
        self._title = ""  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._current_point: Optional[Tuple[float, float]] = None  # 执行当前语句（保持与上文逻辑一致）
        self._color = QColor("#3b82f6")  # 给对象属性 self._color 赋值/初始化（来自当前语句右侧表达式）
        self.setMinimumHeight(220)  # 调用函数/方法执行某个动作或计算
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 调用函数/方法执行某个动作或计算

    def set_points(self, points: Sequence[Tuple[float, float]], *, title: str = "", color: str = "#3b82f6"):  # 定义函数（封装可复用逻辑）
        self._points = [(float(x), float(y)) for x, y in points]  # 给对象属性 self._points 赋值/初始化（来自当前语句右侧表达式）
        self._title = title  # 给对象属性 self._title 赋值/初始化（来自当前语句右侧表达式）
        self._color = QColor(color)  # 给对象属性 self._color 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def set_current_point(self, point: Optional[Tuple[float, float]]):  # 定义函数（封装可复用逻辑）
        self._current_point = (float(point[0]), float(point[1])) if point is not None else None  # 给对象属性 self._current_point 赋值/初始化（来自当前语句右侧表达式）
        self.update()  # 调用函数/方法执行某个动作或计算

    def paintEvent(self, event):  # 定义函数（封装可复用逻辑）
        p = QPainter(self)  # 将 p 设为一次调用/构造的返回值
        p.setRenderHint(QPainter.Antialiasing, True)  # 调用函数/方法执行某个动作或计算
        rect = self.rect()  # 将 rect 设为一次调用/构造的返回值
        # No background fill

        margin = 12  # 将表达式计算结果赋给变量 margin
        title_h = 18 if self._title else 0  # 将表达式计算结果赋给变量 title_h
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)  # 将 plot 设为一次调用/构造的返回值

        if self._title:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#93c5fd")))  # 调用函数/方法执行某个动作或计算
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))  # 调用函数/方法执行某个动作或计算
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)  # 调用函数/方法执行某个动作或计算

        if not self._points:  # 条件分支判断并选择执行路径
            p.setPen(QPen(QColor("#6b7280")))  # 调用函数/方法执行某个动作或计算
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")  # 调用函数/方法执行某个动作或计算
            return  # 从函数返回结果

        xs = [pt[0] for pt in self._points]  # 初始化变量 xs 为一个容器/表达式结果
        ys = [pt[1] for pt in self._points]  # 初始化变量 ys 为一个容器/表达式结果
        x_min, x_max = min(xs), max(xs)  # 调用函数/方法执行某个动作或计算
        y_min, y_max = min(ys), max(ys)  # 调用函数/方法执行某个动作或计算
        
        # Add some padding
        pad_x = (x_max - x_min) * 0.1 if x_max > x_min else 10.0  # 初始化变量 pad_x 为一个容器/表达式结果
        pad_y = (y_max - y_min) * 0.1 if y_max > y_min else 10.0  # 初始化变量 pad_y 为一个容器/表达式结果
        x_min -= pad_x  # 执行当前语句（保持与上文逻辑一致）
        x_max += pad_x  # 执行当前语句（保持与上文逻辑一致）
        y_min -= pad_y  # 执行当前语句（保持与上文逻辑一致）
        y_max += pad_y  # 执行当前语句（保持与上文逻辑一致）

        def to_pos(px, py):  # 定义函数（封装可复用逻辑）
            ix = plot.left() + (px - x_min) / (x_max - x_min) * plot.width()  # 将 ix 设为一次调用/构造的返回值
            iy = plot.top() + (py - y_min) / (y_max - y_min) * plot.height()  # 将 iy 设为一次调用/构造的返回值
            return QPointF(ix, iy)  # 从函数返回结果

        # Draw scatter
        p.setPen(Qt.NoPen)  # 调用函数/方法执行某个动作或计算
        c = QColor(self._color)  # 将 c 设为一次调用/构造的返回值
        c.setAlpha(100)  # 调用函数/方法执行某个动作或计算
        p.setBrush(c)  # 调用函数/方法执行某个动作或计算
        for pt in self._points:  # 循环遍历序列/迭代器
            pos = to_pos(pt[0], pt[1])  # 将 pos 设为一次调用/构造的返回值
            p.drawEllipse(pos, 3, 3)  # 调用函数/方法执行某个动作或计算

        # Draw Convex Hull if possible
        if SCIPY_AVAILABLE and len(self._points) >= 3:  # 条件分支判断并选择执行路径
            try:  # 开始异常捕获保护块
                hull = ConvexHull(self._points)  # 将 hull 设为一次调用/构造的返回值
                poly = QPolygonF()  # 将 poly 设为一次调用/构造的返回值
                for v in hull.vertices:  # 循环遍历序列/迭代器
                    poly.append(to_pos(self._points[v][0], self._points[v][1]))  # 调用函数/方法执行某个动作或计算
                
                hull_c = QColor(self._color)  # 将 hull_c 设为一次调用/构造的返回值
                hull_c.setAlpha(40)  # 调用函数/方法执行某个动作或计算
                p.setBrush(hull_c)  # 调用函数/方法执行某个动作或计算
                p.setPen(QPen(self._color, 1, Qt.DashLine))  # 调用函数/方法执行某个动作或计算
                p.drawPolygon(poly)  # 调用函数/方法执行某个动作或计算
            except Exception:  # 捕获异常并进行处理
                pass  # 控制流语句：改变当前代码块的执行方式

        # Draw current point
        if self._current_point:  # 条件分支判断并选择执行路径
            pos = to_pos(self._current_point[0], self._current_point[1])  # 将 pos 设为一次调用/构造的返回值
            p.setBrush(Qt.NoBrush)  # 调用函数/方法执行某个动作或计算
            p.setPen(QPen(QColor("#f59e0b"), 2))  # 调用函数/方法执行某个动作或计算
            p.drawLine(QPointF(pos.x() - 8, pos.y()), QPointF(pos.x() + 8, pos.y()))  # 调用函数/方法执行某个动作或计算
            p.drawLine(QPointF(pos.x(), pos.y() - 8), QPointF(pos.x(), pos.y() + 8))  # 调用函数/方法执行某个动作或计算
            p.drawEllipse(pos, 6, 6)  # 调用函数/方法执行某个动作或计算

        p.setPen(QPen(QColor("#1f2937"), 1))  # 调用函数/方法执行某个动作或计算
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)  # 调用函数/方法执行某个动作或计算
