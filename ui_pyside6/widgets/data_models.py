from __future__ import annotations  # 从模块导入符号，供后续调用

from typing import Any, Optional  # 从模块导入符号，供后续调用

import pandas as pd  # 导入模块，供后续使用
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt  # 从模块导入符号，供后续调用


class DataFrameModel(QAbstractTableModel):  # 定义类（封装数据与行为）
    def __init__(self, df: Optional[pd.DataFrame] = None):  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self._df = df if df is not None else pd.DataFrame()  # 给对象属性 self._df 赋值/初始化（来自当前语句右侧表达式）

    def set_dataframe(self, df: pd.DataFrame):  # 定义函数（封装可复用逻辑）
        self.beginResetModel()  # 调用函数/方法执行某个动作或计算
        self._df = df  # 给对象属性 self._df 赋值/初始化（来自当前语句右侧表达式）
        self.endResetModel()  # 调用函数/方法执行某个动作或计算

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # 定义函数（封装可复用逻辑）
        if parent.isValid():  # 条件分支判断并选择执行路径
            return 0  # 从函数返回结果
        return int(self._df.shape[0])  # 从函数返回结果

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # 定义函数（封装可复用逻辑）
        if parent.isValid():  # 条件分支判断并选择执行路径
            return 0  # 从函数返回结果
        return int(self._df.shape[1])  # 从函数返回结果

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # 定义函数（封装可复用逻辑）
        if not index.isValid():  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        if role not in (Qt.DisplayRole, Qt.ToolTipRole):  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        value = self._df.iat[index.row(), index.column()]  # 将 value 设为一次调用/构造的返回值
        if pd.isna(value):  # 条件分支判断并选择执行路径
            return ""  # 从函数返回结果
        return str(value)  # 从函数返回结果

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:  # 定义函数（封装可复用逻辑）
        if role != Qt.DisplayRole:  # 条件分支判断并选择执行路径
            return None  # 从函数返回结果
        if orientation == Qt.Horizontal:  # 条件分支判断并选择执行路径
            if 0 <= section < len(self._df.columns):  # 条件分支判断并选择执行路径
                return str(self._df.columns[section])  # 从函数返回结果
            return ""  # 从函数返回结果
        return str(section)  # 从函数返回结果

