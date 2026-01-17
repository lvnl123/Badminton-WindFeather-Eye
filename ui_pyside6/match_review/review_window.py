from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,   # 从模块导入符号，供后续调用
                               QTabWidget, QLabel, QPushButton, QListWidget, QComboBox, QApplication,   # 执行当前语句（保持与上文逻辑一致）
                               QListWidgetItem, QSplitter, QGroupBox, QGridLayout,   # 执行当前语句（保持与上文逻辑一致）
                               QProgressDialog, QMessageBox)  # 执行当前语句（保持与上文逻辑一致）
from PySide6.QtCore import Qt, QSize  # 从模块导入符号，供后续调用
from PySide6.QtGui import QFont, QColor  # 从模块导入符号，供后续调用
from .engine import MatchEngine  # 从模块导入符号，供后续调用
from .panels import (RadarChart, HeatmapChart, ShotTypePie, FeatureCourt3D,   # 从模块导入符号，供后续调用
                     SpeedHeightScatter, TransitionChordChart, TransitionHeatmap, ThemeRiverChart,   # 执行当前语句（保持与上文逻辑一致）
                     VoronoiMap, HitPoint3D, LoadChart, PhysicalKPI, SpeedHistogram,   # 执行当前语句（保持与上文逻辑一致）
                     AccelTimeline, CoverageQuantile, BarycenterEllipse)  # 执行当前语句（保持与上文逻辑一致）
from .arena import Arena3D  # 从模块导入符号，供后续调用

class MetricCard(QGroupBox):  # 定义类（封装数据与行为）
    def __init__(self, title, value, unit="", parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self.setTitle(title)  # 调用函数/方法执行某个动作或计算
        layout = QVBoxLayout(self)  # 将 layout 设为一次调用/构造的返回值
        self.value_label = QLabel(value)  # 给对象属性 self.value_label 赋值/初始化（来自当前语句右侧表达式）
        self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffcc;")  # 调用函数/方法执行某个动作或计算
        self.value_label.setAlignment(Qt.AlignCenter)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(self.value_label)  # 调用函数/方法执行某个动作或计算
        if unit:  # 条件分支判断并选择执行路径
            unit_lbl = QLabel(unit)  # 将 unit_lbl 设为一次调用/构造的返回值
            unit_lbl.setAlignment(Qt.AlignCenter)  # 调用函数/方法执行某个动作或计算
            unit_lbl.setStyleSheet("color: #aaaaaa;")  # 调用函数/方法执行某个动作或计算
            layout.addWidget(unit_lbl)  # 调用函数/方法执行某个动作或计算

class MatchReviewWindow(QMainWindow):  # 定义类（封装数据与行为）
    def __init__(self, parent=None):  # 定义函数（封装可复用逻辑）
        super().__init__(parent)  # 调用函数/方法执行某个动作或计算
        self.setWindowTitle("TrackNetV3 专业比赛复盘系统")  # 调用函数/方法执行某个动作或计算
        self.resize(1600, 900)  # 调用函数/方法执行某个动作或计算
        
        self.engine = None  # 给对象属性 self.engine 赋值/初始化（来自当前语句右侧表达式）
        
        self.central_widget = QTabWidget()  # 给对象属性 self.central_widget 赋值/初始化（来自当前语句右侧表达式）
        container = QWidget()  # 将 container 设为一次调用/构造的返回值
        root_layout = QVBoxLayout(container)  # 将 root_layout 设为一次调用/构造的返回值
        top_bar = QHBoxLayout()  # 将 top_bar 设为一次调用/构造的返回值
        top_bar.addStretch(1)  # 调用函数/方法执行某个动作或计算
        self.dataset_combo = QComboBox(container)  # 给对象属性 self.dataset_combo 赋值/初始化（来自当前语句右侧表达式）
        self.dataset_combo.setMinimumWidth(200)  # 调用函数/方法执行某个动作或计算
        self.dataset_combo.addItem("请选择数据集")  # 调用函数/方法执行某个动作或计算
        for name, path in self._scan_results():  # 循环遍历序列/迭代器
            self.dataset_combo.addItem(name, path)  # 调用函数/方法执行某个动作或计算
        top_bar.addWidget(self.dataset_combo)  # 调用函数/方法执行某个动作或计算
        refresh_btn = QPushButton("刷新", container)  # 将 refresh_btn 设为一次调用/构造的返回值
        refresh_btn.setToolTip("重新扫描结果目录并更新列表")  # 调用函数/方法执行某个动作或计算
        top_bar.addWidget(refresh_btn)  # 调用函数/方法执行某个动作或计算
        root_layout.addLayout(top_bar)  # 调用函数/方法执行某个动作或计算
        root_layout.addWidget(self.central_widget, 1)  # 调用函数/方法执行某个动作或计算
        self.setCentralWidget(container)  # 调用函数/方法执行某个动作或计算
        
        self._init_tabs()  # 调用函数/方法执行某个动作或计算
        self._apply_dark_theme()  # 调用函数/方法执行某个动作或计算
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)  # 调用函数/方法执行某个动作或计算
        refresh_btn.clicked.connect(self._refresh_dataset_list)  # 调用函数/方法执行某个动作或计算

    def _apply_dark_theme(self):  # 定义函数（封装可复用逻辑）
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 10px 20px; }
            QTabBar::tab:selected { background: #555; color: #fff; font-weight: bold; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QListWidget { background-color: #252525; border: 1px solid #444; }
            QListWidget::item:selected { background-color: #00ffcc; color: #000; }
        """)

    def _init_tabs(self):  # 定义函数（封装可复用逻辑）
        # 1. Dashboard
        self.dashboard_tab = QWidget()  # 给对象属性 self.dashboard_tab 赋值/初始化（来自当前语句右侧表达式）
        self.central_widget.addTab(self.dashboard_tab, "比赛概况 (Dashboard)")  # 调用函数/方法执行某个动作或计算
        
        # 2. Rally Analysis
        self.rally_tab = QWidget()  # 给对象属性 self.rally_tab 赋值/初始化（来自当前语句右侧表达式）
        self.central_widget.addTab(self.rally_tab, "战术复盘 (Tactical)")  # 调用函数/方法执行某个动作或计算
        
        # 3. Player Stats
        self.player_tab = QWidget()  # 给对象属性 self.player_tab 赋值/初始化（来自当前语句右侧表达式）
        self.central_widget.addTab(self.player_tab, "球员表现 (Physical)")  # 调用函数/方法执行某个动作或计算
        
        # 4. Tech Stats
        self.tech_tab = QWidget()  # 给对象属性 self.tech_tab 赋值/初始化（来自当前语句右侧表达式）
        self.central_widget.addTab(self.tech_tab, "技术统计 (Technical)")  # 调用函数/方法执行某个动作或计算

        # 5. Deep Tactics
        self.deep_tab = QWidget()  # 给对象属性 self.deep_tab 赋值/初始化（来自当前语句右侧表达式）
        self.central_widget.addTab(self.deep_tab, "深度战术 (Deep Tactics)")  # 调用函数/方法执行某个动作或计算
        
        # 6. Physio Load
        self.load_tab = QWidget()  # 给对象属性 self.load_tab 赋值/初始化（来自当前语句右侧表达式）
        self.central_widget.addTab(self.load_tab, "体能负荷 (Load)")  # 调用函数/方法执行某个动作或计算

    def load_match(self, folder_path):  # 定义函数（封装可复用逻辑）
        try:  # 开始异常捕获保护块
            # Show loading
            progress = QProgressDialog("正在进行深度数据挖掘...", "取消", 0, 100, self)  # 将 progress 设为一次调用/构造的返回值
            progress.setWindowModality(Qt.WindowModal)  # 调用函数/方法执行某个动作或计算
            progress.setWindowTitle("加载比赛数据")  # 调用函数/方法执行某个动作或计算
            progress.setMinimumDuration(0)  # 调用函数/方法执行某个动作或计算
            progress.setAutoClose(True)  # 调用函数/方法执行某个动作或计算
            progress.setAutoReset(True)  # 调用函数/方法执行某个动作或计算
            progress.setStyleSheet("QProgressDialog { background-color: #2b2b2b; color: #ffffff; } QLabel { color: #ffffff; }")  # 调用函数/方法执行某个动作或计算
            progress.show()  # 调用函数/方法执行某个动作或计算
            progress.setValue(10)  # 调用函数/方法执行某个动作或计算
            QApplication.processEvents()  # 调用函数/方法执行某个动作或计算
            
            self.engine = MatchEngine(folder_path)  # 给对象属性 self.engine 赋值/初始化（来自当前语句右侧表达式）
            self.engine.load_data()  # 调用函数/方法执行某个动作或计算
            
            progress.setValue(50)  # 调用函数/方法执行某个动作或计算
            progress.setLabelText("生成可视化图表...")  # 调用函数/方法执行某个动作或计算
            QApplication.processEvents()  # 调用函数/方法执行某个动作或计算
            
            self._clear_tab(self.dashboard_tab)  # 调用函数/方法执行某个动作或计算
            self._clear_tab(self.rally_tab)  # 调用函数/方法执行某个动作或计算
            self._clear_tab(self.player_tab)  # 调用函数/方法执行某个动作或计算
            self._clear_tab(self.tech_tab)  # 调用函数/方法执行某个动作或计算
            self._clear_tab(self.deep_tab)  # 调用函数/方法执行某个动作或计算
            self._clear_tab(self.load_tab)  # 调用函数/方法执行某个动作或计算
            self._build_dashboard()  # 调用函数/方法执行某个动作或计算
            self._build_rally_page()  # 调用函数/方法执行某个动作或计算
            self._build_player_page()  # 调用函数/方法执行某个动作或计算
            self._build_tech_page()  # 调用函数/方法执行某个动作或计算
            self._build_deep_page()  # 调用函数/方法执行某个动作或计算
            self._build_load_page()  # 调用函数/方法执行某个动作或计算
            
            progress.setValue(100)  # 调用函数/方法执行某个动作或计算
            QApplication.processEvents()  # 调用函数/方法执行某个动作或计算
            progress.close()  # 调用函数/方法执行某个动作或计算
            
        except Exception as e:  # 捕获异常并进行处理
            QMessageBox.critical(self, "加载失败", f"无法加载比赛数据:\n{str(e)}")  # 调用函数/方法执行某个动作或计算
    def _scan_results(self):  # 定义函数（封装可复用逻辑）
        from pathlib import Path  # 从模块导入符号，供后续调用
        res = []  # 初始化变量 res 为一个容器/表达式结果
        root = Path(__file__).resolve().parents[2] / "results"  # 将 root 设为一次调用/构造的返回值
        if root.exists():  # 条件分支判断并选择执行路径
            for d in sorted(root.iterdir()):  # 循环遍历序列/迭代器
                if d.is_dir():  # 条件分支判断并选择执行路径
                    csvs = list(d.glob("*_data.csv"))  # 将 csvs 设为一次调用/构造的返回值
                    if csvs:  # 条件分支判断并选择执行路径
                        res.append((d.name, str(d)))  # 调用函数/方法执行某个动作或计算
        return res  # 从函数返回结果
    def _on_dataset_selected(self, idx):  # 定义函数（封装可复用逻辑）
        if idx <= 0:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        path = self.dataset_combo.itemData(idx)  # 将 path 设为一次调用/构造的返回值
        if isinstance(path, str) and path:  # 条件分支判断并选择执行路径
            self.load_match(path)  # 调用函数/方法执行某个动作或计算
    def _refresh_dataset_list(self):  # 定义函数（封装可复用逻辑）
        current_text = self.dataset_combo.currentText()  # 将 current_text 设为一次调用/构造的返回值
        self.dataset_combo.blockSignals(True)  # 调用函数/方法执行某个动作或计算
        self.dataset_combo.clear()  # 调用函数/方法执行某个动作或计算
        self.dataset_combo.addItem("请选择数据集")  # 调用函数/方法执行某个动作或计算
        items = list(self._scan_results())  # 将 items 设为一次调用/构造的返回值
        for name, path in items:  # 循环遍历序列/迭代器
            self.dataset_combo.addItem(name, path)  # 调用函数/方法执行某个动作或计算
        # try to keep previous selection if still exists
        idx = self.dataset_combo.findText(current_text)  # 将 idx 设为一次调用/构造的返回值
        if idx >= 0:  # 条件分支判断并选择执行路径
            self.dataset_combo.setCurrentIndex(idx)  # 调用函数/方法执行某个动作或计算
        else:  # 条件分支的否则路径
            self.dataset_combo.setCurrentIndex(0)  # 调用函数/方法执行某个动作或计算
        self.dataset_combo.blockSignals(False)  # 调用函数/方法执行某个动作或计算
    def _clear_tab(self, tab):  # 定义函数（封装可复用逻辑）
        lay = tab.layout()  # 将 lay 设为一次调用/构造的返回值
        if not lay:  # 条件分支判断并选择执行路径
            return  # 从函数返回结果
        while lay.count():  # 条件循环，直到条件不满足
            item = lay.takeAt(0)  # 将 item 设为一次调用/构造的返回值
            w = item.widget()  # 将 w 设为一次调用/构造的返回值
            if w:  # 条件分支判断并选择执行路径
                w.deleteLater()  # 调用函数/方法执行某个动作或计算

    def _build_dashboard(self):  # 定义函数（封装可复用逻辑）
        layout = self.dashboard_tab.layout() or QVBoxLayout(self.dashboard_tab)  # 将 layout 设为一次调用/构造的返回值
        
        # Top: KPI Cards
        kpi_layout = QHBoxLayout()  # 将 kpi_layout 设为一次调用/构造的返回值
        
        total_rallies = len(self.engine.rallies)  # 将 total_rallies 设为一次调用/构造的返回值
        total_hits = len(self.engine.hits)  # 将 total_hits 设为一次调用/构造的返回值
        avg_rally_len = total_hits / total_rallies if total_rallies else 0  # 将表达式计算结果赋给变量 avg_rally_len
        total_dist = (self.engine.global_player_stats[1]['dist'] + self.engine.global_player_stats[2]['dist']) / 100 # px to m approx
        
        kpi_layout.addWidget(MetricCard("总回合数", str(total_rallies)))  # 调用函数/方法执行某个动作或计算
        kpi_layout.addWidget(MetricCard("总击球数", str(total_hits)))  # 调用函数/方法执行某个动作或计算
        kpi_layout.addWidget(MetricCard("平均拍数", f"{avg_rally_len:.1f}"))  # 调用函数/方法执行某个动作或计算
        kpi_layout.addWidget(MetricCard("总跑动估算(m)", f"{total_dist:.0f}"))  # 调用函数/方法执行某个动作或计算
        
        layout.addLayout(kpi_layout)  # 调用函数/方法执行某个动作或计算
        
        # Middle: Radar & 3D Court
        mid_layout = QHBoxLayout()  # 将 mid_layout 设为一次调用/构造的返回值
        
        radar_group = QGroupBox("能力六维图")  # 将 radar_group 设为一次调用/构造的返回值
        radar_layout = QVBoxLayout(radar_group)  # 将 radar_layout 设为一次调用/构造的返回值
        self.radar_chart = RadarChart()  # 给对象属性 self.radar_chart 赋值/初始化（来自当前语句右侧表达式）
        radar_layout.addWidget(self.radar_chart)  # 调用函数/方法执行某个动作或计算
        
        # Update Radar
        p1_radar = self.engine.get_player_radar_data(1)  # 将 p1_radar 设为一次调用/构造的返回值
        p2_radar = self.engine.get_player_radar_data(2)  # 将 p2_radar 设为一次调用/构造的返回值
        self.radar_chart.plot(p1_radar, p2_radar, list(p1_radar.keys()))  # 调用函数/方法执行某个动作或计算
        
        court3d_group = QGroupBox("击球落点-高度 3D 战术图")  # 将 court3d_group 设为一次调用/构造的返回值
        court3d_layout = QVBoxLayout(court3d_group)  # 将 court3d_layout 设为一次调用/构造的返回值
        self.court3d_chart = FeatureCourt3D()  # 给对象属性 self.court3d_chart 赋值/初始化（来自当前语句右侧表达式）
        court3d_layout.addWidget(self.court3d_chart, alignment=Qt.AlignCenter)  # 调用函数/方法执行某个动作或计算
        self.court3d_chart.plot(self.engine)  # 调用函数/方法执行某个动作或计算
        # Controls
        ctrl_bar = QHBoxLayout()  # 将 ctrl_bar 设为一次调用/构造的返回值
        color_speed_btn = QPushButton("按球速着色", court3d_group)  # 将 color_speed_btn 设为一次调用/构造的返回值
        color_type_btn = QPushButton("按类型着色", court3d_group)  # 将 color_type_btn 设为一次调用/构造的返回值
        hull_toggle_btn = QPushButton("开/关包络", court3d_group)  # 将 hull_toggle_btn 设为一次调用/构造的返回值
        line_toggle_btn = QPushButton("开/关连线", court3d_group)  # 将 line_toggle_btn 设为一次调用/构造的返回值
        zoom_in_btn = QPushButton("缩放 +", court3d_group)  # 将 zoom_in_btn 设为一次调用/构造的返回值
        zoom_out_btn = QPushButton("缩放 -", court3d_group)  # 将 zoom_out_btn 设为一次调用/构造的返回值
        zoom_reset_btn = QPushButton("重置缩放", court3d_group)  # 将 zoom_reset_btn 设为一次调用/构造的返回值
        ctrl_bar.addWidget(color_speed_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addWidget(color_type_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addSpacing(10)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addWidget(hull_toggle_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addWidget(line_toggle_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addSpacing(10)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addWidget(zoom_in_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addWidget(zoom_out_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addWidget(zoom_reset_btn)  # 调用函数/方法执行某个动作或计算
        ctrl_bar.addStretch(1)  # 调用函数/方法执行某个动作或计算
        court3d_layout.addLayout(ctrl_bar)  # 调用函数/方法执行某个动作或计算
        # Bind
        color_speed_btn.clicked.connect(lambda: self.court3d_chart.set_color_mode('speed'))  # 调用函数/方法执行某个动作或计算
        color_type_btn.clicked.connect(lambda: self.court3d_chart.set_color_mode('type'))  # 调用函数/方法执行某个动作或计算
        hull_toggle_btn.clicked.connect(lambda: self.court3d_chart.set_show_hulls(not self.court3d_chart._show_hulls))  # 调用函数/方法执行某个动作或计算
        line_toggle_btn.clicked.connect(lambda: self.court3d_chart.set_show_lines(not self.court3d_chart._show_lines))  # 调用函数/方法执行某个动作或计算
        zoom_in_btn.clicked.connect(lambda: self.court3d_chart.zoom(1.2))  # 调用函数/方法执行某个动作或计算
        zoom_out_btn.clicked.connect(lambda: self.court3d_chart.zoom(1/1.2))  # 调用函数/方法执行某个动作或计算
        zoom_reset_btn.clicked.connect(self.court3d_chart.reset_zoom)  # 调用函数/方法执行某个动作或计算
        
        mid_layout.addWidget(radar_group, 1)  # 调用函数/方法执行某个动作或计算
        mid_layout.addWidget(court3d_group, 2)  # 调用函数/方法执行某个动作或计算
        
        layout.addLayout(mid_layout, 2)  # 调用函数/方法执行某个动作或计算

    def _build_rally_page(self):  # 定义函数（封装可复用逻辑）
        layout = self.rally_tab.layout() or QHBoxLayout(self.rally_tab)  # 将 layout 设为一次调用/构造的返回值
        
        # Left: Rally List
        left_panel = QWidget()  # 将 left_panel 设为一次调用/构造的返回值
        left_layout = QVBoxLayout(left_panel)  # 将 left_layout 设为一次调用/构造的返回值
        left_layout.addWidget(QLabel("回合列表"))  # 调用函数/方法执行某个动作或计算
        
        self.rally_list = QListWidget()  # 给对象属性 self.rally_list 赋值/初始化（来自当前语句右侧表达式）
        for r in self.engine.rallies:  # 循环遍历序列/迭代器
            item = QListWidgetItem(f"Rally {r.id}: {r.hit_count} hits, {r.duration_sec:.1f}s")  # 将 item 设为一次调用/构造的返回值
            item.setData(Qt.UserRole, r.id)  # 调用函数/方法执行某个动作或计算
            self.rally_list.addItem(item)  # 调用函数/方法执行某个动作或计算
            
        self.rally_list.currentRowChanged.connect(self._on_rally_selected)  # 调用函数/方法执行某个动作或计算
        left_layout.addWidget(self.rally_list)  # 调用函数/方法执行某个动作或计算
        
        layout.addWidget(left_panel, 1)  # 调用函数/方法执行某个动作或计算
        
        # Right: 3D Arena & Stats
        right_panel = QWidget()  # 将 right_panel 设为一次调用/构造的返回值
        right_layout = QVBoxLayout(right_panel)  # 将 right_layout 设为一次调用/构造的返回值
        
        self.arena = Arena3D()  # 给对象属性 self.arena 赋值/初始化（来自当前语句右侧表达式）
        right_layout.addWidget(self.arena, 2)  # 调用函数/方法执行某个动作或计算
        
        self.rally_speed_chart = SpeedHeightScatter() # Reusing for rally stats if needed, or create new
        # Let's put text details here instead
        self.rally_info_lbl = QLabel("选择一个回合查看详情")  # 给对象属性 self.rally_info_lbl 赋值/初始化（来自当前语句右侧表达式）
        self.rally_info_lbl.setAlignment(Qt.AlignCenter)  # 调用函数/方法执行某个动作或计算
        right_layout.addWidget(self.rally_info_lbl)  # 调用函数/方法执行某个动作或计算
        
        layout.addWidget(right_panel, 3)  # 调用函数/方法执行某个动作或计算

    def _on_rally_selected(self, row):  # 定义函数（封装可复用逻辑）
        if row < 0: return  # 条件分支判断并选择执行路径
        rally_id = self.rally_list.item(row).data(Qt.UserRole)  # 将 rally_id 设为一次调用/构造的返回值
        rally = next((r for r in self.engine.rallies if r.id == rally_id), None)  # 将 rally 设为一次调用/构造的返回值
        
        if rally:  # 条件分支判断并选择执行路径
            self.arena.plot_rally(rally.trajectory, rally.strokes)  # 调用函数/方法执行某个动作或计算
            
            info = f"""
            <h3>Rally {rally.id}</h3>
            <p>持续时间: {rally.duration_sec:.2f} 秒 | 击球数: {rally.hit_count}</p>
            <p>Player 1 跑动: {rally.player_stats[1]['dist']:.1f} | 平均速度: {rally.player_stats[1]['avg_speed']:.1f}</p>
            <p>Player 2 跑动: {rally.player_stats[2]['dist']:.1f} | 平均速度: {rally.player_stats[2]['avg_speed']:.1f}</p>
            """
            self.rally_info_lbl.setText(info)  # 调用函数/方法执行某个动作或计算

    def _build_player_page(self):  # 定义函数（封装可复用逻辑）
        layout = self.player_tab.layout() or QGridLayout(self.player_tab)  # 将 layout 设为一次调用/构造的返回值
        
        p1_grp = QGroupBox("Player 1 覆盖热区")  # 将 p1_grp 设为一次调用/构造的返回值
        p1_layout = QGridLayout(p1_grp)  # 将 p1_layout 设为一次调用/构造的返回值
        kpi1 = PhysicalKPI()  # 将 kpi1 设为一次调用/构造的返回值
        p1_map = HeatmapChart()  # 将 p1_map 设为一次调用/构造的返回值
        sp1 = SpeedHistogram()  # 将 sp1 设为一次调用/构造的返回值
        ac1 = AccelTimeline()  # 将 ac1 设为一次调用/构造的返回值
        q1 = CoverageQuantile()  # 将 q1 设为一次调用/构造的返回值
        bc1 = BarycenterEllipse()  # 将 bc1 设为一次调用/构造的返回值
        # Top row: 左侧KPI，右侧移动热区
        p1_layout.addWidget(kpi1, 0, 0, 1, 1)  # 调用函数/方法执行某个动作或计算
        p1_layout.addWidget(p1_map, 0, 1, 1, 1)  # 调用函数/方法执行某个动作或计算
        # Second row: 左速度分布，右加速度时间序列
        p1_layout.addWidget(sp1, 1, 0, 1, 1)  # 调用函数/方法执行某个动作或计算
        p1_layout.addWidget(ac1, 1, 1, 1, 1)  # 调用函数/方法执行某个动作或计算
        # Third row: 左覆盖分位线，右质心椭圆
        p1_layout.addWidget(q1, 2, 0, 1, 1)  # 调用函数/方法执行某个动作或计算
        p1_layout.addWidget(bc1, 2, 1, 1, 1)  # 调用函数/方法执行某个动作或计算
        # Stretch to尽可能大
        p1_layout.setColumnStretch(0, 1)  # 调用函数/方法执行某个动作或计算
        p1_layout.setColumnStretch(1, 1)  # 调用函数/方法执行某个动作或计算
        p1_layout.setRowStretch(0, 2)  # 调用函数/方法执行某个动作或计算
        p1_layout.setRowStretch(1, 2)  # 调用函数/方法执行某个动作或计算
        p1_layout.setRowStretch(2, 2)  # 调用函数/方法执行某个动作或计算

        p2_grp = QGroupBox("Player 2 覆盖热区")  # 将 p2_grp 设为一次调用/构造的返回值
        p2_layout = QGridLayout(p2_grp)  # 将 p2_layout 设为一次调用/构造的返回值
        kpi2 = PhysicalKPI()  # 将 kpi2 设为一次调用/构造的返回值
        p2_map = HeatmapChart()  # 将 p2_map 设为一次调用/构造的返回值
        sp2 = SpeedHistogram()  # 将 sp2 设为一次调用/构造的返回值
        ac2 = AccelTimeline()  # 将 ac2 设为一次调用/构造的返回值
        q2 = CoverageQuantile()  # 将 q2 设为一次调用/构造的返回值
        bc2 = BarycenterEllipse()  # 将 bc2 设为一次调用/构造的返回值
        # Top row: 左侧KPI，右侧移动热区
        p2_layout.addWidget(kpi2, 0, 0, 1, 1)  # 调用函数/方法执行某个动作或计算
        p2_layout.addWidget(p2_map, 0, 1, 1, 1)  # 调用函数/方法执行某个动作或计算
        # Second row: 左速度分布，右加速度时间序列
        p2_layout.addWidget(sp2, 1, 0, 1, 1)  # 调用函数/方法执行某个动作或计算
        p2_layout.addWidget(ac2, 1, 1, 1, 1)  # 调用函数/方法执行某个动作或计算
        # Third row: 左覆盖分位线，右质心椭圆
        p2_layout.addWidget(q2, 2, 0, 1, 1)  # 调用函数/方法执行某个动作或计算
        p2_layout.addWidget(bc2, 2, 1, 1, 1)  # 调用函数/方法执行某个动作或计算
        # Stretch to尽可能大
        p2_layout.setColumnStretch(0, 1)  # 调用函数/方法执行某个动作或计算
        p2_layout.setColumnStretch(1, 1)  # 调用函数/方法执行某个动作或计算
        p2_layout.setRowStretch(0, 2)  # 调用函数/方法执行某个动作或计算
        p2_layout.setRowStretch(1, 2)  # 调用函数/方法执行某个动作或计算
        p2_layout.setRowStretch(2, 2)  # 调用函数/方法执行某个动作或计算
        
        # Data
        p1_x = self.engine.df['player1_joint0_x'].dropna().tolist()  # 将 p1_x 设为一次调用/构造的返回值
        p1_y = self.engine.df['player1_joint0_y'].dropna().tolist()  # 将 p1_y 设为一次调用/构造的返回值
        p1_map.plot(p1_x, p1_y, "球员1移动热区")  # 调用函数/方法执行某个动作或计算
        kpi1.plot(self.engine.get_physical_kpis(1), self.engine.get_player_zone_ratios(1), "球员1 运动表现指标")  # 调用函数/方法执行某个动作或计算
        sp1.plot(self.engine.get_speed_series(1), "球员1 速度分布")  # 调用函数/方法执行某个动作或计算
        ac1.plot(self.engine.get_accel_series(1), "球员1 加速度时间序列")  # 调用函数/方法执行某个动作或计算
        q1.plot(p1_x, p1_y, "球员1 覆盖分位等值线")  # 调用函数/方法执行某个动作或计算
        bc1.plot(p1_x, p1_y, self.engine.get_barycenter_cov(1), "球员1 站位质心与稳定性")  # 调用函数/方法执行某个动作或计算
        
        p2_x = self.engine.df['player2_joint0_x'].dropna().tolist()  # 将 p2_x 设为一次调用/构造的返回值
        p2_y = self.engine.df['player2_joint0_y'].dropna().tolist()  # 将 p2_y 设为一次调用/构造的返回值
        p2_map.plot(p2_x, p2_y, "球员2移动热区")  # 调用函数/方法执行某个动作或计算
        kpi2.plot(self.engine.get_physical_kpis(2), self.engine.get_player_zone_ratios(2), "球员2 运动表现指标")  # 调用函数/方法执行某个动作或计算
        sp2.plot(self.engine.get_speed_series(2), "球员2 速度分布")  # 调用函数/方法执行某个动作或计算
        ac2.plot(self.engine.get_accel_series(2), "球员2 加速度时间序列")  # 调用函数/方法执行某个动作或计算
        q2.plot(p2_x, p2_y, "球员2 覆盖分位等值线")  # 调用函数/方法执行某个动作或计算
        bc2.plot(p2_x, p2_y, self.engine.get_barycenter_cov(2), "球员2 站位质心与稳定性")  # 调用函数/方法执行某个动作或计算
        
        layout.addWidget(p1_grp, 0, 0)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(p2_grp, 0, 1)  # 调用函数/方法执行某个动作或计算

    def _build_tech_page(self):  # 定义函数（封装可复用逻辑）
        layout = self.tech_tab.layout() or QGridLayout(self.tech_tab)  # 将 layout 设为一次调用/构造的返回值
        
        # Shot Types P1
        p1_pie = ShotTypePie()  # 将 p1_pie 设为一次调用/构造的返回值
        p1_pie.plot(self.engine.global_player_stats[1]['type_counts'], "Player 1 击球类型")  # 调用函数/方法执行某个动作或计算
        layout.addWidget(p1_pie, 0, 0)  # 调用函数/方法执行某个动作或计算
        
        # Shot Types P2
        p2_pie = ShotTypePie()  # 将 p2_pie 设为一次调用/构造的返回值
        p2_pie.plot(self.engine.global_player_stats[2]['type_counts'], "Player 2 击球类型")  # 调用函数/方法执行某个动作或计算
        layout.addWidget(p2_pie, 0, 1)  # 调用函数/方法执行某个动作或计算
        
        # Speed vs Height
        scatter = SpeedHeightScatter()  # 将 scatter 设为一次调用/构造的返回值
        scatter.plot(self.engine.df)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(scatter, 1, 0, 1, 2)  # 调用函数/方法执行某个动作或计算

    def _build_deep_page(self):  # 定义函数（封装可复用逻辑）
        layout = self.deep_tab.layout() or QGridLayout(self.deep_tab)  # 将 layout 设为一次调用/构造的返回值
        
        # 1. ThemeRiver Flow
        river = ThemeRiverChart()  # 将 river 设为一次调用/构造的返回值
        river.plot(self.engine.get_theme_river_data(window_sec=2.0, player_id=None), "战术类型流图")  # 调用函数/方法执行某个动作或计算
        layout.addWidget(river, 0, 0, 1, 2)  # 调用函数/方法执行某个动作或计算
        
        # 2. Transition Matrix (P1)
        trans_p1 = TransitionHeatmap()  # 将 trans_p1 设为一次调用/构造的返回值
        trans_p1.plot(self.engine.get_transition_matrix(1))  # 调用函数/方法执行某个动作或计算
        layout.addWidget(trans_p1, 1, 0)  # 调用函数/方法执行某个动作或计算
        
        # 3. Voronoi Space Control (Sample from last frame)
        # Just sample one frame for demo
        voronoi = VoronoiMap()  # 将 voronoi 设为一次调用/构造的返回值
        last_row = self.engine.df.iloc[-100] if len(self.engine.df) > 100 else self.engine.df.iloc[0]  # 将 last_row 设为一次调用/构造的返回值
        p1_pos = (last_row['player1_joint0_x'], last_row['player1_joint0_y'])  # 初始化变量 p1_pos 为一个容器/表达式结果
        p2_pos = (last_row['player2_joint0_x'], last_row['player2_joint0_y'])  # 初始化变量 p2_pos 为一个容器/表达式结果
        voronoi.plot(p1_pos, p2_pos)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(voronoi, 1, 1)  # 调用函数/方法执行某个动作或计算

    def _build_load_page(self):  # 定义函数（封装可复用逻辑）
        layout = self.load_tab.layout() or QGridLayout(self.load_tab)  # 将 layout 设为一次调用/构造的返回值
        
        # 1. Load Chart
        load_chart = LoadChart()  # 将 load_chart 设为一次调用/构造的返回值
        load_chart.plot(self.engine.df)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(load_chart, 0, 0, 1, 2)  # 调用函数/方法执行某个动作或计算
        
        # 2. 3D Impact Analysis
        impact_3d = HitPoint3D()  # 将 impact_3d 设为一次调用/构造的返回值
        impact_3d.plot(self.engine.df)  # 调用函数/方法执行某个动作或计算
        layout.addWidget(impact_3d, 1, 0, 1, 2)  # 调用函数/方法执行某个动作或计算
