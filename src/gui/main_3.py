from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import importlib
from tianshou import *
from tianshou.utils.net.common import Net

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if hasattr(sys, '_MEIPASS'):
    # 如果是打包环境，添加 PyInstaller 的临时路径
    sys.path.append(os.path.abspath(os.path.join(sys._MEIPASS, "src")))
else:
    # 如果是开发环境，添加项目的 src 目录
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.F_test_auto_8paras
from src.gui.utils.file_editor import load_file, update_evaluate_method,load_evaluate_method
import src.sac_pso_env
import src.test_sac_pso
import src.sac_pso_env_watch
import src.F_train_5paras
from src.备份 import plot_test
from src.test_sac_pso import get_args,watch,test_global,train_agent
from src.F_train_5paras import train
from src.F_test_auto_8paras import test
from src.config import global_variables
from src.config2 import global_variables2
from PyQt5.QtWidgets import QTextEdit
import io
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit
def capture_plot_wrapper():
    window.capture_plot()

class OutputRedirect(io.StringIO):
    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit

    def write(self, text):
        # 将输出追加到 QTextEdit 中
        self.text_edit.append(text)

    def flush(self):
        pass  # 重定向时不需要实现


from PyQt5.QtCore import QThread, pyqtSignal

# 修改后的TrainThread类（替换原有实现）
class TrainThread(QThread):  
    result_signal = pyqtSignal(object, object)  
    progress_signal = pyqtSignal(str)  

    def __init__(self, args):  
        super().__init__()  
        self.args = args  
        self._tqdm_original_display = None

    def run(self):
        import tqdm
        from weakref import ref
        
        try:
            # **1. 设置PyTorch单线程防止资源竞争**
            
            # 确保设备参数为 CPU
            self.args.device = 'cpu'
            
            # **2. 备份并替换tqdm显示方法**
            self._tqdm_original_display = tqdm.tqdm.display
            
            def custom_display(tqdm_self, msg=None, pos=None):
                try:
                    # **3. 生成标准进度信息**
                    if msg is None:
                        msg = tqdm_self.format_meter(**tqdm_self.format_dict)
                    
                    # **4. 通过信号发送到主线程**
                    self.progress_signal.emit(msg.strip())
                except Exception as e:
                    print(f"tqdm display error: {e}")
                
                # **5. 保留原始显示功能（控制台输出）**
                # return self._tqdm_original_display(tqdm_self, msg, pos)
            
            tqdm.tqdm.display = custom_display
            
            # **6. 重载模块并执行训练**
            importlib.reload(src.sac_pso_env)
            importlib.reload(src.test_sac_pso)
            result, agent = train_agent(self.args,agents=None)
            # train_agent(self.args)
            self.result_signal.emit(result, agent)
            
        except Exception as e:
            # **7. 详细异常捕获和输出**
            import traceback
            error_msg = f"Training Error:\n{traceback.format_exc()}"
            self.progress_signal.emit(error_msg)
        finally:
            # **8. 必须恢复原始tqdm方法**
            if self._tqdm_original_display:
                tqdm.tqdm.display = self._tqdm_original_display
            # **9. 清理PyTorch缓存**
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class WatchThread(QThread):
    # 定义信号，用于在线程完成后通知主线程
    finished_signal = pyqtSignal()
    

    def __init__(self, args, policy_path):
        super().__init__()
        self.args = args
        self.policy_path = policy_path

    def run(self):
        # 在子线程中运行 watch 函数
                    # **6. 重载模块并执行训练**
        importlib.reload(src.sac_pso_env_watch)
        try:
            watch(self.args, self.policy_path)
        except Exception as e:
            print(f"Error in watch: {e}")
        finally:
            # 发射信号，通知主线程任务完成
            self.finished_signal.emit()

class train_cnn(QThread):
    # 定义信号，用于在线程完成后通知主线程
    finished_signal = pyqtSignal()
    

    def __init__(self):
        super().__init__()

    def run(self):
        # 在子线程中运行 watch 函数
        importlib.reload(src.F_train_5paras)
        try:
            train()
        finally:
            # 发射信号，通知主线程任务完成
            self.finished_signal.emit()

import logging

def setup_logging():
    import sys
    import os
    from PyQt5.QtCore import QStandardPaths

    log_dir = os.path.join("sys_logs", "sac_pso_env")
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件路径
    log_file = os.path.join(log_dir, 'app_debug.log')
   
    # 配置日志记录
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s'
    )

    # 打印日志路径，便于调试
    print(f"Logging to: {log_file}")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        setup_logging()
        self.setWindowTitle("PSO GUI Project")
        self.setGeometry(100, 100, 1772, 850)
        self.args = get_args()
        self.initUI()
        
        # 重定向标准输出到 QTextEdit
        self.output_redirect = OutputRedirect(self.output_text)
        sys.stdout = self.output_redirect
               
        # 初始化子图计数器
        self.current_subplot = 0
        self.current_subplot2 = 0
        self.max_subplots = 12  # 假设画布分为 2x2 网格

        # 重定向 matplotlib 的 show 函数
        plt.show = capture_plot_wrapper

        # self.args = None
        self.agent = None
        self.reault = None

        self.args = get_args()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            dummy_tensor = torch.tensor([1.0], device='cuda')  # 触发 CUDA 初始化
            del dummy_tensor

        
    def initUI(self):
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QGridLayout(self.central_widget)
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        # 创建页面
        self.page1 = self.create_page1()
        self.page2 = self.create_page2()
        
        

        # 将页面添加到 QStackedWidget
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)

        # 默认显示第一个页面
        self.stacked_widget.setCurrentWidget(self.page1)
        self.page =1
    def create_page2(self):

        page = QWidget()
        layout = QtWidgets.QGridLayout(page)  
        label = QLabel("这是页面 2")
        layout.addWidget(label)
        layout.setColumnStretch(0, 1)  # output_text所在列
        layout.setColumnStretch(1, 1)  # output_text所在列
        layout.setColumnStretch(2, 1)  # output_text所在列
        layout.setColumnStretch(3, 1)  # output_text所在列
        layout.setColumnStretch(4, 1)  # output_text所在列
        layout.setColumnStretch(5, 1)  # 按钮/输入框所在列
        # 添加 QTextEdit
        # 添加输出区域

        train_button = QtWidgets.QPushButton("train start", self)
        train_button.clicked.connect(self.train_cnn)
        layout.addWidget(train_button, 0, 5,1,2)

        test_button = QtWidgets.QPushButton("test", self)
        # self.plot_button.clicked.connect(self.plot_curve)
        test_button.clicked.connect(self.cnn_test)
        # self.layout.addWidget(self.test_button)
        layout.addWidget(test_button, 1, 5,1,2)

        clear_button = QtWidgets.QPushButton("Clear Canvas", self)
        clear_button.clicked.connect(self.clear_canvas)
        # self.layout.addWidget(self.clear_button)
        layout.addWidget(clear_button, 2, 5,1,2)

        end_button = QtWidgets.QPushButton("end", self)
        end_button.clicked.connect(self.end)
        layout.addWidget(end_button, 3, 5,1,2)

        # 动态生成输入框
        self.input_fields2 = {}  # 存储变量名和对应的输入框

        for id, (var_name, var_value) in enumerate(global_variables2.items()):
            # 添加标签
            self.a = id // 2 * 2
            b = id % 2
            label = QtWidgets.QLabel(f"{var_name} (default: {var_value}):", self)
            # label = QtWidgets.QLabel(f"{var_name}", self)
            layout.addWidget(label, 4 + self.a, 5+b)

            # 根据参数类型动态生成输入框或多选框
            if isinstance(var_value, bool):  # 如果是布尔值，使用 QCheckBox
                checkbox = QtWidgets.QCheckBox(self)
                checkbox.setChecked(var_value)  # 设置默认值
                checkbox.stateChanged.connect(lambda state, name=var_name: self.on_checkbox_changed1(name, state))
                layout.addWidget(checkbox, 5 + self.a, 5+b)
                self.input_fields2[var_name] = checkbox
            # elif var_name == "choose":  # 如果是模型参数，使用 QComboBox
            #     combo_box = QtWidgets.QComboBox(self)
            #     combo_box.addItem("竖直")  # 添加选项
            #     combo_box.addItem("水平")
            #     combo_box.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed1(name, text))
            #     layout.addWidget(combo_box, 5 + self.a, 5+b)
            #     self.input_fields2[var_name] = combo_box
            elif var_name == "model":  # 如果是模型参数，使用 QComboBox
                self.combo_box2 = QtWidgets.QComboBox(self)
                # 指定模型文件夹路径
                model_dir = os.path.join(os.getcwd(), "model")  # 你可以根据实际情况修改路径
                if os.path.exists(model_dir):
                    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    if pt_files:
                        self.combo_box2.addItems(pt_files)
                    else:
                        self.combo_box2.addItem("无可用模型文件")
                else:
                    self.combo_box2.addItem("模型文件夹不存在")
                self.combo_box2.setCurrentText(str(var_value))
                self.combo_box2.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed1(name, text))
                layout.addWidget(self.combo_box2, 5 + self.a, 5+b)
                self.input_fields2[var_name] = self.combo_box2      
            elif var_name == "val_file":  # 如果是模型参数，使用 QComboBox
                self.combo_box4 = QtWidgets.QComboBox(self)
                # 指定模型文件夹路径
                test_file_dir = os.path.join(os.getcwd(), "val_file")  # 你可以根据实际情况修改路径
                if os.path.exists(test_file_dir):
                    test_files = [f for f in os.listdir(test_file_dir) if f.endswith('.xlsx')]
                    if test_files:
                        self.combo_box4.addItems(test_files)
                    else:
                        self.combo_box4.addItem("无可用测试文件")
                else:
                    self.combo_box4.addItem("测试文件夹不存在")
                self.combo_box4.setCurrentText(str(var_value))
                self.combo_box4.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed1(name, text))
                layout.addWidget(self.combo_box4, 5 + self.a, 5+b)
                self.input_fields2[var_name] = self.combo_box4     
            elif var_name == "train_file":  # 如果是模型参数，使用 QComboBox
                self.combo_box6 = QtWidgets.QComboBox(self)
                # 指定模型文件夹路径
                test_file_dir = os.path.join(os.getcwd(), "train_file")  # 你可以根据实际情况修改路径
                if os.path.exists(test_file_dir):
                    test_files = [f for f in os.listdir(test_file_dir) if f.endswith('.xlsx')]
                    if test_files:
                        self.combo_box6.addItems(test_files)
                    else:
                        self.combo_box6.addItem("无可用测试文件")
                else:
                    self.combo_box6.addItem("测试文件夹不存在")
                self.combo_box6.setCurrentText(str(var_value))
                self.combo_box6.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed1(name, text))
                layout.addWidget(self.combo_box6, 5 + self.a, 5+b)
                self.input_fields2[var_name] = self.combo_box6     
            elif var_name == "test_file":  # 如果是模型参数，使用 QComboBox
                self.combo_box7 = QtWidgets.QComboBox(self)
                # 指定模型文件夹路径
                test_file_dir = os.path.join(os.getcwd(), "test_file")  # 你可以根据实际情况修改路径
                if os.path.exists(test_file_dir):
                    test_files = [f for f in os.listdir(test_file_dir) if f.endswith('.xlsx')]
                    if test_files:
                        self.combo_box7.addItems(test_files)
                    else:
                        self.combo_box7.addItem("无可用测试文件")
                else:
                    self.combo_box7.addItem("测试文件夹不存在")
                self.combo_box7.setCurrentText(str(var_value))
                self.combo_box7.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed1(name, text))
                layout.addWidget(self.combo_box7, 5 + self.a, 5+b)
                self.input_fields2[var_name] = self.combo_box7  
            # elif var_name == "direction":  # 如果是模型参数，使用 QComboBox
            #     combo_box = QtWidgets.QComboBox(self)
            #     combo_box.addItem("0")  # 添加选项
            #     combo_box.addItem("90")
            #     combo_box.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed1(name, text))
            #     layout.addWidget(combo_box, 5 + self.a, 5+b)
            #     self.input_fields2[var_name] = combo_box
            else:  # 默认使用 QLineEdit
                input_field = QtWidgets.QLineEdit(self)
                input_field.setText(str(var_value))  # 设置默认值
                layout.addWidget(input_field, 5 + self.a, 5+b)
                self.input_fields2[var_name] = input_field
        # 添加更新按钮
        self.a = (id+1) // 2 * 2
        b = (id+1)  % 2      
        update_button = QtWidgets.QPushButton("Update Global Variables", self)
        update_button.clicked.connect(self.update_global_variables2)
        # self.layout.addWidget(self.update_button)
        layout.addWidget(update_button, 4 + self.a, 5+b,2,2-b)

        # 显示当前全局变量值
        # self.global_label = QtWidgets.QLabel(f"Current Global Variables: {global_variables}", self)
        # # self.layout.addWidget(self.global_label)
        # self.layout.addWidget(self.global_label, 5+self.a+1, 1,1,7)
        self.output_text2 = QtWidgets.QTextEdit(self)
        self.output_text2.setReadOnly(True)  # 设置为只读
        layout.addWidget(self.output_text2, 0, 0, 6+self.a, 5)

        # # # 添加绘图区域
        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        layout.addWidget(self.canvas2, 6+self.a+2, 0, 5, 10)


        button2 = QPushButton("切换到页面 1")
        button2.clicked.connect(self.show_page1)
        layout.addWidget(button2,6+self.a+8, 0)

        # 1. 新增下拉框和删除按钮
        self.combo_box_del1 = QtWidgets.QComboBox(self)
        file_list = []
        exclude_files = {"90.pth", "shuiping_0.pth","shuzhi_0.pth","val_90.xlsx","val_shuiping_0.xlsx","val_shuzhi_0.xlsx","train_90.xlsx","train_shuiping_0.xlsx","train_shuzhi_0.xlsx","test_90.xlsx","test_shuiping_0.xlsx","test_shuzhi_0.xlsx"}

        # 遍历三个文件夹
        for folder in ["model", "val_file","train_file", "test_file"]:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    # 排除指定文件
                    if f not in exclude_files and os.path.isfile(os.path.join(folder_path, f)):
                        file_list.append(f"{folder}/{f}")

        self.combo_box_del1.addItems(file_list)
        layout.addWidget(self.combo_box_del1)  # 你可以调整行列参数

        self.del_button1 = QtWidgets.QPushButton("删除选中文件", self)
        layout.addWidget(self.del_button1)

        self.del_button1.clicked.connect(self.delete_selected_file1)
        return page

    def create_page1(self):
        """创建页面 1，包含一个按钮和一个文本框"""

        page = QWidget()
        layout = QtWidgets.QGridLayout(page)  
        label = QLabel("这是页面 1")
        layout.addWidget(label)
        layout.setColumnStretch(0, 1)  # output_text所在列
        layout.setColumnStretch(6, 3)  # 按钮/输入框所在列
        # 添加 QTextEdit
        self.text_edit = QtWidgets.QTextEdit(self)
        layout.addWidget(self.text_edit, 0, 4, 4, 2)

        # 创建一个 QLabel 用于显示图片
        image_label = QtWidgets.QLabel(self)
        layout.addWidget(image_label, 0, 0, 4, 2)  # 设置位置和大小
        # 加载图片
        pixmap = QtGui.QPixmap("3.png")  # 替换为你的图片路径
        image_label.setPixmap(pixmap)
        # 调整 QLabel 的大小以适应图片
        # image_label.setScaledContents(True)


        # 添加按钮
        load_button = QtWidgets.QPushButton("Load _evaluate Method", self)
        load_button.clicked.connect(self.load_evaluate_method)
        layout.addWidget(load_button, 0, 2)

        save_button = QtWidgets.QPushButton("Save Changes", self)
        save_button.clicked.connect(self.save_changes)
        layout.addWidget(save_button, 0, 3)

        train_button = QtWidgets.QPushButton("train start", self)
        train_button.clicked.connect(self.train)
        layout.addWidget(train_button, 1, 2)

        test_button = QtWidgets.QPushButton("test", self)
        test_button.clicked.connect(self.test)
        layout.addWidget(test_button, 1, 3)

        clear_button = QtWidgets.QPushButton("Clear Canvas", self)
        clear_button.clicked.connect(self.clear_canvas)
        layout.addWidget(clear_button, 2, 2)

        plot_button = QtWidgets.QPushButton("plot", self)
        plot_button.clicked.connect(self.plot_curve)
        layout.addWidget(plot_button, 3, 2)

        sy_button = QtWidgets.QPushButton("sy", self)
        sy_button.clicked.connect(self.sy)
        layout.addWidget(sy_button, 2, 3)

        end_button = QtWidgets.QPushButton("end", self)
        end_button.clicked.connect(self.end)
        layout.addWidget(end_button, 3, 3)

        # 动态生成输入框
        self.input_fields = {}  # 存储变量名和对应的输入框

        for id, (var_name, var_value) in enumerate(global_variables.items()):
            # 添加标签
            self.a = id // 6 * 2
            b = id % 6
            label = QtWidgets.QLabel(f"{var_name} (default: {var_value}):", self)
            layout.addWidget(label, 4 + self.a, b)

            # 根据参数类型动态生成输入框或多选框
            if isinstance(var_value, bool):  # 如果是布尔值，使用 QCheckBox
                checkbox = QtWidgets.QCheckBox(self)
                checkbox.setChecked(var_value)  # 设置默认值
                checkbox.stateChanged.connect(lambda state, name=var_name: self.on_checkbox_changed(name, state))
                layout.addWidget(checkbox, 5 + self.a, b)
                self.input_fields[var_name] = checkbox
            # elif var_name == "choose":  # 如果是模型参数，使用 QComboBox
            #     combo_box = QtWidgets.QComboBox(self)
            #     combo_box.addItem("竖直")  # 添加选项
            #     combo_box.addItem("水平")
            #     combo_box.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed(name, text))
            #     layout.addWidget(combo_box, 5 + self.a, b)
            #     self.input_fields[var_name] = combo_box
            elif var_name == "model":  # 如果是模型参数，使用 QComboBox
                self.combo_box = QtWidgets.QComboBox(self)
                # 指定模型文件夹路径
                
                model_dir = os.path.join(os.getcwd(), "model")  # 你可以根据实际情况修改路径
                if os.path.exists(model_dir):
                    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    if pt_files:
                        self.combo_box.addItems(pt_files)
                    else:
                        self.combo_box.addItem("无可用模型文件")
                else:
                    self.combo_box.addItem("模型文件夹不存在")
                self.combo_box.setCurrentText(str(var_value))
                self.combo_box.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed(name, text))
                layout.addWidget(self.combo_box, 5 + self.a, b)
                self.input_fields[var_name] = self.combo_box
            elif var_name == "model1":  # 如果是模型参数，使用 QComboBox
                self.combo_box3 = QtWidgets.QComboBox(self)
                # 指定模型文件夹路径
                model_dir = os.path.join(os.getcwd(), "model")  # 你可以根据实际情况修改路径
                if os.path.exists(model_dir):
                    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    if pt_files:
                        self.combo_box3.addItems(pt_files)
                    else:
                        self.combo_box3.addItem("无可用模型文件")
                else:
                    self.combo_box3.addItem("模型文件夹不存在")
                self.combo_box3.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed(name, text))
                self.combo_box3.setCurrentText(str(var_value))
                layout.addWidget(self.combo_box3, 5 + self.a, b)
                self.input_fields[var_name] = self.combo_box3
            elif var_name == "sac_model":  # 如果是模型参数，使用 QComboBox
                self.combo_box5 = QtWidgets.QComboBox(self)
                sac_model_dir = os.path.join(os.getcwd(), "sac_model")  # 你可以根据实际情况修改路径
                sac_model_path = os.path.join(self.args.sac_model_dir, "sac")
                import glob
                pattern = f"policy_{self.args.n_pistons}_*.pth"
                search_path = os.path.join(sac_model_path, pattern)
                matched_files = glob.glob(search_path)
                # 指定模型文件夹路径
                if os.path.exists(sac_model_dir):
                    if matched_files:
                        # 只添加文件名，不带路径
                        file_names = [os.path.basename(f) for f in matched_files]
                        self.combo_box5.addItems(file_names)
                    else:
                        self.combo_box5.addItem("无可用模型文件")
                else:
                    self.combo_box5.addItem("模型文件夹不存在")
                self.combo_box5.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed(name, text))
                # self.combo_box5.setCurrentText(str(var_value))
                layout.addWidget(self.combo_box5, 5 + self.a, b)
                self.input_fields[var_name] = self.combo_box5

            else:  # 默认使用 QLineEdit
                input_field = QtWidgets.QLineEdit(self)
                input_field.setText(str(var_value))  # 设置默认值
                layout.addWidget(input_field, 5 + self.a, b)
                self.input_fields[var_name] = input_field
        # 添加更新按钮
        self.a = (id+1) // 6 * 2
        b = (id+1)  % 6       
        update_button = QtWidgets.QPushButton("Update Global Variables", self)
        update_button.clicked.connect(self.update_global_variables)
        layout.addWidget(update_button, 4 + self.a, b,2,6-b)

        # 显示当前全局变量值
        # self.global_label = QtWidgets.QLabel(f"Current Global Variables: {global_variables}", self)
        # # self.layout.addWidget(self.global_label)
        # self.layout.addWidget(self.global_label, 5+self.a+1, 1,1,7)

        # 添加输出区域
        self.output_text = QtWidgets.QTextEdit(self)
        self.output_text.setReadOnly(True)  # 设置为只读
        layout.addWidget(self.output_text, 0, 6, 5+self.a+1, 4)

        # 添加绘图区域
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 5+self.a+2, 0, 5, 10)

        button = QPushButton("切换到页面 2")
        button.clicked.connect(self.show_page2)
        layout.addWidget(button)



        # 1. 新增下拉框和删除按钮
        self.combo_box_del = QtWidgets.QComboBox(self)
        file_list = []
        exclude_files = {"90.pth", "shuiping_0.pth","shuzhi_0.pth","val_90.xlsx","val_shuiping_0.xlsx","val_shuzhi_0.xlsx","train_90.xlsx","train_shuiping_0.xlsx","train_shuzhi_0.xlsx","test_90.xlsx","test_shuiping_0.xlsx","test_shuzhi_0.xlsx"}

        # 遍历三个文件夹
        for folder in ["model", "sac_model/sac"]:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    # 排除指定文件
                    if f not in exclude_files and os.path.isfile(os.path.join(folder_path, f)):
                        file_list.append(f"{folder}/{f}")

        self.combo_box_del.addItems(file_list)
        layout.addWidget(self.combo_box_del)  # 你可以调整行列参数

        self.del_button = QtWidgets.QPushButton("删除选中文件", self)
        layout.addWidget(self.del_button)

        self.del_button.clicked.connect(self.delete_selected_file)

        return page


    def delete_selected_file(self):
        selected = self.combo_box_del.currentText()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择要删除的文件")
            return
        folder, filename = selected.split("/", 1)
        file_path = os.path.join(os.getcwd(), folder, filename)
        reply = QtWidgets.QMessageBox.question(
            self, "确认删除", f"Are you sure remove {filename}?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                os.remove(file_path)
                QtWidgets.QMessageBox.information(self, "删除成功", f"{filename} 已被删除")
                self.combo_box_del.removeItem(self.combo_box_del.currentIndex())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "删除失败", f"删除失败: {e}")

        model_dir = os.path.join(os.getcwd(), "model")  # 你可以根据实际情况修改路径
        # 先保存当前下拉框的选中内容
        combo_box_text = self.combo_box.currentText()
        combo_box_text2 = self.combo_box2.currentText()
        combo_box_text3 = self.combo_box3.currentText()

        if os.path.exists(model_dir):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            # 先清空原有内容
            self.combo_box2.clear()
            self.combo_box.clear()
            self.combo_box3.clear()
            if pt_files:
                self.combo_box3.addItems(pt_files)
                self.combo_box2.addItems(pt_files)
                self.combo_box.addItems(pt_files)
            else:
                self.combo_box3.addItem("无可用模型文件")
                self.combo_box2.addItem("无可用模型文件")
                self.combo_box.addItem("无可用模型文件")

        # 恢复原来的选中项（如果还在新列表中）
        if combo_box_text in [self.combo_box.itemText(i) for i in range(self.combo_box.count())]:
            self.combo_box.setCurrentText(combo_box_text)
        else:
            self.combo_box.setCurrentIndex(0)

        if combo_box_text2 in [self.combo_box2.itemText(i) for i in range(self.combo_box2.count())]:
            self.combo_box2.setCurrentText(combo_box_text2)
        else:
            self.combo_box2.setCurrentIndex(0)

        if combo_box_text3 in [self.combo_box3.itemText(i) for i in range(self.combo_box3.count())]:
            self.combo_box3.setCurrentText(combo_box_text3)
        else:
            self.combo_box3.setCurrentIndex(0)

        val_dir = os.path.join(os.getcwd(), "val_file")  # 你可以根据实际情况修改路径
        if os.path.exists(val_dir):
            val_files = [f for f in os.listdir(val_dir) if f.endswith('.xlsx')]
            # 先清空原有内容
            self.combo_box4.clear()
            if val_files:
                self.combo_box4.addItems(val_files)

        sac_model_dir = os.path.join(os.getcwd(), "sac_model")  # 你可以根据实际情况修改路径
        sac_model_path = os.path.join(self.args.sac_model_dir, "sac")
        import glob
        pattern = f"policy_{self.args.n_pistons}_*.pth"
        search_path = os.path.join(sac_model_path, pattern)
        matched_files = glob.glob(search_path)
        # 指定模型文件夹路径
        self.combo_box5.clear()
        if os.path.exists(sac_model_dir):
            if matched_files:
                # 只添加文件名，不带路径
                file_names = [os.path.basename(f) for f in matched_files]
                self.combo_box5.addItems(file_names)
            else:
                self.combo_box5.addItem("无可用模型文件")
        else:
            self.combo_box5.addItem("模型文件夹不存在")


    def delete_selected_file1(self):
        selected = self.combo_box_del1.currentText()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择要删除的文件")
            return
        folder, filename = selected.split("/", 1)
        file_path = os.path.join(os.getcwd(), folder, filename)
        reply = QtWidgets.QMessageBox.question(
            self, "确认删除", f"Are you sure remove {filename}?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                os.remove(file_path)
                QtWidgets.QMessageBox.information(self, "删除成功", f"{filename} 已被删除")
                self.combo_box_del1.removeItem(self.combo_box_del1.currentIndex())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "删除失败", f"删除失败: {e}")

        model_dir = os.path.join(os.getcwd(), "model")  # 你可以根据实际情况修改路径
        # 先保存当前下拉框的选中内容
        combo_box_text = self.combo_box.currentText()
        combo_box_text2 = self.combo_box2.currentText()
        combo_box_text3 = self.combo_box3.currentText()
        if os.path.exists(model_dir):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            # 先清空原有内容
            self.combo_box2.clear()
            self.combo_box.clear()
            self.combo_box3.clear()
            if pt_files:
                self.combo_box3.addItems(pt_files)
                self.combo_box2.addItems(pt_files)
                self.combo_box.addItems(pt_files)
            else:
                self.combo_box3.addItem("无可用模型文件")
                self.combo_box2.addItem("无可用模型文件")
                self.combo_box.addItem("无可用模型文件")

        # 恢复原来的选中项（如果还在新列表中）
        if combo_box_text in [self.combo_box.itemText(i) for i in range(self.combo_box.count())]:
            self.combo_box.setCurrentText(combo_box_text)
        else:
            self.combo_box.setCurrentIndex(0)

        if combo_box_text2 in [self.combo_box2.itemText(i) for i in range(self.combo_box2.count())]:
            self.combo_box2.setCurrentText(combo_box_text2)
        else:
            self.combo_box2.setCurrentIndex(0)

        if combo_box_text3 in [self.combo_box3.itemText(i) for i in range(self.combo_box3.count())]:
            self.combo_box3.setCurrentText(combo_box_text3)
        else:
            self.combo_box3.setCurrentIndex(0)

        val_dir = os.path.join(os.getcwd(), "val_file")  # 你可以根据实际情况修改路径
        if os.path.exists(val_dir):
            val_files = [f for f in os.listdir(val_dir) if f.endswith('.xlsx')]
            # 先清空原有内容
            self.combo_box4.clear()
            if val_files:
                self.combo_box4.addItems(val_files)

        train_dir = os.path.join(os.getcwd(), "train_file")  # 你可以根据实际情况修改路径
        if os.path.exists(train_dir):
            train_files = [f for f in os.listdir(train_dir) if f.endswith('.xlsx')]
            # 先清空原有内容
            self.combo_box6.clear()
            if train_files:
                self.combo_box6.addItems(train_files)

        test_dir = os.path.join(os.getcwd(), "test_file")  # 你可以根据实际情况修改路径
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) if f.endswith('.xlsx')]
            # 先清空原有内容
            self.combo_box7.clear()
            if test_files:
                self.combo_box7.addItems(test_files)

        sac_model_dir = os.path.join(os.getcwd(), "sac_model")  # 你可以根据实际情况修改路径
        sac_model_path = os.path.join(self.args.sac_model_dir, "sac")
        import glob
        pattern = f"policy_{self.args.n_pistons}_*.pth"
        search_path = os.path.join(sac_model_path, pattern)
        matched_files = glob.glob(search_path)
        # 指定模型文件夹路径
        self.combo_box5.clear()
        if os.path.exists(sac_model_dir):
            if matched_files:
                # 只添加文件名，不带路径
                file_names = [os.path.basename(f) for f in matched_files]
                self.combo_box5.addItems(file_names)
            else:
                self.combo_box5.addItem("无可用模型文件")
        else:
            self.combo_box5.addItem("模型文件夹不存在")


    def show_page1(self):
        """切换到页面 1"""
        self.output_redirect = OutputRedirect(self.output_text)
        sys.stdout = self.output_redirect
        self.page =1
        self.stacked_widget.setCurrentWidget(self.page1)

    def show_page2(self):
        """切换到页面 2"""
        self.output_redirect = OutputRedirect(self.output_text2)
        sys.stdout = self.output_redirect
        self.page =2
        self.stacked_widget.setCurrentWidget(self.page2)

    def on_checkbox_changed(self, var_name, state):
        # 将多选框的状态更新到 input_fields
        self.input_fields[var_name] = state == QtCore.Qt.Checked
        print(f"Checkbox {var_name} changed to: {self.input_fields[var_name]}")
    def on_combo_box_changed(self, var_name, text):
        """
        处理 QComboBox 的选项变化。
        """
        # 更新 input_fields 中的值
        self.input_fields[var_name] = text
        global global_variables
        global_variables[var_name] = text  # 同步更新 global_variables
        # print(f"ComboBox {var_name} changed to: {text}")
    def on_checkbox_changed1(self, var_name, state):
        # 将多选框的状态更新到 input_fields
        self.input_fields2[var_name] = state == QtCore.Qt.Checked
        print(f"Checkbox {var_name} changed to: {self.input_fields2[var_name]}")
    def on_combo_box_changed1(self, var_name, text):
        """
        处理 QComboBox 的选项变化。
        """
        # 更新 input_fields 中的值
        self.input_fields2[var_name] = text
        global global_variables2
        global_variables2[var_name] = text  # 同步更新 global_variables
        # print(f"ComboBox {var_name} changed to: {text}")

    def load_evaluate_method(self):
        import sys
        import os
        # 动态获取文件路径
        if hasattr(sys, '_MEIPASS'):
            # 如果是打包环境，从临时目录加载文件
            file_path = os.path.join(sys._MEIPASS, 'src', 'sac_pso_env.py')            
        else:
            # 如果是开发环境，使用相对路径
            file_path = 'src/sac_pso_env.py'           
        content = load_evaluate_method(file_path)
        self.text_edit.setPlainText(content)
        
    def save_changes(self):
        new_content = self.text_edit.toPlainText()
        if hasattr(sys, '_MEIPASS'):
            # 如果是打包环境，从临时目录加载文件
            file_path = os.path.join(sys._MEIPASS, 'src', 'sac_pso_env.py')
            file_path_watch = os.path.join(sys._MEIPASS, 'src', 'sac_pso_env_watch.py')
        else:
            # 如果是开发环境，使用相对路径
            file_path = 'src/sac_pso_env.py'
            file_path_watch = 'src/sac_pso_env_watch.py'
        update_evaluate_method(file_path, new_content)
        update_evaluate_method(file_path_watch, new_content)
        importlib.reload(src.sac_pso_env)
        importlib.reload(src.sac_pso_env_watch)
        print("Changes saved and module reloaded.")
        # importlib.reload(src.test_sac_pso)
        # update_evaluate_method('src/plot_test.py', new_content)

    def plot_curve(self):
        importlib.reload(plot_test)
        pso = plot_test.PSO()
        positions = np.array([1, 2, 3, 4, 5])
        scores = pso._evaluate(positions)

        # 使用 matplotlib 绘图
        plt.figure()
        plt.plot(positions, scores, 'ro--', label='Example Curve')
        plt.xlabel('Position')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Generated by plot_curve')
        plt.show()  # 调用重定向后的 show 函数

    def sy(self):
        test_global()
        
    def train(self):
        # if self.args is None:
        self.args = get_args()
        # 创建线程实例
        self.train_thread = TrainThread(self.args)
        # 连接线程的信号到主线程的槽函数
        self.train_thread.result_signal.connect(self.on_train_finished)
        self.train_thread.progress_signal.connect(self.update_progress)  # 连接进度信号
        # 启动线程
        self.train_thread.start()
        print("Training started in a separate thread.")

    def on_train_finished(self, result, agent):
        # self.result = result
        # self.agent = agent
        sac_model_dir = os.path.join(os.getcwd(), "sac_model")  # 你可以根据实际情况修改路径
        sac_model_path = os.path.join(self.args.sac_model_dir, "sac")
        import glob
        pattern = f"policy_{self.args.n_pistons}_*.pth"
        search_path = os.path.join(sac_model_path, pattern)
        matched_files = glob.glob(search_path)
        # 指定模型文件夹路径
        self.combo_box5.clear()
        if os.path.exists(sac_model_dir):
            if matched_files:
                # 只添加文件名，不带路径
                file_names = [os.path.basename(f) for f in matched_files]
                self.combo_box5.addItems(file_names)
            else:
                self.combo_box5.addItem("无可用模型文件")
        else:
            self.combo_box5.addItem("模型文件夹不存在")

        file_list = []
        exclude_files = {"90.pth", "shuiping_0.pth","shuzhi_0.pth","val_90.xlsx","val_shuiping_0.xlsx","val_shuzhi_0.xlsx"}
        self.combo_box_del.clear()
        # 遍历三个文件夹
        for folder in ["model", "sac_model/sac"]:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    # 排除指定文件
                    if f not in exclude_files and os.path.isfile(os.path.join(folder_path, f)):
                        file_list.append(f"{folder}/{f}")

        self.combo_box_del.addItems(file_list)
        print("Training finished.")

    def update_progress(self, message):
        # 使用定时器延迟更新防止界面冻结
        QtCore.QTimer.singleShot(0, lambda: 
            self.output_text.append(message)
    )
       
    def train_cnn(self):
        # train()
        self.train_cnn_thread = train_cnn()
        # 连接线程的信号到主线程的槽函数
        self.train_cnn_thread.finished_signal.connect(self.cnn_train_finished)
        # 启动线程
        self.train_cnn_thread.start()
        print("CNN train started in a separate thread.")

    def cnn_train_finished(self):
        model_dir = os.path.join(os.getcwd(), "model")  # 你可以根据实际情况修改路径
        # 先保存当前下拉框的选中内容
        combo_box_text = self.combo_box.currentText()
        combo_box_text2 = self.combo_box2.currentText()
        combo_box_text3 = self.combo_box3.currentText()

        if os.path.exists(model_dir):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            # 先清空原有内容
            self.combo_box2.clear()
            self.combo_box.clear()
            self.combo_box3.clear()
            if pt_files:
                self.combo_box3.addItems(pt_files)
                self.combo_box2.addItems(pt_files)
                self.combo_box.addItems(pt_files)
            else:
                self.combo_box3.addItem("无可用模型文件")
                self.combo_box2.addItem("无可用模型文件")
                self.combo_box.addItem("无可用模型文件")

        # 恢复原来的选中项（如果还在新列表中）
        if combo_box_text in [self.combo_box.itemText(i) for i in range(self.combo_box.count())]:
            self.combo_box.setCurrentText(combo_box_text)
        else:
            self.combo_box.setCurrentIndex(0)

        if combo_box_text2 in [self.combo_box2.itemText(i) for i in range(self.combo_box2.count())]:
            self.combo_box2.setCurrentText(combo_box_text2)
        else:
            self.combo_box2.setCurrentIndex(0)

        if combo_box_text3 in [self.combo_box3.itemText(i) for i in range(self.combo_box3.count())]:
            self.combo_box3.setCurrentText(combo_box_text3)
        else:
            self.combo_box3.setCurrentIndex(0)


        self.combo_box_del.clear()
        self.combo_box_del1.clear()
        file_list = []
        exclude_files = {"90.pth", "shuiping_0.pth","shuzhi_0.pth","val_90.xlsx","val_shuiping_0.xlsx","val_shuzhi_0.xlsx","train_90.xlsx","train_shuiping_0.xlsx","train_shuzhi_0.xlsx","test_90.xlsx","test_shuiping_0.xlsx","test_shuzhi_0.xlsx"}

        # 遍历三个文件夹
        for folder in ["model", "sac_model/sac"]:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    # 排除指定文件
                    if f not in exclude_files and os.path.isfile(os.path.join(folder_path, f)):
                        file_list.append(f"{folder}/{f}")

        self.combo_box_del.addItems(file_list)


        file_list = []
        exclude_files = {"90.pth", "shuiping_0.pth","shuzhi_0.pth","val_90.xlsx","val_shuiping_0.xlsx","val_shuzhi_0.xlsx","train_90.xlsx","train_shuiping_0.xlsx","train_shuzhi_0.xlsx","test_90.xlsx","test_shuiping_0.xlsx","test_shuzhi_0.xlsx"}

        # 遍历三个文件夹
        for folder in ["model", "val_file","train_file", "test_file"]:
            folder_path = os.path.join(os.getcwd(), folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    # 排除指定文件
                    if f not in exclude_files and os.path.isfile(os.path.join(folder_path, f)):
                        file_list.append(f"{folder}/{f}")

        self.combo_box_del1.addItems(file_list)
        print("cnn_train finished.")
  
    def test(self):
        global global_variables
        self.args = get_args()
        # log_path = os.path.join(self.args.logdir, "sac_pso_env", "sac")
        # sac_model_path = os.path.join(self.args.sac_model_dir, "sac")

        # import glob

        # pattern = f"policy_{self.args.n_pistons}_*.pth"
        # search_path = os.path.join(sac_model_path, pattern)
        # matched_files = glob.glob(search_path)

        # if matched_files:
        #     policy_path = matched_files[0].split("\\")[-1]  # 取第一个匹配的文件
        # else:
        #     print("Please train the agent before testing.")
            # return

        # 后续可以用 policy_path
        # policy_path = os.path.join(sac_model_path, "policy"+"_"+str(self.args.n_pistons)+".pth")

        # if not os.path.exists(policy_path):
        #     print("Please train the agent before testing.")
        #     return
        # 创建线程实例
        self.watch_thread = WatchThread(self.args, global_variables["sac_model"])

        # 连接线程的信号到主线程的槽函数
        self.watch_thread.finished_signal.connect(self.on_watch_finished)

        # 启动线程
        self.watch_thread.start()

        print("Testing started in a separate thread.")

    def on_watch_finished(self):
        print("Testing finished.")

    # def train(self):
    #     self.args = get_args()
    #     # self.result, self.agent = train_agent(self.args)
    #     train_agent(self.args)
    # def test(self):
    #     if self.args is None or self.agent is None:
    #         print("Please train the agent before testing.")
    #         return
    #     watch(self.args, self.agent)

    def cnn_test(self):
        importlib.reload(src.F_test_auto_8paras)
        test()

    def end(self):
        if hasattr(self, "train_thread") and self.train_thread.isRunning():
            
            self.train_thread.terminate()
            self.train_thread.wait()  # 等待线程完全终止
            print("stop TrainThread ")
        elif hasattr(self, "watch_thread") and self.watch_thread.isRunning():
            
            self.watch_thread.terminate()
            self.watch_thread.wait()  # 等待线程完全终止
            print("stop watch_thread ")
        elif hasattr(self, "train_cnn_thread") and self.train_cnn_thread.isRunning():
            
            self.train_cnn_thread.terminate()
            self.train_cnn_thread.wait()  # 等待线程完全终止
            print("stop train_cnn_thread ")
        else:
            print("No Thread is  running.")

    def capture_plot(self):
        """
        捕获 matplotlib 的所有图像并嵌入到 PyQt 的画布中。
        """
        print("Capturing plot...")

        for num in plt.get_fignums():
            current_figure = plt.figure(num)
            if self.current_subplot >= self.max_subplots or self.current_subplot2 >= self.max_subplots:
                self.clear_canvas()
            # 计算子图位置
            rows, cols = 3, 4  # 3x4 网格
            if self.page == 1:
                self.current_subplot += 1
                ax = self.figure.add_subplot(rows, cols, self.current_subplot)
            elif self.page == 2:
                self.current_subplot2 += 1
                ax = self.figure2.add_subplot(rows, cols, self.current_subplot2)

            # 绘制折线图
            for line in current_figure.axes[0].lines:
                ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

            # 绘制柱状图
            for bar in current_figure.axes[0].patches:
                x = bar.get_x()
                width = bar.get_width()
                height = bar.get_height()
                ax.bar(x, height, width, label=bar.get_label(), color=bar.get_facecolor())

            # 绘制散点图
            for collection in current_figure.axes[0].collections:
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    x = offsets[:, 0]
                    y = offsets[:, 1]
                    # 获取颜色和标签
                    color = collection.get_facecolor()
                    label = collection.get_label()
                    # 处理颜色（有可能是多色）
                    if color is not None and len(color) > 0:
                        ax.scatter(x, y, color=color[0], label=label)
                    else:
                        ax.scatter(x, y, label=label)

            ax.set_xlabel(current_figure.axes[0].get_xlabel())
            ax.set_ylabel(current_figure.axes[0].get_ylabel())
            ax.set_title(current_figure.axes[0].get_title())
            ax.legend()
            if self.page == 1:
                self.canvas.draw()
            elif self.page == 2:
                self.canvas2.draw()
            plt.close(current_figure)

    def update_global_variables(self):
        global global_variables
        # 遍历所有输入框并更新全局变量
        for var_name, input_field in self.input_fields.items():
            if isinstance(input_field, QtWidgets.QLineEdit):  # 如果是文本框
                global_variables[var_name] = input_field.text()
            # elif isinstance(input_field, QtWidgets.QComboBox):  # 如果是下拉框
            #     global_variables[var_name] = input_field.currentText()
            elif isinstance(input_field, QtWidgets.QCheckBox):  # 如果是多选框
                global_variables[var_name] = input_field.isChecked()
        importlib.reload(src.test_sac_pso)
        print(f"Global variables updated to: {global_variables}")
        self.combo_box5.clear()
        sac_model_dir = os.path.join(os.getcwd(), "sac_model")  # 你可以根据实际情况修改路径
        sac_model_path = os.path.join(self.args.sac_model_dir, "sac")
        import glob
        pattern = f"policy_{global_variables['n_pistons']}_*.pth"
        search_path = os.path.join(sac_model_path, pattern)
        matched_files = glob.glob(search_path)
        # 指定模型文件夹路径
        if os.path.exists(sac_model_dir):
            if matched_files:
                # 只添加文件名，不带路径
                file_names = [os.path.basename(f) for f in matched_files]
                self.combo_box5.addItems(file_names)
            else:
                self.combo_box5.addItem("无可用模型文件")
        else:
            self.combo_box5.addItem("模型文件夹不存在")
            
    def update_global_variables2(self):
        global global_variables2
        # 遍历所有输入框并更新全局变量
        for var_name, input_field in self.input_fields2.items():
            if isinstance(input_field, QtWidgets.QLineEdit):  # 如果是文本框
                global_variables2[var_name] = input_field.text()
            elif isinstance(input_field, QtWidgets.QCheckBox):  # 如果是多选框
                global_variables2[var_name] = input_field.isChecked()
        importlib.reload(src.F_train_5paras)
        importlib.reload(src.F_test_auto_8paras)
        print(f"Global variables updated to: {global_variables2}")

    def clear_canvas(self):
        print("Clearing canvas...")
        if (self.page == 1):
            self.figure.clear()
            self.current_subplot = 0
            self.canvas.draw()
        else:
            self.figure2.clear()
            self.current_subplot2 = 0
            self.canvas2.draw()

    def closeEvent(self, event):
        # 停止所有线程
        self.end()  
        # 等待线程结束
        if hasattr(self, "train_thread"):
            self.train_thread.wait(2000)
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app = QtWidgets.QApplication(sys.argv)
    # args = get_args()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
