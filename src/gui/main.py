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

from src.gui.utils.file_editor import load_file, update_evaluate_method,load_evaluate_method
import src.sac_pso_env
import src.test_sac_pso
from F_model_8paras import F_Net_1D
from src import plot_test
from src.test_sac_pso import get_args,watch,test_global,train_agent
from src.config import global_variables
from PyQt5.QtWidgets import QTextEdit
import io
import torch

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

# class TrainThread(QThread):  
#     result_signal = pyqtSignal(object, object)  
#     progress_signal = pyqtSignal(str)  
    
#     def __init__(self, args):  
#         super().__init__()  
#         self.args = args  
#         # self._tqdm_original_display = None

#     def run(self): 

#         # from weakref import ref
        
#         try:

#             train_agent(self.args)
#         except Exception as e:
#             # **7. 详细异常捕获和输出**
#             import traceback
#             error_msg = f"Training Error:\n{traceback.format_exc()}"
#             self.progress_signal.emit(error_msg)
# import torch.multiprocessing as mp

# class TrainProcess(mp.Process):
#     def __init__(self, args, result_queue, progress_queue):
#         super().__init__()
#         self.args = args
#         self.result_queue = result_queue
#         self.progress_queue = progress_queue
        
#     def run(self):
#         try:
#             # 必须在子进程中设置
#             torch.manual_seed(42)
#             torch.set_num_threads(1)
#             self.args.device = 'cpu'
            
#             # 训练代码
#             result, agent = train_agent(self.args)
            
#             # 将结果放入队列
#             self.result_queue.put((result, agent))
#         except Exception as e:
#             self.progress_queue.put(f"Error: {str(e)}")
class TrainThread(QThread):  
    result_signal = pyqtSignal(object, object)  
    progress_signal = pyqtSignal(str)  
    
    def __init__(self, args):  
        super().__init__()  
        self.args = args
        # 强制使用 CPU
        self.args.device = 'cpu'
        
    def run(self): 
        try:
            # 设置线程局部随机种子
            # torch.manual_seed(42)
            # np.random.seed(42)
            
            # # 确保使用 CPU
            # torch.set_num_threads(1)
            # with torch.no_grad():
            #     train_agent(self.args)

            result, agent = train_agent(self.args)
            self.result_signal.emit(result, agent)
        except Exception as e:
            import traceback
            error_msg = f"Training Error:\n{traceback.format_exc()}"
            self.progress_signal.emit(error_msg)

# def train_agent(args):
#     # 确保设备为 CPU

#     args.device = 'cpu' 
#     # for _ in range(args.n_pistons):
#     try:
#         # net = Net(
#         #     state_shape=(5, 3),
#         #     action_shape=(3, 2),
#         #     hidden_sizes=args.hidden_sizes,
#         #     device=args.device,
#         # ).to(args.device)
#         net = F_Net_1D()
#         print("ok")
#         # 示例：执行一次前向传播验证模型
#         # dummy_input = torch.randn(2,1, 5)
#         # aaa = net(dummy_input)
#         # print(aaa)
#     except RuntimeError as e:
#         print(f"模型创建失败: {e}")
#         raise
#     finally:
#         del net  # 显式释放内存
#         torch.cuda.empty_cache()

class WatchThread(QThread):
    # 定义信号，用于在线程完成后通知主线程
    finished_signal = pyqtSignal()
    

    def __init__(self, args, agent):
        super().__init__()
        self.args = args
        self.agent = agent

    def run(self):
        # 在子线程中运行 watch 函数
        try:
            watch(self.args, self.agent)
        except Exception as e:
            print(f"Error in watch: {e}")
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
        # setup_logging()
        self.setWindowTitle("PSO GUI Project")
        self.setGeometry(100, 100, 1300, 850)

        self.initUI()
        
        # # 重定向标准输出到 QTextEdit
        # self.output_redirect = OutputRedirect(self.output_text)
        # sys.stdout = self.output_redirect

        # 初始化子图计数器
        self.current_subplot = 0
        self.max_subplots = 12  # 假设画布分为 2x2 网格

        # 重定向 matplotlib 的 show 函数
        plt.show = capture_plot_wrapper

        self.args = None
        self.agent = None
        self.reault = None
        # if hasattr(torch, 'cuda') and torch.cuda.is_available():
        #     dummy_tensor = torch.tensor([1.0], device='cuda')  # 触发 CUDA 初始化
        #     del dummy_tensor

    def initUI(self):
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QGridLayout(self.central_widget)
        # 添加 QTextEdit
        self.text_edit = QtWidgets.QTextEdit(self)
        self.layout.addWidget(self.text_edit, 0, 0, 4, 2)

        
        # 添加按钮
        self.load_button = QtWidgets.QPushButton("Load _evaluate Method", self)
        self.load_button.clicked.connect(self.load_evaluate_method)
        self.layout.addWidget(self.load_button, 0, 2)

        self.save_button = QtWidgets.QPushButton("Save Changes", self)
        self.save_button.clicked.connect(self.save_changes)
        self.layout.addWidget(self.save_button, 0, 3)

        self.train_button = QtWidgets.QPushButton("train start", self)
        self.train_button.clicked.connect(self.train)
        self.layout.addWidget(self.train_button, 1, 2)

        self.test_button = QtWidgets.QPushButton("test", self)
        self.test_button.clicked.connect(self.test)
        self.layout.addWidget(self.test_button, 1, 3)

        self.clear_button = QtWidgets.QPushButton("Clear Canvas", self)
        self.clear_button.clicked.connect(self.clear_canvas)
        self.layout.addWidget(self.clear_button, 2, 2)

        self.plot_button = QtWidgets.QPushButton("plot", self)
        self.plot_button.clicked.connect(self.plot_curve)
        self.layout.addWidget(self.plot_button, 2, 3)

        self.sy_button = QtWidgets.QPushButton("sy", self)
        self.sy_button.clicked.connect(self.sy)
        self.layout.addWidget(self.sy_button, 3, 2)

        self.end_button = QtWidgets.QPushButton("end", self)
        self.end_button.clicked.connect(self.end)
        self.layout.addWidget(self.end_button, 3, 3)

        self.input_fields = {}  # 存储变量名和对应的输入框

        # 保存到字典中
        for id, (var_name, var_value) in enumerate(global_variables.items()):
            # 添加标签
            self.a = id // 4 * 2
            b = id % 4
            label = QtWidgets.QLabel(f"{var_name} (default: {var_value}):", self)
            self.layout.addWidget(label, 4 + self.a, b)

            # 根据参数类型动态生成输入框或多选框
            if isinstance(var_value, bool):  # 如果是布尔值，使用 QCheckBox
                checkbox = QtWidgets.QCheckBox(self)
                checkbox.setChecked(var_value)  # 设置默认值
                checkbox.stateChanged.connect(lambda state, name=var_name: self.on_checkbox_changed(name, state))
                self.layout.addWidget(checkbox, 5 + self.a, b)
                self.input_fields[var_name] = checkbox
            elif var_name == "choose":  # 如果是模型参数，使用 QComboBox
                self.combo_box = QtWidgets.QComboBox(self)
                self.combo_box.addItem("竖直")  # 添加选项
                self.combo_box.addItem("水平")
                self.combo_box.currentTextChanged.connect(lambda text, name=var_name: self.on_combo_box_changed(name, text))
                self.layout.addWidget(self.combo_box, 5 + self.a, b)
                self.input_fields[var_name] = self.combo_box
            else:  # 默认使用 QLineEdit
                input_field = QtWidgets.QLineEdit(self)
                input_field.setText(str(var_value))  # 设置默认值
                self.layout.addWidget(input_field, 5 + self.a, b)
                self.input_fields[var_name] = input_field

        # 添加更新按钮
        self.a = (id+1) // 4 * 2
        b = (id+1)  % 4        
        self.update_button = QtWidgets.QPushButton("Update Global Variables", self)
        self.update_button.clicked.connect(self.update_global_variables)
        self.layout.addWidget(self.update_button, 4 + self.a, b,2,4-b)

        self.output_text = QtWidgets.QTextEdit(self)
        self.output_text.setReadOnly(True)  # 设置为只读
        self.layout.addWidget(self.output_text, 0, 4, 5+self.a+1, 4)

        # 添加绘图区域
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, 5+self.a+2, 0, 5, 8)

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
        else:
            # 如果是开发环境，使用相对路径
            file_path = 'src/sac_pso_env.py'
        update_evaluate_method(file_path, new_content)
        importlib.reload(src.sac_pso_env)
        # importlib.reload(src.test_sac_pso)

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
        global global_variables
        # global_variables["window_show"]=self
        self.args = get_args()
        self.args.Window = self
        self.result, self.agent = train_agent(self.args)
        print("Training finished.")
        # train_agent(self.args)        
    # def train(self):
    #     # if self.args is None:
    #     self.args = get_args()

    #     # 创建线程实例
    #     self.train_thread = TrainThread(self.args)

    #     # 连接线程的信号到主线程的槽函数
    #     self.train_thread.result_signal.connect(self.on_train_finished)
    #     self.train_thread.progress_signal.connect(self.update_progress)  # 连接进度信号

    #     # 启动线程
    #     self.train_thread.start()

    #     print("Training started in a separate thread.")
    # def train(self):
    #     self.args = get_args()
    #     self.args.device = 'cpu'
    
    # # 创建通信队列
    #     self.result_queue = mp.Queue()
    #     self.progress_queue = mp.Queue()
    
    # # 创建进程
    #     self.train_process = TrainProcess(
    #         self.args, 
    #         self.result_queue,
    #         self.progress_queue
    #     )
    #     self.train_process.start()
    

    def on_train_finished(self, result, agent):
        self.result = result
        self.agent = agent
        print("Training finished.")
        # print(f"Result: {self.result}, Agent: {self.agent}")

    def update_progress(self, message):
        # 使用定时器延迟更新防止界面冻结
        QtCore.QTimer.singleShot(0, lambda: 
            self.output_text.append(message)
    )

    # def test(self):
    #     if self.args is None or self.agent is None:
    #         print("Please train the agent before testing.")
    #         return

    #     # 创建线程实例
    #     self.watch_thread = WatchThread(self.args, self.agent)

    #     # 连接线程的信号到主线程的槽函数
    #     self.watch_thread.finished_signal.connect(self.on_watch_finished)

    #     # 启动线程
    #     self.watch_thread.start()

    #     print("Testing started in a separate thread.")
    def test(self):
        if self.args is None or self.agent is None:
            print("Please train the agent before testing.")
            return
        # 创建线程实例
        watch(self.args, self.agent)
 
    def on_watch_finished(self):
        print("Testing finished.")

    def end(self):
        if hasattr(self, "train_thread") and self.train_thread.isRunning():
            
            self.train_thread.terminate()
            self.train_thread.wait()  # 等待线程完全终止
            print("stop TrainThread ")
        elif hasattr(self, "watch_thread") and self.watch_thread.isRunning():
            
            self.watch_thread.terminate()
            self.watch_thread.wait()  # 等待线程完全终止
            print("stop watch_thread ")
        else:
            print("No Thread is  running.")

    def capture_plot(self):
        """
        捕获 matplotlib 的所有图像并嵌入到 PyQt 的画布中。
        """
        print("Capturing plot...")

        # 遍历所有 Figure
        for num in plt.get_fignums():
            current_figure = plt.figure(num)
            if self.current_subplot >= self.max_subplots:
                self.clear_canvas()
            # 计算子图位置
            rows, cols = 3, 4  # 3x4 网格
            self.current_subplot += 1
            ax = self.figure.add_subplot(rows, cols, self.current_subplot)

            # 将当前 Figure 的内容绘制到 PyQt 的子图中
            for line in current_figure.axes[0].lines:
                ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

            for bar in current_figure.axes[0].patches:
                x = bar.get_x()
                width = bar.get_width()
                height = bar.get_height()
                ax.bar(x, height, width, label=bar.get_label(), color=bar.get_facecolor())

            ax.set_xlabel(current_figure.axes[0].get_xlabel())
            ax.set_ylabel(current_figure.axes[0].get_ylabel())
            ax.set_title(current_figure.axes[0].get_title())
            ax.legend()

            self.canvas.draw()
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

        print(f"Global variables updated to: {global_variables}")
        importlib.reload(src.test_sac_pso)
    def log_to_text_edit(self, message):
        """将日志信息追加到 QTextEdit 控件中"""
        self.output_text.append(message)

    def clear_canvas(self):
        print("Clearing canvas...")
        self.figure.clear()
        self.current_subplot = 0
        self.canvas.draw()

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
