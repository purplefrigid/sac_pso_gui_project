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
import src.sac_pso_env_watch
from src.备份 import plot_test
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
    

    def __init__(self, args, agent):
        super().__init__()
        self.args = args
        self.agent = agent

    def run(self):
        # 在子线程中运行 watch 函数
                    # **6. 重载模块并执行训练**
        importlib.reload(src.sac_pso_env_watch)
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
        setup_logging()
        self.setWindowTitle("PSO GUI Project")
        self.setGeometry(100, 100, 1772, 850)

        self.initUI()
        
        # 重定向标准输出到 QTextEdit
        self.output_redirect = OutputRedirect(self.output_text)
        sys.stdout = self.output_redirect

        # 初始化子图计数器
        self.current_subplot = 0
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

        # 添加 QTextEdit
        self.text_edit = QtWidgets.QTextEdit(self)
        # self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.text_edit, 0, 4, 4, 2)

        # 创建一个 QLabel 用于显示图片
        self.image_label = QtWidgets.QLabel(self)
        self.layout.addWidget(self.image_label, 0, 0, 4, 2)  # 设置位置和大小
        # 加载图片
        pixmap = QtGui.QPixmap("3.png")  # 替换为你的图片路径
        self.image_label.setPixmap(pixmap)
        # 调整 QLabel 的大小以适应图片
        # self.image_label.setScaledContents(True)


        # 添加按钮
        self.load_button = QtWidgets.QPushButton("Load _evaluate Method", self)
        self.load_button.clicked.connect(self.load_evaluate_method)
        # self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.load_button, 0, 2)

        self.save_button = QtWidgets.QPushButton("Save Changes", self)
        self.save_button.clicked.connect(self.save_changes)
        # self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.save_button, 0, 3)

        self.train_button = QtWidgets.QPushButton("train start", self)
        # self.plot_button.clicked.connect(self.plot_curve)
        self.train_button.clicked.connect(self.train)
        # self.layout.addWidget(self.train_button)
        self.layout.addWidget(self.train_button, 1, 2)

        self.test_button = QtWidgets.QPushButton("test", self)
        # self.plot_button.clicked.connect(self.plot_curve)
        self.test_button.clicked.connect(self.test)
        # self.layout.addWidget(self.test_button)
        self.layout.addWidget(self.test_button, 1, 3)

        self.clear_button = QtWidgets.QPushButton("Clear Canvas", self)
        self.clear_button.clicked.connect(self.clear_canvas)
        # self.layout.addWidget(self.clear_button)
        self.layout.addWidget(self.clear_button, 2, 2)

        self.plot_button = QtWidgets.QPushButton("plot", self)
        self.plot_button.clicked.connect(self.plot_curve)
        self.layout.addWidget(self.plot_button, 3, 2)

        self.sy_button = QtWidgets.QPushButton("sy", self)
        self.sy_button.clicked.connect(self.sy)
        self.layout.addWidget(self.sy_button, 2, 3)

        self.end_button = QtWidgets.QPushButton("end", self)
        self.end_button.clicked.connect(self.end)
        self.layout.addWidget(self.end_button, 3, 3)

        # 动态生成输入框
        self.input_fields = {}  # 存储变量名和对应的输入框

        for id, (var_name, var_value) in enumerate(global_variables.items()):
            # 添加标签
            self.a = id // 6 * 2
            b = id % 6
            label = QtWidgets.QLabel(f"{var_name} (default: {var_value}):", self)
            # label = QtWidgets.QLabel(f"{var_name}", self)
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
        self.a = (id+1) // 6 * 2
        b = (id+1)  % 6       
        self.update_button = QtWidgets.QPushButton("Update Global Variables", self)
        self.update_button.clicked.connect(self.update_global_variables)
        # self.layout.addWidget(self.update_button)
        self.layout.addWidget(self.update_button, 4 + self.a, b,2,6-b)

        # 显示当前全局变量值
        # self.global_label = QtWidgets.QLabel(f"Current Global Variables: {global_variables}", self)
        # # self.layout.addWidget(self.global_label)
        # self.layout.addWidget(self.global_label, 5+self.a+1, 1,1,7)

        # 添加输出区域
        self.output_text = QtWidgets.QTextEdit(self)
        self.output_text.setReadOnly(True)  # 设置为只读
        # self.layout.addWidget(self.output_text)
        self.layout.addWidget(self.output_text, 0, 6, 5+self.a+1, 4)

        # 添加绘图区域
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.canvas, 5+self.a+2, 0, 5, 10)

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
        print("Training finished.")

        # print(f"Result: {self.result}, Agent: {self.agent}")
    # def update_progress(self, message):
    #     # 将进度信息追加到输出区域
    #     self.output_text.append(message)
    def update_progress(self, message):
        # 使用定时器延迟更新防止界面冻结
        QtCore.QTimer.singleShot(0, lambda: 
            self.output_text.append(message)
    )

    # def train(self):
    #     self.args = get_args()
    #     # self.result, self.agent = train_agent(self.args)
    #     train_agent(self.args)
    # def test(self):
    #     if self.args is None or self.agent is None:
    #         print("Please train the agent before testing.")
    #         return
    #     watch(self.args, self.agent)
    def test(self):
        # if self.args is None or self.agent is None:
        #     print("Please train the agent before testing.")
        #     return
        self.args = get_args()
        log_path = os.path.join(self.args.logdir, "sac_pso_env", "sac")
        policy_path = os.path.join(log_path, "policy"+"_"+str(self.args.n_pistons)+".pth")

        if not os.path.exists(policy_path):
            print("Please train the agent before testing.")
            return
        # 创建线程实例
        self.watch_thread = WatchThread(self.args, self.agent)

        # 连接线程的信号到主线程的槽函数
        self.watch_thread.finished_signal.connect(self.on_watch_finished)

        # 启动线程
        self.watch_thread.start()

        print("Testing started in a separate thread.")

    def on_watch_finished(self):
        print("Testing finished.")

    # def capture_plot(self):
    #     """
    #     捕获 matplotlib 的当前图像并嵌入到 PyQt 的画布中。
    #     """
    #     print("Capturing plot...")
    #     if self.current_subplot >= self.max_subplots:
    #         print("Canvas is full. Please clear the canvas to continue.")
    #         return

    #     # 获取当前的 matplotlib Figure
    #     current_figure = plt.gcf()

    #     # 计算子图位置
    #     rows, cols = 3, 4  # 2x2 网格
    #     self.current_subplot += 1
    #     ax = self.figure.add_subplot(rows, cols, self.current_subplot)

    #     # 将当前 Figure 的内容绘制到 PyQt 的子图中
    #     for line in current_figure.axes[0].lines:
    #         ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

    #     ax.set_xlabel(current_figure.axes[0].get_xlabel())
    #     ax.set_ylabel(current_figure.axes[0].get_ylabel())
    #     ax.set_title(current_figure.axes[0].get_title())
    #     ax.legend()

    #     # 刷新画布
    #     self.canvas.draw()

    #     # 清除 matplotlib 的当前 Figure
    #     plt.close(current_figure)
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
        # if self.current_subplot >= self.max_subplots:
        #     self.clear_canvas()
            # print("Canvas is full. Please clear the canvas to continue.")
            # return

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

    # def update_global_variables(self):
    #     global global_variables
    #     # 遍历所有输入框并更新全局变量
    #     for var_name, input_field in self.input_fields.items():
    #         global_variables[var_name] = input_field.text()

    #     # 更新显示标签
    #     # self.global_label.setText(f"Current Global Variables: {global_variables}")
    #     print(f"Global variables updated to: {global_variables}")
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

    def clear_canvas(self):
        print("Clearing canvas...")
        self.figure.clear()
        self.current_subplot = 0
        self.canvas.draw()

    # def closeEvent(self, event):
    #     sys.stdout = sys.__stdout__  # 恢复标准输出
    #     super().closeEvent(event)
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
