from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("页面切换示例")
        self.setGeometry(100, 100, 800, 600)

        # 创建主窗口的布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 创建 QStackedWidget
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

    def create_page1(self):
        """创建页面 1，包含一个按钮和一个文本框"""
        page = QWidget()
        layout = QVBoxLayout(page)

        label = QLabel("这是页面 1")
        layout.addWidget(label)

        text_box = QLineEdit()
        text_box.setPlaceholderText("请输入内容...")
        layout.addWidget(text_box)

        button = QPushButton("切换到页面 2")
        button.clicked.connect(self.show_page2)
        layout.addWidget(button)

        return page

    def create_page2(self):
        """创建页面 2，包含两个按钮"""
        page = QWidget()
        layout = QVBoxLayout(page)

        label = QLabel("这是页面 2")
        layout.addWidget(label)

        button1 = QPushButton("按钮 1")
        layout.addWidget(button1)

        button2 = QPushButton("切换到页面 1")
        button2.clicked.connect(self.show_page1)
        layout.addWidget(button2)

        return page

    def show_page1(self):
        """切换到页面 1"""
        self.stacked_widget.setCurrentWidget(self.page1)

    def show_page2(self):
        """切换到页面 2"""
        self.stacked_widget.setCurrentWidget(self.page2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())