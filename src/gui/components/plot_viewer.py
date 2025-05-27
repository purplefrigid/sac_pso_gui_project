from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import matplotlib.pyplot as plt

class PlotViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotViewer, self).__init__(parent)
        self.setWindowTitle("Plot Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.plot_button = QtWidgets.QPushButton("Plot Original Curve")
        self.plot_button.clicked.connect(self.plot_curve)
        self.layout.addWidget(self.plot_button)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def plot_curve(self):
        # Clear the previous plot
        self.ax.clear()

        # Simulate original _evaluate method output for plotting
        x = np.linspace(0, 10, 100)
        y = np.sin(x)  # Replace with actual evaluation logic

        self.ax.plot(x, y, label='Original Curve', color='blue')
        self.ax.set_title('Original Curve from _evaluate Method')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()