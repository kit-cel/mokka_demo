#!/usr/bin/env python3

from interactive_learning_gui import Ui_MainWindow
from awgn_autoencoder import AWGNAutoencoder

"""Plot Constellation and animate it."""
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QGridLayout,
    QApplication,
    QMainWindow,
    QSlider,
)

import numpy as np

import pyqtgraph as pg
from PySide6 import QtUiTools
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QPropertyAnimation, QObject, Signal, QFile, QThread, Slot

import argparse
import sys
import os
import pandas as pd
import mokka


# The idea here is to split the code two-fold

# The GUI code lives in the main thread and updates the plots etc.
# mokka code lives in the worker thread

# Between the main thread and the worker thread we use Qt signals & slots to
# share configuration, results and progress


class Training(QObject):
    finished = Signal()
    progress_result = Signal(int, float)
    constellation = Signal(object)
    stop = False

    def __init__(self):
        """"""
        super(Training, self).__init__()
        self.model = AWGNAutoencoder()

    @Slot()
    def killed(self):
        print("Handled killed signal")
        self.stop = True

    def run(self):
        epoch = 0
        while True:
            if self.thread().isInterruptionRequested():
                return
            epoch += 1
            bmi = self.model.step()
            self.progress_result.emit(epoch, bmi)
            trained_constellation = (
                self.model.mapper.get_constellation().detach().cpu().numpy()
            )
            self.constellation.emit(trained_constellation)


class Window(QMainWindow, Ui_MainWindow):
    stop_simulation = Signal()

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        # setting title
        self.setWindowTitle("Joint Geometric & Probabilistic Shaping")

        # Set parameters

        # setting geometry
        self.setGeometry(100, 100, 1200, 500)

        # icon
        icon = QIcon("skin.png")

        # setting icon to the window
        self.setWindowIcon(icon)

        # calling method
        # self.UiComponents()
        self.configureGUI()
        self.addQtGraphItems()
        self.hookGUIEvents()

        self.epochs = []
        self.bmi = []

        # showing all the widgets
        self.show()

    def addQtGraphItems(self):
        # Configure scatter_plot
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.constellation_widget.addItem(self.scatter)
        self.constellation_widget.setAspectLocked(1.0)
        self.constellation_widget.setXRange(-1.5, 1.5)
        self.constellation_widget.setYRange(-1.5, 1.5)
        # self.constellation_widget.heightForWidth = lambda self, w: w
        # self.constellation_widget.sizePolicy().setHeightForWidth(True)

        self.performance = pg.PlotCurveItem(size=10)
        self.plot_widget.addItem(self.performance)
        self.plot_widget.setYRange(0, 4)

    def configureGUI(self):
        default_bits_per_symbol = 4
        default_channels = ["AWGN", "Wiener Phase Noise", "Optical Channel"]
        default_shaping_type = ["Geometric", "Probabilistic", "Joint Geometric & Probabilistic"]
        default_objective_functions = ["BMI", "GMI"]

        self.bitpersymbol_box.setValue(default_bits_per_symbol)
        self.channel_box.addItems(default_channels)
        self.shaping_box.addItems(default_shaping_type)
        self.objective_box.addItems(default_objective_functions)

    @Slot()
    def plotConstellation(self, constellation):
        constellation_array = np.concatenate(
            (constellation.real[:, None], constellation.imag[:, None]), axis=1
        )
        # bitstrings = [str(s) for s in mokka.utils.hex2bits(labels, 6)]
        self.scatter.setData(pos=constellation_array)
        # for bitstring, point, label in zip(bitstrings, constellation, self.labeltexts):
        #     label.setText(bitstring)
        #     label.setPos(float(point[0]), float(point[1]))

    @Slot()
    def handleProgress(self, progress, result):
        self.epochs.append(progress)
        self.bmi.append(result)
        self.performance.setData(self.epochs, self.bmi)


    def runTraining(self):
        self.thread = QThread()
        self.worker = Training()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.constellation.connect(self.plotConstellation)
        self.worker.progress_result.connect(self.handleProgress)
        self.stop_simulation.connect(self.worker.killed)
        self.stop_simulation.connect(self.check_signal)
        self.thread.start()

    # Debug
    @Slot()
    def check_signal(self):
        print("Signal emitted!")

    # Below here is all the GUI interaction nonsense

    def hookGUIEvents(self):
        self.run_btn.clicked.connect(self.runBtn_clicked)
        self.reset_btn.clicked.connect(self.resetBtn_clicked)

    @Slot()
    def runBtn_clicked(self):
        if self.run_btn.isFlat():
            self.thread.requestInterruption()
            self.run_btn.setFlat(False)
            return
        self.run_btn.setFlat(True)
        self.runTraining()

    @Slot()
    def resetBtn_clicked(self):
        # ResetPlots
        self.constellation_widget.clear()
        self.plot_widget.clear()
        self.addQtGraphItems()
        self.epochs = []
        self.bmi = []


if __name__ == "__main__":
    # create pyqt5 app
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = Window()
    # start the app
    sys.exit(App.exec())
