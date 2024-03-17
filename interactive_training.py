#!/usr/bin/env python3

from interactive_learning_gui import Ui_MainWindow
from awgn_autoencoder import AWGNAutoencoder, ShapingAutoencoder
from multi_config_dialog import MultiConfigDialog

import settings

"""Plot Constellation and animate it."""
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QGridLayout,
    QApplication,
    QMainWindow,
    QSlider,
    QComboBox,
)

from PySide6.QtPdf import QPdfDocument

import numpy as np

import pyqtgraph as pg
from PySide6 import QtUiTools
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtCore import (
    Qt,
    QPropertyAnimation,
    QObject,
    Signal,
    QFile,
    QThread,
    Slot,
    QSize,
)

import argparse
import sys
import os
import pandas as pd
import mokka

from pyqtconfig import ConfigManager

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOptions(antialias=True)

vhex = np.vectorize(hex)


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

    def __init__(self, settings, max_epoch=None):
        """"""
        super(Training, self).__init__()
        # Here we can handle the settings to configure correct Training
        self.max_epoch = None
        self.settings = settings
        self.model = ShapingAutoencoder(self.settings.as_dict())

    # Configure a Slot to handle changes in the settings object
    @Slot()
    def reconfigure(self, settings):
        self.settings = settings
        self.model.update_settings(self.settings)
        # Simulation type (shaping or adaptive equalization)

    @Slot()
    def killed(self):
        print("Handled killed signal")
        self.stop = True

    def run(self):
        epoch = 0
        while True:
            if self.max_epoch is not None and epoch == self.max_epoch:
                return
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
    simulation_running = Signal(bool)

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        # setting title
        self.setWindowTitle("MOKka Demo")

        # Settings
        self.configureSettings()

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
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(0, 0, 0, 120))
        self.labeltexts = [pg.TextItem() for _ in range(2**4)]
        for label in self.labeltexts:
            self.constellation_widget.addItem(label)

        self.constellation_widget.addItem(self.scatter)
        self.constellation_widget.setAspectLocked(1.0)
        self.constellation_widget.setXRange(-1.5, 1.5)
        self.constellation_widget.setYRange(-1.5, 1.5)
        self.constellation_widget.getPlotItem().setLabel("bottom", "Real Part")
        self.constellation_widget.getPlotItem().setLabel("left", "Imaginary Part")
        self.constellation_widget.getPlotItem().getAxis("left").setTickSpacing(
            major=1.0, minor=0.5
        )
        self.constellation_widget.getPlotItem().getAxis("left").setGrid(128)
        self.constellation_widget.getPlotItem().getAxis("bottom").setTickSpacing(
            major=1.0, minor=0.5
        )
        self.constellation_widget.getPlotItem().getAxis("bottom").setGrid(128)
        # self.constellation_widget.heightForWidth = lambda self, w: w
        # self.constellation_widget.sizePolicy().setHeightForWidth(True)

        self.performance = pg.PlotCurveItem(size=10, pen=pg.mkPen("k", width=3))
        # self.performance.setPen(pg.mkPen("k", width=2.5))
        self.plot_widget.addItem(self.performance)
        self.plot_widget.setYRange(0, 4)
        self.plot_widget.setXRange(0, 1000)
        self.plot_widget.getPlotItem().setLabel("left", "BMI (bit/symbol)")
        self.plot_widget.getPlotItem().setLabel("bottom", "Epoch")
        self.plot_widget.getPlotItem().getAxis("left").setGrid(128)

    def configureSettings(self):
        script_dir = os.path.dirname(__file__)

        default_settings = {"simulation_type": "shaping"}
        self.settings = ConfigManager(
            filename=os.path.join(script_dir, "settings", "settings.json")
        )
        self.settings.set_defaults(default_settings)
        self.shaping_settings = ConfigManager(
            filename=os.path.join(script_dir, "settings", "shaping_settings.json")
        )
        shaping_default_settings = {
            "bits_per_symbol": 4,
            "lr": 1e-3,
            "demapper": settings.Demapper.Neural,
            "channel": settings.ShapingChannel.AWGN,
            "cpe": settings.CPE.NONE,
            "objective": settings.ShapingObjective.BMI,
            "type": settings.ShapingType.Geometric,
        }
        shaping_default_metadata = {
            "channel": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "AWGN": settings.ShapingChannel.AWGN,
                    "Wiener Phase Noise": settings.ShapingChannel.Wiener,
                    "Optical Channel": settings.ShapingChannel.Optical,
                },
            },
            "demapper": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "Neural": settings.Demapper.Neural,
                    "Gaussian": settings.Demapper.Gaussian,
                    "Separated Neural": settings.Demapper.SepNeural,
                    "Separated Gaussian": settings.Demapper.SepGaussian,
                },
            },
            "cpe": {
                "preferred_map_dict": {
                    "None": settings.CPE.NONE,
                    "BPS": settings.CPE.BPS,
                    "V&V": settings.CPE.VV,
                }
            },
            "objective": {
                "preferred_map_dict": {
                    "BMI": settings.ShapingObjective.BMI,
                    "MI": settings.ShapingObjective.MI,
                }
            },
            "type": {
                "preferred_map_dict": {
                    "Geometric": settings.ShapingType.Geometric,
                    "Probabilistic": settings.ShapingType.Probabilistic,
                    "Joint Geometric & Probabilistic": settings.ShapingType.Joint,
                }
            },
        }
        self.shaping_settings.set_many_metadata(shaping_default_metadata)
        self.shaping_settings.set_defaults(shaping_default_settings)
        equalization_default_settings = {}
        self.equalization_settings = ConfigManager(
            filename=os.path.join(script_dir, "settings", "equalization_settings.json")
        )

    def configureGUI(self):
        # Configure options for Shaping
        # self.bitpersymbol_box.setValue(default_bits_per_symbol)
        # self.bitpersymbol_box.setMaximum(max_bits_per_symbol)
        # self.bitpersymbol_box.setMinimum(1)
        # self.channel_box.addItems(default_channels)
        # self.shaping_box.addItems(default_shaping_type)
        # self.objective_box.addItems(default_objective_functions)

        # self.channel_box.addItems(
        # list(self.shaping_settings.get_metadata("channel")["preferred_map_dict"].keys())
        # )
        # Configure options for Equalization

        # Configure other GUI options

        self.settings_group.setTabText(0, "Shaping")
        self.settings_group.setTabText(1, "Equalization")

        # Load logos to QPixMap and display them on the main window
        logo_height = 60
        pdf_doc = QPdfDocument()
        pdf_doc.load("./assets/CEL_logo.pdf")
        logo_size = pdf_doc.pagePointSize(0)
        logo_width = int(logo_height / logo_size.height() * logo_size.width())
        cel_logo = pdf_doc.render(0, QSize(logo_width, logo_height))
        pdf_doc.load("./assets/kitlogo_en_rgb.pdf")
        logo_size = pdf_doc.pagePointSize(0)
        logo_width = int(logo_height / logo_size.height() * logo_size.width())
        kit_logo = pdf_doc.render(0, QSize(logo_width, logo_height))
        pdf_doc.load("./assets/erc_logo.pdf")
        logo_width = int(logo_height / logo_size.height() * logo_size.width())
        erc_logo = pdf_doc.render(0, QSize(logo_width, logo_height))
        self.erc_logo.setPixmap(QPixmap.fromImage(erc_logo))
        self.cel_logo.setPixmap(QPixmap.fromImage(cel_logo))
        self.kit_logo.setPixmap(QPixmap.fromImage(kit_logo))

        # Connect GUI controls to central settings register
        self.shaping_settings.add_handler("bits_per_symbol", self.bitpersymbol_box)
        self.shaping_settings.add_handler(
            "channel", self.channel_box, preferred_mapper=True
        )
        self.shaping_settings.add_handler(
            "type", self.shaping_box, preferred_mapper=True
        )
        self.shaping_settings.add_handler(
            "objective", self.objective_box, preferred_mapper=True
        )

        self.settings_group.setCurrentIndex(0)

    @Slot()
    def settings_tabChngd(self):
        current_id = self.settings_group.currentIndex()
        if current_id == 0:
            self.settings.set("simulation_type", "shaping")
        else:
            self.settings.set("simulation_type", "equalization")

    @Slot()
    def handleSettingsChange(self):
        # Reconfigure GUI following a settings change switch Tab if simulation_type is different
        if self.settings.get("simulation_type") == "shaping":
            self.settings_group.setCurrentIndex(0)
        elif self.settings.get("simulation_type") == "equalization":
            self.settings_group.setCurrentIndex(1)

    @Slot()
    def handleSimulationRunning(self, running):
        if running:
            self.settings_group.setEnabled(False)
            self.settings_btn.setEnabled(False)
        else:
            self.settings_group.setEnabled(True)
            self.settings_btn.setEnabled(True)

    @Slot()
    def plotConstellation(self, constellation):
        constellation_array = np.concatenate(
            (constellation.real[:, None], constellation.imag[:, None]), axis=1
        )
        labels = [s[2:] for s in vhex(np.arange(2**4))]
        bitstrings = [str(s) for s in mokka.utils.hex2bits(labels, 4)]
        self.scatter.setData(pos=constellation_array)
        for bitstring, point, label in zip(bitstrings, constellation, self.labeltexts):
            label.setText(bitstring)
            label.setPos(float(point.real), float(point.imag))

    @Slot()
    def handleProgress(self, progress, result):
        self.epochs.append(progress)
        self.bmi.append(result)
        self.performance.setData(self.epochs, self.bmi)

    def runTraining(self):
        self.thread = QThread()
        self.worker = Training(self.settings)
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
        self.settings_group.currentChanged.connect(self.settings_tabChngd)
        self.settings.updated.connect(self.handleSettingsChange)
        self.simulation_running.connect(self.handleSimulationRunning)
        self.settings_btn.pressed.connect(self.configDialog)

    @Slot()
    def configDialog(self):
        if self.settings.get("simulation_type") == "shaping":
            config_dialog = MultiConfigDialog(
                self.settings, self.shaping_settings, cols=2
            )
            config_dialog.setWindowTitle("Settings")
            config_dialog.accepted.connect(
                lambda: (
                    self.update_config(self.settings, config_dialog.config1),
                    self.update_config(self.shaping_settings, config_dialog.config2),
                )
            )
            config_dialog.exec()
        elif self.settings.get("simulation_type") == "equalization":
            config_dialog = MultiConfigDialog(
                self.settings, self.equalization_settings, cols=2
            )
            config_dialog.setWindowTitle("Settings")
            config_dialog.accepted.connect(
                lambda: (
                    self.update_config(self.settings, config_dialog.config1),
                    self.update_config(
                        self.equalization_settings, config_dialog.config2
                    ),
                )
            )
            config_dialog.exec()

    def update_config(self, config, new_config):
        config.set_many(new_config.as_dict())
        config.save()

    @Slot()
    def runBtn_clicked(self):
        if self.run_btn.isFlat():
            self.thread.requestInterruption()
            self.simulation_running.emit(False)
            self.run_btn.setFlat(False)
            self.run_btn.setText("Run")
            return
        self.run_btn.setFlat(True)
        self.run_btn.setText("Stop && Reset")
        self.resetPlots()
        self.runTraining()
        self.simulation_running.emit(True)

    @Slot()
    def resetBtn_clicked(self):
        # ResetPlots
        self.resetPlots()

    def resetPlots(self):
        self.constellation_widget.clear()
        self.plot_widget.clear()
        self.addQtGraphItems()
        self.epochs = []
        self.bmi = []
        self.results = {}

    def closeEvent(self, event):
        self.settings.save()
        self.shaping_settings.save()
        self.equalization_settings.save()
        event.accept()


if __name__ == "__main__":
    # create pyqt5 app
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = Window()
    # start the app
    sys.exit(App.exec())
