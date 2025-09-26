#!/usr/bin/env python3
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)
torch.set_default_device(device)

from interactive_learning_gui import Ui_MainWindow
from shaping_models import AWGNAutoencoder, ShapingAutoencoder
from equalization_model import EqualizerSimulation
from multi_config_dialog import MultiConfigDialog

from pyqtgraph import PlotWidget

import settings
import time


"""Plot Constellation and animate it."""
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QWidget,
    QLabel,
    QGridLayout,
    QApplication,
    QMainWindow,
    QSlider,
    QComboBox,
    QSizePolicy,
)

from PySide6.QtPdf import QPdfDocument

import numpy as np

import pyqtgraph as pg
from PySide6 import QtUiTools
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtCore import (
    QMutex,
    QMutexLocker,
    Qt,
    QPropertyAnimation,
    QObject,
    Signal,
    QFile,
    QThread,
    Slot,
    QSize,
    QCoreApplication,
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


scatterSizePolicy = QSizePolicy(
    QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
)
scatterSizePolicy.setHorizontalStretch(0)
scatterSizePolicy.setVerticalStretch(0)
scatterSizePolicy.setHeightForWidth(True)


def return_true():
    return True


def resizeEvent(self, event):
    # Create a square base size of 10x10 and scale it to the new size
    # maintaining aspect ratio.
    new_size = QSize(10, 10)
    new_size.scale(event.size(), Qt.AspectRatioMode.KeepAspectRatio)
    self.resize(new_size)


def constructConstellationPlot(parent, name):
    widget = PlotWidget(parent)
    widget.heightForWidth = lambda w: w
    widget.hasHeightForWidth = return_true
    widget.setObjectName(name)
    widget.setSizePolicy(scatterSizePolicy)
    widget.setMinimumSize(QSize(200, 200))
    widget.setBaseSize(QSize(200, 200))
    return widget


def configureScatterPlot(widget, color, size=5, **kwargs):
    scatter = pg.ScatterPlotItem(size=size, brush=pg.mkBrush(*color), **kwargs)
    widget.addItem(scatter)
    widget.setAspectLocked(1.0)
    widget.setXRange(-1.5, 1.5)
    widget.setYRange(-1.5, 1.5)
    widget.getPlotItem().setLabel("bottom", "Real Part")
    widget.getPlotItem().setLabel("left", "Imaginary Part")
    widget.getPlotItem().getAxis("left").setTickSpacing(major=1.0, minor=0.5)
    widget.getPlotItem().getAxis("left").setGrid(128)
    widget.getPlotItem().getAxis("bottom").setTickSpacing(major=1.0, minor=0.5)
    widget.getPlotItem().getAxis("bottom").setGrid(128)
    return scatter


# The idea here is to split the code two-fold

# The GUI code lives in the main thread and updates the plots etc.
# mokka code lives in the worker thread

# Between the main thread and the worker thread we use Qt signals & slots to
# share configuration, results and progress


class Training(QObject):
    finished = Signal()
    progress_result = Signal(int, object)
    symbols1 = Signal(object, object)
    symbols2 = Signal(object)
    channel = Signal(object)
    channel_est = Signal(object)
    stop = False
    config_update = QMutex()

    def __init__(self, settings, max_epoch=None):
        """"""
        super(Training, self).__init__()
        # Here we can handle the settings to configure correct Training
        self.max_epoch = None
        self.settings = settings
        if self.settings["simulation_type"] == "shaping":
            self.model = ShapingAutoencoder(self.settings)
        elif self.settings["simulation_type"] == "equalization":
            self.model = EqualizerSimulation(self.settings)

    # Configure a Slot to handle changes in the settings object
    @Slot()
    def reconfigure(self, settings):
        with QMutexLocker(self.config_update):
            if self.settings["simulation_type"] != settings["simulation_type"]:
                if self.thread() is not None:
                    self.thread().requestInterruption()
                time.sleep(0.1)
                if settings["simulation_type"] == "shaping":
                    self.model = ShapingAutoencoder(settings)
                elif settings["simulation_type"] == "equalization":
                    self.model = EqualizerSimulation(settings)
            self.settings = settings
            self.model.update_config(self.settings)
            if self.settings["simulation_type"] == "equalization":
                # Reset simulation since parameters changed
                self.model.current_frame = 0
        # Simulation type (shaping or adaptive equalization)

    @Slot()
    def run(self):
        epoch = 0
        while True:
            if self.max_epoch is not None and epoch == self.max_epoch:
                return
            if self.thread().isInterruptionRequested():
                self.thread().quit()
                return
            epoch += 1
            with QMutexLocker(self.config_update):
                results = self.model.step()
            if self.settings["simulation_type"] == "shaping":
                if self.settings["objective"] == settings.ShapingObjective.BMI:
                    self.progress_result.emit(epoch, results)
                elif self.settings["objective"] == settings.ShapingObjective.MI:
                    self.progress_result.emit(epoch, results)

                trained_constellation = (
                    self.model.mapper.get_constellation().detach().cpu().numpy()
                )
                probabilities = self.model.mapper.p_symbols.detach().cpu().numpy()

                self.symbols1.emit(trained_constellation, probabilities)
                self.symbols2.emit(results["rx_signal_postcpe"].numpy())
            elif self.settings["simulation_type"] == "equalization":
                self.symbols1.emit(results["rx_signal_posteq"].numpy()[0, :])
                self.symbols2.emit(results["rx_signal_posteq"].numpy()[1, :])
                self.progress_result.emit(epoch, results)


class Window(QMainWindow, Ui_MainWindow):
    stop_simulation = Signal()
    simulation_running = Signal(bool)
    labeltexts = None
    worker_thread: QThread = QThread()

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
        self.setupTraining()
        # self.UiComponents()
        self.configureGUI()
        self.addQtGraphItems()
        self.hookGUIEvents()
        # Configure GUI according to settings
        self.handleSettingsChange()

        self.epochs = []
        self.bmi = []

        # showing all the widgets
        self.show()

    def reconfigureGraphItems(self):
        # This gets called whenever settings change
        if self.labeltexts is not None:
            for label in self.labeltexts:
                self.scatter_widget1.removeItem(label)

        # Reconfigure scatter plots
        if not self.shaping_settings.get("show_labels"):
            return

        if self.settings.get("simulation_type") == "shaping":
            self.labeltexts = [
                pg.TextItem()
                for _ in range(2 ** self.shaping_settings.get("bits_per_symbol"))
            ]
            for label in self.labeltexts:
                self.scatter_widget1.addItem(label)

    def addQtGraphItems(self):
        # Configure plots for Shaping & Eq

        self.scatter_widget1 = constructConstellationPlot(
            self.plot_box, "constellation_widget"
        )
        self.scatter_widget2 = constructConstellationPlot(
            self.plot_box, "rx_symbols_widget"
        )

        self.gridLayout.addWidget(self.scatter_widget1, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.scatter_widget2, 1, 0, 1, 1)

        self.plot_widget = PlotWidget(self.plot_box)
        self.plot_widget.setObjectName("plot_widget")
        sizePolicy1 = QSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding
        )
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        self.plot_widget.setSizePolicy(sizePolicy1)
        self.plot_widget.setMinimumSize(QSize(700, 500))

        self.gridLayout.addWidget(self.plot_widget, 0, 1, 2, 2)
        # Configure scatter_plot
        self.scatter1 = configureScatterPlot(
            self.scatter_widget1, size=5, color=(0, 150, 130, 255)
        )
        # Configure scatter_plot
        self.scatter2 = configureScatterPlot(
            self.scatter_widget2, size=2, color=(0, 150, 130, 120), pen=pg.mkPen()
        )

        self.reconfigureGraphItems()
        self.performance = pg.PlotCurveItem(
            size=10, pen=pg.mkPen((0, 150, 130, 255), width=2)
        )
        self.entropy = pg.InfiniteLine(angle=0, pos=0, pen=pg.mkPen(color="red"))
        # self.performance.setPen(pg.mkPen("k", width=2.5))
        self.plot_widget.addItem(self.performance)
        self.plot_widget.addItem(self.entropy)
        if self.settings.get("simulation_type") == "shaping":
            self.plot_widget.setYRange(0, self.shaping_settings.get("bits_per_symbol"))
        else:
            # Only 64-QAM for Equalization right now
            self.plot_widget.setYRange(0, 6)
        self.plot_widget.setXRange(0, 1000)
        self.plot_widget.getPlotItem().setLabel("left", "BMI (bit/symbol)")
        self.plot_widget.getPlotItem().setLabel("bottom", "Epoch")
        self.plot_widget.getPlotItem().getAxis("left").setGrid(128)
        self.plot_widget.getPlotItem().getAxis("bottom").setGrid(128)

        self.plot_widget.getPlotItem().enableAutoRange(x=True)
        self.plot_widget.getPlotItem().setLimits(
            xMin=0, yMin=0, minYRange=self.shaping_settings.get("bits_per_symbol")
        )
        self.plot_widget.getPlotItem().setAutoPan(x=True)
        self.plot_widget.getPlotItem().setAutoVisible(y=True)

    def configureSettings(self):
        script_dir = os.path.dirname(__file__)

        default_settings = {"simulation_type": "shaping", "symbols_per_step": 2**16}
        default_settings_metadata = {
            "symbols_per_step": {
                "preferred_handler": QSpinBox,
                "preferred_handler_fn": lambda handler: handler.setRange(1, 2**20),
            },
            "simulation_type": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "shaping": "shaping",
                    "equalization": "equalization",
                },
            },
        }
        self.settings = ConfigManager(
            filename=os.path.join(script_dir, "settings", "settings.json")
        )
        self.settings.set_defaults(default_settings)
        self.settings.set_many_metadata(default_settings_metadata)
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
            "SNR": 12.0,
            "LW": 100e3,
            "symbol_rate": 32e9,
            "show_labels": True,
            "qam_init": True,
        }
        shaping_default_metadata = {
            "lr": {
                "preferred_handler": QDoubleSpinBox,
                "preferred_handler_fn": lambda handler: (
                    handler.setRange(0, 1),
                    handler.setDecimals(5),
                    handler.setSingleStep(0.00001),
                ),
            },
            "LW": {
                "preferred_handler": QDoubleSpinBox,
                "preferred_handler_fn": lambda handler: handler.setRange(0, 5e6),
            },
            "symbol_rate": {
                "preferred_handler": QDoubleSpinBox,
                "preferred_handler_fn": lambda handler: handler.setRange(10e9, 100e9),
            },
            "bits_per_symbol": {
                "preferred_handler": QSpinBox,
                "preferred_handler_fn": lambda handler: (
                    handler.setRange(1, 10),
                    handler.setSingleStep(2),
                ),
            },
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
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "None": settings.CPE.NONE,
                    "BPS": settings.CPE.BPS,
                    "V&V": settings.CPE.VV,
                },
            },
            "objective": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "BMI": settings.ShapingObjective.BMI,
                    "MI": settings.ShapingObjective.MI,
                },
            },
            "type": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "Geometric": settings.ShapingType.Geometric,
                    "Probabilistic": settings.ShapingType.Probabilistic,
                    "Joint Geometric & Probabilistic": settings.ShapingType.Joint,
                },
            },
        }
        self.shaping_settings.set_many_metadata(shaping_default_metadata)
        self.shaping_settings.set_defaults(shaping_default_settings)
        equalization_default_settings = {
            "SNR": 24.0,
            "eq_len": 25,
            "channel": "h0",
            "constellation": "64-QAM",
            "num_frames": 100,
            "nu": 0.0270955,
            "batch_len": 200,
            "tau_cd": -2.6,
            "tau_pmd": 5.0,
            "learning_rate": 0.003,
            "var_from_estimate": False,
        }
        equalization_default_metadata = {
            "learning_rate": {
                "preferred_handler": QDoubleSpinBox,
                "preferred_handler_fn": lambda handler: (
                    handler.setRange(0, 1),
                    handler.setDecimals(5),
                    handler.setSingleStep(0.001),
                ),
            },
            "num_frames": {
                "preferred_handler": QSpinBox,
                "preferred_handler_fn": lambda handler: (
                    handler.setRange(1, 1000),
                    handler.setSingleStep(10),
                ),
            },
            "constellation": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "64-QAM": "64-QAM",
                    "16-QAM": "16-QAM",
                    "QPSK": "QPSK",
                },
            },
            "channel": {
                "preferred_handler": QComboBox,
                "preferred_map_dict": {
                    "only optical impairments": "h0",
                    "h1 (Caciularu)": "h1",
                    "h2 (Caciularu)": "h2",
                },
            },
            "nu": {
                "preferred_handler": QDoubleSpinBox,
                "preferred_handler_fn": lambda handler: (
                    handler.setRange(0, 1),
                    handler.setDecimals(5),
                    handler.setSingleStep(0.00001),
                ),
            },
            "batch_len": {
                "preferred_handler": QSpinBox,
                "preferred_handler_fn": lambda handler: (
                    handler.setRange(1, 1000),
                    handler.setSingleStep(100),
                ),
            },
        }
        self.equalization_settings = ConfigManager(
            filename=os.path.join(script_dir, "settings", "equalization_settings.json")
        )
        self.equalization_settings.set_many_metadata(equalization_default_metadata)
        self.equalization_settings.set_defaults(equalization_default_settings)

    def configureGUI(self):
        # Configure options for Shaping
        self.bitpersymbol_box.setSingleStep(2)
        self.bitpersymbol_box.setRange(1, 10)
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

        self.equalization_settings.add_handler(
            "learning_rate", self.eq_learning_rate_box
        )

        self.equalization_settings.add_handler("batch_len", self.eq_block_length_box)

        self.equalization_settings.add_handler("nu", self.eq_shaping_parameter_box)

        self.equalization_settings.add_handler(
            "channel", self.eq_channel_box, preferred_mapper=True
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
        if (
            self.settings.get("simulation_type") == "shaping"
            and self.settings_group.currentIndex() == 1
        ):
            self.settings_group.setCurrentIndex(0)
            # Stop the old simulation and reset the plots
            self.stopSimulation()
            self.resetPlots()
            self.setupTraining()

        elif (
            self.settings.get("simulation_type") == "equalization"
            and self.settings_group.currentIndex() == 0
        ):
            self.settings_group.setCurrentIndex(1)
            # Stop the old simulation and reset the plots
            self.stopSimulation()
            self.resetPlots()
            self.setupTraining()

        self.reconfigureGraphItems()
        if self.settings.get("simulation_type") == "shaping":
            self.worker.reconfigure(
                {**self.settings.as_dict(), **self.shaping_settings.as_dict()}
            )
        elif self.settings.get("simulation_type") == "equalization":
            self.worker.reconfigure(
                {**self.settings.as_dict(), **self.equalization_settings.as_dict()}
            )
        # Handle MI/BMI
        if (
            self.settings.get("simulation_type") == "shaping"
            and self.shaping_settings.get("objective") == settings.ShapingObjective.MI
        ):
            self.plot_widget.getPlotItem().setLabel("left", "MI (bit/symbol)")
        else:
            self.plot_widget.getPlotItem().setLabel("left", "BMI (bit/symbol)")

    @Slot()
    def handleSimulationRunning(self, running):
        # This is called whenever the running state changes
        pass

    @Slot()
    def plotConstellation(self, constellation, probabilities=None):
        constellation_array = np.concatenate(
            (constellation.real[:, None], constellation.imag[:, None]), axis=1
        )
        m = self.shaping_settings.get("bits_per_symbol")
        M = 2**m
        labels = [s[2:] for s in vhex(np.arange(M))]
        bitstrings = [str(s) for s in mokka.utils.hex2bits(labels, m)]
        if probabilities is None:
            scatter_size = 5
        else:
            scatter_size = (2**m) * 5 * probabilities
        self.scatter1.setData(
            pos=constellation_array,
            size=scatter_size,
            color=(0, 150, 130, 255),
        )
        if not self.shaping_settings.get("show_labels"):
            return
        for bitstring, point, label in zip(bitstrings, constellation, self.labeltexts):
            label.setText(bitstring)
            label.setPos(float(point.real), float(point.imag))

    def configureSymbolsPlot(self, scatter):
        @Slot()
        def plotSymbols(symbols):
            symbols_array = np.concatenate(
                (symbols.real[:, None], symbols.imag[:, None]), axis=1
            )
            scatter.clear()
            scatter.setData(pos=symbols_array, size=2, pen=pg.mkPen())

        return plotSymbols

    @Slot()
    def handleProgress(self, progress, result):
        self.epochs.append(progress)
        if self.settings.get("simulation_type") == "shaping" and self.shaping_settings.get("objective") == settings.ShapingObjective.MI:
            self.bmi.append(result["mi"])
        else:
            self.bmi.append(result["bmi"])
        self.performance.setData(self.epochs, self.bmi)
        if self.settings.get("simulation_type") == "equalization":
            self.entropy.setPos(result["entropy"])
        else:
            self.entropy.setPos(self.shaping_settings.get("bits_per_symbol"))

    def setupTraining(self):
        if self.settings.get("simulation_type") == "shaping":
            settings = {**self.settings.as_dict(), **self.shaping_settings.as_dict()}
        elif self.settings.get("simulation_type") == "equalization":
            settings = {
                **self.settings.as_dict(),
                **self.equalization_settings.as_dict(),
            }
        self.worker = Training(settings)

    def runTraining(self):
        self.worker.moveToThread(QCoreApplication.instance().thread())
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        if self.settings.get("simulation_type") == "shaping":
            self.scatter_widget1.getPlotItem().setTitle("Transmit Constellation")
            self.scatter_widget2.getPlotItem().setTitle("Received Symbols")
            self.worker.symbols1.connect(self.plotConstellation)
            self.worker.symbols2.connect(self.configureSymbolsPlot(self.scatter2))
        elif self.settings.get("simulation_type") == "equalization":
            self.scatter_widget1.getPlotItem().setTitle("Received Symbols X-Pol")
            self.scatter_widget2.getPlotItem().setTitle("Received Symbols Y-Pol")
            self.worker.symbols1.connect(self.configureSymbolsPlot(self.scatter1))
            self.worker.symbols2.connect(self.configureSymbolsPlot(self.scatter2))
        self.worker.progress_result.connect(self.handleProgress)
        # self.stop_simulation.connect(self.worker.killed)
        self.stop_simulation.connect(self.check_signal)
        self.worker_thread.start()

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
        self.shaping_settings.updated.connect(self.handleSettingsChange)
        self.equalization_settings.updated.connect(self.handleSettingsChange)
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

    def stopSimulation(self):
        self.worker_thread.requestInterruption()
        self.simulation_running.emit(False)
        self.worker.moveToThread(QCoreApplication.instance().thread())
        self.worker_thread.quit()
        self.run_btn.setFlat(False)
        self.run_btn.setText("Run")

    def startSimulation(self):
        self.run_btn.setFlat(True)
        self.run_btn.setText("Stop && Reset")
        self.resetPlots()
        self.runTraining()
        self.simulation_running.emit(True)

    @Slot()
    def runBtn_clicked(self):
        if self.run_btn.isFlat():
            self.stopSimulation()
            return
        self.startSimulation()

    @Slot()
    def resetBtn_clicked(self):
        # ResetPlots
        self.resetPlots()
        self.setupTraining()

    def resetPlots(self):
        self.scatter_widget1.clear()
        self.scatter_widget2.clear()
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
