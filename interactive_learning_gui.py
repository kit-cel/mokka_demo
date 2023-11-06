# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interactive_learning_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSpinBox,
    QStatusBar, QVBoxLayout, QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.actionChannel_Settings = QAction(MainWindow)
        self.actionChannel_Settings.setObjectName(u"actionChannel_Settings")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMaximumSize(QSize(16777215, 150))
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.frame = QFrame(self.groupBox)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")

        self.verticalLayout_2.addWidget(self.label)

        self.bitpersymbol_box = QSpinBox(self.frame)
        self.bitpersymbol_box.setObjectName(u"bitpersymbol_box")

        self.verticalLayout_2.addWidget(self.bitpersymbol_box)


        self.horizontalLayout_2.addWidget(self.frame)

        self.frame_2 = QFrame(self.groupBox)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_2 = QLabel(self.frame_2)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_3.addWidget(self.label_2)

        self.channel_box = QComboBox(self.frame_2)
        self.channel_box.setObjectName(u"channel_box")

        self.verticalLayout_3.addWidget(self.channel_box)


        self.horizontalLayout_2.addWidget(self.frame_2)

        self.frame_3 = QFrame(self.groupBox)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_3 = QLabel(self.frame_3)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_4.addWidget(self.label_3)

        self.shaping_box = QComboBox(self.frame_3)
        self.shaping_box.setObjectName(u"shaping_box")

        self.verticalLayout_4.addWidget(self.shaping_box)


        self.horizontalLayout_2.addWidget(self.frame_3)

        self.frame_4 = QFrame(self.groupBox)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_4 = QLabel(self.frame_4)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_5.addWidget(self.label_4)

        self.objective_box = QComboBox(self.frame_4)
        self.objective_box.setObjectName(u"objective_box")

        self.verticalLayout_5.addWidget(self.objective_box)


        self.horizontalLayout_2.addWidget(self.frame_4)

        self.frame_5 = QFrame(self.groupBox)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_2.addWidget(self.frame_5)

        self.frame_6 = QFrame(self.groupBox)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_6)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.reset_btn = QPushButton(self.frame_6)
        self.reset_btn.setObjectName(u"reset_btn")

        self.verticalLayout_6.addWidget(self.reset_btn)

        self.run_btn = QPushButton(self.frame_6)
        self.run_btn.setObjectName(u"run_btn")

        self.verticalLayout_6.addWidget(self.run_btn)


        self.horizontalLayout_2.addWidget(self.frame_6)


        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.constellation_widget = PlotWidget(self.groupBox_2)
        self.constellation_widget.setObjectName(u"constellation_widget")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.constellation_widget.sizePolicy().hasHeightForWidth())
        self.constellation_widget.setSizePolicy(sizePolicy)
        self.constellation_widget.setMinimumSize(QSize(500, 500))
        self.constellation_widget.setBaseSize(QSize(300, 300))

        self.horizontalLayout.addWidget(self.constellation_widget)

        self.plot_widget = PlotWidget(self.groupBox_2)
        self.plot_widget.setObjectName(u"plot_widget")

        self.horizontalLayout.addWidget(self.plot_widget)


        self.verticalLayout.addWidget(self.groupBox_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 24))
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo = QMenu(self.menubar)
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.setObjectName(u"menuJoint_Geometric_and_Probabilistic_Shaping_Demo")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.menuAction())
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.addAction(self.actionChannel_Settings)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionChannel_Settings.setText(QCoreApplication.translate("MainWindow", u"Channel Settings", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Controls", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Bits/Symbol", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Channel", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Shaping Type", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Objective function", None))
        self.reset_btn.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.run_btn.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Plots", None))
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
    # retranslateUi

