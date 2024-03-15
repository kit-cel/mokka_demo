# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interactive_learning_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QSpinBox, QStatusBar, QTabWidget, QVBoxLayout,
    QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(810, 740)
        self.actionChannel_Settings = QAction(MainWindow)
        self.actionChannel_Settings.setObjectName(u"actionChannel_Settings")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_4 = QHBoxLayout(self.widget)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.settings_group = QTabWidget(self.widget)
        self.settings_group.setObjectName(u"settings_group")
        self.settings_group.setMaximumSize(QSize(16777215, 150))
        self.settings_group_shaping = QWidget()
        self.settings_group_shaping.setObjectName(u"settings_group_shaping")
        self.horizontalLayout_2 = QHBoxLayout(self.settings_group_shaping)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.frame = QFrame(self.settings_group_shaping)
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

        self.frame_2 = QFrame(self.settings_group_shaping)
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

        self.frame_3 = QFrame(self.settings_group_shaping)
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

        self.frame_4 = QFrame(self.settings_group_shaping)
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

        self.settings_group.addTab(self.settings_group_shaping, "")
        self.settings_group_equalization = QWidget()
        self.settings_group_equalization.setObjectName(u"settings_group_equalization")
        self.settings_group_equalization.setEnabled(True)
        self.horizontalLayout_6 = QHBoxLayout(self.settings_group_equalization)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.frame_7 = QFrame(self.settings_group_equalization)
        self.frame_7.setObjectName(u"frame_7")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.frame_7)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.label_5 = QLabel(self.frame_7)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_7.addWidget(self.label_5)

        self.eq_constellation_box = QComboBox(self.frame_7)
        self.eq_constellation_box.setObjectName(u"eq_constellation_box")

        self.verticalLayout_7.addWidget(self.eq_constellation_box)


        self.horizontalLayout_6.addWidget(self.frame_7)

        self.frame_8 = QFrame(self.settings_group_equalization)
        self.frame_8.setObjectName(u"frame_8")
        sizePolicy.setHeightForWidth(self.frame_8.sizePolicy().hasHeightForWidth())
        self.frame_8.setSizePolicy(sizePolicy)
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_8)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_6 = QLabel(self.frame_8)
        self.label_6.setObjectName(u"label_6")

        self.verticalLayout_8.addWidget(self.label_6)

        self.eq_channel_box = QComboBox(self.frame_8)
        self.eq_channel_box.setObjectName(u"eq_channel_box")

        self.verticalLayout_8.addWidget(self.eq_channel_box)


        self.horizontalLayout_6.addWidget(self.frame_8, 0, Qt.AlignLeft)

        self.settings_group.addTab(self.settings_group_equalization, "")

        self.horizontalLayout_4.addWidget(self.settings_group, 0, Qt.AlignHCenter)

        self.run_group = QGroupBox(self.widget)
        self.run_group.setObjectName(u"run_group")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.run_group.sizePolicy().hasHeightForWidth())
        self.run_group.setSizePolicy(sizePolicy1)
        self.horizontalLayout_5 = QHBoxLayout(self.run_group)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.frame_5 = QFrame(self.run_group)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.settings_btn = QPushButton(self.frame_5)
        self.settings_btn.setObjectName(u"settings_btn")

        self.horizontalLayout_3.addWidget(self.settings_btn)


        self.horizontalLayout_5.addWidget(self.frame_5)

        self.frame_6 = QFrame(self.run_group)
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
        self.run_btn.setMinimumSize(QSize(20, 0))

        self.verticalLayout_6.addWidget(self.run_btn)


        self.horizontalLayout_5.addWidget(self.frame_6)


        self.horizontalLayout_4.addWidget(self.run_group)


        self.verticalLayout.addWidget(self.widget)

        self.plot_box = QWidget(self.centralwidget)
        self.plot_box.setObjectName(u"plot_box")
        self.gridLayout = QGridLayout(self.plot_box)
        self.gridLayout.setObjectName(u"gridLayout")
        self.constellation_widget = PlotWidget(self.plot_box)
        self.constellation_widget.setObjectName(u"constellation_widget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.constellation_widget.sizePolicy().hasHeightForWidth())
        self.constellation_widget.setSizePolicy(sizePolicy2)
        self.constellation_widget.setMinimumSize(QSize(500, 500))
        self.constellation_widget.setBaseSize(QSize(300, 300))

        self.gridLayout.addWidget(self.constellation_widget, 0, 0, 1, 1)

        self.plot_widget = PlotWidget(self.plot_box)
        self.plot_widget.setObjectName(u"plot_widget")
        sizePolicy1.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        self.plot_widget.setSizePolicy(sizePolicy1)
        self.plot_widget.setMinimumSize(QSize(0, 500))

        self.gridLayout.addWidget(self.plot_widget, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.plot_box)

        self.footer_area = QFrame(self.centralwidget)
        self.footer_area.setObjectName(u"footer_area")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.footer_area.sizePolicy().hasHeightForWidth())
        self.footer_area.setSizePolicy(sizePolicy3)
        palette = QPalette()
        brush = QBrush(QColor(128, 128, 128, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.footer_area.setPalette(palette)
        self.footer_area.setAutoFillBackground(False)
        self.footer_area.setStyleSheet(u".QFrame {border-radius: 5px; background-color: grey}\n"
".QLabel {background-color:grey}")
        self.horizontalLayout = QHBoxLayout(self.footer_area)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.acknowledgment_label = QLabel(self.footer_area)
        self.acknowledgment_label.setObjectName(u"acknowledgment_label")
        palette1 = QPalette()
        brush1 = QBrush(QColor(0, 0, 0, 217))
        brush1.setStyle(Qt.SolidPattern)
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Button, brush)
        palette1.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.acknowledgment_label.setPalette(palette1)

        self.horizontalLayout.addWidget(self.acknowledgment_label)

        self.logo_area = QWidget(self.footer_area)
        self.logo_area.setObjectName(u"logo_area")
        sizePolicy3.setHeightForWidth(self.logo_area.sizePolicy().hasHeightForWidth())
        self.logo_area.setSizePolicy(sizePolicy3)
        self.horizontalLayout_7 = QHBoxLayout(self.logo_area)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.erc_logo = QLabel(self.logo_area)
        self.erc_logo.setObjectName(u"erc_logo")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.erc_logo.sizePolicy().hasHeightForWidth())
        self.erc_logo.setSizePolicy(sizePolicy4)
        self.erc_logo.setMinimumSize(QSize(50, 50))
        self.erc_logo.setAutoFillBackground(False)

        self.horizontalLayout_7.addWidget(self.erc_logo)

        self.cel_logo = QLabel(self.logo_area)
        self.cel_logo.setObjectName(u"cel_logo")
        sizePolicy4.setHeightForWidth(self.cel_logo.sizePolicy().hasHeightForWidth())
        self.cel_logo.setSizePolicy(sizePolicy4)
        self.cel_logo.setMinimumSize(QSize(50, 50))
        self.cel_logo.setAutoFillBackground(False)

        self.horizontalLayout_7.addWidget(self.cel_logo)

        self.kit_logo = QLabel(self.logo_area)
        self.kit_logo.setObjectName(u"kit_logo")
        sizePolicy4.setHeightForWidth(self.kit_logo.sizePolicy().hasHeightForWidth())
        self.kit_logo.setSizePolicy(sizePolicy4)
        self.kit_logo.setMinimumSize(QSize(50, 50))
        self.kit_logo.setBaseSize(QSize(0, 0))
        self.kit_logo.setAutoFillBackground(False)

        self.horizontalLayout_7.addWidget(self.kit_logo)


        self.horizontalLayout.addWidget(self.logo_area, 0, Qt.AlignRight)


        self.verticalLayout.addWidget(self.footer_area)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 810, 24))
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo = QMenu(self.menubar)
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.setObjectName(u"menuJoint_Geometric_and_Probabilistic_Shaping_Demo")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.menuAction())
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.addAction(self.actionChannel_Settings)

        self.retranslateUi(MainWindow)

        self.settings_group.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionChannel_Settings.setText(QCoreApplication.translate("MainWindow", u"Channel Settings", None))
#if QT_CONFIG(accessibility)
        self.settings_group_shaping.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
        self.label.setText(QCoreApplication.translate("MainWindow", u"Bits/Symbol", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Channel", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Shaping Type", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Objective function", None))
        self.settings_group.setTabText(self.settings_group.indexOf(self.settings_group_shaping), "")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Constellation", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Channel Type", None))
        self.settings_group.setTabText(self.settings_group.indexOf(self.settings_group_equalization), QCoreApplication.translate("MainWindow", u"Page", None))
        self.run_group.setTitle(QCoreApplication.translate("MainWindow", u"Run Settings", None))
        self.settings_btn.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.reset_btn.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.run_btn.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.acknowledgment_label.setText(QCoreApplication.translate("MainWindow", u"This work has received funding from the European Research Council (ERC)\n"
" under the European Union\u2019s Horizon 2020 research and innovation programme \n"
"(grant agreement No. 101001899).", None))
        self.erc_logo.setText("")
        self.cel_logo.setText("")
        self.kit_logo.setText("")
        self.menuJoint_Geometric_and_Probabilistic_Shaping_Demo.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
    # retranslateUi

