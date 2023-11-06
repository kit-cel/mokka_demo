from PySide6 import QtWidgets
import sys

from pyqtconfig import ConfigManager
from math import ceil

class MultiConfigDialog(QtWidgets.QDialog):
    """
    A Dialog class inheriting from QtWidgets.QDialog. This class creates layout from the input config using
    build_config_layout, as well as QDialogButtonBox with Ok and Cancel buttons.
    """
    def __init__(self, config1, config2, *args, cols=None, **kwargs):
        if "PyQt5" in sys.modules:
            f = kwargs.pop("f", None)
            if f is not None:
                kwargs["flags"] = f
        elif "PySide2" in sys.modules:
            flags = kwargs.pop("flags", None)
            if flags is not None:
                kwargs["f"] = flags

        super().__init__(*args, **kwargs)
        config1_dict = config1.all_as_dict()
        self.config1 = ConfigManager(config1_dict["defaults"])
        self.config1.set_many(config1_dict["config"])
        self.config1.set_many_metadata(config1_dict["metadata"])

        config2_dict = config2.all_as_dict()
        self.config2 = ConfigManager(config2_dict["defaults"])
        self.config2.set_many(config2_dict["config"])
        self.config2.set_many_metadata(config2_dict["metadata"])

        # Build layout from settings
        config_layout_kwargs = {} if cols is None else {"cols": cols}
        config1_layout = build_config_layout(self.config1, **config_layout_kwargs)
        config2_layout = build_config_layout(self.config2, **config_layout_kwargs)

        # Create a button box for the dialog
        if "PyQt6" in sys.modules:
            button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Reset | QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        else:
            button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Reset | QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # QDialogButtonBox places Reset after Ok and Cancel
        button_box.buttons()[2].setText("Reset to Defaults")
        button_box.buttons()[2].clicked.connect(self.show_confirm_reset_dialog)

        # Place everything in a layout in the dialog
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(config1_layout)
        layout.addLayout(config2_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def show_confirm_reset_dialog(self):
        message_box = QtWidgets.QMessageBox(self, text="Are you sure you want to reset to defaults?")
        message_box.setWindowTitle("Warning")
        if "PyQt6" in sys.modules:
            message_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel)
        else:
            message_box.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        message_box.buttonClicked.connect(
            lambda button: ((self.config1.set_many(self.config1.defaults), self.config2.set_many(self.config2.defaults)) if "OK" in button.text() else None))
        message_box.exec()


def build_config_layout(config, cols=2):
    """
    Generate a layout based on the input ConfigManager. The layout consists of a user specified number of columns of
    QFormLayout. In each row of the QFormLayout, the label is the config dict key, and the field is the config handler
    for that key.

    :param config: ConfigManager
    :param cols: Number of columns to use
    :return: QHBoxLayout
    """
    h_layout = QtWidgets.QHBoxLayout()
    forms = [QtWidgets.QFormLayout() for _ in range(cols)]
    for form in forms:
        h_layout.addLayout(form)

    num_items = len(config.get_visible_keys())
    for i, key in enumerate(config.get_visible_keys()):
        # Find which column to put the setting in. Columns are filled equally, with remainder to the left. Each column
        # is filled before proceeding to the next.
        f_index = 0
        for j in range(cols):
            if (i+1) <= ceil((j+1)*num_items/cols):
                f_index = j
                break

        # Get the handler widget for the key
        if key in config.handlers:
            # If we've already defined a handler, use that
            input_widget = config.handlers[key]
        else:
            # Otherwise, try to add a handler. If it fails, skip this row
            config.add_handler(key)
            if key not in config.handlers:
                continue
            else:
                input_widget = config.handlers[key]

        label = QtWidgets.QLabel(key)
        forms[f_index].addRow(label, input_widget)

    return h_layout
