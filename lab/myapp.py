"""
Created on Tue Jan 13 23:47:24 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import config

import sys

from PySide6.QtCore import QLibraryInfo, qVersion, Qt
from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QLabel, QLineEdit, QDockWidget, QTextEdit, QSplitter, QTabWidget, QToolBar)
from PySide6.QtGui import QPixmap, QAction

class MainWindow(QMainWindow):
    def __init__(self, title="main", size=(400, 300)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(size[0], size[1])

class Window(QWidget):
    def __init__(self, title="", size=(200, 200)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(size[0], size[1])

class MyMainWindow(QMainWindow):
    def __init__(self, title="main", size=(1500,1200)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(size[0], size[1])

        #### dock widget
        dock = QDockWidget("side panel", self)
        dock.setWidget(QTextEdit("dock content"))
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        #### splitter
        splitter = QSplitter()
        left = QLabel("left area")
        left.setStyleSheet("background:#ddd; padding;20px;")
        splitter.addWidget(left)
        tabs = QTabWidget()
        page1 = QWidget()
        page1.setLayout(QVBoxLayout())
        page1.layout().addWidget(QLabel("tab1 content"))
        page2 = QWidget()
        page2.setLayout(QVBoxLayout())
        page2.layout().addWidget(QLabel("tab2 content"))
        tabs.addTab(page1, "tab1")
        tabs.addTab(page2, "tab2")
        splitter.addWidget(tabs)
        splitter.setSizes([200, 400])
        self.setCentralWidget(splitter)

        #### toolbar
        toolbar = QToolBar("tool")
        self.addToolBar(toolbar)
        action_msg = QAction("message", self)
        action_msg.triggered.connect(lambda: print("toolbar click"))
        toolbar.addAction(action_msg)


        # #### center widget
        # central = QWidget()
        # self.setCentralWidget(central)

        # #### left
        # self.img_label = QLabel()
        # pixmap = QPixmap(config.ROOT/"assets"/"sample.jpg")
        # self.img_label.setPixmap(pixmap)
        # self.img_label.setScaledContents(True)
        # self.img_label.setFixedSize(400, 400)

        # ### right
        # self.txt_input = QLineEdit()
        # self.txt_input .setPlaceholderText("input here")

        # self.btn_dialog = QPushButton("message")
        # self.btn_dialog.clicked.connect(self.show_message)

        # self.btn_show_text = QPushButton("dispaly input content")
        # self.btn_show_text.clicked.connect(self.show_input_text)

        # right_layout = QVBoxLayout()
        # right_layout.addWidget(self.txt_input)
        # right_layout.addWidget(self.btn_dialog)
        # right_layout.addWidget(self.btn_show_text)
        # right_layout.addStretch()

        # top_layout = QHBoxLayout()
        # top_layout.addWidget(self.img_label)
        # top_layout.addLayout(right_layout)

        # main_layout = QVBoxLayout()
        # main_layout.addLayout(top_layout)
        # central.setLayout(main_layout)

        # #### status bar
        # self.statusBar().showMessage("ready")

        # #### menu bar
        # menu = self.menuBar()
        # file_menu = menu.addMenu("file")
        # action_quit = file_menu.addAction("exit")
        # action_quit.triggered.connect(self.close)


    def show_message(self):
        QMessageBox.information(self, "information", "this is a message dialog")

    def show_input_text(self):
        text = self.txt_input.text()
        QMessageBox.information(self, "content", text)

class MyWindow(QWidget):
    def __init__(self, title="", size=(1500, 1200)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(size[0], size[1])

        #### disp img
        self.image_label = QLabel()
        pixmap = QPixmap(config.ROOT/"assets"/"sample.jpg")
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(200, 200)

        #### input txt
        self.txt_inp = QLineEdit()
        self.txt_inp.setPlaceholderText("input here")

        #### message
        self.btn_dialog = QPushButton("display message")
        self.btn_dialog.clicked.connect(self.show_message)

        #### get input content
        self.btn_get_txt = QPushButton("display input content")
        self.btn_get_txt.clicked.connect(self.show_input_text)

        self.btn_count = QPushButton("count")
        self.btn_count.clicked.connect(self.on_button_clicked)
        self.count = 0

        #### layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.txt_inp)
        layout.addWidget(self.btn_dialog)
        layout.addWidget(self.btn_get_txt)
        layout.addWidget(self.btn_count)
        self.setLayout(layout)

    def on_button_clicked(self):
        self.count += 1
        print(f"you have clicked button: {self.count}")

    def show_message(self):
        QMessageBox.information(self, "information", "this is message dialog")

    def show_input_text(self):
        text = self.txt_inp.text()
        QMessageBox.information(self, "input content", text)


def on_click():
    print("you have clicked!")

if __name__ == "__main__":
    print("---- test ----")

    pyversion = sys.version_info
    print(f"python {pyversion[0]}.{pyversion[1]}.{pyversion[2]}, {sys.platform}")
    print(QLibraryInfo.build())
    print(qVersion())

    app = QApplication(sys.argv)

    # w = MainWindow()
    # w.show()

    # window = Window(title="sub")
    # window.show()

    # count_app = MyWindow()
    # count_app.show()

    main_app = MyMainWindow()
    main_app.show()

    sys.exit(app.exec())



