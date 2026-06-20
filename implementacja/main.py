#!/usr/bin/env python3
"""ConvexML — Convex Optimization Explorer
Entry point for the desktop application.
"""
import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ConvexML")
    app.setOrganizationName("ML Education")

    # High-DPI
    # app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Monospace default font
    font = QFont("JetBrains Mono", 10)
    font.setStyleHint(QFont.StyleHint.Monospace)
    app.setFont(font)

    window = MainWindow()
    window._setup_tab_refresh()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
