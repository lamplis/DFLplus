from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

class QImageDB():
    @staticmethod
    def initialize(image_path):
        QImageDB.intro = QImage ( str(image_path / 'intro.png') )
