import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from localization import StringsDB

from .QXMainWindow import *
            
                
class QActionEx(QAction):
    def __init__(self, icon, text, shortcut=None, trigger_func=None, shortcut_in_tooltip=False, is_checkable=False, is_auto_repeat=False ):
        super().__init__(icon, text)
        if shortcut is not None:
            self.setShortcut(shortcut)
            if shortcut_in_tooltip:
                
                self.setToolTip( f"{text} ( {StringsDB['S_HOT_KEY'] }: {shortcut} )")
                
        if trigger_func is not None:
            self.triggered.connect(trigger_func)
        if is_checkable:            
            self.setCheckable(True)
        self.setAutoRepeat(is_auto_repeat)
            
def QImage_from_np(img):
    if img.dtype != np.uint8:
        raise ValueError("img should be in np.uint8 format")

    if img.ndim == 2:
        img = img[:, :, None]
    # QImage keeps a pointer to this memory and does not copy it.
    # Callers often pass a temporary array, so copy the pixels now.
    img = np.ascontiguousarray(img)
    h, w, c = img.shape
    if c == 1:
        fmt = QImage.Format.Format_Grayscale8
    elif c == 3:
        fmt = QImage.Format.Format_BGR888
    elif c == 4:
        fmt = QImage.Format.Format_ARGB32
    else:
        raise ValueError("unsupported channel count")

    return QImage(img.data, w, h, c * w, fmt).copy()
        
def QImage_to_np(q_img, fmt=QImage.Format.Format_BGR888):
    q_img = q_img.convertToFormat(fmt)

    width = q_img.width()
    height = q_img.height()
    if fmt == QImage.Format.Format_Grayscale8:
        channels = 1
    elif fmt == QImage.Format.Format_ARGB32:
        channels = 4
    else:
        channels = 3

    bits = q_img.constBits()
    bits.setsize(q_img.sizeInBytes())
    # Qt pads each row. Drop that padding before the channel reshape.
    stride = q_img.bytesPerLine()
    rows = np.frombuffer(bits, np.uint8).reshape((height, stride))
    return rows[:, : width * channels].reshape((height, width, channels)).copy()
        
def QPixmap_from_np(img):    
    return QPixmap.fromImage(QImage_from_np(img))
    
def QPoint_from_np(n):
    return QPoint(*n.astype(np.int32))

def QPointF_from_np(n):
    # QPainterPath in PyQt6 accepts QPointF only. QPoint is not converted.
    return QPointF(float(n[0]), float(n[1]))

def QPoint_to_np(q):
    return np.int32( [q.x(), q.y()] )
    
def QSize_to_np(q):
    return np.int32( [q.width(), q.height()] )
    
class QDarkPalette(QPalette):
    def __init__(self):
        super().__init__()
        text_color = QColor(200,200,200)
        self.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.WindowText, text_color )
        self.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        self.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ToolTipBase, text_color )
        self.setColor(QPalette.ColorRole.ToolTipText, text_color )
        self.setColor(QPalette.ColorRole.Text, text_color ) 
        self.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        self.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)