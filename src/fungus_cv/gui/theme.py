"""The app's look: the Nocturne design tokens, a dark palette, one stylesheet and its icons.

Charts keep the report palette (``GRID``, ``INK``, ``INK_2``, ``SERIES``) so what the app
shows matches the exported reports.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

from fungus_cv.analyze.report import GRID, INK, INK_2, SERIES

# --- tokens (Nocturne: styles.css) ---------------------------------------------------------

BG = "#161826"
SURFACE = "#232532"
TEXT = "#e9e9ed"
ACCENT = "#9184d9"
NEUTRAL = {100: "#f3f5fe", 200: "#e4e7f5", 300: "#cfd3e5", 400: "#b2b6ca", 500: "#9397ab",
           600: "#75798c", 700: "#595d6c", 800: "#3f424d", 900: "#292b31"}
ACCENT_RAMP = {100: "#f5f4ff", 200: "#e7e5fe", 300: "#d2cefd", 400: "#b5abfc", 500: "#968ae0",
               600: "#796cbf", 700: "#5d5294", 800: "#423a6a", 900: "#2b2741"}
# Status marks: the ramps carry no red or green, so a failure is a warm tone kept at the
# accent's lightness and a pass is the accent itself.
GOOD = ACCENT_RAMP[300]
BAD = "#e0848f"
WARN = "#d9b36c"
RADIUS = 8


def mix(color: str, alpha: float, over: str | None = None) -> str:
    """``color`` at ``alpha`` — as rgba, or flattened onto ``over`` (an opaque colour)."""
    c = QColor(color)
    if over is None:
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {round(alpha * 255)})"
    o = QColor(over)
    blend = [round(a * alpha + b * (1 - alpha)) for a, b in
             zip((c.red(), c.green(), c.blue()), (o.red(), o.green(), o.blue()))]
    return "#{:02x}{:02x}{:02x}".format(*blend)


DIVIDER = mix(TEXT, 0.16)
MUTED = mix(TEXT, 0.55)

# --- icons (Phosphor, regular weight) ------------------------------------------------------

_PATHS = {
    "sparkle": "M197.58,129.06,146,110l-19-51.62a15.92,15.92,0,0,0-29.88,0L78,110l-51.62,19a15.92,15.92,0,0,0,0,29.88L78,178l19,51.62a15.92,15.92,0,0,0,29.88,0L146,178l51.62-19a15.92,15.92,0,0,0,0-29.88ZM137,164.22a8,8,0,0,0-4.74,4.74L112,223.85,91.78,169A8,8,0,0,0,87,164.22L32.15,144,87,123.78A8,8,0,0,0,91.78,119L112,64.15,132.22,119a8,8,0,0,0,4.74,4.74L191.85,144ZM144,40a8,8,0,0,1,8-8h16V16a8,8,0,0,1,16,0V32h16a8,8,0,0,1,0,16H184V64a8,8,0,0,1-16,0V48H152A8,8,0,0,1,144,40ZM248,88a8,8,0,0,1-8,8h-8v8a8,8,0,0,1-16,0V96h-8a8,8,0,0,1,0-16h8V72a8,8,0,0,1,16,0v8h8A8,8,0,0,1,248,88Z",  # noqa: E501
    "folder": "M216,72H131.31L104,44.69A15.86,15.86,0,0,0,92.69,40H40A16,16,0,0,0,24,56V200.62A15.4,15.4,0,0,0,39.38,216H216.89A15.13,15.13,0,0,0,232,200.89V88A16,16,0,0,0,216,72ZM40,56H92.69l16,16H40ZM216,200H40V88H216Z",  # noqa: E501
    "settings": "M128,80a48,48,0,1,0,48,48A48.05,48.05,0,0,0,128,80Zm0,80a32,32,0,1,1,32-32A32,32,0,0,1,128,160Zm88-29.84q.06-2.16,0-4.32l14.92-18.64a8,8,0,0,0,1.48-7.06,107.21,107.21,0,0,0-10.88-26.25,8,8,0,0,0-6-3.93l-23.72-2.64q-1.48-1.56-3-3L186,40.54a8,8,0,0,0-3.94-6,107.71,107.71,0,0,0-26.25-10.87,8,8,0,0,0-7.06,1.49L130.16,40Q128,40,125.84,40L107.2,25.11a8,8,0,0,0-7.06-1.48A107.6,107.6,0,0,0,73.89,34.51a8,8,0,0,0-3.93,6L67.32,64.27q-1.56,1.49-3,3L40.54,70a8,8,0,0,0-6,3.94,107.71,107.71,0,0,0-10.87,26.25,8,8,0,0,0,1.49,7.06L40,125.84Q40,128,40,130.16L25.11,148.8a8,8,0,0,0-1.48,7.06,107.21,107.21,0,0,0,10.88,26.25,8,8,0,0,0,6,3.93l23.72,2.64q1.49,1.56,3,3L70,215.46a8,8,0,0,0,3.94,6,107.71,107.71,0,0,0,26.25,10.87,8,8,0,0,0,7.06-1.49L125.84,216q2.16.06,4.32,0l18.64,14.92a8,8,0,0,0,7.06,1.48,107.21,107.21,0,0,0,26.25-10.88,8,8,0,0,0,3.93-6l2.64-23.72q1.56-1.48,3-3L215.46,186a8,8,0,0,0,6-3.94,107.71,107.71,0,0,0,10.87-26.25,8,8,0,0,0-1.49-7.06Zm-16.1-6.5a73.93,73.93,0,0,1,0,8.68,8,8,0,0,0,1.74,5.48l14.19,17.73a91.57,91.57,0,0,1-6.23,15L187,173.11a8,8,0,0,0-5.1,2.64,74.11,74.11,0,0,1-6.14,6.14,8,8,0,0,0-2.64,5.1l-2.51,22.58a91.32,91.32,0,0,1-15,6.23l-17.74-14.19a8,8,0,0,0-5-1.75h-.48a73.93,73.93,0,0,1-8.68,0,8,8,0,0,0-5.48,1.74L100.45,215.8a91.57,91.57,0,0,1-15-6.23L82.89,187a8,8,0,0,0-2.64-5.1,74.11,74.11,0,0,1-6.14-6.14,8,8,0,0,0-5.1-2.64L46.43,170.6a91.32,91.32,0,0,1-6.23-15l14.19-17.74a8,8,0,0,0,1.74-5.48,73.93,73.93,0,0,1,0-8.68,8,8,0,0,0-1.74-5.48L40.2,100.45a91.57,91.57,0,0,1,6.23-15L69,82.89a8,8,0,0,0,5.1-2.64,74.11,74.11,0,0,1,6.14-6.14A8,8,0,0,0,82.89,69L85.4,46.43a91.32,91.32,0,0,1,15-6.23l17.74,14.19a8,8,0,0,0,5.48,1.74,73.93,73.93,0,0,1,8.68,0,8,8,0,0,0,5.48-1.74L155.55,40.2a91.57,91.57,0,0,1,15,6.23L173.11,69a8,8,0,0,0,2.64,5.1,74.11,74.11,0,0,1,6.14,6.14,8,8,0,0,0,5.1,2.64l22.58,2.51a91.32,91.32,0,0,1,6.23,15l-14.19,17.74A8,8,0,0,0,199.87,123.66Z",  # noqa: E501
    "caret-down": "M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z",  # noqa: E501
    "caret-up": "M213.66,165.66a8,8,0,0,1-11.32,0L128,91.31,53.66,165.66a8,8,0,0,1-11.32-11.32l80-80a8,8,0,0,1,11.32,0l80,80A8,8,0,0,1,213.66,165.66Z",  # noqa: E501
    "check": "M229.66,77.66l-128,128a8,8,0,0,1-11.32,0l-56-56a8,8,0,0,1,11.32-11.32L96,188.69,218.34,66.34a8,8,0,0,1,11.32,11.32Z",  # noqa: E501
}


def _svg(name: str, color: str) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" fill="{color}">'
            f'<path d="{_PATHS[name]}"/></svg>').encode()


def icon_pixmap(name: str, color: str = TEXT, size: int = 16) -> QPixmap:
    """A Phosphor icon drawn crisply at the screen's pixel density."""
    from PySide6.QtGui import QGuiApplication

    ratio = QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() \
        else 1.0
    pixmap = QPixmap(QSize(round(size * ratio), round(size * ratio)))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(_svg(name, color))).render(painter, QRectF(0, 0, pixmap.width(),
                                                                       pixmap.height()))
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def icon(name: str, color: str = TEXT, size: int = 16) -> QIcon:
    return QIcon(icon_pixmap(name, color, size))


def _icon_files() -> dict[str, str]:
    """Stylesheets take images by path, so the few they use are written out once."""
    folder = Path(tempfile.gettempdir()) / "fungus-cv-theme"
    folder.mkdir(exist_ok=True)
    files = {}
    for key, (name, color) in {"down": ("caret-down", MUTED), "up": ("caret-up", MUTED),
                               "check": ("check", BG)}.items():
        path = folder / f"{key}.svg"
        data = _svg(name, color)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        files[key] = path.as_posix()
    return files


# --- palette and fonts ---------------------------------------------------------------------


def palette() -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.Window: BG, QPalette.WindowText: TEXT, QPalette.Base: SURFACE,
        QPalette.AlternateBase: mix(TEXT, 0.03, SURFACE), QPalette.Text: TEXT,
        QPalette.Button: SURFACE, QPalette.ButtonText: TEXT, QPalette.BrightText: TEXT,
        QPalette.Highlight: ACCENT_RAMP[800], QPalette.HighlightedText: ACCENT_RAMP[100],
        QPalette.ToolTipBase: SURFACE, QPalette.ToolTipText: TEXT, QPalette.Link: ACCENT,
        QPalette.LinkVisited: ACCENT_RAMP[400], QPalette.PlaceholderText: mix(TEXT, 0.4, SURFACE),
        QPalette.Light: NEUTRAL[700], QPalette.Midlight: NEUTRAL[800], QPalette.Mid: NEUTRAL[800],
        QPalette.Dark: NEUTRAL[900], QPalette.Shadow: "#0b0c13",
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(mix(TEXT, 0.4, BG)))
    return pal


def base_font() -> QFont:
    """Inter when it is installed, otherwise the system's own sans-serif."""
    font = QFont()
    if "Inter" in QFontDatabase.families():
        font = QFont("Inter")
    font.setPixelSize(13)
    return font


def heading_font(pixel_size: int) -> QFont:
    font = base_font()
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Medium)
    font.setLetterSpacing(QFont.PercentageSpacing, 98.5)
    return font


def label_font(pixel_size: int = 10) -> QFont:
    """The small uppercase labels (section names, kickers)."""
    font = base_font()
    font.setPixelSize(pixel_size)
    font.setCapitalization(QFont.AllUppercase)
    font.setLetterSpacing(QFont.AbsoluteSpacing, pixel_size * 0.1)
    return font


# --- stylesheet ----------------------------------------------------------------------------


def stylesheet() -> str:
    files = _icon_files()
    hover = mix(TEXT, 0.07)
    pressed = mix(TEXT, 0.14)
    accent_tint = mix(ACCENT, 0.12)
    accent_press = mix(ACCENT, 0.22)
    disabled = mix(TEXT, 0.38)
    return f"""
QMainWindow, QDialog, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BG};
}}
QToolTip {{
    color: {TEXT}; background: {SURFACE}; border: 1px solid {NEUTRAL[700]};
    border-radius: 6px; padding: 5px 8px;
}}
QLabel {{ background: transparent; }}
QLabel:disabled {{ color: {disabled}; }}

/* panels */
QGroupBox {{
    background: {SURFACE}; border: 1px solid {NEUTRAL[800]}; border-radius: {RADIUS}px;
    margin-top: 0; padding: 32px 10px 10px 10px; font-size: 14px; font-weight: 500;
}}
QGroupBox::title {{
    subcontrol-origin: padding; subcontrol-position: top left; left: 12px; top: 9px;
    color: {TEXT}; background: transparent;
}}
QGroupBox::title:disabled {{ color: {disabled}; }}

/* buttons: outlined, never filled */
QPushButton, QToolButton {{
    color: {TEXT}; background: transparent; border: 1px solid {DIVIDER};
    border-radius: {RADIUS}px; padding: 5px 12px; min-height: 20px; font-weight: 500;
}}
QPushButton:hover, QToolButton:hover {{ background: {hover}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {pressed}; }}
QPushButton:checked, QToolButton:checked {{
    color: {ACCENT}; border-color: {ACCENT}; background: {mix(ACCENT, 0.10)};
}}
QPushButton[primary="true"], QPushButton:default {{ color: {ACCENT}; border-color: {ACCENT}; }}
QPushButton[primary="true"]:hover, QPushButton:default:hover {{ background: {accent_tint}; }}
QPushButton[primary="true"]:pressed, QPushButton:default:pressed {{
    background: {accent_press};
}}
QPushButton:disabled, QToolButton:disabled, QPushButton[primary="true"]:disabled {{
    color: {disabled}; border-color: {mix(TEXT, 0.08)}; background: transparent;
}}
QPushButton::menu-indicator {{ image: none; width: 0; }}

/* fields */
QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    color: {TEXT}; background: {SURFACE}; border: 1px solid {DIVIDER};
    border-radius: {RADIUS}px; padding: 4px 8px; min-height: 20px;
    selection-background-color: {ACCENT_RAMP[700]}; selection-color: {ACCENT_RAMP[100]};
}}
QPlainTextEdit, QTextEdit {{ padding: 6px 8px; }}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover,
QPlainTextEdit:hover, QTextEdit:hover {{ border-color: {mix(TEXT, 0.45)}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled,
QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {disabled}; border-color: {mix(TEXT, 0.08)};
}}
QLineEdit:read-only {{ background: transparent; }}
QAbstractSpinBox {{ padding-right: 22px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 20px; border: none; background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin: 2px 2px 0 0; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 2px 2px 0; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {hover}; border-radius: 4px;
}}
QAbstractSpinBox::up-arrow {{ image: url({files["up"]}); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url({files["down"]}); width: 9px; height: 9px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: none; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none;
}}
QComboBox::down-arrow {{ image: url({files["down"]}); width: 10px; height: 10px; }}
QDateTimeEdit::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none;
}}
QDateTimeEdit::down-arrow {{ image: url({files["down"]}); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {NEUTRAL[700]}; border-radius: 6px; padding: 4px;
    outline: 0; selection-background-color: {accent_tint}; selection-color: {ACCENT};
}}

/* checkboxes and radios */
QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {disabled}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px; border: 1.5px solid {mix(TEXT, 0.3)}; background: transparent;
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT}; image: url({files["check"]});
}}
QRadioButton::indicator:checked {{
    border-color: {ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {ACCENT}, stop:0.5 {ACCENT}, stop:0.56 {BG}, stop:1 {BG});
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {mix(TEXT, 0.12)};
}}

/* lists, tables, trees */
QListView, QTreeView, QTableView {{
    background: {SURFACE}; alternate-background-color: {mix(TEXT, 0.025, SURFACE)};
    border: 1px solid {NEUTRAL[800]}; border-radius: {RADIUS}px; outline: 0;
    gridline-color: {mix(TEXT, 0.06)};
    selection-background-color: {accent_tint}; selection-color: {ACCENT_RAMP[200]};
}}
QGroupBox QListView, QGroupBox QTreeView, QGroupBox QTableView {{
    border-color: {mix(TEXT, 0.1)};
}}
QListView::item, QTreeView::item {{ padding: 4px 6px; border-radius: 6px; }}
QListView::item:hover, QTreeView::item:hover, QTableView::item:hover {{
    background: {mix(TEXT, 0.04)};
}}
QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background: {accent_tint}; color: {ACCENT_RAMP[200]};
}}
QTableView::item {{ padding: 2px 6px; border: none; }}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{
    background: {SURFACE}; color: {mix(TEXT, 0.6)}; font-size: 11px; font-weight: 500;
    border: none; border-bottom: 1px solid {DIVIDER}; padding: 6px 6px;
}}
QHeaderView::section:vertical {{ border-bottom: none; border-right: 1px solid {DIVIDER}; }}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

/* tabs: an accent underline, as in the workflow tabs */
QTabWidget::pane {{ border: none; border-top: 1px solid {DIVIDER}; top: -1px; }}
QTabWidget::tab-bar {{ left: 0; }}
QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent; color: {mix(TEXT, 0.66)}; padding: 7px 12px 9px;
    margin-right: 4px; border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom-color: {ACCENT}; }}
QTabBar::tab:disabled {{ color: {disabled}; }}

/* meters */
QProgressBar {{
    background: {mix(TEXT, 0.12)}; border: none; border-radius: 3px;
    max-height: 6px; min-height: 6px; color: transparent; text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 4px; background: {mix(TEXT, 0.12)}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: {ACCENT_RAMP[300]}; border: 2px solid {BG};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_RAMP[200]}; }}
QSlider::groove:horizontal:disabled, QSlider::sub-page:horizontal:disabled {{
    background: {mix(TEXT, 0.08)};
}}
QSlider::handle:horizontal:disabled {{ background: {NEUTRAL[700]}; }}

/* scrollbars and splitters */
QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle {{ background: {NEUTRAL[800]}; border-radius: 3px; }}
QScrollBar::handle:vertical {{ min-height: 24px; }}
QScrollBar::handle:horizontal {{ min-width: 24px; }}
QScrollBar::handle:hover {{ background: {NEUTRAL[700]}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 10px; }}
QSplitter::handle:vertical {{ height: 10px; }}
QGraphicsView {{ border: 1px solid {NEUTRAL[800]}; border-radius: {RADIUS}px; }}

/* menus and the status line */
QMenuBar {{ background: {BG}; color: {TEXT}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {hover}; }}
QMenu {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {NEUTRAL[700]};
    border-radius: {RADIUS}px; padding: 5px;
}}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {accent_tint}; color: {ACCENT_RAMP[200]}; }}
QMenu::item:disabled {{ color: {disabled}; }}
QMenu::separator {{ height: 1px; background: {DIVIDER}; margin: 5px 8px; }}
QStatusBar {{ background: {BG}; color: {MUTED}; font-size: 12px; }}
QStatusBar::item {{ border: none; }}
QMessageBox QLabel {{ color: {TEXT}; }}

/* the shell (main_window.py) */
#topbar, #rail, #pagehead {{ background: {BG}; }}
#brand {{ font-size: 15px; font-weight: 500; }}
#chip {{
    background: {SURFACE}; border: 1px solid {NEUTRAL[800]}; border-radius: {RADIUS}px;
    padding: 4px 10px; font-weight: 400; text-align: left;
}}
#chip:hover {{ background: {mix(TEXT, 0.07, SURFACE)}; }}
#chip:disabled {{ color: {disabled}; }}
QComboBox#chipCombo {{
    background: transparent; border: none; padding: 3px 22px 3px 4px; min-height: 18px;
}}
#chipKey {{ color: {mix(TEXT, 0.5)}; }}
#live {{ color: {mix(TEXT, 0.78)}; }}
#iconButton {{ padding: 0; min-width: 32px; max-width: 32px; min-height: 32px;
               max-height: 32px; }}
#pageSubtitle {{ color: {mix(TEXT, 0.58)}; }}
#rail {{ border: none; }}
#rail::item {{
    padding: 0 9px; margin: 1px 10px 1px 12px; border-radius: {RADIUS}px;
    color: {mix(TEXT, 0.82)}; border: none;
}}
#rail::item:hover {{ background: {hover}; }}
#rail::item:selected {{ background: {accent_tint}; color: {ACCENT}; }}
#rail::item:disabled {{ color: {mix(TEXT, 0.45)}; background: transparent; padding-bottom: 5px; }}
"""


def apply(app) -> None:
    """Give the whole application the Nocturne look."""
    app.setStyle("Fusion")  # stylesheets render the same way on every platform
    app.setPalette(palette())
    app.setFont(base_font())
    app.setStyleSheet(stylesheet())


def mark_primary(*buttons) -> None:
    """The page's main action: outlined in the accent instead of the divider."""
    for button in buttons:
        button.setProperty("primary", True)


__all__ = ["ACCENT", "ACCENT_RAMP", "BAD", "BG", "DIVIDER", "GOOD", "GRID", "INK", "INK_2",
           "MUTED", "NEUTRAL", "SERIES", "SURFACE", "TEXT", "WARN", "apply", "base_font",
           "heading_font", "icon", "icon_pixmap", "label_font", "mark_primary", "mix",
           "stylesheet"]
