"""Desktop app: pages driven like a user would, on a headless (offscreen) display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QPoint, QSettings, Qt  # noqa: E402

from fungus_cv.capture import permissions  # noqa: E402
from fungus_cv.capture.permissions import NOT_APPLICABLE, CameraAccess  # noqa: E402
from fungus_cv.gui.image_view import ImageView  # noqa: E402
from fungus_cv.gui.main_window import MainWindow  # noqa: E402
from fungus_cv.gui.qt_util import bgr_to_qimage  # noqa: E402
from fungus_cv.gui.state import AppState  # noqa: E402
from fungus_cv.measure.geometry import Annotations  # noqa: E402
from fungus_cv.storage import Experiment  # noqa: E402

from . import synthetic as syn  # noqa: E402
from .test_pipeline import build_experiment  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(permissions, "camera_access", lambda **_: CameraAccess(NOT_APPLICABLE))
    from fungus_cv.capture import camera as camera_module

    monkeypatch.setattr(camera_module, "list_cameras", lambda **_: [])
    state = AppState(QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
    win = MainWindow(state)
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.fixture
def dye_experiment(experiment):
    build_experiment(experiment, minutes=range(0, 6), bump_at=-1)
    (experiment.root / "annotations.json").unlink()  # the user will click these
    return Experiment(experiment.root)


def page(win, title):
    win.go_to(title)
    return dict(win.pages)[title]


def test_bgr_to_qimage_colours():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = (255, 0, 0)  # blue in BGR
    q = bgr_to_qimage(img)
    assert (q.width(), q.height()) == (3, 2)
    assert q.pixelColor(0, 0).blue() == 255 and q.pixelColor(0, 0).red() == 0


def test_image_view_click_maps_to_pixels(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    view.show()
    view.set_image(np.zeros((300, 400, 3), np.uint8))
    with qtbot.waitSignal(view.clicked) as sig:
        qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(200, 150))
    x, y, button = sig.args
    assert abs(x - 200) < 3 and abs(y - 150) < 3 and button == Qt.LeftButton.value


def test_open_experiment_and_save_settings(window, dye_experiment, qtbot):
    window.open_experiment(dye_experiment.root)
    assert window.state.experiment.root == dye_experiment.root
    assert dye_experiment.config.name in window.experiment_label.text()

    exp_page = page(window, "Experiment")
    exp_page.f_interval.setText("5m")
    exp_page.f_marker_mm.setValue(syn.MARKER_MM)
    exp_page.f_lighting.setCurrentText("background")
    exp_page.save_settings()
    cfg = Experiment(dye_experiment.root).config
    assert cfg.capture.interval == 300
    assert cfg.analysis.markers.size_mm == syn.MARKER_MM
    assert cfg.analysis.lighting.method == "background"
    assert "# e.g. 5s, 10m, 2h, 1h30m" in dye_experiment.config_path.read_text()
    assert window.state.recent()[0].resolve() == dye_experiment.root.resolve()


def test_full_workflow_setup_analyze_report(window, dye_experiment, qtbot):
    window.open_experiment(dye_experiment.root)
    text = dye_experiment.config_path.read_text().replace("size_mm: null",
                                                          f"size_mm: {syn.MARKER_MM}")
    dye_experiment.config_path.write_text(text)
    window.state.reload()

    setup = page(window, "Set up measurement")
    qtbot.waitUntil(lambda: setup.analyzer is not None, timeout=20000)

    # Click base, tip, then the region corners, as a user would.
    setup._clicked(*syn.BASE_POINT, Qt.LeftButton.value)
    assert setup.tool() == "path"
    setup._clicked(*syn.TIP_POINT, Qt.LeftButton.value)
    for button in setup.tool_group.buttons():
        if button.property("tool") == "roi":
            button.setChecked(True)
    setup._tool_changed()
    for x, y in syn.ROI:
        setup._clicked(x, y, Qt.LeftButton.value)
    setup._clicked(0, 0, Qt.RightButton.value)  # right-click undoes the last corner
    setup._clicked(*syn.ROI[-1], Qt.LeftButton.value)
    setup.save_annotations()
    ann = Annotations.load(dye_experiment.root / "annotations.json")
    assert ann.base == syn.BASE_POINT and ann.tip == syn.TIP_POINT and len(ann.roi) == 4

    # Pick the dye colour on the last frame by dragging a box over it.
    qtbot.waitUntil(lambda: setup.analyzer is not None and setup.frame is not None,
                    timeout=20000)
    setup.frame_slider.setValue(5)
    setup._frame_selected()
    qtbot.waitUntil(lambda: "Loading" not in setup.message.text(), timeout=20000)
    for button in setup.tool_group.buttons():
        if button.property("tool") == "color":
            button.setChecked(True)
    setup._tool_changed()
    setup._dragged(560, 780, 640, 840)
    assert setup.ranges
    setup.save_colors()
    assert "hsv_ranges" in dye_experiment.config_path.read_text()

    analyze = page(window, "Analyze")
    qtbot.waitUntil(lambda: analyze.run_btn.isEnabled(), timeout=20000)
    analyze.run()
    qtbot.waitUntil(lambda: analyze.task is None, timeout=60000)
    assert analyze.table.rowCount() == 6, analyze.message.text()
    analyze.table.selectRow(3)
    assert analyze.overlay._has_image

    report = page(window, "Report")
    report.refresh()
    assert report.make_btn.isEnabled()
    report.make()
    qtbot.waitUntil(lambda: report.result is not None, timeout=60000)
    assert report.fits.rowCount() == 4
    assert report.chart._has_image


def test_setup_field_plots(window, dye_experiment, qtbot):
    window.open_experiment(dye_experiment.root)
    setup = page(window, "Set up measurement")
    qtbot.waitUntil(lambda: setup.analyzer is not None, timeout=20000)
    for button in setup.tool_group.buttons():
        if button.property("tool") == "plot":
            button.setChecked(True)
    setup._tool_changed()
    for poly in ([(100, 100), (300, 100), (300, 300)], [(400, 100), (600, 100), (600, 300)]):
        for x, y in poly:
            setup._clicked(x, y, Qt.LeftButton.value)
        setup.finish_plot()
    setup.plot_list.item(1).setText("north bed")
    ann = setup.build_annotations()
    assert [p.name for p in ann.plots] == ["plot1", "north bed"]
    assert ann.base is None


def test_setup_refuses_incomplete_annotations(window, dye_experiment, qtbot):
    window.open_experiment(dye_experiment.root)
    setup = page(window, "Set up measurement")
    qtbot.waitUntil(lambda: setup.analyzer is not None, timeout=20000)
    with pytest.raises(ValueError, match="base"):
        setup.build_annotations()


def test_doctor_page_lists_checks(window, qtbot):
    doctor = page(window, "Diagnostics")
    qtbot.waitUntil(lambda: doctor.table.rowCount() > 3, timeout=30000)
    names = [doctor.table.item(r, 0).text() for r in range(doctor.table.rowCount())]
    assert any("Camera permission" in n for n in names)


def test_camera_page_reports_no_cameras(window, qtbot):
    camera = page(window, "Camera")
    qtbot.waitUntil(lambda: "No cameras found" in camera.message.text(), timeout=30000)
    assert camera.device.count() == 0


def test_analyze_page_views_and_excludes_frames(window, qtbot, experiment):
    from fungus_cv.analyze import exclusions
    from fungus_cv.analyze.pipeline import analyze

    build_experiment(experiment, minutes=range(0, 4), bump_at=-1)
    analyze(Experiment(experiment.root))
    window.open_experiment(experiment.root)
    analyze_page = page(window, "Analyze")
    analyze_page.load_results()
    assert analyze_page.table.rowCount() == 4

    analyze_page.table.selectRow(2)
    assert analyze_page.overlay._has_image and analyze_page.exclude_btn.isEnabled()
    analyze_page.view_mode.setCurrentText("Aligned frame + mask")
    qtbot.waitUntil(lambda: analyze_page._analyzer is not None, timeout=20000)
    assert analyze_page.overlay._mask_item.pixmap().width() == syn.W

    analyze_page.exclude_selected("finger on the lens")
    marked = exclusions.load(Experiment(experiment.root))
    assert [e.reason for e in marked] == ["finger on the lens"]
    reason_col = [c[0] for c in analyze_page_columns()].index("excluded")
    assert analyze_page.table.item(2, reason_col).text() == "finger on the lens"
    assert analyze_page.table.item(2, 0).font().strikeOut()
    assert analyze_page.include_btn.isEnabled()
    analyze_page.include_selected()
    assert exclusions.load(Experiment(experiment.root)) == []
    assert not analyze_page.table.item(2, 0).font().strikeOut()


def analyze_page_columns():
    from fungus_cv.gui.pages.analyze import COLUMNS

    return COLUMNS
