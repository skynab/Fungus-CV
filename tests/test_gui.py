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


def test_background_results_reach_lambda_callbacks(qtbot):
    """Every background result reaches its callback, including lambdas."""
    from fungus_cv.gui.qt_util import run_task

    results, errors = [], []
    for i in range(30):
        run_task(lambda p, s, i=i: i, lambda r: results.append(r))
        run_task(lambda p, s: 1 / 0, lambda r: None, lambda m: errors.append(m))
    qtbot.waitUntil(lambda: len(results) == 30 and len(errors) == 30, timeout=20000)
    assert sorted(results) == list(range(30)) and "ZeroDivisionError" in errors[0]


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


def test_sidebar_sections_skip_headers(window):
    rows = [window.nav.item(r) for r in range(window.nav.count())]
    headers = [i.text() for i in rows if not i.flags() & Qt.ItemIsSelectable]
    assert headers[:2] == ["CAPTURE", "MEASURE"]
    window.go_to("SAM prompts")
    assert window.stack.currentWidget() is dict(window.pages)["SAM prompts"]
    window.nav.setCurrentRow(0)  # clicking a header changes nothing
    assert window.stack.currentWidget() is dict(window.pages)["SAM prompts"]


def fake_sam(monkeypatch, calls):
    from fungus_cv.gui.pages import prompt as prompt_page
    from fungus_cv.segment.sam2 import Sam2VideoSegmenter

    def segment_single(self, image, prompt):
        calls.append((self.model_name, prompt))
        mask = np.zeros(image.shape[:2], bool)
        x, y = (int(v) for v in prompt.points[0])
        mask[y - 20:y + 20, x - 20:x + 20] = True
        return mask

    monkeypatch.setattr(Sam2VideoSegmenter, "segment_single", segment_single)
    monkeypatch.setattr(prompt_page, "preload_model_modules", lambda methods: None)


def test_prompt_page_clicks_preview_and_save(window, qtbot, experiment, monkeypatch):
    from fungus_cv.segment.prompts import Prompts

    calls = []
    fake_sam(monkeypatch, calls)
    build_experiment(experiment, minutes=range(0, 4), bump_at=-1)
    window.open_experiment(experiment.root)
    prompts_page = page(window, "SAM prompts")
    qtbot.waitUntil(lambda: prompts_page.frame is not None, timeout=20000)
    assert prompts_page.index == 3  # starts on the last frame

    prompts_page.model.setCurrentText("facebook/sam2.1-hiera-tiny")
    prompts_page._clicked(600, 800, Qt.LeftButton.value)
    qtbot.waitUntil(lambda: prompts_page.mask is not None, timeout=20000)
    assert prompts_page.mask.sum() == 1600 and calls[-1][0].endswith("tiny")
    prompts_page._clicked(300, 300, Qt.RightButton.value)
    qtbot.waitUntil(lambda: calls[-1][1].labels == [1, 0], timeout=20000)
    prompts_page.save_prompt()

    saved = Prompts.load(experiment.root / "prompts.json")
    frame_file = Experiment(experiment.root).read_frames()[3]["file"]
    assert [p.frame_file for p in saved.frames] == [frame_file]
    assert saved.frames[0].labels == [1, 0] and prompts_page.prompted.count() == 1

    prompts_page.use_sam()
    cfg = Experiment(experiment.root).config.analysis.target
    assert cfg.method == "sam2" and cfg.sam2.model.endswith("tiny")

    qtbot.waitUntil(lambda: not prompts_page.loading, timeout=20000)
    assert prompts_page.index == 3  # reloading the config keeps the frame the user is on
    prompts_page.show_frame(1)
    qtbot.waitUntil(lambda: prompts_page.index == 1, timeout=20000)
    assert prompts_page.points == [] and not prompts_page.remove_btn.isEnabled()
    prompts_page._jump(prompts_page.prompted.item(0))
    qtbot.waitUntil(lambda: prompts_page.index == 3, timeout=20000)
    assert prompts_page.labels == [1, 0]  # the saved prompt comes back
    prompts_page.remove_prompt()
    assert Prompts.load(experiment.root / "prompts.json").frames == []


def test_image_view_paint_strokes(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    view.show()
    view.set_image(np.zeros((300, 400, 3), np.uint8))
    view.paint_enabled = True
    events = []
    view.stroke.connect(lambda x, y, b, phase: events.append((round(x), round(y), b, phase)))
    clicks = []
    view.clicked.connect(lambda *a: clicks.append(a))
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=QPoint(100, 100))
    qtbot.mouseMove(view.viewport(), QPoint(150, 120))
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=QPoint(150, 120))
    assert [e[3] for e in events][0] == ImageView.STROKE_PRESS
    assert events[-1][3] == ImageView.STROKE_RELEASE and not clicks
    assert abs(events[0][0] - 100) < 3 and abs(events[-1][0] - 150) < 3


def test_labels_page_add_paint_sam_save_rank(window, qtbot, experiment, tmp_path, monkeypatch):
    from fungus_cv.analyze.compare import list_runs
    from fungus_cv.analyze.pipeline import analyze
    from fungus_cv.learn.dataset import Dataset

    from .test_active import SOFT, blueness_prob
    from .test_active import build as build_soft

    calls = []
    fake_sam(monkeypatch, calls)
    monkeypatch.setattr("fungus_cv.gui.pages.labels.preload_model_modules", lambda m: None)
    build_soft(experiment)
    analyze(Experiment(experiment.root))
    text = experiment.config_path.read_text().replace("lower: [95, 60, 30]",
                                                      "lower: [95, 150, 30]", 1)
    experiment.config_path.write_text(text)
    analyze(Experiment(experiment.root))
    loose, tight = (r.run_id for r in list_runs(Experiment(experiment.root)))
    window.open_experiment(experiment.root)

    labels = page(window, "Labels")
    labels.create_dataset(tmp_path / "ds")
    assert labels.dataset is not None and labels.add_btn.isEnabled()
    labels.add_frames(method="suggest", run=loose, against=tight, count=2)
    qtbot.waitUntil(lambda: not labels.busy, timeout=30000)
    assert labels.table.rowCount() == 2, labels.message.text()
    assert labels.index == 0 and labels.image is not None
    assert "priority" in labels.info.text()
    assert len(SOFT) == 2

    # Brush: paint a stroke, erase part of it, undo the erase.
    labels.mask[:] = False
    labels._stroke(20, 20, Qt.LeftButton.value, ImageView.STROKE_PRESS)
    labels._stroke(60, 20, Qt.LeftButton.value, ImageView.STROKE_MOVE)
    labels._stroke(60, 20, Qt.LeftButton.value, ImageView.STROKE_RELEASE)
    painted = labels.mask.sum()
    assert painted > 40 * 17 and labels.mask[20, 40] and labels.dirty
    labels._stroke(40, 20, Qt.RightButton.value, ImageView.STROKE_PRESS)
    labels._stroke(40, 20, Qt.RightButton.value, ImageView.STROKE_RELEASE)
    assert not labels.mask[20, 40]
    labels.undo()
    assert labels.mask.sum() == painted

    # SAM clicks: a 40x40 square proposal, added to the painted stroke.
    labels.sam_mode.setChecked(True)
    assert not labels.view.paint_enabled
    labels._clicked(150, 200, Qt.LeftButton.value)
    qtbot.waitUntil(lambda: labels.proposal is not None, timeout=20000)
    labels.apply_proposal("add")
    assert labels.mask.sum() == painted + 1600 and labels.proposal is None

    first_id = labels.items[0].id
    labels.go(1)  # moving on saves the edited item
    ds = Dataset.open(tmp_path / "ds")
    assert ds.get(first_id).reviewed
    assert ds.load_mask(ds.get(first_id)).sum() == painted + 1600

    labels.unreviewed_only.setChecked(True)
    assert labels.table.rowCount() == 1
    labels.save_item()
    assert labels.table.rowCount() == 0

    labels.unreviewed_only.setChecked(False)
    for item in labels.dataset.items:
        item.reviewed = False
    labels.dataset.save()
    labels.rank(None, prob_fn=blueness_prob, threshold=0.5)
    qtbot.waitUntil(lambda: not labels.busy, timeout=30000)
    priorities = [i.priority for i in labels.items]
    assert all(p is not None for p in priorities) and priorities == sorted(priorities,
                                                                           reverse=True)

    other = LabelsPage_reopen(window, tmp_path / "ds")
    assert other.table.rowCount() == 2


def LabelsPage_reopen(window, path):  # noqa: N802 - reads like the page it builds
    from fungus_cv.gui.pages.labels import LabelsPage

    fresh = LabelsPage(window.state)
    fresh.on_shown()  # reopens the last dataset
    assert fresh.dataset is not None and fresh.dataset.root.resolve() == path.resolve()
    return fresh


def test_train_page_trains_evaluates_uses_and_cancels(window, qtbot, experiment, tmp_path):
    pytest.importorskip("torchvision")
    from fungus_cv.analyze.compare import list_runs
    from fungus_cv.analyze.pipeline import analyze
    from fungus_cv.learn.dataset import Dataset
    from fungus_cv.learn.export import export_from_run

    build_experiment(experiment, minutes=range(0, 6), bump_at=-1)
    exp = Experiment(experiment.root)
    analyze(exp)
    ds = Dataset.create(tmp_path / "datasets" / "dye")
    export_from_run(exp, ds, list_runs(exp)[0].run_id, count=6)
    for item in ds.items:
        item.reviewed = True
    ds.save()
    window.open_experiment(experiment.root)

    train_page = page(window, "Train models")
    train_page.dataset.setText(str(ds.root))
    assert train_page.output.text().endswith("models/dye-v1")
    train_page.encoder.setCurrentText("resnet18")
    train_page.steps.setValue(20)
    train_page.batch.setValue(2)
    train_page.patch.setValue(96)
    train_page.pretrained.setChecked(False)
    train_page.device.setCurrentText("cpu")
    train_page.start()
    qtbot.waitUntil(lambda: train_page.task is None, timeout=120000)
    assert train_page.result is not None, train_page.message.text()
    assert (tmp_path / "models" / "dye-v1" / "model.pt").exists()
    assert len(train_page.history) >= 5 and train_page.chart.image is not None
    assert "validation items" in train_page.card.text()

    train_page.eval_dataset.setText(str(ds.root))
    train_page.evaluate()
    qtbot.waitUntil(lambda: train_page.evaluation is not None, timeout=120000)
    assert train_page.eval_table.rowCount() == 2
    assert train_page.eval_table.item(0, 0).text() == "all (n=6)"

    train_page.use_model()
    target = Experiment(experiment.root).config.analysis.target
    assert target.method == "model" and target.model.path.endswith("dye-v1")

    train_page.dataset.setText(str(ds.root))  # suggests dye-v2 now
    assert train_page.output.text().endswith("models/dye-v2")
    train_page.steps.setValue(100000)
    train_page.start()
    qtbot.waitUntil(lambda: len(train_page.history) >= 1 or train_page.task is None,
                    timeout=120000)
    train_page.cancel()
    qtbot.waitUntil(lambda: train_page.task is None, timeout=120000)
    assert "cancelled" in train_page.message.text()
    assert not (tmp_path / "models" / "dye-v2" / "model.pt").exists()
    if (tmp_path / "models" / "dye-v2" / "checkpoint.pt").exists():  # got past an evaluation
        assert train_page.resume.isVisible() and train_page.resume.isChecked()


def test_study_page_edit_save_run_reopen(window, qtbot, tmp_path):
    from fungus_cv.analyze.study import load_study

    from .test_study import fake_experiment

    for i in range(1, 4):
        fake_experiment(tmp_path, f"control{i}", 70 + i, 0.20 + 0.01 * i, 24, seed=i)
        fake_experiment(tmp_path, f"treated{i}", 70 + i, 0.40 + 0.01 * i, 24, seed=10 + i)
    study = page(window, "Study")
    study.new_study(tmp_path / "moss.yaml")
    for cond in ("control", "treated"):
        for i in range(1, 4):
            study.add_experiment(tmp_path / f"{cond}{i}", condition=cond)
    assert study._cell(0, 0) == "control1"  # stored relative to the study file
    study.metric.setCurrentText("coverage_pct")
    study.params.setText("K, r")
    study.reference.setCurrentText("control")
    study.time_unit.setCurrentText("h")
    study.bootstrap.setValue(50)
    study.run()
    qtbot.waitUntil(lambda: study.result is not None or "b00020" in study.message.text(),
                    timeout=120000)
    assert study.result is not None, study.message.text()
    spec = load_study(tmp_path / "moss.yaml")
    assert spec.reference == "control" and spec.params == ["K", "r"]
    assert len(spec.experiments) == 6 and spec.bootstrap == 50
    assert study.conditions.rowCount() == 4 and study.comparisons.rowCount() == 2
    assert study.replicates.rowCount() == 6
    assert study.curves.image is not None and study.parameters.image is not None
    assert "Welch" in study.methods.toPlainText()

    reopened = StudyPage_fresh(window)
    reopened.open_study(tmp_path / "moss.yaml")
    assert reopened.experiments.rowCount() == 6
    assert reopened.reference.currentText() == "control"
    assert reopened.params.text() == "K, r"

    study.experiments.setItem(0, 1, study.experiments.item(0, 1).__class__(""))
    assert not study.save()  # an empty condition is refused, the file is kept
    assert len(load_study(tmp_path / "moss.yaml").experiments) == 6


def StudyPage_fresh(window):  # noqa: N802
    from fungus_cv.gui.pages.study import StudyPage

    return StudyPage(window.state)


def test_validation_page_hand_measurements_and_suite(window, qtbot, experiment, tmp_path):
    import csv

    import yaml

    from fungus_cv.analyze.pipeline import analyze

    heights = build_experiment(experiment, minutes=range(0, 6), bump_at=-1)
    analyze(Experiment(experiment.root))
    window.open_experiment(experiment.root)
    validation = page(window, "Validation")

    template = tmp_path / "hand.csv"
    validation.template_count.setValue(6)
    validation.make_template(template)
    with open(template, newline="") as f:
        rows = list(csv.DictReader(f))
    files = [r["file"].split("/")[-1] for r in Experiment(experiment.root).read_frames()]
    for row in rows:
        row["value"] = heights[files.index(row["frame"])] * syn.MM_PER_PX
    with open(template, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation.validate()
    qtbot.waitUntil(lambda: validation.agreement is not None, timeout=60000)
    assert validation.agreement.n == 6 and "Bias" in validation.stats.text()
    assert validation.agreement_chart.image is not None

    validation.new_suite(tmp_path / "template_suite.yaml")
    assert (tmp_path / "template_suite.yaml").exists()
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"name": "gui", "cases": [
        {"name": "dye", "experiment": str(experiment.root), "checks": [
            {"type": "measurements", "hand": str(template),
             "expect": {"bias": {"abs_max": 0.5, "max_drift": 0.1}}}]}]}))
    validation.suite_path.setText(str(suite))
    validation.save_baseline.setChecked(True)
    validation.run_suite()
    qtbot.waitUntil(lambda: validation.suite_result is not None, timeout=120000)
    assert "1/1 checks passed" in validation.suite_summary.text()
    assert "baseline saved" in validation.suite_summary.text()
    assert validation.suite_table.rowCount() == 1
    assert validation.suite_table.item(0, 5).text() == "pass"
    validation.only_checked.setChecked(False)
    assert validation.suite_table.rowCount() > 5


def test_capture_page_measures_new_frames_live(window, qtbot, experiment):
    from datetime import timedelta

    from fungus_cv.storage import FrameRecord, encode_image, iso_utc

    from .test_pipeline import T0

    build_experiment(experiment, minutes=range(0, 3), bump_at=-1)
    window.open_experiment(experiment.root)
    capture = page(window, "Capture")
    capture.watch.setChecked(True)
    capture.measure_new_frames()
    capture.measure_new_frames()  # while one runs, the next is queued, not run twice
    qtbot.waitUntil(lambda: capture.watch_task is None and capture.watch_summary is not None,
                    timeout=60000)
    assert capture.watch_summary.processed + capture.watch_summary.skipped_existing == 3
    assert capture.live_chart.image is not None
    assert "Measured" in capture.watch_status.text()

    exp = Experiment(experiment.root)
    for minute in (3, 4):  # two new photos arrive
        ts = T0 + timedelta(minutes=minute)
        path = exp.save_bytes(encode_image(syn.scene(60 * minute), "png"), ts, "cam0", "png")
        exp.append_frame(FrameRecord(timestamp_utc=iso_utc(ts), camera="cam0", status="ok",
                                     file=exp.relative(path), width=syn.W, height=syn.H))
    capture.watch_summary = None
    capture.measure_new_frames()
    qtbot.waitUntil(lambda: capture.watch_summary is not None and capture.watch_task is None,
                    timeout=60000)
    assert capture.watch_summary.processed == 2 and capture.watch_summary.skipped_existing == 3

    (experiment.root / "annotations.json").unlink()
    capture.watch_summary = None
    capture.measure_new_frames()
    qtbot.waitUntil(lambda: capture.watch_task is None and "Not measured" in
                    capture.watch_status.text(), timeout=60000)
    assert "annotate" in capture.watch_status.text()


def test_file_menu_archives_the_experiment(window, qtbot, experiment, tmp_path):
    from fungus_cv.analyze import archive

    build_experiment(experiment, minutes=range(0, 3), bump_at=-1)
    window.open_experiment(experiment.root)
    out = tmp_path / "bundle.zip"
    window.archive_experiment(out=out, frames=False)
    qtbot.waitUntil(lambda: out.exists() and "Wrote" in window.statusBar().currentMessage(),
                    timeout=60000)
    assert archive.verify(out).ok
    assert "0 photo(s)" in window.statusBar().currentMessage()


def test_camera_selector_switches_annotations_and_results(window, qtbot, experiment):
    from fungus_cv.analyze.pipeline import analyze_all_cameras

    from .test_multi_camera import build_two_cameras

    exp = build_two_cameras(experiment)
    analyze_all_cameras(exp)
    window.open_experiment(exp.root)
    assert window.camera_box.isVisible() and window.camera_box.count() == 2
    assert window.state.camera == "cam0"

    analyze_page = page(window, "Analyze")
    analyze_page.load_results()
    assert analyze_page.table.rowCount() == 4
    first = float(analyze_page.rows[-1]["extent_mm"])

    window.camera_box.setCurrentText("cam1")
    assert window.state.camera == "cam1"
    analyze_page.load_results()
    second = float(analyze_page.rows[-1]["extent_mm"])
    assert second < first * 0.9  # cam1 sees the towel foreshortened

    setup = page(window, "Set up measurement")
    qtbot.waitUntil(lambda: setup.analyzer is not None, timeout=20000)
    assert setup.analyzer.camera == "cam1"
    setup.save_annotations()
    assert (exp.root / "annotations_cam1.json").exists()

    report = page(window, "Report")
    report.refresh()
    report.make()
    qtbot.waitUntil(lambda: report.result is not None, timeout=60000)
    assert report.result.files[0].parent.parent.name == "cam1"


def test_file_menu_makes_and_opens_a_demo(window, qtbot, tmp_path):
    window.make_demo(tmp_path / "demo")
    qtbot.waitUntil(lambda: window.state.experiment is not None, timeout=60000)
    assert window.state.experiment.root == tmp_path / "demo"
    assert len(window.state.experiment.read_frames()) == 30


def test_labels_page_edits_one_class_at_a_time(window, qtbot, tmp_path):
    from fungus_cv.learn.dataset import Dataset

    ds = Dataset.create(tmp_path / "ds")
    stem = np.zeros((80, 100), bool)
    stem[:, 40:60] = True
    ds.add(np.full((80, 100, 3), 120, np.uint8), stem, "p1")
    ds.save()
    labels = page(window, "Labels")
    labels.open_dataset(tmp_path / "ds")
    assert not labels.class_box.isEnabled()  # one class: nothing to choose
    labels.add_class("moss")
    assert labels.class_box.isEnabled() and labels.current_class == "moss"
    assert labels.dataset.classes == ["target", "moss"]
    assert not labels.mask.any() and labels.other is not None and labels.other.sum() == 1600

    labels._stroke(50, 20, Qt.LeftButton.value, ImageView.STROKE_PRESS)
    labels._stroke(50, 20, Qt.LeftButton.value, ImageView.STROKE_RELEASE)
    labels.class_box.setCurrentText("target")  # switching keeps the moss edit
    assert labels.mask.sum() == 1600
    labels.save_item()
    saved = Dataset.open(tmp_path / "ds")
    item = saved.get("p1")
    assert saved.load_mask(item, "moss")[20, 50] and saved.load_mask(item, "target").sum() == 1600
    assert item.reviewed


def test_capture_page_shows_run_health(window, qtbot, experiment):
    from fungus_cv.capture.health import write_heartbeat

    build_experiment(experiment, minutes=range(0, 3), bump_at=-1)
    window.open_experiment(experiment.root)
    capture = page(window, "Capture")
    qtbot.waitUntil(lambda: capture.health_report is not None, timeout=20000)
    assert "status.json" in capture.health.text()  # never captured here: a warning
    assert capture.health_timer.isActive()

    write_heartbeat(Experiment(experiment.root), state="finished", stopped_reason="duration")
    capture.health_report = None
    capture.check_health()
    qtbot.waitUntil(lambda: capture.health_report is not None, timeout=20000)
    assert "finished (duration)" in capture.health.text()


def test_report_page_spread_sensitivity_and_summary(window, qtbot, experiment, monkeypatch):
    from fungus_cv.analyze import sensitivity
    from fungus_cv.analyze.pipeline import analyze
    from fungus_cv.gui.pages import report as report_module

    opened = []
    monkeypatch.setattr(report_module.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toLocalFile()))
    build_experiment(experiment, minutes=range(0, 6), bump_at=-1)
    analyze(Experiment(experiment.root))
    window.open_experiment(experiment.root)
    report = page(window, "Report")
    report.refresh()
    assert report.spread_btn.isEnabled()

    report.make_spread()
    qtbot.waitUntil(lambda: report.spread_btn.isEnabled() and "Spread" in
                    report.message.text(), timeout=60000)
    assert report.spread_view._has_image

    # One quick variant keeps the test short; the page runs the default list.
    one = [sensitivity.Variant("open_px_0", "no speck removal",
                               {"target.color.open_px": 0})]
    monkeypatch.setattr(sensitivity, "default_variants", lambda exp: one)
    report.make_sensitivity()
    qtbot.waitUntil(lambda: report.sensitivity_btn.isEnabled() and "Sensitivity of" in
                    report.message.text(), timeout=120000)
    assert "no speck removal" in report.message.text()
    assert report.sensitivity_view._has_image

    report.make_summary()
    qtbot.waitUntil(lambda: bool(opened), timeout=120000)
    assert opened[0].endswith("summary.html") and "Wrote summary.html" in report.message.text()


def test_report_page_hours_and_daily(window, qtbot, tmp_path):
    from .test_daily import outdoor

    exp = outdoor(tmp_path / "outdoor", days=4)
    window.open_experiment(exp.root)
    report = page(window, "Report")
    report.refresh()
    report.metric.setCurrentText("coverage_pct")
    report.hours.setText("25-3")
    report.make()
    assert "hours must look like" in report.message.text()
    report.hours.setText("10-14")
    report.daily.setCurrentText("median")
    report.make()
    qtbot.waitUntil(lambda: report.result is not None, timeout=60000)
    assert report.result.daily == "median" and report.result.n_used == 4
    assert "4 days (daily median) used" in report.message.text()
