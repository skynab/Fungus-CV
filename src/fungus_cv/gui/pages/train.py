"""Train a segmentation model on a labeled dataset, watch it learn, and evaluate it."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fungus_cv.gui import theme
from fungus_cv.gui.chart import ChartLabel
from fungus_cv.gui.qt_util import preload_model_modules, run_task
from fungus_cv.gui.theme import SERIES

METRICS = [("iou", "IoU"), ("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall"),
           ("boundary_f1_2px", "Boundary F1 (2 px)")]


def _folder_row(line: QLineEdit, title: str) -> QHBoxLayout:
    browse = QPushButton("Browse…")

    def choose():
        folder = QFileDialog.getExistingDirectory(None, title, line.text())
        if folder:
            line.setText(folder)

    browse.clicked.connect(choose)
    row = QHBoxLayout()
    row.addWidget(line, 1)
    row.addWidget(browse)
    return row


def next_model_folder(dataset: Path) -> Path:
    """``models/<dataset>-v<N>`` next to the dataset's folder, with the first free N."""
    base = Path(dataset).resolve().parent.parent / "models"
    n = 1
    while (base / f"{Path(dataset).name}-v{n}").exists():
        n += 1
    return base / f"{Path(dataset).name}-v{n}"


class TrainPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.task = None
        self.history: list[dict] = []
        self.result = None
        self.evaluation = None

        self.dataset = QLineEdit()
        self.dataset.textChanged.connect(self._dataset_changed)
        self.output = QLineEdit()
        self.include_unreviewed = QCheckBox("Also train on unreviewed labels")
        self.encoder = QComboBox()
        self.encoder.addItems(["resnet34", "resnet18"])
        self.steps = QSpinBox()
        self.steps.setRange(10, 1_000_000)
        self.steps.setValue(3000)
        self.batch = QSpinBox()
        self.batch.setRange(1, 256)
        self.batch.setValue(8)
        self.patch = QSpinBox()
        self.patch.setRange(64, 2048)
        self.patch.setSingleStep(32)
        self.patch.setValue(384)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(1e-5, 1e-1)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(3e-4)
        self.color_jitter = QDoubleSpinBox()
        self.color_jitter.setRange(0, 5)
        self.color_jitter.setSingleStep(0.5)
        self.color_jitter.setValue(1.0)
        self.flip_vertical = QCheckBox("Allow upside-down flips")
        self.rotate90 = QCheckBox("Allow 90° rotations")
        self.pretrained = QCheckBox("Start from ImageNet weights (downloads once)")
        self.pretrained.setChecked(True)
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "mps", "cpu"])
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setSpecialValueText("off")
        self.patience.setToolTip("Stop when validation IoU hasn't improved for this many "
                                 "evaluations")
        self.resume = QCheckBox("Resume the interrupted run in this folder")
        self.resume.setVisible(False)
        self.output.textChanged.connect(self._output_changed)

        data_box = QGroupBox("Data")
        form = QFormLayout(data_box)
        form.addRow("Dataset", _folder_row(self.dataset, "Dataset folder"))
        form.addRow("Save model to", _folder_row(self.output, "New model folder"))
        form.addRow("", self.include_unreviewed)
        form.addRow("", self.resume)
        settings_box = QGroupBox("Training")
        sform = QFormLayout(settings_box)
        sform.addRow("Encoder", self.encoder)
        sform.addRow("Steps", self.steps)
        sform.addRow("Batch size", self.batch)
        sform.addRow("Patch size (px)", self.patch)
        sform.addRow("Learning rate", self.lr)
        sform.addRow("Colour augmentation ×", self.color_jitter)
        sform.addRow("", self.flip_vertical)
        sform.addRow("", self.rotate90)
        sform.addRow("", self.pretrained)
        sform.addRow("Device", self.device)
        sform.addRow("Stop early after", self.patience)

        self.start_btn = QPushButton("Start training")
        theme.mark_primary(self.start_btn)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.message = QLabel("Choose a dataset with reviewed labels (see the Labels page).")
        self.message.setWordWrap(True)
        self.chart = ChartLabel("The loss and validation IoU appear here while training.")
        self.card = QLabel()
        self.card.setWordWrap(True)
        self.use_btn = QPushButton("Use this model for the experiment's target")
        self.use_btn.clicked.connect(self.use_model)
        self.use_btn.setEnabled(False)
        self.card_btn = QPushButton("Open model card")
        self.card_btn.clicked.connect(self._open_card)
        self.card_btn.setEnabled(False)

        self.eval_model = QLineEdit()
        self.eval_dataset = QLineEdit()
        self.eval_dataset.setPlaceholderText("labeled data NOT used for training")
        self.eval_btn = QPushButton("Evaluate")
        self.eval_btn.clicked.connect(self.evaluate)
        self.eval_table = QTableWidget(0, 1 + len(METRICS))
        self.eval_table.setHorizontalHeaderLabels(["Items"] + [m[1] for m in METRICS])
        self.eval_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.eval_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.eval_table.setMaximumHeight(110)
        eval_box = QGroupBox("Evaluate a model on labeled data")
        eform = QFormLayout(eval_box)
        eform.addRow("Model", _folder_row(self.eval_model, "Model folder"))
        eform.addRow("Dataset", _folder_row(self.eval_dataset, "Dataset folder"))
        eform.addRow(self.eval_btn)
        eform.addRow(self.eval_table)

        left = QVBoxLayout()
        left.addWidget(data_box)
        left.addWidget(settings_box)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        left.addLayout(buttons)
        left.addWidget(self.progress)
        left.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(420)
        right = QVBoxLayout()
        right.addWidget(self.message)
        right.addWidget(self.chart, 1)
        right.addWidget(self.card)
        card_buttons = QHBoxLayout()
        card_buttons.addWidget(self.use_btn)
        card_buttons.addWidget(self.card_btn)
        right.addLayout(card_buttons)
        right.addWidget(eval_box)
        layout = QHBoxLayout(self)
        layout.addWidget(left_widget)
        layout.addLayout(right, 1)
        state.experiment_changed.connect(lambda _: self._update_use())

    def on_shown(self) -> None:
        if not self.dataset.text():
            last = self.state.settings.value("last_dataset", "", str)
            if last:
                self.dataset.setText(last)

    def _output_changed(self, text: str) -> None:
        from fungus_cv.learn.train import CHECKPOINT_NAME

        interrupted = bool(text) and (Path(text) / CHECKPOINT_NAME).exists()
        self.resume.setVisible(interrupted)
        self.resume.setChecked(interrupted)

    def _dataset_changed(self, text: str) -> None:
        if text and (Path(text) / "dataset.json").exists() and self.task is None:
            self.output.setText(str(next_model_folder(Path(text))))

    # --- training --------------------------------------------------------------------------

    def config(self):
        from fungus_cv.learn.train import TrainConfig

        return TrainConfig.with_color_jitter(
            self.color_jitter.value(), encoder=self.encoder.currentText(),
            pretrained=self.pretrained.isChecked(), patch_px=self.patch.value(),
            batch_size=self.batch.value(), steps=self.steps.value(),
            learning_rate=self.lr.value(), reviewed_only=not self.include_unreviewed.isChecked(),
            flip_vertical=self.flip_vertical.isChecked(), rotate90=self.rotate90.isChecked(),
            device=self.device.currentText(), patience=self.patience.value())

    def start(self) -> None:
        from fungus_cv.learn.dataset import Dataset

        try:
            ds = Dataset.open(Path(self.dataset.text()))
        except (FileNotFoundError, ValueError) as exc:
            self._error(str(exc))
            return
        out = Path(self.output.text()) if self.output.text() else None
        if out is None:
            self._error("Choose where to save the model.")
            return
        cfg = self.config()
        cfg.eval_every = max(1, min(250, cfg.steps // 20 or 1))  # enough points for the chart
        self.message.setText("Loading PyTorch…")
        self.repaint()
        preload_model_modules({"model"})
        import fungus_cv.learn.train  # noqa: F401 - import on the main thread (see qt_util)

        self.history, self.result = [], None
        self.chart.set_image(None)
        self.card.setText("")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, cfg.steps)
        self.progress.setValue(0)
        self.message.setText(f"Training on {ds.name}…")

        resume = self.resume.isVisible() and self.resume.isChecked()

        def work(progress, should_stop):
            from fungus_cv.learn.train import train

            return train(ds, out, cfg, progress=progress, should_stop=should_stop,
                         resume=resume)

        self.task = run_task(work, self._trained, self._failed, self._progress)

    def cancel(self) -> None:
        if self.task is not None:
            self.task.stop()
            self.message.setText("Cancelling…")

    def _progress(self, entry: dict) -> None:
        self.history.append(entry)
        self.progress.setValue(entry["step"])
        iou = f", validation IoU {entry['val_iou']:.3f}" if "val_iou" in entry else ""
        self.message.setText(f"Step {entry['step']} of {self.progress.maximum()}: loss "
                             f"{entry['loss']:.4f}{iou}")
        self._draw_history(self.history)

    def _draw_history(self, history: list[dict]) -> None:
        import matplotlib.pyplot as plt

        from fungus_cv.analyze.report import _style

        steps = [h["step"] for h in history]
        fig, ax = plt.subplots(figsize=(7, 3.4), dpi=110)
        _style(ax)
        ax.plot(steps, [h["loss"] for h in history], color=SERIES[0], lw=2, marker="o", ms=3,
                label="training loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        if any("val_iou" in h for h in history):
            ax2 = ax.twinx()
            pts = [(h["step"], h["val_iou"]) for h in history if "val_iou" in h]
            ax2.plot(*zip(*pts), color=SERIES[1], lw=2, marker="o", ms=3,
                     label="validation IoU")
            ax2.set_ylabel("Validation IoU")
            ax2.set_ylim(0, 1)
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8,
                      loc="center right")
        fig.tight_layout()
        self.chart.set_figure(fig)

    def _finish(self) -> None:
        self.task = None
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)

    def _failed(self, message: str) -> None:
        self._finish()
        self._output_changed(self.output.text())  # a cancelled run can be resumed
        self._error(message)

    def _trained(self, result) -> None:
        self._finish()
        self.result = result
        self._dataset_changed(self.dataset.text())  # the next run gets a new folder
        self.show_card(result.out_dir)
        self.eval_model.setText(str(result.out_dir))
        self.message.setText(f"Saved {result.out_dir}. Check it on labeled data it has not "
                             "seen (below) before trusting it.")

    def show_card(self, model_dir: Path) -> None:
        card = json.loads((Path(model_dir) / "model.json").read_text(encoding="utf-8"))
        ds, training = card["dataset"], card["training"]
        val = card["validation"]["summary"]
        lines = [f"<b>{Path(model_dir).name}</b>: {card['architecture']['encoder']}, "
                 f"{len(ds['train_items'])} training / {len(ds['val_items'])} validation "
                 f"items ({ds['split']}), {training['seconds'] / 60:.1f} min on "
                 f"{training['device']}"]
        if val:
            lines.append(f"Validation IoU {val['iou_mean']:.3f} (lowest {val['iou_min']:.3f}), "
                         f"boundary F1 {val['boundary_f1_2px_mean']:.3f}, threshold "
                         f"{card['threshold']:.2f}. Validation also picked the checkpoint, so "
                         "these numbers are optimistic.")
        else:
            lines.append("No validation data: the model is unchecked.")
        per_class = card["validation"].get("per_class") or {}
        if per_class:
            lines.append("Per class: " + ", ".join(
                f"{name} IoU {stats['iou_mean']:.3f}" for name, stats in per_class.items()))
        self.card.setText("<br>".join(lines))
        self._draw_history(training["history"])
        self.card_btn.setEnabled(True)
        self._update_use()

    def _open_card(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.out_dir / "model.json")))

    def _update_use(self) -> None:
        self.use_btn.setEnabled(self.result is not None and self.state.experiment is not None)

    def use_model(self) -> None:
        import yaml

        from fungus_cv.config import ExperimentConfig, set_yaml_value

        exp = self.state.experiment
        if exp is None or self.result is None:
            return
        text = exp.config_path.read_text(encoding="utf-8")
        text = set_yaml_value(text, "analysis.target.method", "model")
        text = set_yaml_value(text, "analysis.target.model.path", str(self.result.out_dir))
        ExperimentConfig.model_validate(yaml.safe_load(text))
        exp.config_path.write_text(text, encoding="utf-8")
        self.state.reload()
        self.message.setText(f"The target is now segmented with {self.result.out_dir.name}. "
                             "Run Analyze.")

    # --- evaluation ------------------------------------------------------------------------

    def evaluate(self) -> None:
        from fungus_cv.learn.dataset import Dataset

        try:
            ds = Dataset.open(Path(self.eval_dataset.text()))
        except (FileNotFoundError, ValueError) as exc:
            self._error(str(exc))
            return
        model = Path(self.eval_model.text())
        preload_model_modules({"model"})
        import fungus_cv.learn.train  # noqa: F401
        self.eval_btn.setEnabled(False)
        self.message.setText("Evaluating…")
        device = self.device.currentText()

        def work(progress, should_stop):
            from fungus_cv.learn.train import evaluate_model

            return evaluate_model(model, ds, device=device)

        run_task(work, self._evaluated, self._eval_failed)

    def _eval_failed(self, message: str) -> None:
        self.eval_btn.setEnabled(True)
        self._error(message)

    def _evaluated(self, result) -> None:
        rows, summary = result
        self.eval_btn.setEnabled(True)
        self.evaluation = summary
        table_rows = [("all", summary["all"]), ("not used in training",
                                                summary["not_in_training_set"])]
        self.eval_table.setRowCount(len(table_rows))
        for r, (name, stats) in enumerate(table_rows):
            self.eval_table.setItem(r, 0, QTableWidgetItem(f"{name} (n={stats['n']})"))
            for c, (key, _) in enumerate(METRICS, 1):
                value = stats.get(f"{key}_mean")
                self.eval_table.setItem(r, c, QTableWidgetItem("" if value is None
                                                               else f"{value:.3f}"))
        note = ""
        if summary["all"]["n"] and not summary["not_in_training_set"]["n"]:
            note = " Every item was used in training, so these numbers are optimistic."
        self.message.setText(f"Evaluated {summary['all']['n']} item(s) at threshold "
                             f"{summary['threshold']:.2f}.{note}")

    def _error(self, message: str) -> None:
        self.message.setText(f"<span style='color:#b00020'>{message}</span>")
