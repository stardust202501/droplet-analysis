"""PyQt6 desktop interface for droplet analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from droplet_analysis import AnalysisConfig, SUPPORTED_EXTENSIONS, analyze_single_image


class AnalysisWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, input_file: Path, output_folder: Path, config: AnalysisConfig):
        super().__init__()
        self.input_file = input_file
        self.output_folder = output_folder
        self.config = config

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.progress.emit(0, 1, self.input_file.name)
            summary = analyze_single_image(
                self.input_file,
                self.output_folder,
                config=self.config,
            )
            self.progress.emit(1, 1, self.input_file.name)
            self.completed.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Droplet Analysis System")
        self.resize(1180, 760)
        self.setMinimumSize(920, 640)
        self.thread: QThread | None = None
        self.worker: AnalysisWorker | None = None
        self.last_output_folder: Path | None = None
        self.preview_path: Path | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("Droplet Analysis System")
        title.setObjectName("title")
        root.addWidget(title)

        controls = QFrame()
        controls.setObjectName("panel")
        control_layout = QGridLayout(controls)
        control_layout.setContentsMargins(14, 12, 14, 12)
        control_layout.setHorizontalSpacing(10)
        control_layout.setVerticalSpacing(10)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select a fluorescence image")
        input_button = QPushButton("Browse")
        input_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        input_button.clicked.connect(self.choose_input_file)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select an output folder")
        output_button = QPushButton("Browse")
        output_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        output_button.clicked.connect(self.choose_output_folder)

        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(20.0, 80.0)
        self.diameter_spin.setValue(40.0)
        self.diameter_spin.setDecimals(1)
        self.diameter_spin.setSuffix(" px")

        control_layout.addWidget(QLabel("Input image"), 0, 0)
        control_layout.addWidget(self.input_edit, 0, 1)
        control_layout.addWidget(input_button, 0, 2)
        control_layout.addWidget(QLabel("Output folder"), 1, 0)
        control_layout.addWidget(self.output_edit, 1, 1)
        control_layout.addWidget(output_button, 1, 2)
        control_layout.addWidget(QLabel("Droplet diameter"), 2, 0)
        control_layout.addWidget(self.diameter_spin, 2, 1)
        control_layout.setColumnStretch(1, 1)
        root.addWidget(controls)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("Start analysis")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.clicked.connect(self.start_analysis)
        self.open_button = QPushButton("Open results")
        self.open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_results)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.open_button)
        action_row.addStretch(1)
        root.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Ready")
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.status_label)
        root.addLayout(progress_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_frame = QFrame()
        preview_frame.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.addWidget(QLabel("Overlay preview"))
        self.preview_label = QLabel("The latest overlay will appear here")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(420, 320)
        self.preview_label.setObjectName("preview")
        preview_layout.addWidget(self.preview_label, 1)

        table_frame = QFrame()
        table_frame.setObjectName("panel")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(10, 10, 10, 10)
        self.summary_label = QLabel("No analysis results")
        table_layout.addWidget(self.summary_label)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Image", "Total", "Positive", "Negative", "Edge", "Separation", "Status"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self.show_selected_overlay)
        table_layout.addWidget(self.table, 1)

        splitter.addWidget(preview_frame)
        splitter.addWidget(table_frame)
        splitter.setSizes([600, 560])
        root.addWidget(splitter, 1)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f8; color: #17212b; font-size: 13px; }
            QLabel#title { font-size: 22px; font-weight: 600; color: #123b52; }
            QFrame#panel { background: white; border: 1px solid #d7dee5; border-radius: 6px; }
            QLineEdit, QDoubleSpinBox { background: white; border: 1px solid #b8c4ce; border-radius: 4px; padding: 7px; }
            QPushButton { background: #ffffff; border: 1px solid #aebbc5; border-radius: 4px; padding: 7px 13px; }
            QPushButton:hover { background: #edf3f6; }
            QPushButton:disabled { color: #8c979f; background: #eef1f3; }
            QPushButton#primaryButton { background: #176b87; color: white; border-color: #176b87; }
            QPushButton#primaryButton:hover { background: #125a72; }
            QLabel#preview { background: #0d1519; color: #d5dde2; border: 1px solid #c6d0d8; border-radius: 4px; }
            QProgressBar { background: white; border: 1px solid #b8c4ce; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: #2c8c77; }
            QTableWidget { background: white; border: 1px solid #c6d0d8; gridline-color: #e3e8ec; }
            QHeaderView::section { background: #eaf0f3; padding: 6px; border: 0; border-right: 1px solid #d4dde3; }
            """
        )

    def choose_input_file(self) -> None:
        extensions = " ".join(f"*{extension}" for extension in sorted(SUPPORTED_EXTENSIONS))
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select fluorescence image",
            "",
            f"Image files ({extensions});;All files (*)",
        )
        if not filename:
            return
        image_path = Path(filename)
        self.input_edit.setText(filename)
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(image_path.parent / "droplet_results"))

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.output_edit.setText(folder)

    def start_analysis(self) -> None:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        input_file = Path(input_text)
        if not input_file.is_file() or input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(self, "Input image", "Select a supported image file.")
            return
        if not output_text:
            QMessageBox.warning(self, "Output folder", "Select an output folder.")
            return
        output_folder = Path(output_text).resolve()
        self.last_output_folder = output_folder
        self.start_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)
        self.summary_label.setText("Analysis in progress")
        config = AnalysisConfig(diameter_px=self.diameter_spin.value())

        self.thread = QThread(self)
        self.worker = AnalysisWorker(input_file, output_folder, config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.analysis_completed)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)
        self.thread.start()

    def update_progress(self, done: int, total: int, name: str) -> None:
        value = int(round(done * 100 / total)) if total else 0
        self.progress_bar.setValue(value)
        self.status_label.setText(f"{done}/{total}  {name}")

    def analysis_completed(self, summary) -> None:
        self.populate_table(summary)
        errors = int((summary["status"] == "error").sum()) if not summary.empty else 0
        positive = int(summary["positive_droplets"].sum()) if not summary.empty else 0
        negative = int(summary["negative_droplets"].sum()) if not summary.empty else 0
        self.summary_label.setText(
            f"Images: {len(summary)}   Positive: {positive}   Negative: {negative}   Errors: {errors}"
        )
        self.progress_bar.setValue(100)
        self.status_label.setText("Completed")
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        successful = summary[summary["status"] == "ok"] if not summary.empty else summary
        if not successful.empty:
            self.preview_path = Path(successful.iloc[-1]["overlay_image"])
            self.refresh_preview()

    def analysis_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis error", message)

    def cleanup_worker(self) -> None:
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

    def populate_table(self, summary) -> None:
        self.table.setRowCount(len(summary))
        for row_index, row in summary.reset_index(drop=True).iterrows():
            values = [
                row["relative_path"],
                row["counted_droplets"],
                row["positive_droplets"],
                row["negative_droplets"],
                row["edge_droplets"],
                f"{row['cluster_separation']:.2f}" if row["status"] == "ok" else "",
                row["status"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, row.get("overlay_image", ""))
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()

    def show_selected_overlay(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.preview_path = Path(path)
            self.refresh_preview()

    def refresh_preview(self) -> None:
        if not self.preview_path or not self.preview_path.exists():
            return
        pixmap = QPixmap(str(self.preview_path))
        if pixmap.isNull():
            return
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refresh_preview()

    def open_results(self) -> None:
        if self.last_output_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_folder)))

    def closeEvent(self, event) -> None:
        if self.worker:
            QMessageBox.information(
                self,
                "Analysis in progress",
                "Wait for the current image analysis to finish before closing the program.",
            )
            event.ignore()
            return
        event.accept()


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("Droplet Analysis System")
    app.setWindowIcon(QIcon())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
