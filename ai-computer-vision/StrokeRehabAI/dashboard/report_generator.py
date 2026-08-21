"""
report_generator.py
======================
Generates clinician-facing rehabilitation reports in four formats —
PDF (ReportLab), Excel (OpenPyXL), CSV, and JSON — each covering:
patient information, exercise summary, recovery metrics, joint
analysis, ROM analysis, compensation analysis, recommendations, and a
session summary.

Every generated report is logged to the `generated_reports` table
(see dashboard/db.py) so past exports are discoverable from the
Reports dashboard page.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dashboard.analytics import RecoveryAnalyticsEngine
from dashboard.clinical_metrics import ClinicalMetricsCalculator
from dashboard.db import get_connection
from dashboard.patient_manager import PatientManager
from dashboard.recommendation_engine import RecommendationEngine
from dashboard.session_manager import SessionManager
from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportDataBuilder:
    """Assembles the full report payload (patient info, exercise summary,
    recovery metrics, joint/ROM/compensation analysis, recommendations,
    session summaries) — shared by every export format below."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.patient_manager = PatientManager(db_path)
        self.session_manager = SessionManager(db_path)
        self.analytics_engine = RecoveryAnalyticsEngine(db_path)
        self.metrics_calculator = ClinicalMetricsCalculator()
        self.recommendation_engine = RecommendationEngine()

    def build(self, patient_id: int, session_limit: int = 50) -> Dict:
        patient = self.patient_manager.get(patient_id)
        if patient is None:
            raise ValueError(f"No patient found with id={patient_id}")

        sessions = self.session_manager.list_sessions(patient_id=patient_id, limit=session_limit)
        analytics = self.analytics_engine.compute_patient_analytics(patient_id, session_limit=session_limit)
        recommendations = self.recommendation_engine.generate(analytics, exercise_display_name=self._most_common_exercise(sessions))

        session_summaries = []
        joint_analysis: Dict[str, List[float]] = {}
        compensation_summary: Dict[str, int] = {}

        for session in sessions:
            detail = self.session_manager.get_session_detail(session.session_id)
            scores = self.metrics_calculator.compute_for_session(detail["frames"], detail["events"], detail["reps"])

            session_summaries.append({
                "session_id": session.session_id,
                "date": session.started_at,
                "duration_seconds": session.duration_seconds,
                "exercises_performed": session.exercise_name,
                "successful_repetitions": detail["successful_repetitions"],
                "incorrect_repetitions": detail["incorrect_repetitions"],
                "average_score": self.analytics_engine._session_mean_quality(detail) * 100.0,
                "recovery_metrics": scores.to_dict(),
            })

            for frame in detail["frames"]:
                for angle_name, value in (frame.get("joint_angles") or {}).items():
                    if value is not None:
                        joint_analysis.setdefault(angle_name, []).append(value)

            for event in detail["events"]:
                if event.get("event_type") == "error" and event.get("error_type") in (
                    "trunk_compensation", "shoulder_hiking", "body_lean", "poor_alignment",
                ):
                    compensation_summary[event["error_type"]] = compensation_summary.get(event["error_type"], 0) + 1

        joint_analysis_summary = {
            name: {"min": min(values), "max": max(values), "avg": sum(values) / len(values), "rom": max(values) - min(values)}
            for name, values in joint_analysis.items()
        }

        return {
            "generated_at": datetime.now().isoformat(),
            "patient_information": self._patient_info(patient),
            "exercise_summary": self._exercise_summary(sessions),
            "recovery_metrics": analytics.to_dict(),
            "joint_analysis": joint_analysis_summary,
            "rom_analysis": {
                "average_rom_deg": analytics.average_rom_deg,
                "max_rom_deg": analytics.max_rom_deg,
                "min_rom_deg": analytics.min_rom_deg,
            },
            "compensation_analysis": compensation_summary,
            "recommendations": [asdict(r) for r in recommendations],
            "session_summary": session_summaries,
        }

    @staticmethod
    def _patient_info(patient) -> Dict:
        return {
            "patient_id": patient.patient_id,
            "full_name": patient.full_name,
            "age": patient.age,
            "gender": patient.gender,
            "affected_side": patient.affected_side,
            "diagnosis": patient.diagnosis,
            "physiotherapist": patient.physiotherapist,
            "treatment_start_date": patient.treatment_start_date,
        }

    @staticmethod
    def _exercise_summary(sessions) -> Dict:
        exercise_counts: Dict[str, int] = {}
        for s in sessions:
            exercise_counts[s.exercise_name] = exercise_counts.get(s.exercise_name, 0) + 1
        return {
            "total_sessions": len(sessions),
            "exercises_practiced": exercise_counts,
            "total_reps": sum(s.total_reps or 0 for s in sessions),
        }

    @staticmethod
    def _most_common_exercise(sessions) -> str:
        if not sessions:
            return "the assigned exercise"
        counts: Dict[str, int] = {}
        for s in sessions:
            counts[s.exercise_name] = counts.get(s.exercise_name, 0) + 1
        return max(counts, key=counts.get)


class ReportGenerator:
    """Exports a ReportDataBuilder payload to PDF, Excel, CSV, or JSON,
    and logs each generated file to the `generated_reports` table."""

    def __init__(self, db_path: str, output_dir: str = "outputs/evaluation_reports/clinical_reports"):
        self.db_path = db_path
        self.output_dir = ensure_dir(output_dir)
        self.data_builder = ReportDataBuilder(db_path)

    @staticmethod
    def _slugify(text: str) -> str:
        """Filesystem-safe (including Windows-safe) slug: keeps only
        alphanumerics, replacing everything else with underscores, and
        collapses repeated underscores. Windows forbids `: * ? " < > |`
        in filenames; patient names can plausibly contain any of these
        (e.g. via free-text entry), so we don't just handle spaces."""
        import re

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
        return slug or "patient"

    def generate(self, patient_id: int, report_format: str, session_limit: int = 50) -> Path:
        data = self.data_builder.build(patient_id, session_limit=session_limit)
        patient_name_slug = self._slugify(data["patient_information"]["full_name"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{patient_name_slug}_report_{timestamp}"

        format_dispatch = {
            "pdf": self._generate_pdf,
            "excel": self._generate_excel,
            "csv": self._generate_csv,
            "json": self._generate_json,
        }
        builder = format_dispatch.get(report_format.lower())
        if builder is None:
            raise ValueError(f"Unknown report format '{report_format}'. Options: {list(format_dispatch.keys())}")

        output_path = builder(data, base_name)
        self._log_report(patient_id, report_format, output_path)
        logger.info("Generated %s report for patient_id=%d: %s", report_format, patient_id, output_path)
        return output_path

    def _log_report(self, patient_id: int, report_format: str, output_path: Path) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO generated_reports (patient_id, report_type, file_path) VALUES (?, ?, ?)",
                (patient_id, report_format.lower(), str(output_path)),
            )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _generate_json(self, data: Dict, base_name: str) -> Path:
        output_path = self.output_dir / f"{base_name}.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return output_path

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _generate_csv(self, data: Dict, base_name: str) -> Path:
        output_path = self.output_dir / f"{base_name}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)

            writer.writerow(["Patient Information"])
            for key, value in data["patient_information"].items():
                writer.writerow([key, value])
            writer.writerow([])

            writer.writerow(["Recovery Metrics"])
            for key, value in data["recovery_metrics"].items():
                writer.writerow([key, value])
            writer.writerow([])

            writer.writerow(["Session Summary"])
            writer.writerow(["Session ID", "Date", "Duration (s)", "Exercise", "Successful Reps", "Incorrect Reps", "Average Score"])
            for session in data["session_summary"]:
                writer.writerow([
                    session["session_id"], session["date"], session["duration_seconds"],
                    session["exercises_performed"], session["successful_repetitions"],
                    session["incorrect_repetitions"], f"{session['average_score']:.1f}",
                ])
            writer.writerow([])

            writer.writerow(["Recommendations"])
            for rec in data["recommendations"]:
                writer.writerow([rec["priority"], rec["category"], rec["text"]])

        return output_path

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------

    def _generate_excel(self, data: Dict, base_name: str) -> Path:
        from openpyxl import Workbook
        from openpyxl.styles import Font

        output_path = self.output_dir / f"{base_name}.xlsx"
        wb = Workbook()

        summary_sheet = wb.active
        summary_sheet.title = "Patient Summary"
        self._write_key_value_sheet(summary_sheet, "Patient Information", data["patient_information"])
        self._write_key_value_sheet(summary_sheet, "Recovery Metrics", data["recovery_metrics"], start_row=summary_sheet.max_row + 2)

        sessions_sheet = wb.create_sheet("Session History")
        headers = ["Session ID", "Date", "Duration (s)", "Exercise", "Successful Reps", "Incorrect Reps", "Average Score"]
        sessions_sheet.append(headers)
        for cell in sessions_sheet[1]:
            cell.font = Font(bold=True)
        for session in data["session_summary"]:
            sessions_sheet.append([
                session["session_id"], session["date"], session["duration_seconds"],
                session["exercises_performed"], session["successful_repetitions"],
                session["incorrect_repetitions"], round(session["average_score"], 1),
            ])

        joints_sheet = wb.create_sheet("Joint Analysis")
        joints_sheet.append(["Joint / Angle", "Min (deg)", "Max (deg)", "Avg (deg)", "ROM (deg)"])
        for cell in joints_sheet[1]:
            cell.font = Font(bold=True)
        for name, stats in data["joint_analysis"].items():
            joints_sheet.append([name, round(stats["min"], 1), round(stats["max"], 1), round(stats["avg"], 1), round(stats["rom"], 1)])

        recs_sheet = wb.create_sheet("Recommendations")
        recs_sheet.append(["Priority", "Category", "Recommendation"])
        for cell in recs_sheet[1]:
            cell.font = Font(bold=True)
        for rec in data["recommendations"]:
            recs_sheet.append([rec["priority"], rec["category"], rec["text"]])

        wb.save(output_path)
        return output_path

    @staticmethod
    def _write_key_value_sheet(sheet, title: str, data: Dict, start_row: int = 1) -> None:
        from openpyxl.styles import Font

        sheet.cell(row=start_row, column=1, value=title).font = Font(bold=True, size=12)
        row = start_row + 1
        for key, value in data.items():
            if isinstance(value, dict):
                continue
            sheet.cell(row=row, column=1, value=str(key))
            sheet.cell(row=row, column=2, value=value if isinstance(value, (int, float, str)) else str(value))
            row += 1

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _generate_pdf(self, data: Dict, base_name: str) -> Path:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        output_path = self.output_dir / f"{base_name}.pdf"
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("StrokeRehabAI — Rehabilitation Progress Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {data['generated_at']}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Patient Information", styles["Heading2"]))
        patient_table_data = [[str(k).replace("_", " ").title(), str(v)] for k, v in data["patient_information"].items()]
        story.append(self._styled_table(patient_table_data))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Recovery Metrics", styles["Heading2"]))
        metrics_table_data = [[str(k).replace("_", " ").title(), f"{v:.1f}" if isinstance(v, float) else str(v)] for k, v in data["recovery_metrics"].items()]
        story.append(self._styled_table(metrics_table_data))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("ROM Analysis", styles["Heading2"]))
        rom_data = [[str(k).replace("_", " ").title(), f"{v:.1f}\u00b0"] for k, v in data["rom_analysis"].items()]
        story.append(self._styled_table(rom_data))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Compensation Analysis", styles["Heading2"]))
        if data["compensation_analysis"]:
            comp_data = [["Compensation Type", "Event Count"]] + [[k.replace("_", " ").title(), v] for k, v in data["compensation_analysis"].items()]
            story.append(self._styled_table(comp_data, header=True))
        else:
            story.append(Paragraph("No compensation events recorded.", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Recommendations", styles["Heading2"]))
        for rec in data["recommendations"]:
            story.append(Paragraph(f"\u2022 [{rec['priority'].upper()}] {rec['text']}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Session Summary", styles["Heading2"]))
        session_header = ["Date", "Duration (s)", "Exercise", "Success", "Incorrect", "Score"]
        session_rows = [session_header] + [
            [s["date"][:10], f"{s['duration_seconds'] or 0:.0f}", s["exercises_performed"],
             s["successful_repetitions"], s["incorrect_repetitions"], f"{s['average_score']:.0f}"]
            for s in data["session_summary"][:20]
        ]
        story.append(self._styled_table(session_rows, header=True))

        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        doc.build(story)
        return output_path

    @staticmethod
    def _styled_table(table_data: List[List], header: bool = False):
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        table = Table(table_data)
        style_commands = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        if header:
            style_commands += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        table.setStyle(TableStyle(style_commands))
        return table
